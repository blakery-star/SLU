#coding=utf8
import sys, os, time, gc, json

install_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(install_path)

from utils.args import init_args
from utils.initialization import *
from utils.batch import from_example_list
from utils.vocab import PAD
from utils.example_for_csc import CSCDataset

# initialization params, output path, logger, random seed and torch.device
args = init_args(sys.argv[1:])

if args.csc_model != 'sound':
    if args.csc_train:
        raise ValueError("Only similar-sound-csc model can be trained.")
elif (not args.csc_train) and (args.csc_pretrained is None):
        print("Default pretrained vocab for similar-sound-csc will be loaded.")

set_random_seed(args.seed)
print("Initialization finished ...")
print("Random seed is set to %d" % (args.seed))
print("Use GPU with index %s" % (args.device) if args.device >= 0 else "Use CPU as target torch device")

start_time = time.time()
train_path = os.path.join(args.dataroot, 'train.json')
dev_path = os.path.join(args.dataroot, 'development.json')

CSCDataset.configuration(args.dataroot, train_path=train_path, word2vec_path=args.word2vec_path)
csc_dev = CSCDataset.load_dataset(dev_path, mode="pred")
csc_train_d = CSCDataset.load_dataset(train_path, mode="pred")
if args.csc_model == 'sound':
    csc_train = CSCDataset.load_dataset(train_path, mode="train")

print("Load dataset and database finished, cost %.4fs ..." % (time.time() - start_time))
if args.csc_model == 'sound':
    print("Dataset size: train -> %d ; dev -> %d" % (len(csc_train), len(csc_dev)))
else:
    print("Dataset size: dev_1 -> %d ; dev_2 -> %d " % (len(csc_train_d), len(csc_dev)))

args.vocab_size = CSCDataset.word_vocab.vocab_size
args.pad_idx = CSCDataset.word_vocab[PAD]
args.num_tags = CSCDataset.label_vocab.num_tags
args.tag_pad_idx = CSCDataset.label_vocab.convert_tag_to_idx(PAD)

print("Creating corrector...")
testonasr_mode = False
if args.csc_model == 'Ernie4CSC':
    """
    Use our implementation of Ernie4CSC with torch.
    """

    """ from model.ErnieCSC.model import Ernie
    csc_model = Ernie()
    if args.csc_pretrained is None:
        args.csc_pretrained = "checkpoint/Ernie4CSC_model_best.pth"
    para_dict = torch.load(args.csc_pretrained)
    csc_model.load_state_dict(para_dict)
    csc_model.to(args.device) """

    raise ValueError("\'Ernie\' is recommended.")

elif args.csc_model == "MacBERT":
    """
    Use pycorrector-implemented MacBERT4CSC.
    """

    from pycorrector.macbert.macbert_corrector import MacBertCorrector
    if args.csc_pretrained is None:
        args.csc_pretrained = "shibing624/macbert4csc-base-chinese"
    csc_model = MacBertCorrector(args.csc_pretrained)
    corrector = csc_model.macbert_correct

elif args.csc_model == "Ernie":
    """
    Use pycorrector-implemented Ernie4CSC with paddle.
    """

    from pycorrector.ernie_csc.ernie_csc_corrector import ErnieCSCCorrector
    if args.csc_pretrained is None:
        args.csc_pretrained = "csc-ernie-1.0"
    csc_model = ErnieCSCCorrector(args.csc_pretrained)
    corrector = csc_model.ernie_csc_correct

elif args.csc_model == "sound":
    """
    Use our implementation of our method, i.e. similar-sound-CSC.
    """

    from model.correcting_model import SimCSC
    csc_model = SimCSC(log="checkpoints/corrected.txt")
    if args.csc_train:
        print("training similar-sound-csc....")
        csc_model.learn(csc_train)
        csc_model.save_vocab()
    else:
        if args.csc_pretrained is None:
            csc_model.load_vocab()
        else:
            csc_model.load_vocab(args.csc_pretrained)
        print("similar-sound-csc loaded.")
    
    corrector = csc_model.correct

elif args.csc_model == "NoCSC":
    """
    Test on asr_1best and manuscript.
    """

    testonasr_mode = True

    
csc_train = csc_train_d
print("Corrector prepared.")

csc_train_name = "train_{}_csc".format(args.csc_model)
csc_dev_name = "dev_{}_csc".format(args.csc_model)
if args.use_history:
    csc_train_name += "_his"
    csc_dev_name += "_his"
csc_train_name = csc_train_name + ".json"
csc_dev_name = csc_dev_name + ".json"


def csc_process(dataset, path, save_path=None, filename=None, save=True):
    if (save and (save_path is None or filename is None)):
        raise ValueError("save_path or filename is None.")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_path = os.path.join(save_path, filename)

    corrections = []
    if save:
        csc_json = json.load(open(path, 'r', encoding='utf8'))
    labels = []
    for i in range(len(dataset)):
        batch = dataset[i]
        for j in range(len(batch)):
            data = batch[j]
            if not testonasr_mode:
                pred, _ = corrector(data.utt)
            else:
                pred = data.utt
            label = data.label
            corrections.append(pred)
            labels.append(label)
            if save:
                csc_json[i][j]['asr_1best'] = pred
    
    if save:
        json.dump(csc_json, open(save_path, 'w', encoding='utf8'), ensure_ascii=False, indent=4)

    return CSCDataset.evaluator.acc(corrections, labels)


print("Evaluating on training dataset (as dev_1)...")
if args.csc_model == "sound":
    with open("checkpoints/corrected.txt", 'a', encoding='utf8') as f:
        f.write("======= train ========\n")
metrics_train = csc_process(csc_train, train_path, args.dataroot, csc_train_name, save=args.csc_save)
print("Evaluating on development dataset (as dev_2)...")
if args.csc_model == "sound":
    with open("checkpoints/corrected.txt", 'a', encoding='utf8') as f:
        f.write("======== dev =========\n")
metrics_dev = csc_process(csc_dev, dev_path, args.dataroot, csc_dev_name, save=args.csc_save)

acc, fscore = metrics_train['acc'], metrics_train['fscore']
print("Dev_1 acc: %.2f\tDev_1 fscore(p/r/f): (%.2f/%.2f/%.2f)" % \
        (acc, fscore['precision'], fscore['recall'], fscore['fscore']))

acc, fscore = metrics_dev['acc'], metrics_dev['fscore']
print("Dev_2 acc: %.2f\tDev_2 fscore(p/r/f): (%.2f/%.2f/%.2f)" % \
        (acc, fscore['precision'], fscore['recall'], fscore['fscore']))
#coding=utf8
import sys, os, time, gc, json
from torch.optim import Adam,AdamW

install_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(install_path)

from utils.args import init_args
from utils.initialization import *

from utils.batch import from_example_list
from utils.vocab import PAD
from utils.example import Example
from model.slu_baseline_tagging import SLUTagging
from model.slu_bert_bertvocab import SLUBert_bertvocab

# initialization params, output path, logger, random seed and torch.device
args = init_args(sys.argv[1:])

set_random_seed(args.seed)
device = set_torch_device(args.device)
print("Initialization finished ...")
print("Random seed is set to %d" % (args.seed))
print("Use GPU with index %s" % (args.device) if args.device >= 0 else "Use CPU as target torch device")

start_time = time.time()
Example.configuration(args, train_path=os.path.join(args.dataroot, 'train.json'))

if args.train_data == "asr" or args.train_data == "manu":
    train_path = os.path.join(args.dataroot, 'train.json')
else:
    train_path = os.path.join(args.dataroot, "train_{}_csc.json".format(args.csc_model))

train_dataset = Example.load_dataset(train_path,mode=args.train_data)

if args.dev_data == "asr":
    dev_path = os.path.join(args.dataroot, 'development.json')
else:
    dev_path = os.path.join(args.dataroot, "dev_{}_csc.json".format(args.csc_model))
dev_dataset = Example.load_dataset(dev_path,mode=args.dev_data)


print("Load dataset and database finished, cost %.4fs ..." % (time.time() - start_time))
print("Dataset size: train -> %d ; dev -> %d" % (len(train_dataset), len(dev_dataset)))

args.vocab_size = Example.word_vocab.vocab_size
args.pad_idx = Example.word_vocab[PAD]
args.num_tags = Example.label_vocab.num_tags
args.tag_pad_idx = Example.label_vocab.convert_tag_to_idx(PAD)


if args.model=="baseline":
    TagModel = SLUTagging(args).to(device)
    Example.word2vec.load_embeddings(TagModel.word_embed, Example.word_vocab, device=device)
elif args.model == "bert":
    TagModel = SLUBert_bertvocab(args).to(device)
else:
    raise ValueError("No such tagging model")

if not os.path.exists("./save_model"):
    os.makedirs("./save_model")

model_file_path = os.path.join('./save_model',args.model+"_"+args.decode+"_"+args.train_data+"_"+args.encoder_cell+"_tune_batch_size{}_lr{}_max_epoch{}_dropout{}_embed_size{}_hidden_size{}_num_layer{}_seed{}".format(args.tune,args.batch_size,args.lr,args.max_epoch,args.dropout,args.embed_size,args.hidden_size,args.num_layer,args.seed))
model_path=os.path.join(model_file_path,"model.bin")

if args.testing:
    check_point = torch.load(open(model_path, 'rb'), map_location=device)
    TagModel.load_state_dict(check_point['TagModel'])
    print("Load saved TagModel from root path")

def set_optimizer(model, args):
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    grouped_params = [{'params': list(set([p for n, p in params]))}]
    optimizer = Adam(grouped_params, lr=args.lr)
    return optimizer


def decode(choice):
    assert choice in ['train', 'dev']
    TagModel.eval()
    dataset = train_dataset if choice == 'train' else dev_dataset
    predictions, labels = [], []
    total_loss, count = 0, 0
    with torch.no_grad():
        for i in range(0, len(dataset), args.batch_size):
            cur_dataset = dataset[i: i + args.batch_size]
            current_batch = from_example_list(args, cur_dataset, device, train=True, use_bert = args.model =='bert')
            pred, label, loss = TagModel.decode(Example.label_vocab, current_batch)
            for j in range(len(current_batch)):
                if any([l.split('-')[-1] not in current_batch.utt[j] for l in pred[j]]):
                    print(current_batch.utt[j], pred[j], label[j])
            predictions.extend(pred)
            labels.extend(label)
            total_loss += loss
            count += 1
        metrics = Example.evaluator.acc(predictions, labels)
    torch.cuda.empty_cache()
    gc.collect()
    return metrics, total_loss / count


def predict():
    TagModel.eval()
    test_path = os.path.join(args.dataroot, 'test_unlabelled.json')
    test_dataset = Example.load_dataset(test_path)
    predictions = {}
    with torch.no_grad():
        for i in range(0, len(test_dataset), args.batch_size):
            cur_dataset = test_dataset[i: i + args.batch_size]
            current_batch = from_example_list(args, cur_dataset, device, train=False, use_bert = args.model =='bert')
            pred = TagModel.decode(Example.label_vocab, current_batch)
            for pi, p in enumerate(pred):
                did = current_batch.did[pi]
                predictions[did] = p
    test_json = json.load(open(test_path, 'r'))
    ptr = 0
    for ei, example in enumerate(test_json):
        for ui, utt in enumerate(example):
            utt['pred'] = [pred.split('-') for pred in predictions[f"{ei}-{ui}"]]
            ptr += 1
    json.dump(test_json, open(os.path.join(args.dataroot, 'prediction.json'), 'w'), indent=4, ensure_ascii=False)


if not args.testing:
    num_training_steps = ((len(train_dataset) + args.batch_size - 1) // args.batch_size) * args.max_epoch
    print('Total training steps: %d' % (num_training_steps))
    optimizer = set_optimizer(TagModel, args)
    nsamples, best_result = len(train_dataset), {'dev_acc': 0., 'dev_f1': 0.}
    train_index, step_size = np.arange(nsamples), args.batch_size
    print('Start training ......')
    for i in range(args.max_epoch):
        start_time = time.time()
        epoch_loss = 0
        np.random.shuffle(train_index)
        TagModel.train()
        count = 0
        for j in range(0, nsamples, step_size):
            cur_dataset = [train_dataset[k] for k in train_index[j: j + step_size]]
            current_batch = from_example_list(args, cur_dataset, device, train=True, use_bert = args.model =='bert')
            output, loss = TagModel(current_batch)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            count += 1
        print('Training: \tEpoch: %d\tTime: %.4f\tTraining Loss: %.4f' % (i, time.time() - start_time, epoch_loss / count))
        torch.cuda.empty_cache()
        gc.collect()

        start_time = time.time()
        metrics, dev_loss = decode('dev')
        dev_acc, dev_fscore = metrics['acc'], metrics['fscore']
        print('Evaluation: \tEpoch: %d\tTime: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)' % (i, time.time() - start_time, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore']))
        if dev_acc > best_result['dev_acc']:
            best_result['dev_loss'], best_result['dev_acc'], best_result['dev_f1'], best_result['iter'] = dev_loss, dev_acc, dev_fscore, i
            if not os.path.exists(model_file_path):os.mkdir(model_file_path)
            torch.save({
                'epoch': i, 'TagModel': TagModel.state_dict(),
                'optim': optimizer.state_dict(),
            }, open(model_path, 'wb'))
            print('NEW BEST TAGMODEL: \tEpoch: %d\tDev loss: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)' % (i, dev_loss, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore']))

    print('FINAL BEST RESULT: \tEpoch: %d\tDev loss: %.4f\tDev acc: %.4f\tDev fscore(p/r/f): (%.4f/%.4f/%.4f)' % (best_result['iter'], best_result['dev_loss'], best_result['dev_acc'], best_result['dev_f1']['precision'], best_result['dev_f1']['recall'], best_result['dev_f1']['fscore']))
else:
    start_time = time.time()
    metrics, dev_loss = decode('dev')
    dev_acc, dev_fscore = metrics['acc'], metrics['fscore']
    predict()
    print("Evaluation costs %.2fs ; Dev loss: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)" % (time.time() - start_time, dev_loss, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore']))

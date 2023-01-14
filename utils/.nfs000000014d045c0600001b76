import json

from utils.vocab import Vocab, LabelVocab
from utils.word2vec import Word2vecUtils
from utils.evaluator import Evaluator

class CSCDataset():

    @classmethod
    def configuration(cls, root, train_path=None, word2vec_path=None):
        cls.evaluator = Evaluator()
        cls.word_vocab = Vocab(padding=True, unk=True, filepath=train_path)
        cls.word2vec = Word2vecUtils(word2vec_path)
        cls.label_vocab = LabelVocab(root)

    @classmethod
    def load_dataset(cls, data_path, mode="train"):
        if (mode != "train") and (mode != 'pred'):
            raise ValueError("mode for CSCDataset.load_dataset must be train or pred.")
        dataset = json.load(open(data_path, 'r', encoding='utf-8'))
        examples = []
        if mode == "train":
            for _, data in enumerate(dataset):
                for _, utt in enumerate(data):
                    ex = cls(utt)
                    examples.append(ex)
        else:
            for _, data in enumerate(dataset):
                expl = []
                for _, utt in enumerate(data):
                    ex = cls(utt)
                    expl.append(ex)
                examples.append(expl)
                
        return examples

    def __init__(self, ex: dict):
        super(CSCDataset, self).__init__()
        self.ex = ex

        self.utt = ex['asr_1best']
        self.label = ex['manual_transcript']
        self.slot = dict()
        for label in ex['semantic']:
            act_slot = f'{label[0]}-{label[1]}'
            if len(label) == 3:
                self.slot[act_slot] = label[2]

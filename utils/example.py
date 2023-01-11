import json

from utils.vocab import Vocab, LabelVocab
from utils.word2vec import Word2vecUtils
<<<<<<< HEAD
=======
from utils.bert2embd import Bert2vecUtils
>>>>>>> origin/backup_gu
from utils.evaluator import Evaluator

class Example():

    @classmethod
<<<<<<< HEAD
    def configuration(cls, root, train_path=None, word2vec_path=None):
        cls.evaluator = Evaluator()
        cls.word_vocab = Vocab(padding=True, unk=True, filepath=train_path)
        cls.word2vec = Word2vecUtils(word2vec_path)
=======
    def configuration(cls, root, train_path=None, word2vec_path=None, embedding_type='Bert_pretrained'):
        cls.evaluator = Evaluator()
        cls.word_vocab = Vocab(padding=True, unk=True, filepath=train_path)
        if embedding_type == 'Bert_pretrained':
            cls.word2vec = Bert2vecUtils()
        elif embedding_type == 'WordVab_embedding':
            cls.word2vec = Word2vecUtils(word2vec_path)
        else:
            raise NotImplementedError
>>>>>>> origin/backup_gu
        cls.label_vocab = LabelVocab(root)

    @classmethod
    def load_dataset(cls, data_path,mode="asr"):
        dataset = json.load(open(data_path, 'r',encoding='utf-8'))
        examples = []
        for di, data in enumerate(dataset):
            for ui, utt in enumerate(data):
                ex = cls(utt, f'{di}-{ui}',mode=mode)
                examples.append(ex)
        return examples

    def __init__(self, ex: dict, did, mode):
        super(Example, self).__init__()
        self.ex = ex
        self.did = did

        if mode =="manu":
            self.utt = ex['manual_transcript']
        elif mode == "asr":
            self.utt = ex['asr_1best']
        else:
            raise ValueError("No such training_data")
        self.slot = {}
        for label in ex['semantic']:
            act_slot = f'{label[0]}-{label[1]}'
            if len(label) == 3:
                self.slot[act_slot] = label[2]
        self.tags = ['O'] * len(self.utt)
        for slot in self.slot:
            value = self.slot[slot]
            bidx = self.utt.find(value)
            if bidx != -1:
                self.tags[bidx: bidx + len(value)] = [f'I-{slot}'] * len(value)
                self.tags[bidx] = f'B-{slot}'
        self.slotvalue = [f'{slot}-{value}' for slot, value in self.slot.items()]
        self.input_idx = [Example.word_vocab[c] for c in self.utt]
        l = Example.label_vocab
        self.tag_id = [l.convert_tag_to_idx(tag) for tag in self.tags]

import json

from utils.vocab import Vocab, LabelVocab
from utils.vocab_for_onei import LabelVocab_for_onei
from utils.word2vec import Word2vecUtils
from utils.bert2embd import Bert2vecUtils
from utils.evaluator import Evaluator
from transformers import BertTokenizer, BertConfig, BertForMaskedLM, BertForNextSentencePrediction
from transformers import BertModel,AutoTokenizer,AutoModelForTokenClassification



class Example():

    @classmethod
    def configuration(cls, args, train_path=None):
        root = args.dataroot
        word2vec_path = args.word2vec_path
        cls.evaluator = Evaluator()
        cls.word_vocab = Vocab(padding=True, unk=True, filepath=train_path)
        cls.word2vec = Word2vecUtils(word2vec_path)
        if args.decode =="onei":
            cls.label_vocab = LabelVocab_for_onei(root)
        else:
            cls.label_vocab = LabelVocab(root)
        cls.args = args
        cls.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)

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
        else:
            self.utt = ex['asr_1best']
        if Example.args.out_blank:
            self.utt = self.utt.replace(' ','')
        
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
                if Example.args.decode == "onei":
                    self.tags[bidx: bidx + len(value)] = [f'I'] * len(value)
                else:
                    self.tags[bidx: bidx + len(value)] = [f'I-{slot}'] * len(value)
                self.tags[bidx] = f'B-{slot}'
        self.slotvalue = [f'{slot}-{value}' for slot, value in self.slot.items()]
        self.input_idx = [Example.word_vocab[c] for c in self.utt]
        l = Example.label_vocab
        self.tag_id = [l.convert_tag_to_idx(tag) for tag in self.tags]
      

        if Example.args.model == "bert":
            utt_list = [word for word in self.utt]
            for idx,word in enumerate(utt_list):
                if word.isupper():
                    word = word.lower()
                    utt_list[idx] = word
                if word == ' ':
                    utt_list[idx] = '[SEP]'
                
            # bert_token = tokenizer.tokenize(self.utt)
            bert_id = Example.tokenizer.convert_tokens_to_ids(utt_list)
            self.bert_id = bert_id

            
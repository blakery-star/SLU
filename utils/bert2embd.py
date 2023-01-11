#coding=utf8

from utils.vocab import PAD, UNK
import numpy as np
import torch
from transformers import BertTokenizer, BertConfig, BertForMaskedLM, BertForNextSentencePrediction
from transformers import BertModel,AutoTokenizer,AutoModelForTokenClassification
import sys


class Bert2vecUtils():

    def __init__(self):
        super(Bert2vecUtils, self).__init__()
        self.Bert = {}
        self.read_model()

    def load_embeddings(self, module, vocab, device='cpu'):
        """ Initialize the embedding with glove and char embedding
        """
        emb_size = module.weight.data.size(-1)
        outliers = 0

        for index, word in enumerate (vocab.word2id):
            if word == PAD : # PAD symbol is always 0-vector
                module.weight.data[vocab[PAD]] = torch.zeros(emb_size, dtype=torch.float, device=device)
                continue
            if word == UNK:
                word = '[UNK]'
            if word == ' ':
                module.weight.data[vocab[word]] = torch.zeros(emb_size, dtype=torch.float, device=device)
                continue
            # use pretrained bert to get the word embedding 
            tokens = self.tokenizer.tokenize(word)
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_ids = torch.tensor(input_ids).long().unsqueeze(0)

            with torch.no_grad():
                out = self.bert_model(input_ids)
            module.weight.data[vocab[word]] = out.last_hidden_state[0]
           



        return 1 - outliers / float(len(vocab))

    def read_model(self):
        model_name = 'chinese-bert-wwm-ext'
        MODEL_PATH = '/chinese-bert-wwm-ext'
 
        # self.tokenizer = BertTokenizer.from_pretrained(model_name)
        # model_config = BertConfig.from_pretrained(model_name)
        # model_config.output_hidden_states = True
        # model_config.output_attentions = True
        # self.bert_model = BertModel.from_pretrained(MODEL_PATH, config = model_config)

        self.tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")

        self.bert_model = BertModel.from_pretrained("hfl/chinese-bert-wwm-ext")

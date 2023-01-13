#coding=utf8
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from transformers import BertTokenizer, BertConfig, BertForMaskedLM, BertForNextSentencePrediction
from transformers import BertModel,AutoTokenizer,AutoModelForTokenClassification
from utils.vocab import PAD, UNK
from utils.decoder import decode_baseline,decode_new,decode_onei



class SLUBert_bertvocab(nn.Module):

    def __init__(self, config):
        super(SLUBert_bertvocab, self).__init__()
        self.config = config
        self.cell = config.encoder_cell
        self.word_embed = nn.Embedding(config.vocab_size, config.embed_size, padding_idx=0)
        self.rnn = getattr(nn, self.cell)(config.embed_size, config.hidden_size // 2, num_layers=config.num_layer, bidirectional=True, batch_first=True)
        self.dropout_layer = nn.Dropout(p=config.dropout)
        self.output_layer = TaggingFNNDecoder(config.hidden_size, config.num_tags, config.tag_pad_idx)
        self.bert_model = BertModel.from_pretrained(config.pretrained_model)
        self.device = config.device

    def forward(self, batch):
        tag_ids = batch.tag_ids
        B = len(tag_ids)
        tag_mask = batch.tag_mask
        input_ids = batch.input_ids
        lengths = batch.lengths
        utt = batch.utt
        embeds = torch.zeros(B,max(lengths),self.config.embed_size).to(self.device)
        
        for batch_id,input_id in enumerate(input_ids[:]):
            if self.config.tune:
                embed = self.bert_model(input_id.unsqueeze(0)).last_hidden_state[0]
                embeds[batch_id] = embed
            else :
                with torch.no_grad():
                    embed = self.bert_model(input_id.unsqueeze(0)).last_hidden_state[0]
                    embeds[batch_id] = embed
        
        packed_inputs = rnn_utils.pack_padded_sequence(embeds, lengths, batch_first=True, enforce_sorted=True)
        packed_rnn_out, h_t_c_t = self.rnn(packed_inputs)  # bsize x seqlen x dim
        rnn_out, unpacked_len = rnn_utils.pad_packed_sequence(packed_rnn_out, batch_first=True)


        hiddens = self.dropout_layer(rnn_out)
        tag_output = self.output_layer(hiddens, tag_mask, tag_ids)

        return tag_output

    def decode(self, label_vocab, batch):
        if self.config.decode == "baseline":
            return decode_baseline(self, label_vocab, batch)
        elif self.config.decode == "newdecode":
            return decode_new(self, label_vocab, batch)
        elif self.config.decode == "onei":
            return decode_onei(self, label_vocab, batch)
        else:
            raise NotImplementedError("No such decoder")


class TaggingFNNDecoder(nn.Module):

    def __init__(self, input_size, num_tags, pad_id):
        super(TaggingFNNDecoder, self).__init__()
        self.num_tags = num_tags
        self.output_layer = nn.Linear(input_size, num_tags)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, hiddens, mask, labels=None):
        logits = self.output_layer(hiddens)
        logits += (1 - mask).unsqueeze(-1).repeat(1, 1, self.num_tags) * -1e32
        prob = torch.softmax(logits, dim=-1)
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
            return prob, loss
        return (prob, )

import json

import jieba
from jieba import posseg as pseg
from pypinyin import lazy_pinyin, TONE2
import difflib

class SimCSC():
    def __init__(self):
        super(SimCSC, self).__init__()
        self.vocab = dict()

    def learn(self, dataset):
        for data in dataset:
            slots = data.slot
            for slot in slots:
                values = pseg.cut(slots[slot])
                for value in values:
                    if len(value.word)<2: continue
                    pinyin = lazy_pinyin(value.word, style=TONE2)
                    pinyin = ''.join(pinyin)
                    if pinyin not in self.vocab:
                        self.vocab[pinyin] = [value.word]
                    elif value.word not in self.vocab[pinyin]:
                        self.vocab[pinyin] += [value.word]

    def save_vocab(self, file_dir="checkpoints", filename="slotvalue_vocab.json"):
        json.dump(self.vocab, open(file_dir+"/"+filename, 'w', encoding='utf8'), ensure_ascii=False, indent=4)

    def load_vocab(self, path="checkpoints/slotvalue_vocab.json"):
        self.vocab = json.load(open(path, 'r', encoding='utf-8'))

    def correct(self, utterance, cutoff_sound=0.8, cutoff_look=0.5, use_history=True, ext=True):
            
        def crt(utt, self_vocab=self.vocab, use_history=False):
            words = pseg.cut(utt)
            for word in words:
                if len(word.word)<2: continue
                if word.flag == "ns" or word.flag == "nr":
                    pinyin = lazy_pinyin(word.word, style=TONE2)
                    pinyin = ''.join(pinyin)
                    correct = difflib.get_close_matches(pinyin, set(self_vocab.keys()), cutoff=cutoff_sound)
                    if len(correct) > 0:
                        best_match = difflib.get_close_matches(word.word, self_vocab[correct[0]], cutoff=cutoff_look)
                        if len(best_match) > 0:
                            utt = utt.replace(word.word, best_match[0])
                        else:
                            if use_history:
                                self_vocab[pinyin] = [word.word]
                    else:
                        if use_history:
                            self_vocab[pinyin] = [word.word]
            return utt
        
        if isinstance(utterance, str):
            if ext:
                return crt(utterance), None
            else:
                return crt(utterance)
        elif isinstance(utterance, list):
            if (isinstance(utterance[0], str) and use_history and (len(utterance) > 1)):
                answer = []
                for utt in utterance:
                    temp_vocab = self.vocab.copy()
                    answer += [crt(utt, temp_vocab, use_history)]
                if ext:
                    return answer, None
                else:
                    return answer
            else:
                return [correct(utt, ext=False) for utt in utterance], None
        else:
            raise ValueError("faulty type of utterance.")
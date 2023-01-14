import json

import jieba
from jieba import posseg as pseg
from pypinyin import lazy_pinyin
from pypinyin import TONE2 as tone
import difflib

class SimCSC():
    def __init__(self, log=None):
        super(SimCSC, self).__init__()
        self.vocab = dict()
        self.log = log
        with open(self.log, 'w', encoding='utf8') as f:
            pass

    def learn(self, dataset):
        for data in dataset:
            slots = data.slot
            for slot in slots:
                if len(slots[slot]) < 2: continue
                pinyin = lazy_pinyin(slots[slot], style=tone)
                pinyin = ''.join(pinyin)
                if pinyin not in self.vocab:
                    self.vocab[pinyin] = [slots[slot]]
                elif slots[slot] not in self.vocab[pinyin]:
                    self.vocab[pinyin] += [slots[slot]]

    def save_vocab(self, file_dir="checkpoints", filename="slotvalue_vocab.json"):
        json.dump(self.vocab, open(file_dir+"/"+filename, 'a', encoding='utf8'), ensure_ascii=False, indent=4)

    def load_vocab(self, path="checkpoints/slotvalue_vocab.json"):
        self.vocab = json.load(open(path, 'r', encoding='utf-8'))

    def correct(self, utterance, cutoff_sound=0.9, cutoff_look=0.75, use_history=True, ext=True):
            
        def crt(utt, self_vocab=self.vocab, use_history=False):
            words = pseg.cut(utt)
            current_pinyin = ""
            current_word = ""
            start_flag = False
            for word in words:
                if len(word.word)<2: continue
                if word.flag == "ns" or word.flag == "nr":
                    start_flag = True
                if start_flag:
                    pinyin = lazy_pinyin(word.word, style=tone)
                    pinyin = ''.join(pinyin)
                    current_pinyin += pinyin
                    current_word += word.word
            correct = difflib.get_close_matches(current_pinyin, set(self_vocab.keys()), cutoff=cutoff_sound)
            if len(correct) > 0:
                best_match = difflib.get_close_matches(current_word, self_vocab[correct[0]], cutoff=cutoff_look)
                if len(best_match) > 0:
                    if (best_match[0].find(current_word) >= 0) or (current_word.find(best_match[0]) >= 0):
                        return utt
                    utt = utt.replace(current_word, best_match[0])
                    if self.log is not None:
                        with open(self.log, 'a', encoding='utf8') as f:
                            f.write(current_word + " -> " + best_match[0] + "\n")
                else:
                    if use_history:
                        self_vocab[current_pinyin] = [current_word.word]
            if use_history:
                self_vocab[current_pinyin] = [current_word]
            current_pinyin = ""
            current_word = ""

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
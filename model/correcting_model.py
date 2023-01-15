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

    def correct(self, utterance, cutoff_sound=0.9, use_history=True, ext=True):
            
        def crt(utt, self_vocab=self.vocab, use_history=False):
            words = list(pseg.cut(utt))
            current_pinyin = ""
            current_word = ""
            start_flag = False
            for i in range(len(words)):
                if words[i].flag == "ns" or words[i].flag == "nr":
                    break

            for j in range(len(words)-1,-1,-1):
                if words[j].flag.find("n") == 0:
                    break

            if (j<i):
                j = len(words)-1

            for k in range(i,j+1):
                pinyin = lazy_pinyin(words[k].word, style=tone)
                pinyin = ''.join(pinyin)
                current_pinyin += pinyin
                current_word += words[k].word

            correct = difflib.get_close_matches(current_pinyin, set(self_vocab.keys()), cutoff=cutoff_sound)
            if len(correct) > 0:
                best_match = self_vocab[correct[0]]
                if len(best_match) > 0:
                    if (best_match[0].find(current_word) < 0) and (current_word.find(best_match[0]) < 0) \
                                                              and abs(len(current_word) - len(best_match[0])) < 2:
                        utt = utt.replace(current_word, best_match[0])
                        current_word = best_match[0]
                        if self.log is not None:
                            with open(self.log, 'a', encoding='utf8') as f:
                                f.write(current_word + " -> " + best_match[0] + "\n")

            if use_history:
                self_vocab[current_pinyin] = [current_word]

            return utt
        
        if isinstance(utterance, str):
            if ext:
                return crt(utterance, use_history=use_history), None
            else:
                return crt(utterance, use_history=use_history)
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
                return [correct(utt, use_history=use_history, ext=False) for utt in utterance], None
        else:
            raise ValueError("faulty type of utterance.")
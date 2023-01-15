import torch

def decode_baseline(self, label_vocab, batch):
    batch_size = len(batch)
    labels = batch.labels
    output = self.forward(batch)
    prob = output[0]
    predictions = []
    for i in range(batch_size):
        pred = torch.argmax(prob[i], dim=-1).cpu().tolist()
        pred_tuple = []
        idx_buff, tag_buff, pred_tags = [], [], []
        pred = pred[:len(batch.utt[i])]
        for idx, tid in enumerate(pred):
            tag = label_vocab.convert_idx_to_tag(tid)
            pred_tags.append(tag)
            if (tag == 'O' or tag.startswith('B')) and len(tag_buff) > 0:
                slot = '-'.join(tag_buff[0].split('-')[1:])
                value = ''.join([batch.utt[i][j] for j in idx_buff])
                idx_buff, tag_buff = [], []
                pred_tuple.append(f'{slot}-{value}')
                if tag.startswith('B'):
                    idx_buff.append(idx)
                    tag_buff.append(tag)
            elif tag.startswith('I') or tag.startswith('B'):
                idx_buff.append(idx)
                tag_buff.append(tag)
        if len(tag_buff) > 0:
            slot = '-'.join(tag_buff[0].split('-')[1:])
            value = ''.join([batch.utt[i][j] for j in idx_buff])
            pred_tuple.append(f'{slot}-{value}')
        predictions.append(pred_tuple)
    if len(output) == 1:
        return predictions
    else:
        loss = output[1]
        return predictions, labels, loss.cpu().item()




def decode_new(self, label_vocab, batch):
    batch_size = len(batch)
    labels = batch.labels
    output = self.forward(batch)
    prob = output[0]
    predictions = []
    for i in range(batch_size):
        pred = torch.argmax(prob[i], dim=-1).cpu().tolist()
        pred_tuple = []
        idx_buff, tag_buff, pred_tags = [], [], []
        pred = pred[:len(batch.utt[i])]
        for idx, tid in enumerate(pred):
            tag = label_vocab.convert_idx_to_tag(tid)
            pred_tags.append(tag)


            if (tag == 'O' or tag.startswith('B')) and len(tag_buff) > 0:
                value = ''.join([batch.utt[i][j] for j in idx_buff])
                idx_buff, tag_buff = [], []
                pred_tuple.append(f'{slot}-{value}')
                if tag.startswith('B'):
                    idx_buff.append(idx)
                    tag_buff.append(tag)
                    slot = '-'.join(tag.split('-')[1:])
            elif tag.startswith('B'):
                idx_buff.append(idx)
                tag_buff.append(tag)
                slot = '-'.join(tag.split('-')[1:])
            elif tag.startswith('I'): 
                if len(tag_buff)==0:continue
                slot2='-'.join(tag.split('-')[1:])
                if slot2==slot:
                    idx_buff.append(idx)
                    tag_buff.append(tag)
                else:
                    print(batch[i].utt)
                    print([label_vocab.convert_idx_to_tag(x) for x in pred])
                    print(batch[i].tags)
                    slot_prob,slot2_prob=1,1
                    for k in range(len(tag_buff)):
                        slot_idx=label_vocab.convert_tag_to_idx(tag_buff[k][0]+'-'+slot)
                        slot2_idx=label_vocab.convert_tag_to_idx(tag_buff[k][0]+'-'+slot2)

                        slot_prob=slot_prob*prob[i][idx][slot_idx]
                        slot2_prob=slot2_prob*prob[i][idx][slot2_idx]

                    if slot_prob<slot2_prob:
                        slot=slot2
                    idx_buff.append(idx)
                    tag_buff.append(tag)
                    
                    
        if len(tag_buff) > 0:
            slot = '-'.join(tag_buff[0].split('-')[1:])
            value = ''.join([batch.utt[i][j] for j in idx_buff])
            pred_tuple.append(f'{slot}-{value}')
        predictions.append(pred_tuple)
    if len(output) == 1:
        return predictions
    else:
        loss = output[1]
        return predictions, labels, loss.cpu().item()


def decode_onei(self, label_vocab, batch):
    batch_size = len(batch)
    labels = batch.labels
    output = self.forward(batch)
    prob = output[0]
    predictions = []
    for i in range(batch_size):
        pred = torch.argmax(prob[i], dim=-1).cpu().tolist()
        pred_tuple = []
        idx_buff, tag_buff, pred_tags = [], [], []
        pred = pred[:len(batch.utt[i])]
        for idx, tid in enumerate(pred):
            tag = label_vocab.convert_idx_to_tag(tid)
    
            pred_tags.append(tag)
            if len(tag_buff)==0:
                if tag.startswith('B'):
                    idx_buff.append(idx)
                    tag_buff.append(tag)
                else:
                    continue
            else:
                if tag == 'I':
                    idx_buff.append(idx)
                    tag_buff.append(tag)    
                elif tag == 'O':
                    slot = '-'.join(tag_buff[0].split('-')[1:])
                    value = ''.join([batch.utt[i][j] for j in idx_buff])
                    idx_buff, tag_buff = [], []
                    pred_tuple.append(f'{slot}-{value}')
                else:
                    slot = '-'.join(tag_buff[0].split('-')[1:])
                    value = ''.join([batch.utt[i][j] for j in idx_buff])
                    idx_buff, tag_buff = [], []
                    pred_tuple.append(f'{slot}-{value}')
                    idx_buff.append(idx)
                    tag_buff.append(tag)
        if len(tag_buff) > 0:
            slot = '-'.join(tag_buff[0].split('-')[1:])
            value = ''.join([batch.utt[i][j] for j in idx_buff])
            pred_tuple.append(f'{slot}-{value}')
        predictions.append(pred_tuple)
    if len(output) == 1:
        return predictions
    else:
        loss = output[1]
        return predictions, labels, loss.cpu().item()
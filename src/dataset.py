import torch
import pandas as pd
import numpy as np

from torch.utils.data import Dataset
from torch.distributions.categorical import Categorical

import tqdm

def createDataCSV(dataset, a, b):
    labels = []
    texts = []
    dataType = []
    label_map = {}

    label_freq = {}

    name_map = {'wiki31k': 'Wiki10-31K',
                'wiki500k': 'Wiki-500K',
                'amazoncat13k': 'AmazonCat-13K',
                'amazon670k': 'Amazon-670K',
                'eurlex4k': 'Eurlex-4K'}

    assert dataset in name_map
    dataset = name_map[dataset]

    fext = '_texts.txt' if dataset == 'Eurlex-4K' else '_raw_texts.txt'
    with open(f'./data/{dataset}/train{fext}') as f:
        for i in tqdm.tqdm(f):
            texts.append(i.replace('\n', ''))
            dataType.append('train')

    with open(f'./data/{dataset}/test{fext}') as f:
        for i in tqdm.tqdm(f):
            texts.append(i.replace('\n', ''))
            dataType.append('test')

    with open(f'./data/{dataset}/train_labels.txt') as f:
        for i in tqdm.tqdm(f):
            for l in i.replace('\n', '').split():
                label_map[l] = 0
                if l not in label_freq.keys():
                    label_freq[l] = 1
                else:
                    label_freq[l] += 1
            labels.append(i.replace('\n', ''))


    with open(f'./data/{dataset}/test_labels.txt') as f:
        print(len(label_map))
        for i in tqdm.tqdm(f):
            for l in i.replace('\n', '').split():
                label_map[l] = 0
            labels.append(i.replace('\n', ''))
        print(len(label_map))

    assert len(texts) == len(labels) == len(dataType)

    df_row = {'text': texts, 'label': labels, 'dataType': dataType}

    for i, k in enumerate(sorted(label_map.keys())):
        label_map[k] = i
    df = pd.DataFrame(df_row)

    freq = np.zeros(len(label_map))
    for l, f in label_freq.items():
        freq[label_map[l]] = f
        
    c = (np.log(np.sum(df.dataType == 'train')) - 1) * np.power(b+1, a)
    inv_prop = 1 + c * np.power(freq + b, -a)

    print('label map', len(label_map))

    return df, label_map, inv_prop, freq


class MDataset(Dataset):
    def __init__(self, df, mode, tokenizer, label_map, max_length,
                 token_type_ids=None, group_y=None, candidates_num=None, return_group=True):
        assert mode in ["train", "valid", "test"]
        self.mode = mode
        self.df, self.n_labels, self.label_map = df[df.dataType == self.mode], len(label_map), label_map
        self.len = len(self.df)
        self.tokenizer, self.max_length, self.group_y = tokenizer, max_length, group_y
        self.multi_group = False
        self.token_type_ids = token_type_ids
        self.candidates_num = candidates_num
        self.return_group = return_group

        if group_y is not None:
            # group y mode
            self.candidates_num, self.group_y, self.n_group_y_labels = candidates_num, [], group_y.shape[0]
            self.map_group_y = np.empty(len(label_map), dtype=np.long)
            for idx, labels in enumerate(group_y):
                self.group_y.append([])
                for label in labels:
                    self.group_y[-1].append(label_map[label])
                self.map_group_y[self.group_y[-1]] = idx
                self.group_y[-1]  = np.array(self.group_y[-1])
            self.group_y = np.array(self.group_y)
    

    def _get_docs(self, idx):
        
        max_len = self.max_length
        review = self.df.text.values[idx].lower()

        review = ' '.join(review.split()[:max_len])

        text = review
        if self.token_type_ids is not None:
            input_ids = self.token_type_ids[idx]
            if input_ids[-1] == 0:
                input_ids = input_ids[input_ids != 0]
            input_ids = input_ids.tolist()
        elif hasattr(self.tokenizer, 'encode_plus'):
            input_ids = self.tokenizer.encode(
                'filling empty' if len(text) == 0 else text,
                add_special_tokens=True,
                max_length=max_len,
                truncation=True
            )
        else:
            # fast 
            input_ids = self.tokenizer.encode(
                'filling empty' if len(text) == 0 else text,
                add_special_tokens=True
            ).ids

        if len(input_ids) == 0:
            print('zero string')
            assert 0
        if len(input_ids) > self.max_length:
            input_ids[self.max_length-1] = input_ids[-1]
            input_ids = input_ids[:self.max_length]

        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)

        padding_length = self.max_length - len(input_ids)
        input_ids = input_ids + ([0] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)

        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        token_type_ids = torch.tensor(token_type_ids)

        return input_ids, attention_mask, token_type_ids


    def __getitem__(self, idx):

        input_ids, attention_mask, token_type_ids = self._get_docs(idx)
        
        labels = [self.label_map[i] for i in self.df.label.values[idx].split() if i in self.label_map]


        if self.group_y is not None and self.return_group:
            label_ids = torch.zeros(self.n_labels)
            label_ids = label_ids.scatter(0, torch.tensor(labels),
                                          torch.tensor([1.0 for i in labels]))
            group_labels = self.map_group_y[labels]
            if self.multi_group:
                group_labels = np.concatenate(group_labels)
            group_label_ids = torch.zeros(self.n_group_y_labels)
            group_label_ids = group_label_ids.scatter(0, torch.tensor(group_labels),
                                                      torch.tensor([1.0 for i in group_labels]))
            candidates = np.concatenate(self.group_y[group_labels], axis=0) # mohamm: These candidates are actually positive candidates

            if len(candidates) < self.candidates_num:
                sample = np.random.randint(self.n_group_y_labels, size=self.candidates_num - len(candidates))
                candidates = np.concatenate([candidates, sample])
            elif len(candidates) > self.candidates_num:
                candidates = np.random.choice(candidates, self.candidates_num, replace=False)

            if self.mode == 'train':
                return input_ids, attention_mask, token_type_ids,\
                    label_ids[candidates], group_label_ids, candidates
            else:
                return input_ids, attention_mask, token_type_ids,\
                    label_ids, group_label_ids, candidates

        label_ids = torch.zeros(self.n_labels)
        label_ids = label_ids.scatter(0, torch.tensor(labels),
                                      torch.tensor([1.0 for i in labels]))
        return input_ids, attention_mask, token_type_ids, label_ids
    
    def __len__(self):
        return self.len



class DatasetUniform(MDataset):

    def __init__(self, df, mode, tokenizer, label_map, max_length,
                 token_type_ids=None, group_y=None, candidates_num=None,
                 return_labels=True, dist_weights=None):
        super().__init__(df, mode, tokenizer, label_map, max_length,
                         token_type_ids, group_y, candidates_num)
        self.return_labels = return_labels
        if return_labels:
            self.uniform = False
            if dist_weights is None:
                self.uniform = True
            else:
                # dist_weights = torch.ones(self.n_labels) if dist_weights is None else dist_weights
                self.cat_dist = Categorical(logits=dist_weights)


    def _get_labels(self, idx, labels_pos):

        if self.uniform:
            neg_labels = np.random.randint(self.n_labels, size=self.candidates_num+50)
        else:
            neg_labels = self.cat_dist.sample((self.candidates_num+50,)).numpy()
        neg_labels = neg_labels[~np.in1d(neg_labels, labels_pos)]
        lbl_indices = np.concatenate((labels_pos, neg_labels))[:self.candidates_num]
        if len(lbl_indices)<self.candidates_num:
            neg_more = np.random.randint(self.n_labels, size=self.candidates_num-len(lbl_indices))
            lbl_indices = np.concatenate((lbl_indices, neg_more))
        
        lbl_one_hot = np.zeros(self.candidates_num)
        lbl_one_hot[:len(labels_pos)] = 1.0

        return lbl_one_hot, lbl_indices
    


    def __getitem__(self, idx):
    
        input_ids, attention_mask, token_type_ids = self._get_docs(idx)

        if self.return_labels:
      
            labels_pos = [self.label_map[i] for i in self.df.label.values[idx].split() if i in self.label_map]
            
            lbl_one_hot, lbl_indices = self._get_labels(idx, labels_pos)

            return input_ids, attention_mask, token_type_ids, lbl_one_hot, lbl_indices

        else:
            return input_ids, attention_mask, token_type_ids




class DatasetMips(DatasetUniform):

    def __init__(self, df, mode, tokenizer, label_map, max_length,
                 token_type_ids=None, group_y=None, candidates_num=None,
                 return_labels=True, dist_weights=None):
        super().__init__(df, mode, tokenizer, label_map, max_length,
                         token_type_ids, group_y, candidates_num,
                         return_labels, dist_weights)

    def _get_labels(self, labels_pos):
        
        if self.uniform:
            labels_rand = torch.tensor(np.random.randint(self.n_labels, size=self.candidates_num+50))
        else:
            labels_rand = self.cat_dist.sample((self.candidates_num+50,))
        labels_rand = labels_rand[~np.in1d(labels_rand, labels_pos)]
        labels_rand = labels_rand[:self.candidates_num]
        if len(labels_rand)<self.candidates_num:
            neg_more = torch.tensor(np.random.randint(self.n_labels, size=self.candidates_num-len(labels_rand)))
            labels_rand = torch.cat((labels_rand, neg_more))

        return labels_rand
        


    def __getitem__(self, idx):
    
        input_ids, attention_mask, token_type_ids = self._get_docs(idx)
        
        if self.return_labels:
      
            labels_pos = [self.label_map[i] for i in self.df.label.values[idx].split() if i in self.label_map]
            
            labels_rand = self._get_labels(labels_pos)

            return input_ids, attention_mask, token_type_ids, labels_pos, labels_rand

        else:
            return input_ids, attention_mask, token_type_ids

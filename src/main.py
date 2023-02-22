import sys
import random
import numpy as np
from apex import amp
from model import LightXML

from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader

from transformers import AdamW

import torch

from torch.utils.data import DataLoader
from dataset import MDataset, createDataCSV, \
                    DatasetUniform, DatasetMips
from log import Logger
import json

def load_group(dataset, group_tree=0):
    name_map = {'wiki31k': 'Wiki10-31K',
                'wiki500k': 'Wiki-500K',
                'amazoncat13k': 'AmazonCat-13K',
                'amazon670k': 'Amazon-670K',
                'eurlex4k': 'Eurlex-4K'}
    return np.load(f'./data/{name_map[dataset]}/label_group{group_tree}.npy', allow_pickle=True)

def collate_uniform_pos(batch):
    input_ids = torch.vstack([item[0] for item in batch])
    attention_mask = torch.vstack([item[1] for item in batch])
    token_type_ids = torch.vstack([item[2] for item in batch])
    if len(batch)==3:
        return input_ids, attention_mask, token_type_ids
    else:
        labels_pos = [torch.LongTensor([item[3]]).squeeze(0) for item in batch]
        labels_rand = torch.vstack([item[4] for item in batch])
        return input_ids, attention_mask, token_type_ids, labels_pos, labels_rand


def train(model, df, label_map):
    tokenizer = model.get_tokenizer()

    if args.model_type=='light':
        group_y = load_group(args.dataset, args.group_y_group)
        train_d = MDataset(df, 'train', tokenizer, label_map, args.max_len, group_y=group_y,
                           candidates_num=args.group_y_candidate_num)
        test_d = MDataset(df, 'test', tokenizer, label_map, args.max_len, candidates_num=args.group_y_candidate_num,
                          return_group=args.eval_type=='original')

        if args.valid:
            valid_d = MDataset(df, 'valid', tokenizer, label_map, args.max_len, group_y=group_y,
                               candidates_num=args.group_y_candidate_num)
            validloader = DataLoader(valid_d, batch_size=args.batch, num_workers=0, 
                                     shuffle=False)
    else:

        test_d = MDataset(df, 'test', tokenizer, label_map, args.max_len)

        if args.model_type=='full':
            train_d = MDataset(df, 'train', tokenizer, label_map, args.max_len)

        elif args.model_type=='uniform':
            train_d = DatasetUniform(df, 'train', tokenizer, label_map, args.max_len,
                                     candidates_num=args.group_y_candidate_num)


        elif args.model_type == 'mips':
            train_d = DatasetMips(df, 'train', tokenizer, label_map, args.max_len,
                                        candidates_num=args.group_y_candidate_num)


    if args.model_type == 'mips':
        trainloader = DataLoader(train_d, batch_size=args.batch, num_workers=5,
                                 shuffle=True, collate_fn=collate_uniform_pos)
    else:
        trainloader = DataLoader(train_d, batch_size=args.batch, num_workers=5,
                                 shuffle=True)
    testloader = DataLoader(test_d, batch_size=args.batch, num_workers=5,
                            shuffle=False)
        


    model.cuda()
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)#, eps=1e-8)
        
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    max_only_p5 = 0
    ev_result = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    for epoch in range(0, args.epoch+5):
        train_loss = model.one_epoch(epoch, trainloader, optimizer, mode='train',
                                     eval_loader=validloader if args.valid else testloader,
                                     eval_step=args.eval_step, log=LOG)

        if args.valid:
            ev_result = model.one_epoch(epoch, validloader, optimizer, mode='eval')
        else:
            ev_result = model.one_epoch(epoch, testloader, optimizer, mode='eval')

        g_p1, g_p3, g_p5, p1, p3, p5, psp1, psp3, psp5 = ev_result

        log_str = f'{epoch:>2}: {p1:.4f}, {p3:.4f}, {p5:.4f}, {psp1:.4f}, {psp3:.4f}, {psp5:.4f}, train_loss:{train_loss}'

        if args.model_type=='light' \
           and args.eval_type=='original':
            log_str += f' {g_p1:.4f}, {g_p3:.4f}, {g_p5:.4f}'
        if args.valid:
            log_str += ' valid'
        LOG.log(log_str)

        if max_only_p5 < p5:
            max_only_p5 = p5
            model.save_model(f'models/model-{get_exp_name()}.bin')

        if epoch >= args.epoch + 5 and max_only_p5 != p5:
            break


def get_exp_name():
    name = [args.dataset, '' if args.bert == 'bert-base' else args.bert]
    name.append(args.model_type)
    if args.model_type.startswith('ivfflat') or args.model_type.startswith('exact'):
        name.extend(['num-neg-mips', str(args.num_neg_mips)])
        name.extend(['step', str(args.mips_preprocess_step)])
        if args.model_type.startswith('ivfflat'):
            name.extend(['nlist', str(args.nlist)])
            name.extend(['nprobe', str(args.nprobe_train)])
            name.extend(['automatic-search' if args.automatic_search else 'manual-search'])
        name.extend(['eth', str(args.eth_in_epoch)])
    # name.append(args.eval_type)
    # if args.dataset in ['wiki500k', 'amazon670k']:
    if args.model_type=='light': # mohamm
        name.append('t'+str(args.group_y_group))

    return '_'.join([i for i in name if i != ''])


def init_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--batch', type=int, required=False, default=16)
parser.add_argument('--update_count', type=int, required=False, default=1)
parser.add_argument('--lr', type=float, required=False, default=0.0001)
parser.add_argument('--seed', type=int, required=False, default=6088)
parser.add_argument('--epoch', type=int, required=False, default=20)
parser.add_argument('--dataset', type=str, required=False, default='eurlex4k')
parser.add_argument('--bert', type=str, required=False, default='bert-base')

parser.add_argument('--max_len', type=int, required=False, default=512)

parser.add_argument('--valid', action='store_true')

parser.add_argument('--swa', action='store_true')
parser.add_argument('--swa_warmup', type=int, required=False, default=10)
parser.add_argument('--swa_step', type=int, required=False, default=100)

parser.add_argument('--group_y_group', type=int, default=0)
parser.add_argument('--group_y_candidate_num', type=int, required=False, default=3000)
parser.add_argument('--group_y_candidate_topk', type=int, required=False, default=10)

parser.add_argument('--eval_step', type=int, required=False, default=100000)

parser.add_argument('--hidden_dim', type=int, required=False, default=300)

parser.add_argument('--eval_model', action='store_true')

parser.add_argument('--eval_type', type=str, choices=['plain', 'original'], default='original',
                    help='Type of evaluation')
parser.add_argument('--model_type', type=str, choices=['light', 'full', 'mips', 'uniform'], default='light')

parser.add_argument('--a', default=0.55, type=float,
                    help='"a" hyperparameter for computing the inverse propensities')
parser.add_argument('--b', default=1.5, type=float,
                    help='"b" hyperparameter for computing the inverse propensities')

parser.add_argument('--num_neg_mips', default=5, type=int,
                    help='number of negative labels from mips in *-in-epoch-* samplings')
parser.add_argument('--mips_preprocess_step', default=1000, type=int,
                    help='step for preprocessing weights of the calssifier using mips in *-in-epoch-*')
parser.add_argument('--mips_preprocess_epoch', default=1, type=int,
                    help='epoch interval for preprocessing weights of the calssifier using mips in *-in-epoch-*')
parser.add_argument('--eth_in_epoch', default=0, type=int,
                    help='the epoch at which in-epoch-prop starts (before this epoch, only prop is used)')
parser.add_argument('--nlist', default=818, type=int,
                    help='the nlist parameter in faiss')
parser.add_argument('--nprobe_train', default=64, type=int,
                    help='the nprobe parameter in faiss for training')
parser.add_argument('--nprobe_eval', default=350, type=int,
                    help='the nprobe parameter in faiss for evaluation')


args = parser.parse_args()
print(json.dumps(args.__dict__, indent=4))

if __name__ == '__main__':
    init_seed(args.seed)

    print(get_exp_name())

    LOG = Logger('log_'+get_exp_name())
    
    print(f'load {args.dataset} dataset...')
    df, label_map, inv_prop, freq = createDataCSV(args.dataset, args.a, args.b)
    if args.valid:
        train_df, valid_df = train_test_split(df[df['dataType'] == 'train'],
                                              test_size=4000,
                                              random_state=1240)
        df.iloc[valid_df.index.values, 2] = 'valid'
        print('valid size', len(df[df['dataType'] == 'valid']))

    print(f'load {args.dataset} dataset with '
          f'{len(df[df.dataType =="train"])} train {len(df[df.dataType =="test"])} test with {len(label_map)} labels done')

    # if args.dataset in ['wiki500k', 'amazon670k']:
    if args.model_type=='light': # mohamm
        group_y = load_group(args.dataset, args.group_y_group)
        _group_y = []
        for idx, labels in enumerate(group_y):
            _group_y.append([])
            for label in labels:
                _group_y[-1].append(label_map[label])
            _group_y[-1] = np.array(_group_y[-1])
        group_y = np.array(_group_y)

        model = LightXML(n_labels=len(label_map), group_y=group_y, bert=args.bert,
                         update_count=args.update_count, use_swa=args.swa,
                         swa_warmup_epoch=args.swa_warmup, swa_update_step=args.swa_step,
                         candidates_topk=args.group_y_candidate_topk, candidates_num=args.group_y_candidate_num,
                         hidden_dim=args.hidden_dim, model_type=args.model_type, eval_type=args.eval_type, 
                         inv_prop=inv_prop)
    else:
        model = LightXML(n_labels=len(label_map), bert=args.bert,
                         update_count=args.update_count, use_swa=args.swa,
                         swa_warmup_epoch=args.swa_warmup, swa_update_step=args.swa_step,
                         candidates_num=args.group_y_candidate_num, model_type=args.model_type,
                         eval_type=args.eval_type, inv_prop=inv_prop, num_neg_mips=args.num_neg_mips,
                         mips_preprocess_step=args.mips_preprocess_step, mips_preprocess_epoch=args.mips_preprocess_epoch,
                         eth_in_epoch=args.eth_in_epoch, nlist=args.nlist, nprobe_train=args.nprobe_train,
                         nprobe_eval=args.nprobe_eval)

    if args.eval_model and args.model_type=='light':
        print(f'load models/model-{get_exp_name()}.bin')
        testloader = DataLoader(MDataset(df, 'test', model.get_tokenizer(), label_map, args.max_len, # get_fast_tokenizer() -> get_tokenizer()
                                         candidates_num=args.group_y_candidate_num),
                                batch_size=256, num_workers=0, shuffle=False)

        group_y = load_group(args.dataset, args.group_y_group)
        # validloader = DataLoader(MDataset(df, 'valid', model.get_fast_tokenizer(), label_map, args.max_len, group_y=group_y,
        #                                   candidates_num=args.group_y_candidate_num),
        #                          batch_size=256, num_workers=0, shuffle=False)
        model.load_state_dict(torch.load(f'models/model-{get_exp_name()}.bin'))
        model = model.cuda()

        print(len(df[df.dataType == 'test']))
        # model.one_epoch(0, validloader, None, mode='eval')

        pred_scores, pred_labels = model.one_epoch(0, testloader, None, mode='test')
        np.save(f'results/{get_exp_name()}-labels.npy', np.array(pred_labels))
        np.save(f'results/{get_exp_name()}-scores.npy', np.array(pred_scores))
        sys.exit(0)

    train(model, df, label_map)
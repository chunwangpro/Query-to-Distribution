import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
import glob
import re

import psqlparse
import itertools

import sys
sys.path.insert(0,'..')

import common
import datasets
import made
#import transformer
import math
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline

import pandas as pd
import IPython as ip
import copy

import estimators as estimators_lib

#from bloomfilter import BloomFilter
#from earlystopping import EarlyStopping

from graphviz import Digraph
import torch
from torch.autograd import Variable
# make_dot was moved to https://github.com/szagoruyko/pytorchviz
from torchviz import make_dot

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='wine', help='Dataset.')
parser.add_argument('--device', type=str, default='cuda:0', help='Device')
parser.add_argument('--query-size', type=int, default=1000, help='query size')
parser.add_argument('--scale', type=float, default=0.1, help='scale of generated table')
parser.add_argument('--num-conditions', type=int, default=2, help='num of conditions')
parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
parser.add_argument('--num-gpus', type=int, default=0, help='#gpus.')
parser.add_argument('--bs', type=int, default=4096, help='Batch size.')
parser.add_argument(
    '--warmups',
    type=int,
    default=0,
    help='Learning rate warmup steps.  Crucial for Transformer.')
parser.add_argument('--epochs',
                    type=int,
                    default=30,
                    help='Number of epochs to train for.')
parser.add_argument('--constant-lr',
                    type=float,
                    default=None,
                    help='Constant LR?')
parser.add_argument(
    '--column-masking',
    action='store_true',
    help='Column masking training, which permits wildcard skipping'\
    ' at querying time.')

# MADE.
parser.add_argument('--fc-hiddens',
                    type=int,
                    default=64,
                    help='Hidden units in FC.')
parser.add_argument('--emb-size',
                    type=int,
                    default=16,
                    help='embeding size')
parser.add_argument('--layers', type=int, default=5, help='# layers in FC.')
parser.add_argument('--residual', action='store_true', help='ResMade?')
parser.add_argument('--direct-io', action='store_true', help='Do direct IO?')
parser.add_argument(
    '--inv-order',
    action='store_true',
    help='Set this flag iff using MADE and specifying --order. Flag --order '\
    'lists natural indices, e.g., [0 2 1] means variable 2 appears second.'\
    'MADE, however, is implemented to take in an argument the inverse '\
    'semantics (element i indicates the position of variable i).  Transformer'\
    ' does not have this issue and thus should not have this flag on.')
parser.add_argument(
    '--input-encoding',
    type=str,
    default= 'embed',
    help='Input encoding for MADE/ResMADE, {binary, one_hot, embed,None}.')
parser.add_argument(
    '--output-encoding',
    type=str,
    default='embed',
    help='Iutput encoding for MADE/ResMADE, {one_hot, embed}.  If embed, '
    'then input encoding should be set to embed as well')
# Transformer.
parser.add_argument(
    '--heads',
    type=int,
    default=0,
    help='Transformer: num heads.  A non-zero value turns on Transformer'\
    ' (otherwise MADE/ResMADE).'
)
parser.add_argument('--blocks',
                    type=int,
                    default=2,
                    help='Transformer: num blocks.')
parser.add_argument('--dmodel',
                    type=int,
                    default=32,
                    help='Transformer: d_model.')
parser.add_argument('--dff', type=int, default=128, help='Transformer: d_ff.')
parser.add_argument('--transformer-act',
                    type=str,
                    default='gelu',
                    help='Transformer activation.')

# Ordering.
parser.add_argument('--num-orderings',
                    type=int,
                    default=1,
                    help='Number of orderings.')
parser.add_argument(
    '--order',
    nargs='+',
    type=int,
    required=False,
    help=
    'Use a specific ordering.  '\
    'Format: e.g., [0 2 1] means variable 2 appears second.'
)

parser.add_argument('--inference-opts',
                    action='store_true',
                    help='Tracing optimization for better latency.')

parser.add_argument('--num-queries', type=int, default=10, help='# queries.')
parser.add_argument('--num-cpu', type=int, default=16, help='# cpus.')

parser.add_argument('--err-csv',
                    type=str,
                    default='results.csv',
                    help='Save result csv to what path?')
parser.add_argument('--glob',
                    type=str,
                    help='Checkpoints to glob under models/.')
parser.add_argument('--blacklist',
                    type=str,
                    help='Remove some globbed checkpoint files.')
parser.add_argument('--psample',
                    type=int,
                    default=2048,
                    help='# of progressive samples to use per query.')
               

# Estimators to enable.
parser.add_argument('--run-sampling',
                    action='store_true',
                    help='Run a materialized sampler?')
parser.add_argument('--run-maxdiff',
                    action='store_true',
                    help='Run the MaxDiff histogram?')
parser.add_argument('--run-bn',
                    action='store_true',
                    help='Run Bayes nets? If enabled, run BN only.')
parser.add_argument('--run-CDF',
                    action='store_true',
                    help='Run CDF learner?')

# Bayes nets.
parser.add_argument('--bn-samples',
                    type=int,
                    default=200,
                    help='# samples for each BN inference.')
parser.add_argument('--bn-root',
                    type=int,
                    default=0,
                    help='Root variable index for chow liu tree.')
# Maxdiff
parser.add_argument(
    '--maxdiff-limit',
    type=int,
    default=30000,
    help='Maximum number of partitions of the Maxdiff histogram.')

args = parser.parse_args()


OPS = {
    '>': np.greater,
    '<': np.less,
    '>=': np.greater_equal,
    '<=': np.less_equal,
    '=': np.equal
}

def ReportModel(model, blacklist=None):
    ps = []
    for name, p in model.named_parameters():
        #print(p.requires_grad)
        if blacklist is None or blacklist not in name:
            ps.append(np.prod(p.size()))
    num_params = sum(ps)
    mb = num_params * 4 / 1024 / 1024
    print('Number of model parameters: {} (~= {:.1f}MB)'.format(num_params, mb))
    # print(model)
    return mb

def InitWeight(m):
    if type(m) == made.MaskedLinear or type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    if type(m) == nn.Embedding:
        nn.init.normal_(m.weight, std=0.02)

class DataGenerator:
    def __init__(self, table_meta, args=args, seed=0, verbose=True):
        torch.manual_seed(0)
        np.random.seed(0)
        self.DEVICE = args.device
        self.est_DEVICE = args.device
        args.column_masking=True
        self.seed = seed
        self.args = args
        self.verbose = verbose
        self.table = table_meta
        self.cols_to_train_size = table_meta.split_col_size
        self.fixed_ordering = None
        if args.order is not None:
            if verbose:
                print('Using passed-in order:', args.order)
            self.fixed_ordering = args.order

        if args.heads > 0:
            self.model = self.MakeTransformer()
        else:
            self.model = self.MakeMade()
            self.model_no_update = self.MakeMade()
        
        self.mb = ReportModel(self.model)

    def merge_vals(self,val1,val2,size2):
        for i in range(len(val1)):
            val1[i] = val1[i] * size2 + val2[i]
        return val1

    def _sample_n_split(self,
                num_samples,
                this_model,
                table,
                inp=None,):
        ncols = len(table.split_col_size)
        print ('total columns: {}'.format(nclols))
        logits = this_model(torch.zeros(1, this_model.nin, device=self.DEVICE, requires_grad=True))
        # print (logits.requires_grad)
        if inp is None:
            inp = self.inp[:num_samples]
        sub_col = [False] * ncols
        i = 0 
        while i < ncols:
            natural_idx = ordering[i] if ordering else i 
            column_idx = self.table.rev_index[natural_idx] 
            # Column i.
            op = operators[column_idx]
            # print (i, natural_idx, column_idx, op)
            all_val = columns[column_idx].all_distinct_values
            if op is not None:
                # There exists a filter.
                #no split col 
                if self.table.split_col_size[natural_idx] == len(all_val):
                    i += 1 
                # split col 
                else:
                    sub_col[natural_idx+1] = True 
                    i += 2
            else:
                i += 1
        column_values = []
        column_index = []
        for i in range(ncols):
            natural_idx = i
            column_idx = self.table.rev_index[natural_idx] 
            # If wildcard enabled, 'logits' wasn't assigned last iter.
            probs_i = torch.softmax(
                this_model.logits_for_col(natural_idx, logits), 1)
            # Num samples to draw for column i.
            if i != 0:
                num_i = 1
            else:
                num_i = num_samples 
            # print ('probs_i.shape', probs_i.shape)
            import random
            paths_vanished = (probs_i.sum(1) <= 0).view(-1, 1)
            probs_i = probs_i.masked_fill_(paths_vanished, 1.0)
            samples_i = torch.multinomial(
                probs_i, num_samples=num_i,
                replacement=True).view(-1, 1)  # [bs, num_i]
            column_index.append(samples_i)
            y_hard = torch.zeros((samples_i.shape[0], probs_i.shape[1]), device=self.DEVICE)
            y_hard.scatter_(1, samples_i, 1)
            inp = this_model.EncodeInput(
                y_hard,
                natural_col=natural_idx,
                out=inp).detach()
            if i < ncols - 1:
                logits = this_model.forward_with_encoded_input(inp)
            
        for i in range(ncols):
            natural_idx = i
            column_idx = self.table.rev_index[natural_idx]
            all_val = table.columns[column_idx].all_distinct_values
            if self.table.split_col_size[natural_idx] == len(all_val):
                # no split
                column_values.append([all_val[xx] for xx in column_index[natural_idx]])
            elif sub_col[natural_idx]:
                val1 = column_index[natural_idx-1]
                val2 = column_index[natural_idx]
                size2 = self.table.split_col_size[natural_idx]
                val = self.merge_vals(val1,val2,size2)
                column_values.append([all_val[xx] for xx in val])
        return column_values

    def _sample_n(self,
                num_samples,
                this_model,
                table,
                inp=None,
                primary_data=None,
                primary_meta=None,
                mapping=None):
        ncols = len(table.split_col_size)
        print ('total columns: {}'.format(ncols))
        logits = this_model(torch.zeros(1, this_model.nin, device=self.DEVICE, requires_grad=True))
        print ('input shape: {}'.format(inp.shape))
        if inp is None:
            inp = self.inp[:num_samples]
        column_values = []
        column_indexes = []
        probs = []
        emb_size = []
        for i in range(ncols):
            print ('column {} is {}'.format(i, table.columns[i].name))
            natural_idx = i
            column_idx = self.table.rev_index[natural_idx]
            all_val = table.columns[column_idx].all_distinct_values
            if primary_meta is None or i >= len(primary_meta.columns):
                # Generate foreign table according to the generated primary table.
                # If wildcard enabled, 'logits' wasn't assigned last iter.
                print ('logits: {}'.format(logits.shape))
                probs_i = torch.softmax(
                    this_model.logits_for_col(natural_idx, logits), 1)
                probs.append(probs_i.sum(1))
                # Num samples to draw for column i.
                if i != 0:
                    num_i = 1
                else:
                    num_i = num_samples 
                # print ('probs_i.shape', probs_i.shape)
                import random
                paths_vanished = (probs_i.sum(1) <= 0).view(-1, 1)
                probs_i = probs_i.masked_fill_(paths_vanished, 1.0)
                samples_i = torch.multinomial(
                    probs_i, num_samples=num_i,
                    replacement=True).view(-1, 1)  # [bs, num_i]
                column_values.append([all_val[xx.item()] for xx in samples_i])
                y_hard = torch.zeros((samples_i.shape[0], probs_i.shape[1]), device=self.DEVICE)
                column_indexes.append(samples_i)
                y_hard.scatter_(1, samples_i, 1)
                inp = this_model.EncodeInput(
                    y_hard,
                    natural_col=natural_idx,
                    out=inp).detach()
                if i < ncols - 1:
                    logits = this_model.forward_with_encoded_input(inp)
            else:
                probs_i = torch.softmax(
                    this_model.logits_for_col(natural_idx, logits), 1)
                probs.append(probs_i.sum(1))
                emb_size.append(probs_i.shape[1])
                samples_i = primary_data[i]
                if i == len(primary_meta.columns)-1:
                    print ('primary_meta.cardinality: {}'.format(primary_meta.cardinality))
                    valid_i = torch.ones(int(primary_meta.cardinality))
                    for k, primary_col in enumerate(primary_data):
                        exist_values = set(table.columns[k].all_distinct_values)
                        for kk, idx_k in enumerate(primary_col.data):
                            if primary_meta.columns[k].all_distinct_values[idx_k.item()] not in exist_values:
                                valid_i[kk] = 0
                    print ('valid_i: {}'.format(valid_i.shape))
                    sel = probs[0]
                    for p in probs[1:]:
                        sel = sel * p
                    print ('sel: {}'.format(sel.shape))
                    sel = sel[:int(primary_meta.cardinality)] * valid_i
                    expand_sample_i = torch.multinomial(
                                sel, num_samples=num_samples,
                                replacement=True)
                    print ('sel: {}, expand_sample_i: {}'.format(sel.shape, expand_sample_i.shape))
                    for k, primary_col in enumerate(primary_data):
                        print ('retry: column {} is {}'.format(k, table.columns[k].name))
                        samples_k = primary_col[expand_sample_i]
                        print ('samples_k: {}'.format(samples_k.shape))
                        column_idx = self.table.rev_index[k]
                        all_val = table.columns[column_idx].all_distinct_values
                        print (len(all_val))
                        column_values.append([all_val[xx.item()] for xx in samples_k])
                        column_indexes.append(samples_k)
                        y_hard = torch.zeros((num_samples, emb_size[k]), device=self.DEVICE)
                        y_hard.scatter_(1, samples_k, 1)
                        inp = this_model.EncodeInput(
                            y_hard,
                            natural_col=k,
                            out=inp).detach()
                    column_values.append(expand_sample_i.view(-1).tolist())
                    column_indexes.append(samples_k)
                else:
                    y_hard = torch.zeros((num_samples, probs_i.shape[1]), device=self.DEVICE)
                    print ('samples_i.shape: ', samples_i.shape)
                    samples_i = torch.cat([samples_i, torch.zeros(num_samples - samples_i.shape[0], 1).long()], dim=0)
                    print ('samples_i.shape: ', samples_i.shape)
                    print ('y_hard.shape: ', y_hard.shape)
                    y_hard.scatter_(1, samples_i, 1)
                    inp = this_model.EncodeInput(
                        y_hard,
                        natural_col=natural_idx,
                        out=inp).detach()
                if i < ncols - 1:
                    logits = this_model.forward_with_encoded_input(inp) 
        return column_values, column_indexes
    
    def generate_data(self, scale=1.0, path=None, path_naru=None, primary_data=None, primary_meta=None, mapping=None):
        if path_naru is None:
            PATH = '../models/{}-{:.1f}MB-{}-{}epochs-seed{}-querysize{}-numcondition{}.pt'.format(
                path, self.mb,self.model.name(),
                self.args.epochs, self.seed, self.args.query_size, self.args.num_conditions)
        else:
            PATH = path_naru
        print ('Loading From PATH {}'.format(PATH))
        self.model.load_state_dict(torch.load(PATH, map_location=torch.device(self.DEVICE)))
        self.model.eval()
        self.kZeros = torch.zeros(int(self.table.cardinality * scale),
                                    self.model.nin,
                                    device=self.DEVICE)
        self.inp = self.model.EncodeInput(self.kZeros)
        self.inp = self.inp.view(int(self.table.cardinality * scale), -1)
        return self._sample_n(int(self.table.cardinality * scale),
                    self.model,
                    self.table,
                    inp=self.inp,
                    primary_data=primary_data,
                    primary_meta=primary_meta,
                    mapping=mapping)

    def Query(self,
          query=None,
          do_print=True):
        assert query is not None
        cols, idxs, ops, vals, true_sel = query
        return self.estimator.Query(cols, ops, vals), true_sel

    def sample(self,num_sample):
        gen_data = self.model.sample(num_sample,device = self.DEVICE).cpu().numpy() 
        gen_data = rev_discrete(gen_data,self.table.columns)
        return  gen_data.values

    def MakeMade(self):
        if self.args.inv_order:
            print('Inverting order!')
            self.fixed_ordering = InvertOrder(self.fixed_ordering)

        model = made.MADE(
            nin=len(self.cols_to_train_size),
            hidden_sizes=[self.args.fc_hiddens] *
            self.args.layers if self.args.layers > 0 else [512, 256, 512, 128, 1024],
            nout=sum(self.cols_to_train_size),
            input_bins=self.cols_to_train_size,
            input_encoding=self.args.input_encoding,
            output_encoding=self.args.output_encoding,
            embed_size=self.args.emb_size,
            seed=self.seed,
            do_direct_io_connections=self.args.direct_io,
            natural_ordering=False if self.seed is not None and self.seed != 0 else True,
            residual_connections=True,
            fixed_ordering=self.fixed_ordering,
            column_masking=self.args.column_masking,
        ).to(self.DEVICE)
        return model

    def MakeTransformer(self):
        return transformer.Transformer(
            num_blocks=self.args.blocks,
            d_model=self.args.dmodel,
            d_ff=self.args.dff,
            num_heads=self.args.heads,
            nin=len(self.cols_to_train_size),
            input_bins=self.cols_to_train_size,
            use_positional_embs=True,
            activation=self.args.transformer_act,
            fixed_ordering=self.fixed_ordering,
            column_masking=self.args.column_masking,
            seed=self.seed,
        ).to(self.DEVICE)


if __name__ == '__main__':
    # args.dataset='wine'
    # args.fc_hiddens = 128
    # args.layers = 5
    # args.device = 'cuda:5'
    # args.query_size = 10000
    # args.lr = 5e-3
    # args.num_conditions = 3

    args.input_encoding = 'embed'
    args.output_encoding = 'embed'
    args.emb_size = 32
    args.direct_io = False
    args.column_masking = True
    torch.set_grad_enabled(False)
    print (args.device, args.query_size, args.lr, args.num_conditions)
    
    title_table = datasets.LoadTitle()
    joined_table = datasets.LoadTitleJoinCastinfo()

    idx_mapping_title2join = [{} for _ in title_table.columns]
    for k, c in enumerate(title_table.columns):
        # print (c.all_distinct_values)
        # print (joined_table.columns[k].all_distinct_values)
        distinct = joined_table.columns[k].all_distinct_values.tolist()
        for idx, v in enumerate(c.all_distinct_values):
            try:
                idx_j = distinct.index(v)
                idx_mapping_title2join[k][idx] = idx_j
            except:
                pass

    print (idx_mapping_title2join)

    ## Generate Primary Table `Title`
    table_train = title_table
    train_data = common.TableDataset(table_train)
    model = DataGenerator(title_table,seed=0)
    generated_data_title, generated_data_title_idx = model.generate_data(path='title', path_naru=None)
    print (len(generated_data_title))

    for k in range(len(generated_data_title_idx)):
        for kk in range(len(generated_data_title_idx[k])):
            key = generated_data_title_idx[k][kk].item()
            if key in idx_mapping_title2join[k]:
                generated_data_title_idx[k][kk][0] = idx_mapping_title2join[k][key]
            else:
                generated_data_title_idx[k][kk][0] = -1

    ## Generate Joined Table `TitleCastInfo`
    table_train = joined_table
    train_data = common.TableDataset(table_train)
    model = DataGenerator(joined_table,seed=0)
    generated_data_join, generated_data_join_idx = model.generate_data(scale=args.scale, path='title_cast_info', path_naru=None, primary_data=generated_data_title_idx, primary_meta=title_table, mapping=idx_mapping_title2join)
    print (len(generated_data_join))

    with open('../datasets/generated_title_castinfo_join.csv', 'w') as f:
        if args.dataset == 'dmv':
            titles = []
            for n in joined_table.columns:
                joined_table.append(n.pg_name)
            f.write(','.join(titles))
            f.write('\n')
        for row in range(len(generated_data_join[0])):
            for col in range(len(generated_data_join)):
                f.write(str(generated_data_join[col][row]))
                if col < len(generated_data_join) - 1:
                    f.write(',')
            f.write('\n')
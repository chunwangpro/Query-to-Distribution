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

# import psqlparse
import itertools

import common
import datasets
import made
# import transformer
import math
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline

import pandas as pd
import pdb
import IPython as ip
import copy

import estimators as estimators_lib

# from bloomfilter import BloomFilter
# from earlystopping import EarlyStopping

from graphviz import Digraph
import torch
from torch.autograd import Variable
# make_dot was moved to https://github.com/szagoruyko/pytorchviz
from torchviz import make_dot

# added by LPALG
import sys
# import numpy as np
# import scipy as sc
# import pandas as pd
from scipy import optimize
from copy import deepcopy

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='wine', help='Dataset.')
parser.add_argument('--device', type=str, default='cuda:0', help='Device')
parser.add_argument('--query-size', type=int, default=1000, help='query size')
parser.add_argument('--reload', type=bool, default=False, help='query size')
parser.add_argument('--num-conditions',
                    type=int,
                    default=1000,
                    help='num of conditions')
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
parser.add_argument('--emb-size', type=int, default=16, help='embeding size')
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
    default='embed',
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
parser.add_argument('--run-CDF', action='store_true', help='Run CDF learner?')

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
        # print(p.requires_grad)
        if blacklist is None or blacklist not in name:
            ps.append(np.prod(p.size()))
    num_params = sum(ps)
    mb = num_params * 4 / 1024 / 1024
    print('Number of model parameters: {} (~= {:.1f}MB)'.format(
        num_params, mb))
    # print(model)
    return mb


def InitWeight(m):
    if type(m) == made.MaskedLinear or type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    if type(m) == nn.Embedding:
        nn.init.normal_(m.weight, std=0.02)


def Oracle(table, query):
    cols, idxs, ops, vals = query
    oracle_est = estimators_lib.Oracle(table)

    return oracle_est.Query(cols, ops, vals)


def cal_true_card(query, table):
    cols, idxs, ops, vals = query
    ops = np.array(ops)
    probs = Oracle(table, (cols, idxs, ops, vals))
    return probs


def GenerateQuery(table, min_num_filters, max_num_filters, rng, dataset):
    """Generate a random query."""
    num_filters = rng.randint(max_num_filters - 1, max_num_filters)
    # print (235)
    cols, idxs, ops, vals = SampleTupleThenRandom(table, num_filters, rng,
                                                  dataset)
    # print (vals)
    sel = cal_true_card(
        (cols, idxs, ops, vals), table) / float(table.cardinality)
    return cols, idxs, ops, vals, sel


def SampleTupleThenRandom(table, num_filters, rng, dataset):
    vals = []
    # new_table = table.data.dropna(axis=0,how='any')
    new_table = table.data
    # print (248)
    s = new_table.iloc[rng.randint(0, new_table.shape[0])]
    vals = s.values
    # print (251)
    if dataset in ['dmv', 'dmv-tiny', 'order_line']:
        vals[6] = vals[6].to_datetime64()
    elif dataset in ['orders1', 'orders']:
        vals[4] = vals[4].to_datetime64()
    elif dataset == 'lineitem':
        vals[10] = vals[10].to_datetime64()
        vals[11] = vals[11].to_datetime64()
        vals[12] = vals[12].to_datetime64()
    # print (260)
    idxs = rng.choice(len(table.columns), replace=False, size=num_filters)
    # idxs = [12]
    cols = np.take(table.columns, idxs)
    # print (264)
    # print (cols)
    # If dom size >= 10, okay to place a range filter.
    # Otherwise, low domain size columns should be queried with equality.
    ops = rng.choice(['<=', '>=', '='], size=num_filters)
    ops_all_eqs = ['='] * num_filters
    sensible_to_do_range = [c.DistributionSize() >= 10 for c in cols]
    # print (271)
    ops = np.where(sensible_to_do_range, ops, ops_all_eqs)
    # print (273)
    # if num_filters == len(table.columns):
    #     return table.columns,np.arange(len(table.columns)), ops, vals
    # print (276)
    vals = vals[idxs]
    op_a = []
    val_a = []
    for i in range(len(vals)):
        val_a.append([vals[i]])
        op_a.append([ops[i]])
    # print (283)
    return cols, idxs, pd.DataFrame(op_a).values, pd.DataFrame(val_a).values


class Query2Distribute:
    def __init__(self, table_meta, args=args, seed=0, verbose=True):
        torch.manual_seed(0)
        np.random.seed(0)
        self.DEVICE = args.device if torch.cuda.is_available() else 'cpu'
        self.est_DEVICE = args.device
        args.column_masking = True
        self.seed = seed
        self.args = args
        self.verbose = verbose
        self.table = table_meta
        self.cols_to_train_size = table.split_col_size
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

    def fit(self, data, query_set=None, path=None, reload=False):
        if self.fixed_ordering is None:
            if self.seed is not None:
                PATH = 'models/{}-{:.1f}MB-{}-{}epochs-seed{}-querysize{}-numcondition{}.pt'.format(
                    path, self.mb, self.model.name(), self.args.epochs,
                    self.seed, self.args.query_size, self.args.num_conditions)
            else:
                PATH = 'models/{}-{:.1f}MB-{}-{}epochs-seed{}-{}.pt'.format(
                    path, self.mb, self.model.name(), self.args.epochs,
                    self.seed, time.time())
        else:
            annot = ''
            if self.args.inv_order:
                annot = '-invOrder'
            PATH = 'models/{}-{:.1f}MB-{}-{}epochs-seed{}-order{}{}.pt'.format(
                path, self.mb, self.model.name(), self.args.epochs, self.seed,
                '_'.join(map(str, self.fixed_ordering)), annot)

        if reload and os.path.exists(PATH):
            print('Loading from {}'.format(PATH))
            self.model.load_state_dict(torch.load(PATH))
            print('Completed!')
        self.model.apply(InitWeight)
        self.model.train()
        # opt = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), 2e-4)
        opt = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            self.args.lr)
        # oracle_est = estimators_lib.Oracle(table)
        # query_set = query_set[4:5]
        for epoch in range(self.args.epochs):
            total_loss = None
            cnt = 0
            for query in query_set:
                self.estimator = estimators_lib.ProgressiveSampling(
                    self.model,
                    self.args.output_encoding,
                    self.table,
                    self.args.psample,
                    device=self.est_DEVICE,
                    shortcircuit=self.args.column_masking)
                print(query)
                pred, true = self.Query(query)
                # pred, true = self.Query(query, forward_which=1)
                # print (pred.item(), true.item())

                loss = F.mse_loss(
                    torch.log(pred.unsqueeze(0)),
                    torch.log(torch.FloatTensor([true]).to(self.DEVICE)))
                # make_dot(loss).view()
                if total_loss is None:
                    total_loss = loss
                else:
                    total_loss = total_loss + loss
                cnt += 1
                if cnt % 64 == 0 or cnt == len(query_set):
                    total_loss = total_loss / (64 if cnt % 64 == 0 else cnt %
                                               64)
                    opt.zero_grad()
                    total_loss.backward()
                    # for name, parms in self.model.named_parameters():
                    #     print('-->name:', name, '-->grad_requirs:',parms.requires_grad, \
                    #     ' -->grad_value:',parms.grad, parms)
                    print(epoch, total_loss.item())
                    total_loss = None
                    opt.step()

        os.makedirs(os.path.dirname(PATH), exist_ok=True)
        torch.save(self.model.state_dict(), PATH)
        if self.verbose:
            print('Saved to:')
            print(PATH)

    def eval_model(self, data, path=None, reload=False):
        if self.fixed_ordering is None:
            if self.seed is not None:
                PATH = 'models/{}-{:.1f}MB-{}-{}epochs-seed{}-querysize{}-numcondition{}.pt'.format(
                    path, self.mb, self.model.name(), self.args.epochs,
                    self.seed, self.args.query_size, self.args.num_conditions)
            else:
                PATH = 'models/{}-{:.1f}MB-{}-{}epochs-seed{}-{}.pt'.format(
                    path, self.mb, self.model.name(), self.args.epochs,
                    self.seed, time.time())
        else:
            annot = ''
            if self.args.inv_order:
                annot = '-invOrder'

            PATH = 'models/{}-{:.1f}MB-{}-{}epochs-seed{}-order{}{}.pt'.format(
                path, self.mb, self.model.name(), self.args.epochs, self.seed,
                '_'.join(map(str, self.fixed_ordering)), annot)

        if reload and os.path.exists(PATH):
            print('Load ', PATH)
            self.model.load_state_dict(torch.load(PATH))
        else:
            if not isinstance(self.model, transformer.Transformer):
                if self.verbose:
                    print('Applying InitWeight()')
                self.model.apply(InitWeight)

            if isinstance(self.model, transformer.Transformer):
                opt = torch.optim.Adam(
                    list(self.model.parameters()),
                    2e-4,
                    betas=(0.9, 0.98),
                    eps=1e-9,
                )
            else:
                opt = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    2e-4)

            bs = self.args.bs
            log_every = 200

            train_losses = []
            train_start = time.time()
            # early_stopping = EarlyStopping(patience=3, verbose=False, delta=0.05)

            if self.verbose:
                print('Training done; evaluating likelihood on full data:')
        all_losses = self.RunEpoch('test',
                                   opt=None,
                                   train_data=data,
                                   val_data=data,
                                   batch_size=1024,
                                   log_every=500,
                                   return_losses=True)
        model_nats = np.mean(all_losses)
        model_bits = model_nats / np.log(2)
        self.model.model_bits = model_bits
        print(model_nats, model_bits)
        # os.makedirs(os.path.dirname(PATH), exist_ok=True)
        # torch.save(self.model.state_dict(), PATH)
        # if self.verbose:
        #     print('Saved to:')
        #     print(PATH)

    def RunEpoch(
            self,
            split,
            opt,
            train_data,
            val_data=None,
            batch_size=100,
            upto=None,
            epoch_num=None,
            verbose=False,
            log_every=10,
            return_losses=False,
    ):
        torch.set_grad_enabled(split == 'train')
        self.model.train() if split == 'train' else self.model.eval()
        dataset = train_data if split == 'train' else val_data
        losses = []
        # print (dataset[0])
        loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=(split == 'train'))

        # How many orderings to run for the same batch?
        nsamples = 1
        if hasattr(self.model, 'orderings'):
            nsamples = len(self.model.orderings)

        for step, xb in enumerate(loader):
            #start_time_step = time.time()

            if split == 'train':
                base_lr = 8e-4
                for param_group in opt.param_groups:
                    if self.args.constant_lr:
                        lr = args.constant_lr
                    elif self.args.warmups:
                        t = self.args.warmups
                        d_model = self.model.embed_size
                        global_steps = len(loader) * epoch_num + step + 1
                        lr = (d_model**-0.5) * min(
                            (global_steps**-.5), global_steps * (t**-1.5))
                    else:
                        lr = 1e-2

                    param_group['lr'] = lr

            if upto and step >= upto:
                break

            xb = xb.to(self.DEVICE).to(torch.float32)
            # print (xb[0])
            # Forward pass, potentially through several orderings.
            xbhat = None
            model_logits = []
            num_orders_to_forward = 1
            if split == 'test' and nsamples > 1:
                # At test, we want to test the 'true' nll under all orderings.
                num_orders_to_forward = nsamples

            for i in range(num_orders_to_forward):
                if hasattr(self.model, 'update_masks'):
                    # We want to update_masks even for first ever batch.
                    self.model.update_masks()

                model_out = self.model(xb)
                # print (model_out.shape)
                model_logits.append(model_out)
                if xbhat is None:
                    xbhat = torch.zeros_like(model_out)
                xbhat += model_out
            # print (xbhat.shape, xb.shape)
            if xbhat.shape == xb.shape:
                if mean:
                    xb = (xb * std) + mean
                loss = F.binary_cross_entropy_with_logits(
                    xbhat, xb, size_average=False) / xbhat.size()[0]
            else:
                if self.model.input_bins is None:
                    # NOTE: we have to view() it in this order due to the mask
                    # construction within MADE.  The masks there on the output unit
                    # determine which unit sees what input vars.
                    xbhat = xbhat.view(-1, self.model.nout // self.model.nin,
                                       self.model.nin)
                    # Equivalent to:
                    loss = F.cross_entropy(xbhat, xb.long(), reduction='none') \
                            .sum(-1).mean()
                else:
                    if num_orders_to_forward == 1:

                        loss, acc = self.model.nll(xbhat, xb)
                        loss = loss.mean()

                    else:
                        # Average across orderings & then across minibatch.
                        #
                        #   p(x) = 1/N sum_i p_i(x)
                        #   log(p(x)) = log(1/N) + log(sum_i p_i(x))
                        #             = log(1/N) + logsumexp ( log p_i(x) )
                        #             = log(1/N) + logsumexp ( - nll_i (x) )
                        #
                        # Used only at test time.
                        logps = []  # [batch size, num orders]
                        assert len(model_logits) == num_orders_to_forward, len(
                            model_logits)
                        for logits in model_logits:
                            # Note the minus.
                            loss = -self.model.nll(logits, xb)
                            logps.append(loss)
                        logps = torch.stack(logps, dim=1)
                        logps = logps.logsumexp(dim=1) + torch.log(
                            torch.tensor(1.0 / nsamples, device=logps.device))
                        loss = (-logps).mean()
            losses.append(loss.item())

            if self.verbose:
                if step % log_every == 0:
                    if split == 'train':
                        print(
                            'Epoch {} Iter {}, {} entropy gap {:.4f} bits (loss {:.3f}, data {:.3f}) {:.5f} lr'
                            .format(epoch_num, step, split,
                                    loss.item() / np.log(2) - self.table_bits,
                                    loss.item() / np.log(2), self.table_bits,
                                    lr))
                    else:
                        print(
                            'Epoch {} Iter {}, {} loss {:.4f} nats / {:.4f} bits'
                            .format(epoch_num, step, split, loss.item(),
                                    loss.item() / np.log(2)))

            if split == 'train':
                opt.zero_grad()
                loss.backward()
                opt.step()

            # if self.verbose:
            #     print('%s epoch average loss: %f' % (split, np.mean(losses)))
            # end_time_step = time.time()

            # print((end_time_step-start_time_step)*len(loader))
            # print(aa)
        if return_losses:
            return losses
        return np.mean(losses)

    def Query(self, query=None, do_print=True):
        assert query is not None
        cols, idxs, ops, vals, true_sel = query
        return self.estimator.Query(cols, ops, vals), true_sel

    def sample(self, num_sample):
        gen_data = self.model.sample(num_sample,
                                     device=self.DEVICE).cpu().numpy()
        gen_data = rev_discrete(gen_data, self.table.columns)
        return gen_data.values

    def MakeMade(self):
        if self.args.inv_order:
            print('Inverting order!')
            self.fixed_ordering = InvertOrder(self.fixed_ordering)

        model = made.MADE(
            nin=len(self.cols_to_train_size),
            hidden_sizes=[self.args.fc_hiddens] * self.args.layers
            if self.args.layers > 0 else [512, 256, 512, 128, 1024],
            nout=sum(self.cols_to_train_size),
            input_bins=self.cols_to_train_size,
            input_encoding=self.args.input_encoding,
            output_encoding=self.args.output_encoding,
            embed_size=self.args.emb_size,
            seed=self.seed,
            do_direct_io_connections=self.args.direct_io,
            natural_ordering=False
            if self.seed is not None and self.seed != 0 else True,
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


# LPALG
def dictionary_column_interval(table_size, query_set):
    n_column = table_size[1]
    column_interval = {}
    for i in range(n_column):
        column_interval[i] = set([0, sys.maxsize])
    for query in query_set:
        col_idxs = query[1]
        ops = query[2]
        vals = query[3]
        sel = query[4]
        for i in range(len(col_idxs)):
            column_interval[col_idxs[i]].add(vals[i][0])
    for k, v in column_interval.items():
        column_interval[k] = sorted(list(v))
    return column_interval


def dictionary_column_variable(column_to_interval):
    total_variables = 0
    column_to_variable = {}
    for k, v in column_to_interval.items():
        count = len(v)
        column_to_variable[k] = [total_variables + i for i in range(count)]
        total_variables += count
    return total_variables, column_to_variable


def dictionary_variable_interval(column_to_interval, column_to_variable):
    variable_to_interval = {}
    for column, variable in column_to_variable.items():
        for i in range(len(variable)):
            variable_to_interval[variable[i]] = column_to_interval[column][i]
    return variable_to_interval


def op_to_variables(column_to_variable, column, x_index, op):
    variable = np.array(column_to_variable[column])
    if op == '>':  # 从现在+1个区间，到无穷区间
        return list(variable[x_index + 1:])
    elif op == '>=':  # 从现在的区间，到无穷区间
        return list(variable[x_index:])
    elif op == '=':  # 现在的区间
        return [variable[x_index]]
    elif op == '<':  # 第一个区间到现在-1个区间
        return list(variable[:x_index])
    elif op == '<=':  # 第一个区间到现在这个区间
        return list(variable[:x_index + 1])


def dictionary_query_to_variable(query_set, column_to_interval,
                                 column_to_variable):
    query_to_variable = {}
    for i in range(len(query_set)):
        col_idxs = query_set[i][1]
        ops = query_set[i][2]
        vals = query_set[i][3]
        sel = query_set[i][4]
        query_to_variable[i] = []
        for j in range(len(col_idxs)):
            column = col_idxs[j]
            x_index = column_to_interval[column].index(vals[j][0])
            query_to_variable[i].append(
                op_to_variables(column_to_variable, column, x_index,
                                ops[j][0]))
    return query_to_variable


def seperate_column_variable(column_to_variable, query_to_variable):
    used_variable = set()
    for k, v in query_to_variable.items():
        for x in v:
            used_variable.update(set(x))

    column_used_variable = deepcopy(column_to_variable)
    for k, v in column_used_variable.items():
        column_used_variable[k] = sorted(list(set(v) & used_variable))

    column_unused_variable = deepcopy(column_to_variable)
    for k, v in column_unused_variable.items():
        column_unused_variable[k] = sorted(list(set(v) - used_variable))
    return used_variable, column_used_variable, column_unused_variable


def fun():
    def error(x):
        total_error = 0
        for value in column_to_variable.values():
            column_error = 0
            for x_idx in value:
                column_error += x[x_idx]
            total_error += (column_error - n_row)**2
        return total_error

    return error


def query_constraints(query_set, query_to_variable):
    query_constraints_list = []
    for key, values in query_to_variable.items():
        sel = query_set[key][4]
        for value in values:

            def value_constraints(x, sel=sel, value=value):
                error = 0
                for idx in value:
                    error += x[idx]
                return error - sel * n_row

            query_constraints_list.append(value_constraints)
    return query_constraints_list


def table_constraints(column_to_variable):
    table_constraints_list = []
    for value in column_to_variable.values():

        def column_constraints(x, value=value):
            error = 0
            for idx in value:
                error += x[idx]
            return error - n_row

        table_constraints_list.append(column_constraints)
    return table_constraints_list


def constraints():
    return query_eq_constraints + table_ineq_constraints


def randomized_rouding(x):
    int_x = deepcopy(x)
    for i in range(len(x)):
        xi = x[i]
        floor = np.floor(xi)
        ceil = np.ceil(xi)
        if not floor == ceil:
            int_x[i] = np.random.choice([floor, ceil],
                                        p=[xi - floor, ceil - xi])
    return int_x


if __name__ == '__main__':
    pdb.set_trace()
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
    args.direct_io = True
    args.column_masking = True
    print(args.device, args.query_size, args.lr, args.num_conditions,
          args.reload)
    #assert args.dataset in ['dmv-tiny', 'dmv']
    if args.dataset == 'dmv-tiny':
        table = datasets.LoadDmv('dmv-tiny.csv')
    elif args.dataset == 'dmv':
        table = datasets.LoadDmv()
    else:
        type_casts = {}
        if args.dataset in ['orders1']:
            type_casts = {4: np.datetime64, 5: np.float}
        elif args.dataset in ['orders']:
            type_casts = {4: np.datetime64}
        elif args.dataset == 'lineitem':
            type_casts = {
                10: np.datetime64,
                11: np.datetime64,
                12: np.datetime64
            }
        elif args.dataset == 'order_line':
            type_casts = {6: np.datetime64}

        table = datasets.LoadDataset(args.dataset + '.csv',
                                     args.dataset,
                                     type_casts=type_casts)

    # table_bits = Entropy(
    #     table,
    #     table.data.fillna(value=0).groupby([c.name for c in table.columns
    #                                        ]).size(), [2])[0]
    print(table.data.shape)
    table_train = table
    train_data = common.TableDataset(table_train)
    query_set = None
    model = Query2Distribute(table, seed=0)
    if not args.reload:
        print('Begin Generating Queries ...')
        rng = np.random.RandomState(1234)
        query_set = [
            GenerateQuery(table, 2, args.num_conditions + 1, rng, args.dataset)
            for i in range(args.query_size)
        ]
        pdb.set_trace()
        print('Complete Generating Queries ...')
        # model.fit(train_data, query_set, path=args.dataset, reload=args.reload)
        # model.eval_model(train_data, path=args.dataset, reload=True)
    # model.load_est(reload_model=True)
    # query

    # LPALG
    '''
    (array([Column(10, distribution_size=111),Column(0, distribution_size=106)], dtype=object), 
    array([10,  0]), 
    array([['<='],['=']], dtype=object), 
    array([[12.3],[ 6.5]]), 
    0.03340003078343851)
    query_set = [([10, 0], [['<='], ['=']], [[12.3],[6.5]], 0.03340003078343851),
                  ([8, 5], [['='], ['<=']], [[3.04],
                                             [47.]], 0.013544712944435893),
                  ([8, 2], [['>='], ['<=']], [[3.09],
                                              [0.36]], 0.5704171155918116)]
    '''
    table_size = table.data.shape  # (6497, 13)
    n_row = table_size[0]
    n_column = table_size[1]

    column_to_interval = dictionary_column_interval(table_size, query_set)
    print(column_to_interval)
    total_variables, column_to_variable = dictionary_column_variable(
        column_to_interval)
    variable_to_interval = dictionary_variable_interval(
        column_to_interval, column_to_variable)
    query_to_variable = dictionary_query_to_variable(query_set,
                                                     column_to_interval,
                                                     column_to_variable)
    used_x, col_used_x, col_unused_x = seperate_column_variable(
        column_to_variable, query_to_variable)

    query_constraints_list = query_constraints(query_set, query_to_variable)
    query_eq_constraints = [{
        'type': 'ineq',
        'fun': constraint
    } for constraint in query_constraints_list]
    table_constraints_list = table_constraints(column_to_variable)
    table_ineq_constraints = [{
        'type': 'ineq',
        'fun': constraint
    } for constraint in table_constraints_list]
    x0 = np.array([1] * total_variables)
    bounds = np.array([[0, None]] * total_variables)
    res = optimize.minimize(fun(),
                            x0,
                            method='SLSQP',
                            constraints=constraints(),
                            bounds=bounds)
    print(res)
    int_x = randomized_rouding(res.x).astype(int)
    print(int_x)
    column_length = [func(int_x) + n_row for func in table_constraints_list]
    print(column_length)
    table_generate_length = max(column_length)
    print(table_generate_length)

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
import random

# import psqlparse
import itertools

# import common
# import datasets
import made
import datetime
from schema import Query
from schema import Schema
from schema import Column
# import transformer
import math
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline

import pandas as pd
import IPython as ip
import copy

import sampling as estimators_lib

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
parser.add_argument('--goon', type=bool, default=False, help='query size')
parser.add_argument('--generate', type=bool, default=False, help='query size')
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
        if blacklist is None or blacklist not in name:
            ps.append(np.prod(p.size()))
    num_params = sum(ps)
    mb = num_params * 4 / 1024 / 1024
    print('Number of model parameters: {} (~= {:.1f}MB)'.format(
        num_params, mb))
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
    def __init__(self, schema, args=args, seed=0, verbose=True):
        torch.manual_seed(0)
        np.random.seed(0)
        self.DEVICE = args.device if torch.cuda.is_available() else 'cpu'
        self.est_DEVICE = args.device
        args.column_masking = True
        self.seed = seed
        self.args = args
        self.verbose = verbose
        self.table = schema
        self.cols_to_train_size = schema.split_col_size
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

    def fit(self, schema, queries=None, path=None, reload=False, goon=False):
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
        if args.generate and os.path.exists(PATH):
            print('Loading from {}'.format(PATH))
            self.model.load_state_dict(torch.load(PATH))
            print('Completed!')
            self.model.eval()
            self.estimator = estimators_lib.ProgressiveSampling(
                self.model,
                self.args.output_encoding,
                schema,
                self.args.psample,
                device=self.est_DEVICE,
                shortcircuit=self.args.column_masking)
            return self.estimator.generate()
        elif reload and os.path.exists(PATH):
            print('Loading from {}'.format(PATH))
            self.model.load_state_dict(torch.load(PATH))
            print('Completed!')
            self.model.eval()
            errors = []
            for query in queries:
                self.estimator = estimators_lib.ProgressiveSampling(
                    self.model,
                    self.args.output_encoding,
                    schema,
                    self.args.psample,
                    device=self.est_DEVICE,
                    shortcircuit=self.args.column_masking)
                pred, true = self.Query(query)
                print(pred.item() + 1, true)
                if (true / float(pred.item() + 1)) > 3 or (
                        float(pred.item() + 1) / true) > 3:
                    print(query.query)
                if pred.item() + 1 < true:
                    errors.append(true / float(pred.item() + 1))
                else:
                    errors.append(float(pred.item() + 1) / true)
            print(
                'mean: {}, median: {}, max: {}, 90th: {}, 95th: {}, 99th: {}'.
                format(np.mean(errors), np.median(errors), np.max(errors),
                       np.percentile(errors, 90), np.percentile(errors, 95),
                       np.percentile(errors, 99)))
        else:
            if goon and os.path.exists(PATH):
                print('Loading from {}'.format(PATH))
                self.model.load_state_dict(torch.load(PATH))
                print('Completed!')
            else:
                self.model.apply(InitWeight)
            self.model.train()
            opt = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                self.args.lr)
            for epoch in range(self.args.epochs):
                total_loss = None
                cnt = 0
                for query in queries:
                    # print (query.query)
                    self.estimator = estimators_lib.ProgressiveSampling(
                        self.model,
                        self.args.output_encoding,
                        schema,
                        self.args.psample,
                        device=self.est_DEVICE,
                        shortcircuit=self.args.column_masking)
                    pred, true = self.Query(query)
                    # print (pred, true)
                    loss = F.mse_loss(
                        torch.log(pred.unsqueeze(0) + 1e-5),
                        torch.log(torch.FloatTensor([true]).to(self.DEVICE)))
                    # if pred.item()+1 > true:
                    #     loss = pred.unsqueeze(0)+1 / true
                    # else:
                    #     loss = torch.FloatTensor([true]).to(self.DEVICE) / (pred.unsqueeze(0)+1)
                    # loss = F.mse_loss(pred.unsqueeze(0), torch.FloatTensor([true]).to(self.DEVICE))
                    # print (loss)
                    # make_dot(loss).view()
                    if total_loss is None:
                        total_loss = loss
                    else:
                        total_loss = total_loss + loss
                    # print ('query id: {}'.format(cnt))
                    cnt += 1
                    if cnt % 64 == 0 or cnt == len(queries):
                        total_loss = total_loss / (64 if cnt %
                                                   64 == 0 else cnt % 64)
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

        if return_losses:
            return losses
        return np.mean(losses)

    def Query(self, query=None, do_print=True):
        assert query is not None
        return self.estimator.estimate(query) * float(
            self.table.cardinality), query.true_cardinality

    def Generate(self):
        return self.estimator.generate()

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
    args.direct_io = True
    args.column_masking = True
    print(args.device, args.query_size, args.lr, args.num_conditions,
          args.reload)

    cardinality = 13364709
    is_main = Column('is_main', 0, 1, int)
    biz_order_id = Column('biz_order_id', 31635, 1146860131006968630, int)
    pay_status = Column('pay_status', 1, 12, int)
    is_detail = Column('is_detail', 0, 1, int)
    auction_id = Column('auction_id', 0, 220463047172604086, int)
    biz_type = Column('biz_type', 100, 52001, int)
    buyer_flag = Column('buyer_flag', 0, 205, int)
    options = Column('options', 0, 4611686022722355200, int)
    buyer_id = Column('buyer_id', 21006, 2208724496142, int)
    seller_id = Column('seller_id', 73, 2208694966044, int)
    attribute4 = Column('attribute4', 0, 2, int)
    logistics_status = Column('logistics_status', 1, 8, int)
    status = Column('status', 0, 1, int)
    s_time = datetime.datetime.strptime('2007-01-26 22:49:39',
                                        '%Y-%m-%d %H:%M:%S').timestamp()
    e_time = datetime.datetime.strptime('2020-09-01 17:18:58',
                                        '%Y-%m-%d %H:%M:%S').timestamp()
    gmt_create = Column('gmt_create', s_time, e_time, datetime)
    s_time = datetime.datetime.strptime('2007-01-26 22:49:39',
                                        '%Y-%m-%d %H:%M:%S').timestamp()
    e_time = datetime.datetime.strptime('2020-09-01 17:18:58',
                                        '%Y-%m-%d %H:%M:%S').timestamp()
    end_time = Column('end_time', s_time, e_time, datetime)
    s_time = datetime.datetime.strptime('2008-10-04 17:50:50',
                                        '%Y-%m-%d %H:%M:%S').timestamp()
    e_time = datetime.datetime.strptime('2020-07-30 17:18:59',
                                        '%Y-%m-%d %H:%M:%S').timestamp()
    pay_time = Column('pay_time', s_time, e_time, datetime)
    from_group = Column('from_group', 0, 4, int)
    sub_biz_type = Column('sub_biz_type', 0, 5007, int)
    attributes = Column('attributes', '', '', str)
    buyer_rate_status = Column('buyer_rate_status', 4, 7, int)
    parent_id = Column('parent_id', 0, 1146860131006968630, int)
    refund_status = Column('refund_status', 0, 14, int)
    # columns = [biz_order_id, is_detail, seller_id, auction_id, biz_type, pay_status, options, buyer_id, status, gmt_create, from_group]
    columns = [
        biz_order_id, refund_status, gmt_create, parent_id, sub_biz_type,
        is_detail, seller_id, auction_id, biz_type, pay_status, is_main,
        status, from_group, buyer_id, buyer_flag
    ]
    # columns = [biz_order_id, is_detail, auction_id, pay_status, options, buyer_id, from_group]
    # columns = [biz_order_id]
    table = Schema(columns, cardinality)
    queries = []
    types = {}
    for col in columns:
        types[col.name] = col.type
    # sql = "SELECT count(*) FROM tc_biz_order_0526 AS tc_biz_order WHERE is_main = 1 AND biz_type IN (5000, 1110, 6001, 8001, 760, 9999, 6868, 3600, 2100, 3000, 2700, 2600, 1400, 1410, 6800, 2500, 150, 3800, 3300, 3500, 2000, 110, 1102, 10000, 2410, 2400, 1500, 1201, 1200, 900, 620, 610, 600, 710, 500, 300, 200, 100) AND (options & 134217728 <> 134217728 OR options & 268435456 <> 268435456) AND buyer_id = 3893680462 AND options & 72057594037927936 <> 72057594037927936 AND options & 4503599627370496 <> 4503599627370496 AND options & 34359738368 <> 34359738368 AND options & 281474976710656 <> 281474976710656 AND options & 68719476736 <> 68719476736 AND options & 1073741824 <> 1073741824 AND status = 0 AND buyer_flag IN (5, 4, 3, 2, 1, 0) AND from_group = 0 AND attributes NOT LIKE '%;tbpwBizType:c2b2c;%' AND IFNULL(attribute4, 0) <> 1 AND IFNULL(attribute4, 0) <> 2"
    # queries.append(Query(sql, types, 3))
    with open(
            '/data1/jisun.sj/query2distribute/query2distribute/datasets/buyers_0032.tc_biz_order_0526.dedup.card.in3',
            'r') as f:
        lines = f.readlines()
        random.shuffle(lines)
        for query in lines:
            print(query)
            if int(query.split(',')[-1]) > 0:
                queries.append(
                    Query(','.join(query.split(',')[:-1]), types,
                          int(query.split(',')[-1])))
                # if len(queries) > 256:
                #     break
    table.extract_distinct_vals(queries)
    table.preprocess_cols()
    print('distinct vals: ', table.distinct_vals)

    model = Query2Distribute(table, seed=0)
    if not args.generate:
        model.fit(table,
                  queries,
                  path=args.dataset,
                  reload=args.reload,
                  goon=args.goon)
    else:
        dataset = model.fit(table, queries, path=args.dataset)
        print(len(dataset), len(dataset[0]))
        with open('{}_generated_9.csv'.format(args.dataset), 'w') as f:
            # f.write(','.join([c.name for c in columns]))
            # f.write('\n')
            for i in range(len(dataset[0])):
                f.write(','.join([str(d[i]) for d in dataset]))
                f.write('\n')
        print('Written!!')

    # model.eval_model(train_data, path=args.dataset,reload=True)
    # model.load_est(reload_model=True)
    ##query
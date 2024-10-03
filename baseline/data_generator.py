import argparse
import collections
import copy
import glob
import itertools

# import transformer
import math
import os
import pdb
import re
import time

import IPython as ip
import numpy as np
import pandas as pd
import psqlparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from graphviz import Digraph
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, OneHotEncoder
from torch.autograd import Variable

# make_dot was moved to https://github.com/szagoruyko/pytorchviz
from torchviz import make_dot

import common
import datasets
import estimators as estimators_lib
import made

# from bloomfilter import BloomFilter
# from earlystopping import EarlyStopping


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="wine", help="Dataset.")
parser.add_argument("--device", type=str, default="cuda:0", help="Device")
parser.add_argument("--query-size", type=int, default=1000, help="query size")
parser.add_argument("--num-conditions", type=int, default=1000, help="num of conditions")
parser.add_argument("--lr", type=float, default=5e-3, help="learning rate")
parser.add_argument("--num-gpus", type=int, default=0, help="#gpus.")
parser.add_argument("--bs", type=int, default=4096, help="Batch size.")
parser.add_argument(
    "--warmups", type=int, default=0, help="Learning rate warmup steps.  Crucial for Transformer."
)
parser.add_argument("--epochs", type=int, default=30, help="Number of epochs to train for.")
parser.add_argument("--constant-lr", type=float, default=None, help="Constant LR?")
parser.add_argument(
    "--column-masking",
    action="store_true",
    help="Column masking training, which permits wildcard skipping" " at querying time.",
)

# MADE.
parser.add_argument("--fc-hiddens", type=int, default=64, help="Hidden units in FC.")
parser.add_argument("--emb-size", type=int, default=16, help="embeding size")
parser.add_argument("--layers", type=int, default=5, help="# layers in FC.")
parser.add_argument("--residual", action="store_true", help="ResMade?")
parser.add_argument("--direct-io", action="store_true", help="Do direct IO?")
parser.add_argument(
    "--inv-order",
    action="store_true",
    help="Set this flag iff using MADE and specifying --order. Flag --order "
    "lists natural indices, e.g., [0 2 1] means variable 2 appears second."
    "MADE, however, is implemented to take in an argument the inverse "
    "semantics (element i indicates the position of variable i).  Transformer"
    " does not have this issue and thus should not have this flag on.",
)
parser.add_argument(
    "--input-encoding",
    type=str,
    default="embed",
    help="Input encoding for MADE/ResMADE, {binary, one_hot, embed,None}.",
)
parser.add_argument(
    "--output-encoding",
    type=str,
    default="embed",
    help="Iutput encoding for MADE/ResMADE, {one_hot, embed}.  If embed, "
    "then input encoding should be set to embed as well",
)
# Transformer.
parser.add_argument(
    "--heads",
    type=int,
    default=0,
    help="Transformer: num heads.  A non-zero value turns on Transformer"
    " (otherwise MADE/ResMADE).",
)
parser.add_argument("--blocks", type=int, default=2, help="Transformer: num blocks.")
parser.add_argument("--dmodel", type=int, default=32, help="Transformer: d_model.")
parser.add_argument("--dff", type=int, default=128, help="Transformer: d_ff.")
parser.add_argument("--transformer-act", type=str, default="gelu", help="Transformer activation.")

# Ordering.
parser.add_argument("--num-orderings", type=int, default=1, help="Number of orderings.")
parser.add_argument(
    "--order",
    nargs="+",
    type=int,
    required=False,
    help="Use a specific ordering.  " "Format: e.g., [0 2 1] means variable 2 appears second.",
)

parser.add_argument(
    "--inference-opts", action="store_true", help="Tracing optimization for better latency."
)

parser.add_argument("--num-queries", type=int, default=10, help="# queries.")
parser.add_argument("--num-cpu", type=int, default=16, help="# cpus.")

parser.add_argument(
    "--err-csv", type=str, default="results.csv", help="Save result csv to what path?"
)
parser.add_argument("--glob", type=str, help="Checkpoints to glob under models/.")
parser.add_argument("--blacklist", type=str, help="Remove some globbed checkpoint files.")
parser.add_argument(
    "--psample", type=int, default=2048, help="# of progressive samples to use per query."
)


# Estimators to enable.
parser.add_argument("--run-sampling", action="store_true", help="Run a materialized sampler?")
parser.add_argument("--run-maxdiff", action="store_true", help="Run the MaxDiff histogram?")
parser.add_argument(
    "--run-bn", action="store_true", help="Run Bayes nets? If enabled, run BN only."
)
parser.add_argument("--run-CDF", action="store_true", help="Run CDF learner?")

# Bayes nets.
parser.add_argument("--bn-samples", type=int, default=200, help="# samples for each BN inference.")
parser.add_argument("--bn-root", type=int, default=0, help="Root variable index for chow liu tree.")
# Maxdiff
parser.add_argument(
    "--maxdiff-limit",
    type=int,
    default=30000,
    help="Maximum number of partitions of the Maxdiff histogram.",
)

args = parser.parse_args()


OPS = {">": np.greater, "<": np.less, ">=": np.greater_equal, "<=": np.less_equal, "=": np.equal}


def ReportModel(model, blacklist=None):
    ps = []
    for name, p in model.named_parameters():
        # print(p.requires_grad)
        if blacklist is None or blacklist not in name:
            ps.append(np.prod(p.size()))
    num_params = sum(ps)
    mb = num_params * 4 / 1024 / 1024
    print("Number of model parameters: {} (~= {:.1f}MB)".format(num_params, mb))
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
        args.column_masking = True
        self.seed = seed
        self.args = args
        self.verbose = verbose
        self.table = table_meta
        self.cols_to_train_size = table.split_col_size
        self.fixed_ordering = None
        if args.order is not None:
            if verbose:
                print("Using passed-in order:", args.order)
            self.fixed_ordering = args.order

        if args.heads > 0:
            self.model = self.MakeTransformer()
        else:
            self.model = self.MakeMade()
            self.model_no_update = self.MakeMade()

        self.mb = ReportModel(self.model)

    def merge_vals(self, val1, val2, size2):
        for i in range(len(val1)):
            val1[i] = val1[i] * size2 + val2[i]
        return val1

    def _sample_n_split(
        self,
        num_samples,
        this_model,
        table,
        inp=None,
    ):
        ncols = len(table.split_col_size)
        print("total columns: {}".format(nclols))
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
                # no split col
                if self.table.split_col_size[natural_idx] == len(all_val):
                    i += 1
                # split col
                else:
                    sub_col[natural_idx + 1] = True
                    i += 2
            else:
                i += 1
        column_values = []
        column_index = []
        for i in range(ncols):
            natural_idx = i
            column_idx = self.table.rev_index[natural_idx]
            # If wildcard enabled, 'logits' wasn't assigned last iter.
            probs_i = torch.softmax(this_model.logits_for_col(natural_idx, logits), 1)
            # Num samples to draw for column i.
            if i != 0:
                num_i = 1
            else:
                num_i = num_samples
            # print ('probs_i.shape', probs_i.shape)
            import random

            paths_vanished = (probs_i.sum(1) <= 0).view(-1, 1)
            probs_i = probs_i.masked_fill_(paths_vanished, 1.0)
            samples_i = torch.multinomial(probs_i, num_samples=num_i, replacement=True).view(
                -1, 1
            )  # [bs, num_i]
            column_index.append(samples_i)
            y_hard = torch.zeros((samples_i.shape[0], probs_i.shape[1]), device=self.DEVICE)
            y_hard.scatter_(1, samples_i, 1)
            inp = this_model.EncodeInput(y_hard, natural_col=natural_idx, out=inp)
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
                val1 = column_index[natural_idx - 1]
                val2 = column_index[natural_idx]
                size2 = self.table.split_col_size[natural_idx]
                val = self.merge_vals(val1, val2, size2)
                column_values.append([all_val[xx] for xx in val])
        return column_values

    def _sample_n(
        self,
        num_samples,
        this_model,
        table,
        inp=None,
    ):
        ncols = len(table.split_col_size)
        print("total columns: {}".format(ncols))
        logits = this_model(torch.zeros(1, this_model.nin, device=self.DEVICE, requires_grad=True))
        print("input shape: {}".format(inp.shape))
        if inp is None:
            inp = self.inp[:num_samples]
        column_values = []
        for i in range(ncols):
            print("column {}".format(i))
            natural_idx = i
            column_idx = self.table.rev_index[natural_idx]
            all_val = table.columns[column_idx].all_distinct_values
            # If wildcard enabled, 'logits' wasn't assigned last iter.
            probs_i = torch.softmax(
                this_model.logits_for_col(natural_idx, logits), 1
            )  # 该行不同取值的概率
            # Num samples to draw for column i.
            if i != 0:
                num_i = 1
            else:
                num_i = num_samples
            # print ('probs_i.shape', probs_i.shape)
            import random

            paths_vanished = (probs_i.sum(1) <= 0).view(-1, 1)
            probs_i = probs_i.masked_fill_(paths_vanished, 1.0)
            samples_i = torch.multinomial(probs_i, num_samples=num_i, replacement=True).view(
                -1, 1
            )  # [bs, num_i] #采样
            column_values.append([all_val[xx] for xx in samples_i])
            y_hard = torch.zeros((samples_i.shape[0], probs_i.shape[1]), device=self.DEVICE)
            y_hard.scatter_(1, samples_i, 1)
            pdb.set_trace()
            inp = this_model.EncodeInput(y_hard, natural_col=natural_idx, out=inp)
            if i < ncols - 1:
                logits = this_model.forward_with_encoded_input(inp)
        return column_values

    def generate_data(self, path=None, path_naru=None):
        if path_naru is None:
            PATH = "models/{}-{:.1f}MB-{}-{}epochs-seed{}-querysize{}-numcondition{}.pt".format(
                path,
                self.mb,
                self.model.name(),
                self.args.epochs,
                self.seed,
                self.args.query_size,
                self.args.num_conditions,
            )
        else:
            PATH = path_naru
        print("Loading From PATH {}".format(PATH))
        # self.model.load_state_dict(torch.load(PATH, map_location=torch.device(self.DEVICE)))
        self.model.eval()
        self.kZeros = torch.zeros(self.table.cardinality, self.model.nin, device=self.DEVICE)
        self.inp = self.model.EncodeInput(self.kZeros)
        self.inp = self.inp.view(self.table.cardinality, -1)
        return self._sample_n(
            self.table.cardinality,
            self.model,
            self.table,
            inp=self.inp,
        )

    def Query(self, query=None, do_print=True):
        assert query is not None
        cols, idxs, ops, vals, true_sel = query
        return self.estimator.Query(cols, ops, vals), true_sel

    def sample(self, num_sample):
        gen_data = self.model.sample(num_sample, device=self.DEVICE).cpu().numpy()
        gen_data = rev_discrete(gen_data, self.table.columns)
        return gen_data.values

    def MakeMade(self):
        if self.args.inv_order:
            print("Inverting order!")
            self.fixed_ordering = InvertOrder(self.fixed_ordering)

        model = made.MADE(
            nin=len(self.cols_to_train_size),
            hidden_sizes=(
                [self.args.fc_hiddens] * self.args.layers
                if self.args.layers > 0
                else [512, 256, 512, 128, 1024]
            ),
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


if __name__ == "__main__":
    # args.dataset='wine'
    # args.fc_hiddens = 128
    # args.layers = 5
    # args.device = 'cuda:5'
    # args.query_size = 10000
    # args.lr = 5e-3
    # args.num_conditions = 3

    args.input_encoding = "embed"
    args.output_encoding = "embed"
    args.emb_size = 32
    args.direct_io = True
    args.column_masking = True
    torch.set_grad_enabled(False)
    print(args.device, args.query_size, args.lr, args.num_conditions)
    # assert args.dataset in ['dmv-tiny', 'dmv']
    if args.dataset == "dmv-tiny":
        table = datasets.LoadDmv("dmv-tiny.csv")
    elif args.dataset == "dmv":
        table = datasets.LoadDmv()
    else:
        type_casts = {}
        if args.dataset in ["orders1"]:
            type_casts = {4: np.datetime64, 5: np.float}
        elif args.dataset in ["orders"]:
            type_casts = {4: np.datetime64}
        elif args.dataset == "lineitem":
            type_casts = {10: np.datetime64, 11: np.datetime64, 12: np.datetime64}
        elif args.dataset == "order_line":
            type_casts = {6: np.datetime64}

        table = datasets.LoadDataset(args.dataset + ".csv", args.dataset, type_casts=type_casts)

    print(table.data.shape)
    table_train = table
    # pdb.set_trace()
    train_data = common.TableDataset(table_train)
    model = DataGenerator(table, seed=0)
    generated_data = model.generate_data(path=args.dataset, path_naru=None)
    print(len(generated_data), len(generated_data[1]))
    with open("datasets/generated_" + args.dataset + "_random.csv", "w") as f:
        if args.dataset == "dmv":
            titles = []
            for n in table.columns:
                titles.append(n.pg_name)
            f.write(",".join(titles))
            f.write("\n")
        for row in range(len(generated_data[0])):
            for col in range(len(generated_data)):
                f.write(str(generated_data[col][row]))
                if col < len(generated_data) - 1:
                    f.write(",")
            f.write("\n")

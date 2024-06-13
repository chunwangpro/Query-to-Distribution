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
parser.add_argument("--test-query-num", type=int, default=100, help="num of tested queries")
parser.add_argument(
    "--test-num-conditions", type=int, default=5, help="num of conditions in each tested query"
)
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


def Oracle(_table, query):
    cols, idxs, ops, vals = query
    oracle_est = estimators_lib.Oracle(_table)
    cols = np.take(_table.columns, idxs)
    return oracle_est.Query(cols, ops, vals)


def cal_true_card(query, _table):
    cols, idxs, ops, vals = query
    ops = np.array(ops)
    probs = Oracle(_table, (cols, idxs, ops, vals))
    return probs


def GenerateQuery(table_origin, min_num_filters, max_num_filters, rng, dataset, table_generate):
    """Generate a random query."""
    num_filters = rng.randint(max_num_filters - 1, max_num_filters)

    cols, idxs, ops, vals = SampleTupleThenRandom(table_origin, num_filters, rng, dataset)
    # print (vals)
    sel = cal_true_card((cols, idxs, ops, vals), table_origin)
    sel2 = cal_true_card((cols, idxs, ops, vals), table_generate)
    return cols, idxs, ops, vals, sel, sel2


def SampleTupleThenRandom(table, num_filters, rng, dataset):
    vals = []
    # new_table = table.data.dropna(axis=0,how='any')
    new_table = table.data
    s = new_table.iloc[rng.randint(0, new_table.shape[0])]
    vals = s.values

    if dataset in ["dmv", "dmv-tiny", "order_line"]:
        vals[6] = vals[6].to_datetime64()
    elif dataset in ["orders1", "orders"]:
        vals[4] = vals[4].to_datetime64()
    elif dataset == "lineitem":
        vals[10] = vals[10].to_datetime64()
        vals[11] = vals[11].to_datetime64()
        vals[12] = vals[12].to_datetime64()

    idxs = rng.choice(len(table.columns), replace=False, size=num_filters)
    # idxs = [12]
    cols = np.take(table.columns, idxs)
    # print (cols)
    # If dom size >= 10, okay to place a range filter.
    # Otherwise, low domain size columns should be queried with equality.
    ops = rng.choice(["<=", ">=", "="], size=num_filters)
    ops_all_eqs = ["="] * num_filters
    sensible_to_do_range = [c.DistributionSize() >= 10 for c in cols]

    ops = np.where(sensible_to_do_range, ops, ops_all_eqs)

    # if num_filters == len(table.columns):
    #     return table.columns,np.arange(len(table.columns)), ops, vals

    vals = vals[idxs]
    op_a = []
    val_a = []
    for i in range(len(vals)):
        val_a.append([vals[i]])
        op_a.append([ops[i]])

    return cols, idxs, pd.DataFrame(op_a).values, pd.DataFrame(val_a).values


if __name__ == "__main__":

    args.input_encoding = "embed"
    args.output_encoding = "embed"
    args.emb_size = 32
    args.direct_io = False
    args.column_masking = True
    print(args.device, args.query_size, args.lr, args.num_conditions)
    # assert args.dataset in ['dmv-tiny', 'dmv']
    if args.dataset == "dmv-tiny":
        table = datasets.LoadDmv("dmv-tiny.csv")
    elif args.dataset == "dmv":
        table_origin = datasets.LoadDmv()
        table_generate = datasets.LoadDmv("generated_dmv_random.csv")
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

        table_origin = datasets.LoadDataset(
            args.dataset + ".csv", args.dataset, type_casts=type_casts
        )
        table_generate = datasets.LoadDataset(
            "generated_" + args.dataset + "_random.csv",
            "generated_" + args.dataset,
            type_casts=type_casts,
        )

    # table_bits = Entropy(
    #     table,
    #     table.data.fillna(value=0).groupby([c.name for c in table.columns
    #                                        ]).size(), [2])[0]
    print(table_origin.data.shape, table_generate.data.shape)
    rng = np.random.RandomState(100)
    diff = []
    print(args.test_query_num, args.test_num_conditions)

    for i in range(args.test_query_num):
        cols, idxs, ops, vals, sel, sel2 = GenerateQuery(
            table_origin, 2, args.test_num_conditions + 1, rng, args.dataset, table_generate
        )
        # print (cols, idxs ,ops, vals)
        if sel == 0:
            sel += 1
        if sel2 == 0:
            sel2 += 1
        elif sel < sel2:
            diff.append(float(sel2) / sel)
        else:
            diff.append(float(sel) / sel2)
        # print (sel, sel2)
    print(
        np.mean(diff),
        np.median(diff),
        np.percentile(diff, 90),
        np.percentile(diff, 95),
        np.percentile(diff, 99),
        np.max(diff),
    )

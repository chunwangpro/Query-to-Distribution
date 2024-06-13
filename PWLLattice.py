import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_lattice as tfl
from matplotlib import pyplot as plt
from tqdm import tqdm

from models import *
from query_func import *

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="wine2", help="Dataset.")
parser.add_argument("--query-size", type=int, default=1000, help="query size")
parser.add_argument("--min-conditions", type=int, default=1, help="min num of conditions")
parser.add_argument("--max-conditions", type=int, default=2, help="max num of conditions")
parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train for.")
parser.add_argument("--bs", type=int, default=10000, help="Batch size.")
parser.add_argument("--loss", type=str, default="MSE", help="Loss.")
parser.add_argument("--lattice-size", type=int, default=2, help="Lattice size.")
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
parser.add_argument("--seed", type=int, default=42, help="Random seed")

try:
    args = parser.parse_args()
except:
    # args = parser.parse_args([])
    args, unknown = parser.parse_known_args()


def make_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


OPS = {">": np.greater, "<": np.less, ">=": np.greater_equal, "<=": np.less_equal, "=": np.equal}

FilePath = (
    f"{args.dataset}_{args.query_size}_{args.min_conditions}_{args.max_conditions}_{args.loss}"
)
resultsPath = f"results/{FilePath}"
modelPath = f"saved_models/{FilePath}"
make_directory(resultsPath)
make_directory(modelPath)

print("\nBegin Loading Data ...")
print(f"{args.dataset}.csv")
table = np.loadtxt(f"datasets/{args.dataset}.csv", delimiter=",")
np.savetxt(f"{resultsPath}/original_table.csv", table, delimiter=",")
print("Done.\n")

print("Begin Generating Queries Set ...")
table_size = table.shape
rng = np.random.RandomState(args.seed)
query_set = [
    generate_random_query(table, args.min_conditions, args.max_conditions + 1, rng)
    for _ in tqdm(range(args.query_size))
]
print("Done.\n")

print("Begin Intervalization ...")
unique_intervals = column_intervalization(table_size, query_set)
unique_intervals
column_interval_number = count_column_unique_interval(unique_intervals)
print(f"{column_interval_number=}")
print("Done.\n")

# 修改 x = [sys.maxsize] * n_column     # 这里使用每个col_unique_interval的最后一个元素即可
# 如果使用两个input的话，一个修改为最大，一个修改为最小
train_X = []
n_column = table_size[1]
for query in query_set:
    x = [sys.maxsize] * n_column  # 这里使用每个col_unique_interval的最后一个元素即可
    idxs, _, vals, _ = query
    for i in range(len(idxs)):
        x[idxs[i]] = vals[i]
    train_X.append(x)
train_X = np.array(train_X).astype(np.float32)
train_Y = np.array([[query[-1]] for query in query_set], dtype=np.float32)


# 可以PWL改成三次样条吗
m = PWLLattice(
    modelPath,
    table_size,
    unique_intervals,
    pwl_keypoints=None,
    lattice_size=args.lattice_size,
)

m.fit(train_X, train_Y, lr=args.lr, bs=args.bs, epochs=args.epochs, loss=args.loss)
m.load()

dataNew = m.generate(unique_intervals, batch_size=10000)
np.savetxt(f"{resultsPath}/generated_table.csv", dataNew, delimiter=",")

Q_error = calculate_Q_error(dataNew, query_set, table_size)
print_Q_error(Q_error, args, resultsPath)

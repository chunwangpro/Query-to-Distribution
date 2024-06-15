# PWL - Lattice numpy Version
# PWL - Lattice model is an 1-input model

# Query Phase:
## calculate_query_cardinality_numpy

# Generation Phase:
## generate_from_batches / np.concatenate

# no Plottings
import warnings

warnings.filterwarnings("ignore")
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
parser.add_argument("--dataset", type=str, default="wine3", help="Dataset.")
parser.add_argument("--query-size", type=int, default=10000, help="query size")
parser.add_argument("--min-conditions", type=int, default=1, help="min num of query conditions")
parser.add_argument("--max-conditions", type=int, default=2, help="max num of query conditions")
parser.add_argument("--lattice-size", type=int, default=2, help="Lattice size for each column.")
parser.add_argument("--pwl-n", type=int, default=1, help="pwl layer number for each column.")

parser.add_argument("--pwl-tanh", type=bool, default=False, help="tanh layer after pwl.")
parser.add_argument("--boundary", type=bool, default=False, help="add boundary point to train set.")
parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs to train for.")
parser.add_argument("--bs", type=int, default=1000, help="Batch size.")
parser.add_argument("--loss", type=str, default="MSE", help="Loss.")
parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")

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
rng = np.random.RandomState(42)
query_set = [
    generate_random_query(table, args.min_conditions, args.max_conditions, rng)
    for _ in tqdm(range(args.query_size))
]
print("Done.\n")


print("Begin Intervalization ...")
unique_intervals = column_intervalization(query_set, table_size)
unique_intervals
column_interval_number = count_column_unique_interval(unique_intervals)
print(f"{column_interval_number=}")
print("Done.\n")


print("Begin Building Train set and Model ...")
X, y = build_train_set_1_input(query_set, unique_intervals)
if args.boundary:
    X, y = add_boundary_1_input(X, y, unique_intervals)

m = PWLLattice(
    modelPath,
    table_size,
    unique_intervals,
    pwl_keypoints=None,
    pwl_n=args.pwl_n,
    lattice_size=args.lattice_size,
    pwl_tanh=args.pwl_tanh,
)
print("Done.\n")


m.fit(X, y, lr=args.lr, bs=args.bs, epochs=args.epochs, loss=args.loss)
m.load()


dataNew = m.generate_by_row(unique_intervals, batch_size=10000)
np.savetxt(f"{resultsPath}/generated_table.csv", dataNew, delimiter=",")


Q_error = calculate_Q_error(dataNew, query_set, table_size)
print_Q_error(Q_error, args, resultsPath)

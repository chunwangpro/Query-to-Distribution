# rewrite by numpy to achieve faster speed
# PWL - Lattice model is an 1-input model
# PWL-Lattice-Copula model is an 2-input model

# Query Phase:
## calculate_query_cardinality: numpy version

# Generation Phase:
## generate_by_row / generate_by_col

# no Plottings
import argparse
import os

import numpy as np
from tqdm import tqdm

from CDF_models import *
from dataset import *
from generators import *
from utils import *

np.random.seed(42)


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="1-input", help="model type")
parser.add_argument("--dataset", type=str, default="wine5", help="Dataset.")
parser.add_argument("--query-size", type=int, default=100, help="query size")
parser.add_argument("--min-conditions", type=int, default=1, help="min num of query conditions")
parser.add_argument("--max-conditions", type=int, default=5, help="max num of query conditions")

parser.add_argument("--lattice-size", type=int, default=2, help="Lattice size for each column.")
parser.add_argument(
    "--last-lattice-size", type=int, default=2, help="Lattice size for Joint CDF model."
)
parser.add_argument(
    "--use-last-pwl", type=bool, default=False, help="whether use pwl layer after model output."
)
parser.add_argument(
    "--boundary", type=bool, default=False, help="whether add boundary point to train set."
)
parser.add_argument(
    "--unique-train", type=bool, default=False, help="whether make train set unique."
)
parser.add_argument("--pwl-n", type=int, default=1, help="Number of PWL layer for each column.")
parser.add_argument(
    "--pwl-tanh", type=bool, default=False, help="whether add tanh activation layer after pwl."
)
parser.add_argument("--epochs", type=int, default=2000, help="Number of train epochs.")
parser.add_argument("--bs", type=int, default=1000, help="Batch size.")
parser.add_argument("--loss", type=str, default="MSE", help="Loss.")
parser.add_argument("--opt", type=str, default="adamax", help="Optimizer.")
parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")


try:
    args = parser.parse_args()
except:
    # args = parser.parse_args([])
    args, unknown = parser.parse_known_args()


FilePath = (
    f"{args.dataset}_{args.query_size}_{args.min_conditions}_{args.max_conditions}_{args.model}"
)


def make_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


resultsPath = f"results/{FilePath}"
modelPath = f"models/{FilePath}"
make_directory(resultsPath)
make_directory(modelPath)


OPS = {
    ">": np.greater,
    "<": np.less,
    ">=": np.greater_equal,
    "<=": np.less_equal,
    "=": np.equal,
}


print("\nBegin Loading Data ...")
table, original_table_columns, sorted_table_columns, max_decimal_places = load_and_process_dataset(
    args.dataset, resultsPath
)
table_size = table.shape
print(f"{args.dataset}.csv,    shape: {table_size}")
print("Done.\n")


print("Begin Generating Queries Set ...")
rng = np.random.RandomState(42)
query_set = [generate_random_query(table, args, rng) for _ in tqdm(range(args.query_size))]
print("Done.\n")


print("Begin Intervalization ...")
unique_intervals = column_intervalization(query_set, table_size)
column_interval_number = count_column_unique_interval(unique_intervals)
print(f"{column_interval_number=}")
print("Done.\n")


print("Begin Building Train set and Model ...")
X, y, m, values = setup_train_set_and_model(
    args, query_set, unique_intervals, modelPath, table_size
)
# m.show_all_attributes()


# here use lattice layer as the last layer to learn the joint CDF
# can be replaced by other layers, e.g., AutoRegressive Model
# last_lattice_size = args.lattice_size if args.last_lattice_size == 0 else args.last_lattice_size

# JointCDFodel = Lattice(
#     lattice_size=[last_lattice_size] * table_size[1],
#     monotonicities=["increasing"] * table_size[1],
#     col_idx="Joint-CDF",
# )

JointCDFodel = ResNetCDFLayer


m.build_model(JointCDFodel)
print("Done.\n")


m.fit(X, y, args)
m.load()


Table_Generated = m.generate_table_by_row(values, batch_size=10000)
Q_error = calculate_Q_error(Table_Generated, query_set)
print_Q_error(Q_error, args, resultsPath)
print(f"\n Original table shape : {table_size}")
print(f"Generated table shape : {Table_Generated.shape}")

recovered_Table_Generated = recover_table_as_original(
    Table_Generated, original_table_columns, sorted_table_columns, max_decimal_places
)
recovered_Table_Generated.to_csv(f"{resultsPath}/generated_table.csv", index=False, header=False)

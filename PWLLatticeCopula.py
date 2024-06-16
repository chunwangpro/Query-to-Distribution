# PWL-Lattice-Copula numpy Version
# PWL-Lattice-Copula model is an 2-input model

# Query Phase:
## calculate_query_cardinality_numpy

# Generation Phase:
## generate_by_row / generate_row_batch_table / np.concatenate

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
from pyDOE import lhs
from tqdm import tqdm

# import common
# import datasets
# import estimators as estimators_lib
from models import *
from query_func import *

# def Oracle(table, query):
#     cols, idxs, ops, vals = query
#     oracle_est = estimators_lib.Oracle(table)
#     return oracle_est.Query(cols, ops, vals)


# def cal_true_card(query, table):
#     cols, idxs, ops, vals = query
#     ops = np.array(ops)
#     probs = Oracle(table, (cols, idxs, ops, vals))
#     return probs


# def GenerateQuery(table, min_num_filters, max_num_filters, rng, dataset):
#     """Generate a random query."""
#     num_filters = rng.randint(max_num_filters - 1, max_num_filters)
#     cols, idxs, ops, vals = SampleTupleThenRandom(table, num_filters, rng, dataset)
#     sel = cal_true_card((cols, idxs, ops, vals), table) / float(table.cardinality)
#     return cols, idxs, ops, vals, sel


# def SampleTupleThenRandom(table, num_filters, rng, dataset):
#     vals = []
#     new_table = table.data
#     s = new_table.iloc[rng.randint(0, new_table.shape[0])]
#     vals = s.values

#     idxs = rng.choice(len(table.columns), replace=False, size=num_filters)
#     cols = np.take(table.columns, idxs)
#     # If dom size >= 10, okay to place a range filter.
#     # Otherwise, low domain size columns should be queried with equality.
#     # ops = rng.choice(['='], size=num_filters)
#     # ops = rng.choice(['<=', '>=', '>', '<'], size=num_filters)
#     # ops = rng.choice(['<=', '>='], size=num_filters)
#     ops = rng.choice(["<="], size=num_filters)
#     # ops_all_eqs = ['='] * num_filters
#     # sensible_to_do_range = [c.DistributionSize() >= 10 for c in cols]
#     # ops = np.where(sensible_to_do_range, ops, ops_all_eqs)
#     # if num_filters == len(table.columns):
#     #     return table.columns,np.arange(len(table.columns)), ops, vals
#     vals = vals[idxs]
#     op_a = []
#     val_a = []
#     for i in range(len(vals)):
#         val_a.append([vals[i]])
#         op_a.append([ops[i]])
#     return cols, idxs, pd.DataFrame(op_a).values, pd.DataFrame(val_a).values


# def dictionary_column_interval(table_size, query_set):
#     # Traverse all queries to apply the intervalization skill for each column
#     # deal with columns with all positive values
#     n_column = table_size[1]
#     column_interval = {}
#     for i in range(n_column):
#         column_interval[i] = set()
#     for query in query_set:
#         _, idxs, _, vals, _ = query
#         for i in range(len(idxs)):
#             column_interval[idxs[i]].add(vals[i][0])
#     for k, v in column_interval.items():
#         column_interval[k] = sorted(list(v))
#         least, great = column_interval[k][0], column_interval[k][-1]
#         column_interval[k] = sorted([0, least / 2] + column_interval[k] + [great + 1])
#     return column_interval


# def count_column_unique_interval(unique_intervals):
#     # count unique query interval in each column
#     return [len(v) for v in unique_intervals.values()]


# def process_train_data(unique_intervals, query_set):
#     X, Y = [], []
#     origin = np.array([[0, v[-1]] for v in unique_intervals.values()]).ravel()
#     for query in query_set:
#         x = list(origin)
#         _, idxs, ops, vals, sel = query
#         for i in range(len(idxs)):
#             if ops[i][0] == "<=":
#                 x[idxs[i] * 2 + 1] = vals[i][0]
#             elif ops[i][0] == "<":
#                 ind = unique_intervals[idxs[i]].index(vals[i][0]) - 1
#                 x[idxs[i] * 2 + 1] = unique_intervals[idxs[i]][ind]
#             elif ops[i][0] == ">":
#                 x[idxs[i] * 2] = vals[i][0]
#             elif ops[i][0] == ">=":
#                 ind = unique_intervals[idxs[i]].index(vals[i][0]) + 1
#                 x[idxs[i] * 2] = unique_intervals[idxs[i]][ind]
#             elif ops[i][0] == "=":
#                 ind = unique_intervals[idxs[i]].index(vals[i][0]) - 1
#                 x[idxs[i] * 2] = unique_intervals[idxs[i]][ind]
#                 x[idxs[i] * 2 + 1] = vals[i][0]
#         X.append(x)
#         Y.append(sel)
#     X = np.array(X).astype(np.float32)
#     Y = np.array(Y).astype(np.float32).reshape(-1, 1)
#     total = np.concatenate((X, Y), axis=1)
#     # total = np.unique(total, axis=0)
#     #     choose = np.random.choice(total.shape[0], size=round(total.shape[0]*train_size), replace=False)
#     #     others = list(set(range(total.shape[0])) - set(choose))
#     #     train, test = total[choose], total[others]
#     #     df_train = pd.DataFrame(train, columns=[f'col_{i}' for i in range(total.shape[1])])
#     df_train = pd.DataFrame(total, columns=[f"col_{i}" for i in range(total.shape[1])])
#     # boundary
#     df_train.loc[len(df_train.index)] = [0] * total.shape[1]
#     zero = [[v[-1], 0] for v in unique_intervals.values()]
#     df_train.loc[len(df_train.index)] = list(np.array(zero).ravel()) + [0.0]
#     one = [[0, v[-1]] for v in unique_intervals.values()]
#     df_train.loc[len(df_train.index)] = list(np.array(one).ravel()) + [1.0]

#     new_train = np.array(df_train.sort_values(by=list(df_train.columns)[:-1]))
#     train_X, train_Y = np.hsplit(new_train, [-1])

#     #     df_test = pd.DataFrame(test, columns=[f'col_{i}' for i in range(total.shape[1])])
#     #     new_test = np.array(df_test.sort_values(by=list(df_test.columns)[:-1]))
#     #     test_X, test_Y = np.hsplit(new_test, [-1])
#     return train_X, train_Y  # , test_X, test_Y


def generate_data_new(grid, model):
    assert grid.shape[1] == n_column
    # transform 1-input grid to 2-input extend grid
    length = grid.shape[0]
    inf = [0] * length
    grid_dict = {}
    for i in range(n_column):
        grid_dict[f"col_{i}_inf"] = inf
        grid_dict[f"col_{i}_sup"] = grid[:, i]
    extend_grid = np.array(pd.DataFrame(grid_dict))
    print("Begin model inference")
    pred = model.inference(extend_grid)
    print("Done")
    # newpred is the predict cardinality
    newpred = np.round(pred * n_row)
    # newpred = np.round(pred)
    # delete all the zero cardinality rows
    line = pd.DataFrame(
        np.concatenate([grid, newpred], axis=1),
        columns=[f"col_{i}" for i in range(n_column)] + ["card"],
    )
    nozero = (line == 0).sum(axis=1)
    line = line[nozero == 0].reset_index(drop=True)
    grid, pred = np.hsplit(np.array(line), [-1])
    pred = pred.astype(int)
    # generate dataNew
    print("\nBegin generating table...")
    dataNew = pd.DataFrame(columns=[f"col_{i}" for i in range(n_column)], index=range(n_row))
    count = 0
    for i in trange(grid.shape[0]):
        df = dataNew
        grid_value = grid[i]
        for j in range(n_column):
            df = df.query(f"col_{j} <= {grid_value[j]}")
        card = pred[i][0] - df.shape[0]
        if card > 0:
            # df3 = pd.DataFrame({f"col_{k}": [grid_value[k]] * card for k in range(n_column)})
            # dataNew = dataNew.append(df3, ignore_index = True)
            dataNew.iloc[count : count + card, :] = grid_value
            count += card
            if count > n_row:
                print("Reached table length in ", i, grid.shape[0])
                break
        # print table length every 5000
        if i % 5000 == 0:
            print(count)
    dataNew.dropna(axis=0, how="all", inplace=True)
    return dataNew


# def execute_query(dataNew, query_set):
#     diff = []
#     for query in tqdm(query_set):
#         df = dataNew
#         _, idxs, ops, vals, sel = query
#         for i in range(len(idxs)):
#             op = "==" if ops[i][0] == "=" else ops[i][0]
#             df = df.query(f"col_{idxs[i]} {op} {vals[i][0]}")
#         card = 1 if round(sel * n_row) == 0 else round(sel * n_row)
#         card2 = 1 if df.shape[0] == 0 else df.shape[0]
#         diff.append(max(card / card2, card2 / card))
#     return diff


# def print_error(diff, args):
#     print(
#         f"\n\n Q-error of Lattice (query size={args.query_size}, condition={args.num_conditions}, epoches={args.epochs}):\n"
#     )
#     print(f"min:    {np.min(diff)}")
#     print(f"10:     {np.percentile(diff, 10)}")
#     print(f"20:     {np.percentile(diff, 20)}")
#     print(f"30:     {np.percentile(diff, 30)}")
#     print(f"40:     {np.percentile(diff, 40)}")
#     print(f"median: {np.median(diff)}")
#     print(f"60:     {np.percentile(diff, 60)}")
#     print(f"70:     {np.percentile(diff, 70)}")
#     print(f"80:     {np.percentile(diff, 80)}")
#     print(f"90:     {np.percentile(diff, 90)}")
#     print(f"95:     {np.percentile(diff, 95)}")
#     print(f"max:    {np.max(diff)}")
#     print(f"mean:   {np.mean(diff)}")


class LatticeCDFLayer(tf.keras.Model):
    def __init__(self, dim, lattice_size=2):
        super().__init__()
        self.dim = dim
        self.lattice_size = lattice_size

        self.copula_lattice = tfl.layers.Lattice(
            lattice_sizes=[self.lattice_size] * self.dim,
            interpolation="hypercube",  # simplex
            monotonicities=["increasing"] * self.dim,
            output_min=0.0,
            output_max=1.0,
            name="lattice",
        )

    def call(self, x):
        y = self.copula_lattice(x)
        grad = y
        for i in range(self.dim):
            grad = tf.gradients(grad, x[i])  # , stop_gradients=[a, b])
        return y, x, grad


class CopulaModel(LatticeCDFLayer):
    def __init__(self, dim, lattice_size=2, pwl_keypoints=None):
        super().__init__(dim=dim, lattice_size=lattice_size)
        self.pwl_calibration_input_keypoints = (
            unique_intervals if pwl_keypoints is None else pwl_keypoints
        )

        self.model_inputs = []
        for i in range(self.dim):
            self.model_inputs.append(tf.keras.layers.Input(shape=[1], name="col_%s_inf" % i))
            self.model_inputs.append(tf.keras.layers.Input(shape=[1], name="col_%s_sup" % i))

        self.calibrators = []
        for i in range(self.dim):
            self.calibrators.append(
                tfl.layers.PWLCalibration(
                    input_keypoints=np.array(self.pwl_calibration_input_keypoints[i]),
                    dtype=tf.float32,
                    output_min=0.0,
                    output_max=1.0,
                    clamp_min=True,
                    clamp_max=True,
                    monotonicity="decreasing",
                    name="col_%s_inf_pwl" % i,
                )(self.model_inputs[2 * i])
            )
            self.calibrators.append(
                tfl.layers.PWLCalibration(
                    input_keypoints=np.array(self.pwl_calibration_input_keypoints[i]),
                    # input_keypoints=np.linspace(
                    #     feat_mins[i],
                    #     feat_maxs[i],
                    #     num=pwl_calibration_num_keypoints),
                    dtype=tf.float32,
                    output_min=0.0,
                    output_max=1.0,
                    clamp_min=True,
                    clamp_max=True,
                    monotonicity="increasing",
                    name="col_%s_sup_pwl" % i,
                )(self.model_inputs[2 * i + 1])
            )

        self.lattice_cdf = []
        for i in range(self.dim):
            self.lattice_cdf.append(
                tfl.layers.Lattice(
                    lattice_sizes=[lattice_size] * 2,
                    interpolation="hypercube",  # simplex
                    monotonicities=["increasing"] * 2,
                    output_min=0.0,
                    output_max=1.0,
                    name="lattice_col_%s" % i,
                )([self.calibrators[2 * i], self.calibrators[2 * i + 1]])
            )

        self.model = tf.keras.models.Model(
            inputs=self.model_inputs, outputs=self.copula_lattice(self.lattice_cdf)
        )

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred, lattice_inputs, lattice_grad = self(x, training=True)  # , training=True)
            loss1 = self.compiled_loss(y, y_pred)
            loss2 = 100000  # min(x)
            loss3 = 1  # max(sum(x)-self.dim+1, 0)
            loss4 = 1
            loss = loss1 + loss2 + loss3 + loss4
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}


# loss 可以在complie里传入，但是不能通过重写fit方法来载入


class Trainer_Lattice(CopulaModel):
    def __init__(
        self, name, dim, pwl_keypoints=None, lattice_size=2  # also can input table unique values
    ):
        super().__init__(dim=dim, lattice_size=lattice_size)
        self.model_path = "./models/Lattice/model/" + name
        self.weight_path = "./models/Lattice/weight/" + name
        self.model.save("%s.hdf5" % self.model_path)
        self.model.summary()

    def fit(
        self,
        X,
        y,
        lr=0.01,
        bs=1000,
        epochs=1000,
        reduceLR_factor=0.1,
        reduceLR_patience=10,
        ESt_patience=20,
        verbose=1,
        loss="MSE",
        opt="Adam",
    ):
        assert X.shape[0] == y.shape[0]

        X = X.astype(np.float32)
        y = y.astype(np.float32)

        features = [X[:, i] for i in range(X.shape[1])]
        target = y

        Loss = {
            "MAE": tf.keras.losses.mean_absolute_error,
            "MSE": tf.keras.losses.mean_squared_error,
            "MAPE": tf.keras.losses.mean_absolute_percentage_error,
        }

        Opt = {
            "Adam": tf.keras.optimizers.Adam(lr),
            "Nadam": tf.keras.optimizers.Nadam(),
            "Adagrad": tf.keras.optimizers.Adagrad(),
            "Adadelta": tf.keras.optimizers.Adadelta(),
            "Adamax": tf.keras.optimizers.Adamax(),
            "RMSprop": tf.keras.optimizers.RMSprop(),
        }

        self.model.compile(loss=Loss[loss], optimizer=Opt[opt])

        earlyStopping = tf.keras.callbacks.EarlyStopping(
            restore_best_weights=True,
            monitor="loss",
            mode="min",
            patience=ESt_patience,
            verbose=verbose,
        )
        mcp_save = tf.keras.callbacks.ModelCheckpoint(
            filepath=f"{self.weight_path}.h5",
            save_weights_only=True,
            save_best_only=True,
            monitor="loss",
            mode="min",
            verbose=verbose,
        )
        reduce_lr_loss = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="loss",
            factor=reduceLR_factor,
            patience=reduceLR_patience,
            verbose=verbose,
            min_delta=1e-8,
            mode="min",
        )

        self.model.fit(
            features,
            target,
            epochs=epochs,
            batch_size=bs,
            verbose=1,
            callbacks=[earlyStopping, mcp_save, reduce_lr_loss],
        )

    def load(self):
        self.model = tf.keras.models.load_model("%s.hdf5" % self.model_path)
        self.model.load_weights("%s.hdf5" % self.weight_path)

    def inference(self, grid):
        # assert grid.shape[1] == self.dim * 2
        pred = self.model.predict(np.hsplit(grid, self.dim * 2))
        return pred


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="wine3", help="Dataset.")
parser.add_argument("--query-size", type=int, default=1000, help="query size")
parser.add_argument("--min-conditions", type=int, default=1, help="min num of query conditions")
parser.add_argument("--max-conditions", type=int, default=2, help="max num of query conditions")
# parser.add_argument("--lattice-size", type=int, default=2, help="Lattice size for each column.")
parser.add_argument("--pwl-n", type=int, default=1, help="pwl layer number for each column.")
parser.add_argument("--pwl-tanh", type=bool, default=False, help="tanh layer after pwl.")
parser.add_argument("--boundary", type=bool, default=False, help="add boundary point to train set.")
parser.add_argument("--epochs", type=int, default=1000, help="Number of train epochs.")
parser.add_argument("--bs", type=int, default=1000, help="Batch size.")
parser.add_argument("--loss", type=str, default="MSE", help="Loss.")
parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")


# parser.add_argument("--opt", type=str, default="Adam", help="Optimizer.")
# parser.add_argument("--lhs-n", type=int, default=10000, help="LHS sample number.")


try:
    args = parser.parse_args()
except:
    # args = parser.parse_args([])
    args, unknown = parser.parse_known_args()


def make_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


FilePath = (
    f"{args.dataset}_{args.query_size}_{args.min_conditions}_{args.max_conditions}_{args.loss}"
)
resultsPath = f"results/{FilePath}"
modelPath = f"saved_models/{FilePath}"
make_directory(resultsPath)
make_directory(modelPath)

OPS = {
    ">": np.greater,
    "<": np.less,
    ">=": np.greater_equal,
    "<=": np.less_equal,
    "=": np.equal,
}

# bs = int(args.bs)
# lr = float(args.lr)
# train_size = float(args.train_size)
# epochs = int(args.epochs)
# lattice = int(args.lattice)
# sample = int(args.sample)
# lhs_n = int(args.lhs_n)

print("\nBegin Loading Data ...")
table = np.loadtxt(f"datasets/{args.dataset}.csv", delimiter=",")
np.savetxt(f"{resultsPath}/original_table.csv", table, delimiter=",")
table_size = table.shape
print(f"{args.dataset}.csv  shape: {table_size}")
print("Done.\n")


# table = datasets.LoadDataset(args.dataset + ".csv", args.dataset)

# print("Begin Generating Queries ...")
# rng = np.random.RandomState(args.seed)
# query_set = [
#     GenerateQuery(table, 2, args.num_conditions + 1, rng, args.dataset)
#     for i in tqdm(range(args.query_size))
# ]
# print("Complete Generating Queries.")


print("Begin Generating Queries Set ...")
rng = np.random.RandomState(42)
query_set = [
    generate_random_query(table, args.min_conditions, args.max_conditions, rng)
    for _ in tqdm(range(args.query_size))
]
print("Done.\n")


# print("\n\nCalculating intervalization...")
# table_size = table.data.shape
# n_row, n_column = table_size[0], table_size[1]
# unique_intervals = dictionary_column_interval(table_size, query_set)
# column_interval_number = count_column_unique_interval(unique_intervals)
# print("\nColumn intervals", column_interval_number, np.product(column_interval_number))


print("Begin Intervalization ...")
unique_intervals = column_intervalization(query_set, table_size)
column_interval_number = count_column_unique_interval(unique_intervals)
print(f"{column_interval_number=}")
print("Done.\n")

# 这里还没改
print("Begin Building Train set and Model ...")
X, y = build_train_set_2_input(query_set, unique_intervals)
if args.boundary:
    X, y = add_boundary_2_input(X, y, unique_intervals)

# 这里开始要改
X, Y = process_train_data(unique_intervals, query_set)


# model = LatticeCDF(unique_intervals, pwl_keypoints=None)
m = Trainer_Lattice(modelPath, table_size, pwl_keypoints=None)
print("Done.\n")

# 下面两句已经改好了
m.fit(X, Y, lr=args.lr, bs=args.bs, epochs=args.epochs, loss=args.loss, opt=args.opt)
m.load()  # 要增加 load 方法


# Full-Factorial net of unique intervals
#     values = [v for v in unique_intervals.values()]
#     mesh = np.meshgrid(*values)
#     grid = np.array(mesh).T.reshape(-1, len(values)).astype(np.float32)

# Latin Hypercube sampling
#     lb = np.array([v[1] for v in unique_intervals.values()])
#     ub = np.array([v[-1] for v in unique_intervals.values()])
#     lhs_sample = lhs(n_column, samples=10000, criterion='center')
#     sample_df = pd.DataFrame(lb + (ub-lb)*lhs_sample, columns=[f'col_{i}' for i in range(n_column)])
#     grid = np.array(sample_df.sort_values(by=list(sample_df.columns)))
lb = np.array([1] * n_column)
ub = np.array(column_interval_number) - 1
lhs_sample = lb + (ub - lb) * lhs(n_column, samples=lhs_n, criterion="center")
index = np.round(lhs_sample).astype(int)
grid_mesh = np.empty_like(index, dtype=np.float32)
for i in range(lhs_n):
    idx = index[i, :]
    grid_mesh[i] = [unique_intervals[j][idx[j]] for j in range(n_column)]
sample_df = pd.DataFrame(grid_mesh, columns=[f"col_{i}" for i in range(n_column)])
grid_a = np.array(sample_df.sort_values(by=list(sample_df.columns)))
greatest = np.array([v[-1] for v in unique_intervals.values()]).reshape(1, -1)
grid = np.concatenate([grid_a, greatest], axis=0)

# 下面这句话是要改的
dataNew = generate_data_new(grid, model=m)

# 这里还没改
dataNew = m.generate_by_row(unique_intervals, batch_size=10000)
np.savetxt(f"{resultsPath}/generated_table.csv", dataNew, delimiter=",")

# 下面已经改好了，让输入的dataNew是numpy就行
Q_error = calculate_Q_error(dataNew, query_set, table_size)
print_Q_error(Q_error, args, resultsPath)
print(f"\n Original table shape : {table_size}")
print(f"Generated table shape : {dataNew.shape}")

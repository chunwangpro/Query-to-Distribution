import warnings

warnings.filterwarnings("ignore")
import itertools

import numpy as np
import tensorflow as tf
import tensorflow_lattice as tfl
from tqdm import tqdm

from query_func import *


def build_train_set_1_input(query_set, unique_intervals):
    X = []
    for query in query_set:
        x = [v[-3] for v in unique_intervals.values()]
        idxs, _, vals, _ = query
        for i, v in zip(idxs, vals):
            x[i] = v
        X.append(x)
    X = np.array(X, dtype=np.float32)
    y = np.array([query[-1] for query in query_set], dtype=np.float32).reshape(-1, 1)
    return X, y


def add_boundary_1_input(X, y, unique_intervals):
    percent = 0.4
    # make train set unique
    train = np.hstack((X, y))
    train = np.unique(train, axis=0)
    # create boundary set
    min_x = [v[0] for v in unique_intervals.values()]
    max_x = [v[-3] for v in unique_intervals.values()]
    border_x = np.array([min_x, max_x])
    border_y = np.array([[0], [1]])
    border = np.hstack((border_x, border_y))
    # repeat boundary to raise weight
    k = int(train.shape[0] / border.shape[0] * percent)
    repeated_border = np.tile(border, (k, 1))
    train = np.vstack((train, repeated_border))
    # shuffle and split
    np.random.shuffle(train)
    X, y = np.hsplit(train, [X.shape[1]])
    return X, y


def build_train_set_2_input(query_set, unique_intervals):

    def process_op_lt(x, idx, val):
        x[idx * 2 + 1] = val

    def process_op_le(x, idx, val):
        ind = unique_intervals[idx].index(val) + 1
        x[idx * 2 + 1] = unique_intervals[idx][ind]

    def process_op_ge(x, idx, val):
        x[idx * 2] = val

    def process_op_gt(x, idx, val):
        ind = unique_intervals[idx].index(val) + 1
        x[idx * 2] = unique_intervals[idx][ind]

    def process_op_eq(x, idx, val):
        ind = unique_intervals[idx].index(val) + 1
        x[idx * 2] = val
        x[idx * 2 + 1] = unique_intervals[idx][ind]

    op_functions = {
        "<": process_op_lt,
        "<=": process_op_le,
        ">=": process_op_ge,
        ">": process_op_gt,
        "=": process_op_eq,
    }
    X = []
    origin = [[v[0], v[-1]] for v in unique_intervals.values()]
    for query in query_set:
        x = np.array(origin).ravel()
        idxs, ops, vals, _ = query
        for i, o, v in zip(idxs, ops, vals):
            op_functions[o](x, i, v)
        X.append(x)
    X = np.array(X, dtype=np.float32)
    y = np.array([query[-1] for query in query_set], dtype=np.float32).reshape(-1, 1)
    return X, y


def add_boundary_2_input(X, y, unique_intervals):
    percent = 0.4
    # make train set unique
    train = np.hstack((X, y))
    train = np.unique(train, axis=0)
    # create boundary set
    # min_x = [v[0] for v in unique_intervals.values()]
    # max_x = [v[-3] for v in unique_intervals.values()]
    # border_x = np.array([min_x, max_x])
    # border_y = np.array([[0], [1]])
    # border = np.hstack((border_x, border_y))
    # repeat boundary to raise weight
    k = int(train.shape[0] / border.shape[0] * percent)
    repeated_border = np.tile(border, (k, 1))
    train = np.vstack((train, repeated_border))
    # shuffle and split
    np.random.shuffle(train)
    X, y = np.hsplit(train, [X.shape[1]])
    return X, y


class PWLLattice:
    def __init__(
        self,
        path,
        table_size,
        unique_intervals,
        pwl_keypoints=None,  # also can input table unique values
        pwl_n=3,
        lattice_size=2,
        pwl_tanh=False,
    ):
        self.name = "PWLLattice"
        self.path = path
        self.model_path = f"{self.path}/{self.name}_model"
        self.weight_path = f"{self.path}/{self.name}_weight"

        self.n_row, self.n_column = table_size
        self.dim = self.n_column
        self.unique_intervals = unique_intervals

        self.pwl_n = pwl_n
        self.pwl_keypoints = unique_intervals if pwl_keypoints is None else pwl_keypoints
        self.pwl_tanh = pwl_tanh

        self.lattice_size = (
            [len(v) for v in unique_intervals.values()]
            if lattice_size == 0
            else [lattice_size] * self.dim
        )

        self.model_inputs = tf.keras.layers.Input(shape=[self.dim], name="input_layer")

        def PWL(input_keypoints, col_idx, PWL_idx):
            return tfl.layers.PWLCalibration(
                input_keypoints=np.array(input_keypoints),
                dtype=tf.float32,
                output_min=0.0,
                output_max=1.0,
                monotonicity="increasing",
                name=f"col_{col_idx}_PWL_{PWL_idx}",
            )

        def column_PWL_layer(input_layer, col_idx, pwl_n, activation=self.pwl_tanh):
            pwl_input = tf.keras.layers.Lambda(
                lambda x: tf.expand_dims(x[:, col_idx], axis=-1), name=f"lambda_col_{col_idx}"
            )(input_layer)
            pwl_output = pwl_input
            for j in range(pwl_n):
                if j == 0:
                    keypoints = self.pwl_keypoints[col_idx]
                else:
                    keypoints = np.linspace(0, 1, num=len(self.pwl_keypoints[col_idx]))
                pwl_output = PWL(keypoints, col_idx, j + 1)(pwl_output)
                if activation and j < pwl_n - 1:
                    pwl_output = tf.keras.layers.Activation(
                        "tanh", name=f"col_{col_idx}_tanh_{j+1}"
                    )(pwl_output)
            return pwl_output

        self.lattice_inputs = [
            column_PWL_layer(self.model_inputs, i, self.pwl_n) for i in range(self.dim)
        ]

        self.lattice = tfl.layers.Lattice(
            lattice_sizes=self.lattice_size,
            interpolation="simplex",  # "hypercube",
            monotonicities=["increasing"] * self.dim,
            output_min=0.0,
            output_max=1.0,
            name="lattice",
        )

        self.model_output = self.lattice(self.lattice_inputs)

        self.model = tf.keras.models.Model(
            inputs=self.model_inputs,
            outputs=self.model_output,
        )
        self.model.summary()

    def fit(
        self,
        X,
        y,
        lr=0.001,
        bs=1000,
        epochs=2000,
        verbose=1,
        reduceLR_factor=0.1,
        reduceLR_patience=10,
        ESt_patience=20,
        loss="MSE",
    ):
        # assert X.shape[0] == y.shape[0]
        # assert X.shape[1] == self.dim * 2

        Loss = {
            "MAE": tf.keras.losses.mean_absolute_error,
            "MSE": tf.keras.losses.mean_squared_error,
            "MAPE": tf.keras.losses.mean_absolute_percentage_error,
        }

        # self.model.compile(loss=lossFunc, optimizer="adam", metrics=["accuracy"])
        self.model.compile(loss=Loss[loss], optimizer=tf.keras.optimizers.Adamax(lr))

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

        # X = [X[:, i] for i in range(X.shape[1])]

        self.model.fit(
            X,
            y,
            epochs=epochs,
            batch_size=bs,
            verbose=verbose,
            callbacks=[earlyStopping, mcp_save, reduce_lr_loss],
        )
        self.model.save(f"{self.model_path}")

    def load(self, modelPath=None, summary=False):
        Path = modelPath if modelPath else self.path
        self.model = tf.keras.models.load_model(f"{Path}/{self.name}_model")
        # self.model.load_weights(f"{Path}/{self.name}_weight.h5")
        if summary:
            print(self.model.summary())

    def predict(self, grid):
        # assert grid.shape[1] == self.dim
        # return self.model.predict(np.hsplit(grid, self.dim))
        return self.model.predict(grid, verbose=0)

    def generate_by_row(self, unique_intervals, batch_size=10000):

        values = [v for v in unique_intervals.values()]
        total_combinations = np.prod([len(v) for v in values])
        batch_num = (total_combinations // batch_size) + 1
        print(f"\nBegin Generating Table from Batches ({batch_size=}, {batch_num=}) ...")

        ArrayNew = None
        for grid_batch in tqdm(self._yield_row_grid_batches(values, batch_size), total=batch_num):
            pred_batch = self.predict(grid_batch)
            ArrayNew = self._generate_row_batch_table(grid_batch, pred_batch, ArrayNew)
        if ArrayNew.shape[0] < self.n_row:
            print(
                f"Generated table row length({ArrayNew.shape[0]}) is less than the original table row length({self.n_row})."
            )
        print("Done.\n")
        return ArrayNew

    def _yield_row_grid_batches(self, values, batch_size):
        # use batches to avoid memory error
        iterator = itertools.product(*values)
        while True:
            batch = list(itertools.islice(iterator, batch_size))
            if not batch:
                break
            yield np.array(batch, dtype=np.float32)

    def _generate_row_batch_table(self, grid_batch, pred_batch, ArrayNew=None):
        # generate by row, one query may generate several rows
        count = 0 if ArrayNew is None else ArrayNew.shape[0]
        ops = ["<="] * self.n_column
        pred_batch = (pred_batch * self.n_row).astype(int)  # Case 1: change 0.8 to 0, 1.8 to 1
        for i in range(grid_batch.shape[0]):
            vals = grid_batch[i]
            card = pred_batch[i, 0] - calculate_query_cardinality(ArrayNew, ops, vals)
            if card >= 1:
                array3 = np.repeat(vals, card).reshape(self.n_column, card).T
                ArrayNew = (
                    array3 if ArrayNew is None else np.concatenate((ArrayNew, array3), axis=0)
                )
                count += card
                if count > self.n_row:
                    print(
                        f"Reached table max row length({self.n_row}) in {i}-th row of grid with grid value of {vals}, stop generation."
                    )
                    break
        else:
            return ArrayNew
        return ArrayNew[: self.n_row, :]

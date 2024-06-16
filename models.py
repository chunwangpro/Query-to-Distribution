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


def add_boundary_1_input(X, y, unique_intervals, unique_train=False):
    # set percentage of boundary points
    alpha = 0.1
    # make train set unique
    train = np.hstack((X, y))
    if unique_train:
        train = np.unique(train, axis=0)
    # create boundary set
    min_x = [v[0] for v in unique_intervals.values()]
    max_x = [v[-3] for v in unique_intervals.values()]
    border_x = np.array([min_x, max_x])
    border_y = np.array([[0], [1]])
    border = np.hstack((border_x, border_y))
    # repeat boundary to raise weight
    k = int(train.shape[0] / border.shape[0] * alpha)
    repeated_border = np.tile(border, (k, 1))
    train = np.vstack((train, repeated_border))
    # shuffle and split
    np.random.shuffle(train)
    X, y = np.hsplit(train, [-1])
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


def add_boundary_2_input(X, y, unique_intervals, unique_train=False):
    alpha = 0.1
    # make train set unique
    train = np.hstack((X, y))
    if unique_train:
        train = np.unique(train, axis=0)
    # create boundary set
    # 1. one point
    one = np.array([[v[0], v[-1]] for v in unique_intervals.values()]).ravel()
    one = np.append(one, 1)
    k = int(train.shape[0] * alpha)
    repeated_one = np.tile(one, (k, 1))
    # 2. two zero points
    zero_0 = np.array([[v[0]] * 2 for v in unique_intervals.values()]).ravel()
    zero_1 = np.array([[v[-1]] * 2 for v in unique_intervals.values()]).ravel()
    zero = np.vstack((zero_0, zero_1))
    k = int(train.shape[0] / 2 * alpha)
    repeated_zero = np.tile(zero, (k, 1))
    zero_y = np.zeros((2 * k, 1))
    repeated_zero = np.hstack((repeated_zero, zero_y))
    # 3. other zero points
    k = int(train.shape[0] * alpha)
    other_zero = [[[np.random.choice(v)] * 2 for v in unique_intervals.values()] for _ in range(k)]
    other_zero = [np.array(v).ravel() for v in other_zero]
    other_zero = np.hstack((np.array(other_zero), np.zeros((k, 1))))
    train = np.vstack((train, repeated_one, repeated_zero, other_zero))
    # shuffle and split
    np.random.shuffle(train)
    X, y = np.hsplit(train, [-1])
    return X, y


def PWL(input_keypoints, col_idx, PWL_idx, monotonicity, suffix=""):
    # single PieceWise Linear Calibration Layer
    return tfl.layers.PWLCalibration(
        input_keypoints=np.array(input_keypoints),
        dtype=tf.float32,
        output_min=0.0,
        output_max=1.0,
        monotonicity=monotonicity,  # "increasing", "decreasing"
        name=f"col_{col_idx}{suffix}_PWL_{PWL_idx}",
    )


def column_PWL_layer(input_layer, pwl_keypoints, col_idx, pwl_n, input_type, activation=False):
    # One column with Multiple PWL layers and optional activation
    layer_input = tf.keras.layers.Lambda(
        lambda x: tf.expand_dims(x[:, col_idx], axis=-1), name=f"lambda_col_{col_idx}"
    )(input_layer)
    x = layer_input
    if input_type == "1-input":
        monotonicity = "increasing"
    elif input_type == "2-input":
        monotonicity = "decreasing" if col_idx % 2 == 0 else "increasing"
        suffix = "_inf" if col_idx % 2 == 0 else "_sup"
        col_idx = col_idx // 2
    else:
        raise ValueError("Invalid input_type. Must be '1-input' or '2-input'.")

    for j in range(pwl_n):
        if j == 0:
            # first PWL layer
            keypoints = pwl_keypoints[col_idx]
        else:
            # other PWL layers
            keypoints = np.linspace(0, 1, num=len(pwl_keypoints[col_idx]))
        x = PWL(keypoints, col_idx, j, monotonicity, suffix)(x)
        if activation and j < pwl_n - 1:
            # add activation after PWL layer except the last PWL layer
            x = tf.keras.layers.Activation("tanh", name=f"col_{col_idx}{suffix}_tanh_{j}")(x)
    return x


def Lattice(lattice_size, monotonicities, col_idx, interpolation="simplex"):
    return tfl.layers.Lattice(
        lattice_sizes=lattice_size,
        interpolation=interpolation,  # "simplex", "hypercube"
        monotonicities=monotonicities,
        output_min=0.0,
        output_max=1.0,
        name=f"lattice_col_{col_idx}",
    )


class PWLLattice:
    # PWL + Lattice: 1-input Model
    def __init__(
        self,
        path,
        table_size,
        unique_intervals,
        pwl_keypoints=None,
        pwl_n=3,
        lattice_size=2,
        pwl_tanh=False,
    ):
        self.name = "PWLLattice"
        self.path = path

        self.n_row, self.n_column = table_size
        self.dim = self.n_column
        self.unique_intervals = unique_intervals

        self.pwl_n = pwl_n
        self.pwl_keypoints = unique_intervals if pwl_keypoints is None else pwl_keypoints
        self.pwl_tanh = pwl_tanh

        self.lattice_size = lattice_size
        # self.copula_lattice_size is used in the last layer of the model, if it is a lattice layer.
        self.copula_lattice_size = (
            [len(v) for v in unique_intervals.values()]
            if self.lattice_size == 0
            else [self.lattice_size] * self.n_column
        )

    def build_model(self):
        self.model_inputs = tf.keras.layers.Input(shape=[self.dim], name="input_layer")

        self.column_PWL = [
            column_PWL_layer(
                self.model_inputs, self.pwl_keypoints, i, self.pwl_n, "1-input", self.pwl_tanh
            )
            for i in range(self.dim)
        ]

        self.last_lattice_layer = Lattice(
            self.copula_lattice_size, monotonicities=["increasing"] * self.n_column, col_idx=1
        )

        self.model_output = self.last_lattice_layer(self.column_PWL)

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
        # assert X.shape[1] == self.dim
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
            filepath=f"{self.path}/{self.name}_weight.h5",
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
        self.model.save(f"{self.path}/{self.name}_model")

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


class PWLLatticeCopula(PWLLattice):
    # PWL + Lattice + Copula: 2-input Model
    def __init__(
        self,
        path,
        table_size,
        unique_intervals,
        pwl_keypoints=None,
        pwl_n=3,
        lattice_size=2,
        pwl_tanh=False,
    ):
        super().__init__(
            path, table_size, unique_intervals, pwl_keypoints, pwl_n, lattice_size, pwl_tanh
        )

        self.name = "PWLLatticeCopula"
        self.dim = self.n_column * 2

    def build_model(self):
        self.model_inputs = tf.keras.layers.Input(shape=[self.dim], name="input_layer")

        self.column_PWL = [
            column_PWL_layer(
                self.model_inputs, self.pwl_keypoints, i, self.pwl_n, "2-input", self.pwl_tanh
            )
            for i in range(self.dim)
        ]

        # 2-input lattice layers for each column
        self.col_lattice_layers = [
            Lattice(
                lattice_size=[self.lattice_size] * 2,
                monotonicities=["increasing"] * 2,
                col_idx=i // 2,
            )([self.column_PWL[i], self.column_PWL[i + 1]])
            for i in range(0, self.dim, 2)
        ]
        print(f"self.dim: {self.dim}")
        print(f"self.n_column: {self.n_column}")
        print(f"len(self.col_lattice_layers): {len(self.col_lattice_layers)}")

        # Create last_lattice_layer connecting all col_lattice_layers
        self.last_lattice_layer = Lattice(
            self.copula_lattice_size,
            monotonicities=["increasing"] * self.n_column,
            col_idx="Joint-CDF",
        )

        # use lattice layer as the last layer to learn the joint CDF
        # can be replaced by other layers, e.g., AutoRegressive Model
        self.JointCDFModel = self.last_lattice_layer

        self.model_output = self.JointCDFModel(self.col_lattice_layers)

        self.model = tf.keras.models.Model(
            inputs=self.model_inputs,
            outputs=self.model_output,
        )
        self.model.summary()

    # 下面的代码需要修改成 2-input 的格式
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

    def generate_by_col():
        pass

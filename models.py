import warnings

warnings.filterwarnings("ignore")
import itertools

import numpy as np
import tensorflow as tf
import tensorflow_lattice as tfl
from tqdm import tqdm

from query_func import *


def build_train_set_1_input(query_set, unique_intervals, args, table_size):
    X = []
    for query in query_set:
        x = [v[-3] for v in unique_intervals.values()]
        idxs, _, vals, _ = query
        for i, v in zip(idxs, vals):
            x[i] = v
        X.append(x)
    X = np.array(X, dtype=np.float32)
    y = np.array([query[-1] for query in query_set], dtype=np.float32).reshape(-1, 1)
    y /= table_size[0]
    train = np.hstack((X, y))

    # make train set unique
    if args.unique_train:
        train = np.unique(train, axis=0)

    # add boundary
    if args.boundary:
        train = add_boundary_1_input(train, unique_intervals, args.boundary)

    # shuffle and split
    np.random.shuffle(train)
    X, y = np.hsplit(train, [-1])
    return X, y


def add_boundary_1_input(train, unique_intervals, alpha=0.1):
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

    return train


def build_train_set_2_input(query_set, unique_intervals, args, table_size):

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
    y /= table_size[0]
    train = np.hstack((X, y))

    # make train set unique
    if args.unique_train:
        train = np.unique(train, axis=0)

    # add boundary
    if args.boundary:
        train = add_boundary_2_input(train, unique_intervals, args.boundary)

    # shuffle and split
    np.random.shuffle(train)
    X, y = np.hsplit(train, [-1])
    return X, y


def add_boundary_2_input(train, unique_intervals, alpha=0.1):
    # add total k = int(train.shape[0] * alpha) boundary points
    # 1/4 for one point, 1/4 for two zero points, 1/2 for other zero points
    # create boundary set

    # 1. one point
    one = np.array([[v[0], v[-1]] for v in unique_intervals.values()]).ravel()
    one = np.append(one, 1)
    k = int(train.shape[0] * alpha / 4)
    repeated_one = np.tile(one, (k, 1))

    # 2. two zero points
    zero_0 = np.array([[v[0]] * 2 for v in unique_intervals.values()]).ravel()
    zero_1 = np.array([[v[-1]] * 2 for v in unique_intervals.values()]).ravel()
    zero = np.vstack((zero_0, zero_1))
    zero_y = np.zeros((2, 1))
    zero = np.hstack((zero, zero_y))
    k = int(train.shape[0] * alpha / 8)
    repeated_zero = np.tile(zero, (k, 1))

    # 3. other zero points
    k = int(train.shape[0] * alpha / 2)
    other_zero = [[[np.random.choice(v)] * 2 for v in unique_intervals.values()] for _ in range(k)]
    other_zero = [np.array(v).ravel() for v in other_zero]
    other_zero = np.hstack((np.array(other_zero), np.zeros((k, 1))))

    train = np.vstack((train, repeated_one, repeated_zero, other_zero))
    return train


def setup_train_set_and_model(args, query_set, unique_intervals, modelPath, table_size):
    """
    Setup the training set and model based on the model type.
    X: Train X, query intervals. e.g. [a,b) for each column in 2-input model; (-inf, a] for each column in 1-input model.
    y: Train y, cardinality.
    m: Model.
    values: Unique intervals of each column, it will be used to generate grid intervals in table generation phase after model is well-trained. e.g. [a,b) for each column in 2-input model; (-inf, a] for each column in 1-input model.
    """
    if args.model == "1-input":
        X, y = build_train_set_1_input(query_set, unique_intervals, args, table_size)
        m = PWLLattice(
            args,
            modelPath,
            table_size,
            unique_intervals,
            pwl_keypoints=None,
            pwl_n=args.pwl_n,
            lattice_size=args.lattice_size,
            pwl_tanh=args.pwl_tanh,
        )
        values = [v for v in unique_intervals.values()]

    elif args.model == "2-input":
        X, y = build_train_set_2_input(query_set, unique_intervals, args, table_size)
        # model = LatticeCDF(unique_intervals, pwl_keypoints=None)
        # m = Trainer_Lattice(modelPath, table_size, pwl_keypoints=None)
        m = PWLLatticeCopula(
            args,
            modelPath,
            table_size,
            unique_intervals,
            pwl_keypoints=None,
            pwl_n=args.pwl_n,
            lattice_size=args.lattice_size,
            pwl_tanh=args.pwl_tanh,
        )
        values = [[(v[i], v[i + 1]) for i in range(len(v) - 1)] for v in unique_intervals.values()]
    else:
        raise ValueError("Invalid model type. Please use '1-input' or '2-input'.")
    return X, y, m, values


def PWL(input_keypoints, col_idx, PWL_idx, monotonicity, suffix=""):
    # single PieceWise Linear Calibration Layer
    return tfl.layers.PWLCalibration(
        input_keypoints=np.array(input_keypoints),
        units=1,  # output dimension
        dtype=tf.float32,
        output_min=0.0,
        output_max=1.0,
        monotonicity=monotonicity,  # "increasing" or "decreasing" or "none"
        kernel_initializer="equal_heights",  # "equal_heights" or "equal_slopes"
        kernel_regularizer=None,  # ("laplacian", 0.5, 0.5),  # ("laplacian", l1, l2)
        num_projection_iterations=8,
        input_keypoints_type="fixed",  # "fixed" or "learned_interior"
        name=f"col_{col_idx}{suffix}_PWL_{PWL_idx}",
    )


def column_PWL_layer(input_layer, pwl_keypoints, dim_idx, pwl_n, input_type, activation=False):
    # One column with Multiple PWL layers and optional activation
    layer_input = tf.keras.layers.Lambda(
        lambda x: tf.expand_dims(x[:, dim_idx], axis=-1), name=f"lambda_dim_{dim_idx}"
    )(input_layer)
    x = layer_input

    if input_type == "1-input":
        monotonicity = "increasing"
        suffix = ""
        col_idx = dim_idx
    elif input_type == "2-input":
        monotonicity = "increasing" if dim_idx % 2 else "decreasing"
        suffix = "_sup" if dim_idx % 2 else "_inf"
        col_idx = dim_idx // 2
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
        units=1,  # output dimension
        interpolation=interpolation,  # "simplex", "hypercube"
        monotonicities=monotonicities,  # "increasing", "none"
        output_min=0.0,
        output_max=1.0,
        num_projection_iterations=10,
        monotonic_at_every_step=True,
        name=f"lattice_col_{col_idx}",
    )


class PWLLattice:
    # PWL + Lattice: 1-input Model
    def __init__(
        self,
        args,
        path,
        table_size,
        unique_intervals,
        pwl_keypoints=None,
        pwl_n=3,
        lattice_size=2,
        pwl_tanh=False,
    ):
        self.name = "PWLLattice"
        self.args = args
        self.path = path

        self.n_row, self.n_column = table_size
        self.dim = self.n_column
        self.unique_intervals = unique_intervals

        self.pwl_keypoints = (
            unique_intervals if pwl_keypoints is None else pwl_keypoints
        )  # also can be unique values of each column distribution
        self.pwl_n = pwl_n
        self.pwl_tanh = pwl_tanh

        # self.last_lattice_size is used in the last layer of the model, if it is a lattice layer.
        self.last_lattice_size = (
            [len(v) for v in unique_intervals.values()]
            if lattice_size == 0
            else [lattice_size] * self.n_column
        )

    def show_all_attributes(self):
        for key, value in self.__dict__.items():
            print(f"{key}: {value}")

    def build_model(self):
        self.model_inputs = tf.keras.layers.Input(shape=[self.dim], name="input_layer")

        self.column_PWL = [
            column_PWL_layer(
                self.model_inputs, self.pwl_keypoints, i, self.pwl_n, "1-input", self.pwl_tanh
            )
            for i in range(self.dim)
        ]

        # here use lattice layer as the last layer to learn the CDF, can be replaced by other layers, e.g., AutoRegressive Model
        self.JointCDFModel = Lattice(
            self.last_lattice_size,
            monotonicities=["increasing"] * self.n_column,
            col_idx="Joint-CDF",
        )

        self.model_output = self.JointCDFModel(self.column_PWL)

        self.model = tf.keras.models.Model(
            inputs=self.model_inputs,
            outputs=self.model_output,
        )
        self.model.summary()

    def fit(
        self,
        X,
        y,
        args,
        reduceLR_factor=0.1,
        reduceLR_patience=10,
        ESt_patience=20,
        verbose=1,
    ):

        Loss = {
            "MSE": tf.keras.losses.mean_squared_error,
            "MAE": tf.keras.losses.mean_absolute_error,
            "MAPE": tf.keras.losses.mean_absolute_percentage_error,
        }
        Opt = {
            "adam": tf.keras.optimizers.Adam,
            "adamax": tf.keras.optimizers.Adamax,
            "rmsprop": tf.keras.optimizers.RMSprop,
        }

        self.model.compile(loss=Loss[args.loss], optimizer=Opt[args.opt](args.lr))

        earlyStopping = tf.keras.callbacks.EarlyStopping(
            patience=ESt_patience,
            restore_best_weights=True,
            monitor="loss",
            mode="min",
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
            factor=reduceLR_factor,
            patience=reduceLR_patience,
            min_delta=1e-8,
            monitor="loss",
            mode="min",
            verbose=verbose,
        )

        self.model.fit(
            X,
            y,
            epochs=args.epochs,
            batch_size=args.bs,
            callbacks=[earlyStopping, mcp_save, reduce_lr_loss],
            verbose=verbose,
        )
        self.model.save(f"{self.path}/{self.name}_model")

    def predict(self, x, verbose=0):
        return self.model.predict(x, verbose=verbose)

    def load(self, modelPath=None, summary=False):
        Path = modelPath if modelPath else self.path
        self.model = tf.keras.models.load_model(f"{Path}/{self.name}_model")
        # self.model.load_weights(f"{Path}/{self.name}_weight.h5")
        if summary:
            print(self.model.summary())

    def generate_table_by_row(self, values, batch_size=10000, test_table=None):
        total_combinations = np.prod([len(v) for v in values])
        batch_number = (total_combinations // batch_size) + 1
        print(f"\nBegin Generating Table by Row Batches ({batch_number=}, {batch_size=}) ...")

        Table_Generated = np.empty((0, self.n_column), dtype=np.float32)
        for row_batch in tqdm(self._yield_row_batch(values, batch_size), total=batch_number):
            # pred_batch = self.model.predict(row_batch, verbose=0)
            # # Case 1: change 0.8 to 0, 1.8 to 1
            # pred_batch = (pred_batch * self.n_row).astype(int)

            # only for test: begin test
            # print(f"row_batch: {row_batch}")
            table = test_table

            if self.args.model == "1-input":
                ops = ["<="] * self.n_column
                new_table = table
            elif self.args.model == "2-input":
                ops = [">=", "<"] * self.n_column
                rows, cols = table.shape
                new_table = np.zeros((rows, 2 * cols))
                for i in range(cols):
                    new_table[:, 2 * i] = table[:, i]
                    new_table[:, 2 * i + 1] = table[:, i]
            pred_batch = np.array(
                [calculate_query_cardinality(new_table, ops, row) for row in row_batch]
            ).reshape(-1, 1)
            ##### test end

            Table_Generated = self._generate_subtable_by_row_batch(
                row_batch, pred_batch, Table_Generated
            )
            if Table_Generated.shape[0] > self.n_row:
                print(f"Reached table max row length({self.n_row}), stop generation.")
                break
        else:
            if Table_Generated.shape[0] < self.n_row:
                print(
                    f"Generated table row length({Table_Generated.shape[0]}) is less than the original table row length({self.n_row})."
                )
            else:
                print("Done.\n")
            return Table_Generated
        return Table_Generated[: self.n_row, :]

    def _yield_row_batch(self, values, batch_size):
        # yield batches to avoid large memory usage
        iterator = itertools.product(*values)
        while True:
            batch = list(itertools.islice(iterator, batch_size))
            if not batch:
                break
            yield np.array(batch, dtype=np.float32).reshape(len(batch), -1)

    def _generate_subtable_by_row_batch(self, row_batch, pred_batch, Table_Generated):
        """
        Using inclusion-exclusion principle is time-consuming. Here we query once before generate to calculate the shortfall cardinality. One query may generate several rows.
        """
        ops = ["<="] * self.n_column
        for i in range(row_batch.shape[0]):
            vals = row_batch[i]
            card = pred_batch[i, 0] - calculate_query_cardinality(Table_Generated, ops, vals)

            if card < 1:
                continue

            subtable = np.tile(vals, (card, 1))
            Table_Generated = np.concatenate((Table_Generated, subtable), axis=0)
        return Table_Generated

    def generate_table_by_col():
        pass


class PWLLatticeCopula(PWLLattice):
    # PWL + Lattice + Copula: 2-input Model
    def __init__(
        self,
        args,
        path,
        table_size,
        unique_intervals,
        pwl_keypoints=None,
        pwl_n=3,
        lattice_size=2,
        pwl_tanh=False,
    ):
        super().__init__(
            args,
            path,
            table_size,
            unique_intervals,
            pwl_keypoints,
            pwl_n,
            lattice_size,
            pwl_tanh,
        )

        self.name = "PWLLatticeCopula"
        self.dim = self.n_column * 2
        self.col_lattice_size = 2 if lattice_size == 0 else lattice_size

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
                lattice_size=[self.col_lattice_size] * 2,
                monotonicities=["increasing"] * 2,
                col_idx=i // 2,
            )([self.column_PWL[i], self.column_PWL[i + 1]])
            for i in range(0, self.dim, 2)
        ]

        # here use lattice layer as the last layer to learn the joint CDF
        # can be replaced by other layers, e.g., AutoRegressive Model
        self.JointCDFModel = Lattice(
            self.last_lattice_size,
            monotonicities=["increasing"] * self.n_column,
            col_idx="Joint-CDF",
        )

        self.model_output = self.JointCDFModel(self.col_lattice_layers)

        self.model = tf.keras.models.Model(
            inputs=self.model_inputs,
            outputs=self.model_output,
        )
        self.model.summary()

    def _generate_subtable_by_row_batch(self, row_batch, pred_batch, Table_Generated=None):
        """
        Using inclusion-exclusion principle is time-consuming. Here we query once before generate to calculate the shortfall cardinality. One query may generate several rows.
        """
        valid_indices = np.where(pred_batch[:, 0] >= 1)[0]
        for i in valid_indices:
            card = pred_batch[i, 0]
            vals = row_batch[i]

            # [::2] use the left interval of each column pair
            subtable = np.tile(vals[::2], (card, 1))
            Table_Generated = np.concatenate((Table_Generated, subtable), axis=0)
            # if Table_Generated is None:
            #     Table_Generated = subtable
            # else:
            #     Table_Generated = np.concatenate((Table_Generated, subtable), axis=0)

        #     count += card
        #     if count > self.n_row:
        #         print(
        #             f"Reached table max row length({self.n_row}) in {i}-th row of grid with grid value of {vals}, stop generation."
        #         )
        #         break
        # else:
        #     return Table_Generated
        # return Table_Generated[: self.n_row, :]
        return Table_Generated

    def generate_table_by_col(self, unique_intervals, values, column_one_point, batch_size=10000):
        print(f"\nBegin Generating Table by Column ...")

        Table_Generated = np.empty(
            (0, 1), dtype=np.float32
        )  ## 这里需要改，看看后续predict和添加表之后怎么改

        for col_idx in self.column:
            new_values = self._process_grid_values(
                Table_Generated, unique_intervals, values, col_idx
            )

            total_combinations = np.prod([len(v) for v in new_values])
            batch_number = (total_combinations // batch_size) + 1
            print(
                f"\nGenerating Column {col_idx}/{self.column} by Batches ({batch_number=}, {batch_size=}) ..."
            )

            for batch in tqdm(self._yield_col_batch(new_values, batch_size), total=batch_number):
                col_batch = self._assemble_batch_with_back_column(batch, col_idx, column_one_point)
                pred_batch = self.model.predict(col_batch, verbose=0)
                # Case 1: change 0.8 to 0, 1.8 to 1
                pred_batch = (pred_batch * self.n_row).astype(int)
                Table_Generated = self._generate_subtable_by_col_batch(
                    col_batch, pred_batch, Table_Generated
                )
            ###### break 条件要重新设计
        return Table_Generated

    def _yield_col_batch(self, values, batch_size):
        # 只适用于 2-input
        # yield batches to avoid large memory usage
        iterator = itertools.product(*values)
        while True:
            batch = list(itertools.islice(iterator, batch_size))
            if not batch:
                break
            np_batch = np.array([np.concatenate(b) for b in batch], dtype=np.float32)
            yield np_batch.reshape(len(batch), -1)

    def _process_grid_values(self, Table_Generated, unique_intervals, values, col_idx):
        # 只适用于 2-input
        if col_idx == 0:
            new_values = [values[0]]
        else:
            Table_Generated = np.unique(Table_Generated, axis=0)
            Table_G_size = Table_Generated.shape
            front_column = np.zeros(
                (Table_G_size[0], Table_G_size[1] * 2), dtype=Table_Generated.dtype
            )
            front_column[:, 0::2] = Table_Generated

            for j in range(Table_G_size[1]):
                interval = np.array(unique_intervals[j])
                idx = np.searchsorted(interval, Table_Generated[:, j])
                front_column[:, j * 2 + 1] = interval[idx + 1]
            new_values = [front_column, values[col_idx]]
        return new_values

    def _assemble_batch_with_back_column(self, batch, col_idx, column_one_point):
        back_column = column_one_point[2 * col_idx + 2 :]
        repeated_back_columns = np.tile(back_column, (len(batch), 1))
        col_batch = np.hstack(batch, repeated_back_columns)
        return col_batch

    def _generate_subtable_by_col_batch(self, col_batch, pred_batch, Table_Generated):
        pass

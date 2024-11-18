import itertools
import warnings

import numpy as np
from tqdm import tqdm

from model import *
from preprocessing import *
from util import *

np.random.seed(42)
warnings.filterwarnings("ignore")


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
        m = Generator_1_input(
            args,
            modelPath,
            table_size,
            unique_intervals,
            pwl_keypoints=None,
        )
        values = [v for v in unique_intervals.values()]

    elif args.model == "2-input":
        X, y = build_train_set_2_input(query_set, unique_intervals, args, table_size)
        # model = LatticeCDF(unique_intervals, pwl_keypoints=None)
        # m = Trainer_Lattice(modelPath, table_size, pwl_keypoints=None)
        m = Generator_2_input(
            args,
            modelPath,
            table_size,
            unique_intervals,
            pwl_keypoints=None,
        )
        values = [[(v[i], v[i + 1]) for i in range(len(v) - 1)] for v in unique_intervals.values()]
    else:
        raise ModelTypeError()
    return X, y, m, values


class Generator_1_input(BaseModel):
    # PWL + Lattice: 1-input Model
    def __init__(
        self,
        args,
        path,
        table_size,
        unique_intervals,
        pwl_keypoints=None,
    ):
        super().__init__(
            args,
            path,
            table_size,
            unique_intervals,
            pwl_keypoints,
        )

        self.name = "Generator_1_input"

    def generate_table_by_row(self, values, batch_size=10000):
        batch_number = self._calculate_batch_number(values, batch_size)
        print(f"\nBegin Generating Table by Row Batches ({batch_number=}, {batch_size=}) ...")
        Table_Generated = np.empty((0, self.n_column), dtype=np.float32)
        for row_batch in tqdm(self._yield_row_batch(values, batch_size), total=batch_number):
            pred_batch = self.model.predict(row_batch, verbose=0)
            # Case 1: change 0.8 to 0, 1.8 to 1
            pred_batch = (pred_batch * self.n_row).astype(int)
            Table_Generated = self._generate_subtable_by_row_batch(
                row_batch, pred_batch, Table_Generated
            )

            if Table_Generated.shape[0] > self.n_row:
                Table_Generated = Table_Generated[: self.n_row, :]
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

    def _calculate_batch_number(self, values, batch_size):
        total_combinations = np.prod([len(v) for v in values])
        batch_number = (total_combinations // batch_size) + 1
        return batch_number

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

    def Test_generate_table_by_row(self, values, batch_size=10000, test_table=None):
        batch_number = self._calculate_batch_number(values, batch_size)

        print(f"\nBegin Generating Table by Row Batches ({batch_number=}, {batch_size=}) ...")

        Table_Generated = np.empty((0, self.n_column), dtype=np.float32)

        for row_batch in tqdm(self._yield_row_batch(values, batch_size), total=batch_number):
            # pred_batch = self.model.predict(row_batch, verbose=0)
            # # Case 1: change 0.8 to 0, 1.8 to 1
            # pred_batch = (pred_batch * self.n_row).astype(int)

            # only for test: begin test
            # print(f"row_batch: {row_batch}")
            if self.args.model == "1-input":
                ops = ["<="] * self.n_column
                new_table = test_table
            elif self.args.model == "2-input":
                ops = [">=", "<"] * self.n_column
                rows, cols = test_table.shape
                new_table = np.zeros((rows, 2 * cols))
                for i in range(cols):
                    new_table[:, 2 * i] = test_table[:, i]
                    new_table[:, 2 * i + 1] = test_table[:, i]
            pred_batch = np.array(
                [calculate_query_cardinality(new_table, ops, row) for row in row_batch]
            ).reshape(-1, 1)
            ##### test end

            Table_Generated = self._generate_subtable_by_row_batch(
                row_batch, pred_batch, Table_Generated
            )
            if Table_Generated.shape[0] > self.n_row:
                Table_Generated = Table_Generated[: self.n_row, :]
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


class Generator_2_input(Generator_1_input):
    # PWL + Lattice + Copula: 2-input Model
    def __init__(
        self,
        args,
        path,
        table_size,
        unique_intervals,
        pwl_keypoints=None,
    ):
        super().__init__(
            args,
            path,
            table_size,
            unique_intervals,
            pwl_keypoints,
        )

        self.name = "Generator_2_input"
        self.dim = self.n_column * 2

    def _generate_subtable_by_row_batch(self, row_batch, pred_batch, Table_Generated=None):

        valid_indices = np.where(pred_batch[:, 0] >= 1)[0]
        for i in valid_indices:
            card = pred_batch[i, 0]
            vals = row_batch[i]
            # [::2] use the left interval of each column pair
            subtable = np.tile(vals[::2], (card, 1))
            Table_Generated = np.concatenate((Table_Generated, subtable), axis=0)
        return Table_Generated

    def generate_table_by_col(self, values, batch_size=10000):
        print(f"\nBegin Generating Table by Column, Total Column: {self.n_column}")

        column_one_point = np.array([[v[0], v[-1]] for v in self.unique_intervals.values()]).ravel()

        Table_Generated = None
        for col_idx in range(self.n_column):
            new_values = self._process_front_new_values_grid(Table_Generated, values, col_idx)

            batch_number = self._calculate_batch_number(new_values, batch_size)

            print(f"\nGenerating Column {col_idx}:")

            back_column = column_one_point[2 * col_idx + 2 :]

            New_Table_Generated = np.empty((0, 2 * col_idx + 2), dtype=np.float32)

            for batch in tqdm(self._yield_col_batch(new_values, batch_size), total=batch_number):

                col_batch = self._assemble_batch_with_back_columns(batch, back_column)

                pred_batch = self.model.predict(col_batch, verbose=0)
                # Case 1: change 0.8 to 0, 1.8 to 1
                pred_batch = (pred_batch * self.n_row).astype(int)

                New_Table_Generated = self._generate_subtable_by_col_batch(
                    batch, pred_batch, New_Table_Generated
                )

                if New_Table_Generated.shape[0] > self.n_row:
                    New_Table_Generated = New_Table_Generated[: self.n_row, :]
                    print(f"Reached table max row length({self.n_row}), stop generation.")
                    break
            Table_Generated = New_Table_Generated
        return Table_Generated[:, ::2]

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

    def _process_front_new_values_grid(self, Table_Generated, values, col_idx):
        # 只适用于 2-input
        if col_idx == 0:
            new_values = [values[col_idx]]
        else:
            front_column = np.unique(Table_Generated, axis=0)
            new_values = [front_column, values[col_idx]]
        return new_values

    def _assemble_batch_with_back_columns(self, batch, back_column):
        repeated_back_columns = np.tile(back_column, (len(batch), 1))
        col_batch = np.hstack((batch, repeated_back_columns))
        return col_batch

    def _generate_subtable_by_col_batch(self, batch, pred_batch, New_Table_Generated):

        valid_indices = np.where(pred_batch[:, 0] >= 1)[0]
        for i in valid_indices:
            card = pred_batch[i, 0]
            vals = batch[i]
            subtable = np.tile(vals, (card, 1))
            New_Table_Generated = np.concatenate((New_Table_Generated, subtable), axis=0)
        return New_Table_Generated

    def Test_generate_table_by_col(self, values, batch_size=10000, test_table=None):
        print(f"\nBegin Generating Table by Column, Total Column: {self.n_column}")

        column_one_point = np.array([[v[0], v[-1]] for v in self.unique_intervals.values()]).ravel()

        # test
        if self.args.model == "1-input":
            pass
            # ops = ["<="] * self.n_column
            # new_table = test_table
        elif self.args.model == "2-input":
            ops = [">=", "<"] * self.n_column
            rows, cols = test_table.shape
            new_table = np.zeros((rows, 2 * cols))
            for i in range(cols):
                new_table[:, 2 * i] = test_table[:, i]
                new_table[:, 2 * i + 1] = test_table[:, i]
        # test end

        Table_Generated = None
        for col_idx in range(self.n_column):
            new_values = self._process_front_new_values_grid(Table_Generated, values, col_idx)

            batch_number = self._calculate_batch_number(new_values, batch_size)

            print(f"\nGenerating Column {col_idx}:")

            back_column = column_one_point[2 * col_idx + 2 :]

            New_Table_Generated = np.empty((0, 2 * col_idx + 2), dtype=np.float32)

            for batch in tqdm(self._yield_col_batch(new_values, batch_size), total=batch_number):

                col_batch = self._assemble_batch_with_back_columns(batch, back_column)

                # pred_batch = self.model.predict(col_batch, verbose=0)
                # # Case 1: change 0.8 to 0, 1.8 to 1
                # pred_batch = (pred_batch * self.n_row).astype(int)

                # test begin
                # print(f"batch: {batch}")
                pred_batch = np.array(
                    [calculate_query_cardinality(new_table, ops, col_val) for col_val in col_batch]
                ).reshape(-1, 1)
                ##### test end

                New_Table_Generated = self._generate_subtable_by_col_batch(
                    batch, pred_batch, New_Table_Generated
                )

                if New_Table_Generated.shape[0] > self.n_row:
                    New_Table_Generated = New_Table_Generated[: self.n_row, :]
                    print(f"Reached table max row length({self.n_row}), stop generation.")
                    break
            Table_Generated = New_Table_Generated
            print(f"Generated table row length: {Table_Generated.shape[0]}")
        return Table_Generated[:, ::2]

    # def _process_front_new_values_grid(self, Table_Generated, unique_intervals, values, col_idx):
    #     # 只适用于 2-input
    #     if col_idx == 0:
    #         new_values = [values[0]]
    #     else:
    #         Table_Generated = np.unique(Table_Generated, axis=0)
    #         Table_G_size = Table_Generated.shape
    #         front_column = np.zeros(
    #             (Table_G_size[0], Table_G_size[1] * 2), dtype=Table_Generated.dtype
    #         )
    #         front_column[:, 0::2] = Table_Generated

    #         for j in range(Table_G_size[1]):
    #             interval = np.array(unique_intervals[j])
    #             idx = np.searchsorted(interval, Table_Generated[:, j])
    #             front_column[:, j * 2 + 1] = interval[idx + 1]
    #         new_values = [front_column, values[col_idx]]
    #     return new_values

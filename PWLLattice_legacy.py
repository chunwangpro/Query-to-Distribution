# PWL - Lattice Legacy Version
# PWL - Lattice model is an 1-input model

# Query Phase:
## common.CSVTable / cal_true_card

# Generation Phase:
## generate_5 / calculate_query_cardinality_numpy / np.concatenate

# with Plottings
import warnings

warnings.filterwarnings("ignore")
import argparse
import itertools
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_lattice as tfl
from tqdm import tqdm

import common
import datasets
import estimators as estimators_lib

OPS = {">": np.greater, "<": np.less, ">=": np.greater_equal, "<=": np.less_equal, "=": np.equal}


def Oracle(table, query):
    cols, idxs, ops, vals = query
    oracle_est = estimators_lib.Oracle(table)
    return oracle_est.Query(cols, ops, vals)


def cal_true_card(query, table):
    cols, idxs, ops, vals = query
    ops = np.array(ops)
    probs = Oracle(table, (cols, idxs, ops, vals))
    return probs


def GenerateQuery(table, min_num_filters, max_num_filters, rng, dataset):
    """Generate a random query."""
    num_filters = rng.randint(min_num_filters, max_num_filters)
    cols, idxs, ops, vals = SampleTupleThenRandom(table, num_filters, rng, dataset)
    sel = cal_true_card((cols, idxs, ops, vals), table) / table.data.shape[0]
    return cols, idxs, ops, vals, sel


def SampleTupleThenRandom(table, num_filters, rng, dataset):
    idxs = rng.choice(len(table.columns), replace=False, size=num_filters)
    idxs = np.sort(idxs)
    cols = np.take(table.columns, idxs)

    # If dom size >= 10, okay to place a range filter.
    # Otherwise, low domain size columns should be queried with equality.
    # ops = rng.choice(['<=', '>=', '='], size=num_filters)
    # ops = rng.choice(['<=', '>'], size=num_filters)
    ops = rng.choice(["<="], replace=True, size=num_filters)

    #     ops_all_eqs = ['='] * num_filters
    #     sensible_to_do_range = [c.DistributionSize() >= 10 for c in cols]
    #     ops = np.where(sensible_to_do_range, ops, ops_all_eqs)

    # if num_filters == len(table.columns):
    #     return table.columns,np.arange(len(table.columns)), ops, vals

    vals = []
    s = table.data.iloc[rng.randint(0, table.data.shape[0])]
    vals = s[idxs]
    vals = np.array(vals)

    op_a = []
    val_a = []
    for i in range(len(vals)):
        val_a.append([vals[i]])
        op_a.append([ops[i]])

    return cols, idxs, pd.DataFrame(op_a).values, pd.DataFrame(val_a).values


# 修改区间对 > >= < = 均适用
# 修改 初始interval
# 修改空interval使用【1】
# 修改最小值最大值为：0～min， max～max+max-max_2
def dictionary_column_interval(table_size, query_set):
    # Traverse all queries to apply the intervalization skill for each column
    n_column = table_size[1]
    column_interval = {}
    for i in range(n_column):
        column_interval[i] = set()  # use set([0, sys.maxsize]) to adapt '>' and '<'.
    for query in query_set:
        _, col_idxs, _, vals, _ = query
        for i in range(len(col_idxs)):
            column_interval[col_idxs[i]].add(vals[i][0])
    for k, v in column_interval.items():
        if not v:
            column_interval[k] = [0]  # use [0] to represent empty column interval
        else:
            column_interval[k] = sorted(list(v))
    return column_interval


def count_column_unique_interval(unique_intervals):
    # count unique query interval for each column
    return [len(v) for v in unique_intervals.values()]


def calculate_query_cardinality_df(df, ops, vals):
    # use pd.DataFrame
    # assert len(df.columns) == len(ops) == len(vals)
    bools = np.ones(len(df), dtype=bool)
    for (_, c), o, v in zip(df.items(), ops, vals):
        bools &= OPS[o](c, v)
    return bools.sum()


# inclusion-exclusion principle is time-consuming
# here we query once before generate to calculate the shortfall cardinality


def calculate_query_cardinality(data, ops, vals):
    # use pure numpy
    if data is None:
        return 0
    # assert data.shape[1] == len(ops) == len(vals)
    bools = np.ones(data.shape[0], dtype=bool)
    for i, (o, v) in enumerate(zip(ops, vals)):
        bools &= OPS[o](data[:, i], v)
    return bools.sum()


# n_column = 3
# data = np.array([[1, 2, 3, 4, 5], [10, 20, 30, 40, 50], [10, 20, 30, 40, 50]]).T
# df = pd.DataFrame(data, columns=[f"col_{i}" for i in range(n_column)])
# ops = [">=", ">=", ">="]
# vals = [3, 20, 20]
# print(calculate_query_cardinality_df(df, ops, vals))
# print(calculate_query_cardinality(data, ops, vals))


def calculate_Q_error_2(dataNew, query_set, n_row):
    Q_error = []
    dataNew = common.CsvTable("dataNew", dataNew, dataNew.columns)
    dataNew_cols = dataNew.columns
    our_table_row = dataNew.data.shape[0]

    for query in tqdm(query_set):
        _, idxs, ops, vals, _ = query
        cols = np.take(dataNew_cols, idxs)
        sel_pred = cal_true_card((cols, _, ops, vals), dataNew) / our_table_row
        sel_true = query[-1]
        if sel_pred == 0 and sel_pred == 0:
            Q_error.append(1)
            continue
        if sel_pred == 0:
            sel_pred = 1 / our_table_row
        elif sel_true == 0:
            sel_true = 1 / n_row
        Q_error.append(max(sel_pred / sel_true, sel_true / sel_pred))
    return Q_error


def calculate_Q_error_old(dataNew, query_set, n_row):
    Q_error = []
    our_table_row = dataNew.shape[0]
    for query in tqdm(query_set):
        df = dataNew.copy()
        for i in range(len(query[0])):
            if query[2][i][0] == "=":
                op = "=="
            else:
                op = query[2][i][0]
            df = df.query(f"col_{query[1][i]} {op} {query[3][i][0]}")
        sel = df.shape[0] / our_table_row

        sel1 = 1 / our_table_row if sel == 0 else sel
        sel2 = 1 / n_row if query[4] == 0 else query[4]
        Q_error.append(max(sel1 / sel2, sel2 / sel1))
    return Q_error


def execute_query(dataNew, query_set):
    diff = []
    for query in query_set:
        sentence = ""
        for i in range(len(query[0])):
            if i != 0:
                sentence += " and "
            sentence += f"col_{query[1][i]}"
            if query[2][i][0] == "=":
                sentence += "=="
            else:
                sentence += query[2][i][0]
            sentence += f"{query[3][i][0]}"
        sel = dataNew.query(sentence).shape[0] / dataNew.shape[0]
        sel2 = query[4]  # round(query[4] * n_row)
        if sel == 0:
            sel += 1 / dataNew.shape[0]
        if sel2 == 0:
            sel2 += 1 / n_row
        if sel < sel2:
            diff.append(sel2 / sel)
        else:
            diff.append(sel / sel2)
    return diff


def print_Q_error(Q_error, args):
    print(
        f"\n\n Q-error of Lattice (dataset={args.dataset}, query size={args.query_size}, condition=[{args.min_conditions}, {args.max_conditions}], loss={args.loss}):\n"
    )
    print(f"min:    {np.min(Q_error)}")
    print(f"10:     {np.percentile(Q_error, 10)}")
    print(f"20:     {np.percentile(Q_error, 20)}")
    print(f"30:     {np.percentile(Q_error, 30)}")
    print(f"40:     {np.percentile(Q_error, 40)}")
    print(f"median: {np.median(Q_error)}")
    print(f"60:     {np.percentile(Q_error, 60)}")
    print(f"70:     {np.percentile(Q_error, 70)}")
    print(f"80:     {np.percentile(Q_error, 80)}")
    print(f"90:     {np.percentile(Q_error, 90)}")
    print(f"95:     {np.percentile(Q_error, 95)}")
    print(f"99:     {np.percentile(Q_error, 99)}")
    print(f"max:    {np.max(Q_error)}")
    print(f"mean:   {np.mean(Q_error)}")


class PWLLattice:
    def __init__(
        self,
        path,
        table_shape,
        unique_intervals,
        pwl_keypoints=None,  # also can input table unique values
        lattice_size=2,
    ):
        self.name = "PWLLattice"
        self.path = path
        self.n_row = table_shape[0]
        self.n_column = table_shape[1]
        self.dim = len(unique_intervals.keys())
        self.lattice_size = lattice_size
        self.unique_intervals = unique_intervals
        self.pwl_calibration_input_keypoints = (
            unique_intervals if pwl_keypoints is None else pwl_keypoints
        )
        self.model_path = f"{self.path}/{self.name}_model"
        self.weight_path = f"{self.path}/{self.name}_weight"
        # self.sample_feat = None

        self.model_inputs = []
        for i in range(self.dim):
            self.model_inputs.append(tf.keras.layers.Input(shape=[1], name="col_%s" % i))
            # self.model_inputs.append(
            #     tf.keras.layers.Input(shape=[1], name='col%s_l' % i))
            # self.model_inputs.append(
            #     tf.keras.layers.Input(shape=[1], name='col%s_u' % i))

        self.calibrators = []
        for i in range(self.dim):
            # self.calibrators.append(
            #     tfl.layers.PWLCalibration(
            #         input_keypoints=np.linspace(
            #             feat_mins[i],
            #             feat_maxs[i],
            #             num=pwl_calibration_num_keypoints),
            #         dtype=tf.float32,
            #         output_min=0.0,
            #         output_max=lattice_size - 1.0,
            #         monotonicity='decreasing',
            #     ))
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
                    monotonicity="increasing",
                )
            )

        self.lattice = tfl.layers.Lattice(
            lattice_sizes=[lattice_size] * self.dim,  # (self.dim * 2),
            interpolation="simplex",  # 可以尝试别的插值类型
            monotonicities=["increasing"] * self.dim,  # (self.dim * 2),
            output_min=0.0,
            output_max=1.0,
            name="lattice",
        )

        # self.output1 = tfl.layers.PWLCalibration(
        #     input_keypoints=np.linspace(0.0,
        #                                 np.log(self.n_row),
        #                                 num=pwl_calibration_num_keypoints),
        #     dtype=tf.float32,
        #     output_min=0.0,
        #     output_max=np.log(self.n_row),
        #     name='output1_calib',
        # )

        # self.output2 = tfl.layers.PWLCalibration(
        #     input_keypoints=np.linspace(0.0,
        #                                 self.n_row,
        #                                 num=pwl_calibration_num_keypoints),
        #     dtype=tf.float32,
        #     output_min=0.0,
        #     output_max=self.n_row,
        #     name='output2_calib',
        # )

        # self.lattice_inputs = []
        # for i in range(self.dim):  # (self.dim) * 2):
        #     self.lattice_inputs.append(self.calibrators[i](
        #         self.model_inputs[i]))
        # self.model_output = self.output2(
        #     tf.keras.backend.exp(
        #         self.output1(self.lattice(self.lattice_inputs))))

        self.lattice_inputs = []
        for i in range(self.dim):  # (self.dim) * 2):
            self.lattice_inputs.append(self.calibrators[i](self.model_inputs[i]))
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
        reduceLR_factor=0.5,
        reduceLR_patience=20,
        verbose=1,
        loss="MSE",
    ):
        assert X.shape[0] == y.shape[0]
        # assert X.shape[1] == self.dim * 2

        X = X.astype(np.float32)
        y = y.astype(np.float32)

        # for i in range(self.dim):
        #     self.calibrators[i].input_keypoints = compute_quantiles(
        #         X[:, i].ravel(),
        #         num_keypoints=self.pwl_calibration_num_keypoints,
        #     )
        # self.calibrators[i * 2].input_keypoints = compute_quantiles(
        #     X[:, [i * 2, i * 2 + 1]].ravel(),
        #     num_keypoints=self.pwl_calibration_num_keypoints,
        # )
        # self.calibrators[i * 2 + 1].input_keypoints = compute_quantiles(
        #     X[:, [i * 2, i * 2 + 1]].ravel(),
        #     num_keypoints=self.pwl_calibration_num_keypoints,
        # )

        features = [X[:, i] for i in range(X.shape[1])]
        target = y

        if loss == "KL":
            lossFunc = tf.keras.losses.KLDivergence()
        if loss == "MAE":
            lossFunc = tf.keras.losses.mean_absolute_error
        if loss == "MSE":
            lossFunc = tf.keras.losses.mean_squared_error
        if loss == "MAPE":
            lossFunc = tf.keras.losses.mean_absolute_percentage_error

        self.model.compile(loss=lossFunc, optimizer=tf.keras.optimizers.Adamax(lr))

        earlyStopping = tf.keras.callbacks.EarlyStopping(
            restore_best_weights=True,
            monitor="loss",
            mode="min",
            patience=100,
            verbose=verbose,
        )
        mcp_save = tf.keras.callbacks.ModelCheckpoint(
            filepath=f"{self.weight_path}.h5",
            save_weights_only=True,
            save_best_only=True,
            monitor="loss",
            mode="min",
            verbose=1,
        )
        reduce_lr_loss = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="loss",
            factor=reduceLR_factor,
            patience=reduceLR_patience,
            verbose=verbose,
            min_delta=1e-10,
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
        self.model.save(f"{self.model_path}")

    def load(self, modelPath=None, summary=False):
        Path = modelPath if modelPath else self.path
        self.model = tf.keras.models.load_model(f"{Path}/{self.name}_model")
        # self.model.load_weights(f"{Path}/{self.name}_weight.h5")
        if summary:
            print(self.model.summary())

    def predict(self, grid):
        # assert grid.shape[1] == self.dim
        pred = self.model.predict(np.hsplit(grid, self.dim))
        return pred

    def generate(self, grid, pred=None):
        return self.generate_5(grid, pred)

    def generate_0(self, grid):
        assert grid.shape[1] == self.dim
        pred = m.model.predict(np.hsplit(grid, self.dim))
        dataNew = pd.DataFrame(
            columns=[f"col_{i}" for i in range(n_column)],
            index=[i for i in range(np.prod(column_interval_number))],
        )

        # generate 一条再query一条，再generate下一条
        count = 0
        for i in trange(grid.shape[0]):
            sentence = ""
            for j in range(grid.shape[1]):
                if j != 0:
                    sentence += " and "
                sentence += f"col_{j}"
                sentence += " <= "
                sentence += f"{grid[i][j]:f}"

            xi = pred[i][0] * n_row - dataNew.query(sentence).shape[0]

            if int(xi) > 0:
                floor = np.floor(xi)
                ceil = np.ceil(xi)
                if floor == ceil:
                    card = int(xi)
                else:
                    card = np.random.choice([floor, ceil], p=[xi - floor, ceil - xi]).astype("int")
                dataNew.iloc[count : count + card, :] = grid[i]
                count += card
        dataNew.dropna(axis=0, how="all", inplace=True)
        return dataNew

    def generate_1(self, grid, pred=None):
        # 使用 df.query 生成数据
        if pred is None:
            pred = self.predict(grid)
        assert pred.shape[0] == grid.shape[0]
        # generate one query means generate one row, then next query and next row
        dataNew = pd.DataFrame(
            columns=[f"col_{i}" for i in range(self.n_column)],
            index=[i for i in range(self.n_row)],
        )
        count = 0
        for i in tqdm(range(grid.shape[0])):
            df = dataNew
            grid_value = grid[i]
            for j in range(self.n_column):
                # use <= to filter the data
                df = df.query(f"col_{j} <= {grid_value[j]}", inplace=False)
            xi = pred[i][0] * self.n_row - df.shape[0]
            # Case 1: change 0.8 to 0, 1.8 to 1,
            card = int(xi)

            if card >= 1:
                # Case 2: change 0.8 to 0, 1.8 to 2
                card = np.round(xi).astype("int")
                floor = np.floor(xi)
                ceil = np.ceil(xi)
                if floor == ceil:
                    card = int(xi)
                else:
                    card = np.random.choice([floor, ceil], p=[xi - floor, ceil - xi]).astype("int")
                ## end

                if count + card > self.n_row or (
                    count + card == self.n_row and i != grid.shape[0] - 1
                ):
                    print(
                        f"Reached table max row length({self.n_row}) in {i}-th row of grid with grid value of {grid[i]}, stop generation."
                    )
                    dataNew.iloc[count:, :] = grid_value
                    break
                else:
                    dataNew.iloc[count : count + card, :] = grid_value
                    count += card

        else:
            print("Finished table generation")
        dataNew.dropna(axis=0, how="all", inplace=True)
        return dataNew

    def generate_2(self, grid, pred=None):
        # common.CsvTable / cal_true_card / dataNew.iloc
        if pred is None:
            pred = self.predict(grid)
        assert pred.shape[0] == grid.shape[0]
        # generate one query means generate one row, then next query and next row
        dataNew = pd.DataFrame(
            columns=[f"col_{i}" for i in range(self.n_column)],
            index=[i for i in range(self.n_row)],
        )
        dataNew[:] = sys.maxsize
        ops = np.array([["<="]] * self.n_column, dtype=object)
        count = 0
        for i in tqdm(range(grid.shape[0])):
            grid_value = grid[i]
            vals = grid_value.reshape(-1, 1)
            table = common.CsvTable("dataNew", dataNew, dataNew.columns)
            dataNew_cols = table.columns
            card_current = cal_true_card((dataNew_cols, None, ops, vals), table)
            xi = pred[i][0] * self.n_row - card_current
            # Case 1: change 0.8 to 0, 1.8 to 1,
            card = int(xi)
            if card >= 1:
                if count + card > self.n_row or (
                    count + card == self.n_row and i != grid.shape[0] - 1
                ):
                    print(
                        f"Reached table max row length({self.n_row}) in {i}-th row of grid with grid value of {grid[i]}, stop generation."
                    )
                    dataNew.iloc[count:, :] = grid_value
                    break
                else:
                    dataNew.iloc[count : count + card, :] = grid_value
                    count += card
        else:
            print("Finished table generation")
        dataNew.dropna(axis=0, how="all", inplace=True)
        return dataNew

    def generate_3(self, grid, pred=None):
        # common.CsvTable / cal_true_card / pd.concat
        if pred is None:
            pred = self.predict(grid)
        assert pred.shape[0] == grid.shape[0]
        # generate by row, one query may generate several rows
        dataNew = pd.DataFrame(columns=[f"col_{i}" for i in range(self.n_column)])
        ops = np.array([["<="]] * self.n_column, dtype=object)
        count = 0
        for i in tqdm(range(grid.shape[0])):
            grid_value = grid[i]
            vals = grid_value.reshape(-1, 1)
            table = common.CsvTable("dataNew", dataNew, dataNew.columns)
            dataNew_cols = table.columns
            card_current = cal_true_card((dataNew_cols, None, ops, vals), table)
            xi = pred[i][0] * self.n_row - card_current
            # Case 1: change 0.8 to 0, 1.8 to 1,
            card = int(xi)
            if card >= 1:
                # Case 2: change 0.8 to 0, 1.8 to 2
                # if int(xi) >= 1:
                # card = np.round(xi).astype("int")
                # floor = np.floor(xi)
                # ceil = np.ceil(xi)
                # if floor == ceil:
                #     card = int(xi)
                # else:
                #     card = np.random.choice([floor, ceil], p=[xi - floor, ceil - xi]).astype("int")
                if count + card > self.n_row or (
                    count + card == self.n_row and i != grid.shape[0] - 1
                ):
                    print(
                        f"Reached table max row length({self.n_row}) in {i}-th row of grid with grid value of {grid[i]}, stop generation."
                    )
                    left_row = self.n_row - count
                    df3 = pd.DataFrame(
                        {f"col_{k}": np.tile(grid_value[k], left_row) for k in range(self.n_column)}
                    )
                    dataNew = pd.concat([dataNew, df3], ignore_index=True)
                    break
                else:
                    df3 = pd.DataFrame(
                        {f"col_{k}": np.tile(grid_value[k], card) for k in range(self.n_column)}
                    )
                    dataNew = pd.concat([dataNew, df3], ignore_index=True)
                    count += card
        else:
            print("Finished table generation")
        return dataNew

    def generate_4(self, grid, pred=None):
        # df / calculate_query_cardinality / np.concatenate
        if pred is None:
            pred = self.predict(grid)
        assert pred.shape[0] == grid.shape[0]
        # generate by row, one query may generate several rows
        column_names = [f"col_{i}" for i in range(self.n_column)]
        dataNew = pd.DataFrame(columns=column_names)
        ops = ["<="] * self.n_column

        count = 0
        ArrayNew = None
        pred = (pred * self.n_row).astype(int)  # Case 1: change 0.8 to 0, 1.8 to 1
        for i in tqdm(range(grid.shape[0])):
            vals = grid[i]
            card = pred[i, 0] - calculate_query_cardinality_df(dataNew, ops, vals)

            if card >= 1:
                array3 = np.repeat(vals, card).reshape(self.n_column, card).T
                ArrayNew = (
                    array3 if ArrayNew is None else np.concatenate((ArrayNew, array3), axis=0)
                )
                dataNew = pd.DataFrame(ArrayNew, columns=column_names)
                count += card
                if count > self.n_row:
                    print(
                        f"Reached table max row length({self.n_row}) in {i}-th row of grid with grid value of {vals}, stop generation."
                    )
                    break
        else:
            print("Completed table generation")
            # if count < n_row:
            #     print(
            #         f"Reached table max row length({n_row}) in the last row of grid, stop generation."
            #     )
            #     # 如果不足需要补 系统最大值
            #     # dataNew = pd.DataFrame(ArrayNew, columns=column_names)
            return dataNew
        return dataNew.iloc[: self.n_row, :]

    def generate_5(self, grid, pred=None):
        # numpy / calculate_query_cardinality_numpy / np.concatenate
        if pred is None:
            pred = self.predict(grid)
        assert pred.shape[0] == grid.shape[0]
        # generate by row, one query may generate several rows
        print("Begin Generating Table ...")
        count = 0
        ArrayNew = None
        ops = ["<="] * self.n_column
        pred = (pred * self.n_row).astype(int)  # Case 1: change 0.8 to 0, 1.8 to 1
        for i in tqdm(range(grid.shape[0])):
            vals = grid[i]
            card = pred[i, 0] - calculate_query_cardinality(ArrayNew, ops, vals)
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
            print("Done.\n")
            # if count < self.n_row:
            #     print(
            #         f"Generated table row length({count}) is less than the original table row length({self.n_row})."
            #     )
            #     # 如果不足,补系统最大值吗？
            return ArrayNew
        return ArrayNew[: self.n_row, :]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="wine2", help="Dataset.")
    parser.add_argument("--query-size", type=int, default=1000, help="query size")
    parser.add_argument("--min-conditions", type=int, default=2, help="min num of conditions")
    parser.add_argument("--max-conditions", type=int, default=2, help="max num of conditions")
    parser.add_argument("--epochs", type=int, default=2000, help="Number of epochs to train for.")
    parser.add_argument("--bs", type=int, default=1000, help="Batch size.")
    parser.add_argument("--loss", type=str, default="MSE", help="Loss.")
    parser.add_argument("--lattice-size", type=int, default=2, help="Lattice size.")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    try:
        args = parser.parse_args()
    except:
        # args = parser.parse_args([])
        args, unknown = parser.parse_known_args()

    def make_directory(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    OPS = {
        ">": np.greater,
        "<": np.less,
        ">=": np.greater_equal,
        "<=": np.less_equal,
        "=": np.equal,
    }
    FilePath = (
        f"{args.dataset}_{args.query_size}_{args.min_conditions}_{args.max_conditions}_{args.loss}"
    )
    resultsPath = f"results/{FilePath}"
    modelPath = f"saved_models/{FilePath}"
    make_directory(resultsPath)
    make_directory(modelPath)

    table = datasets.LoadDataset(f"{args.dataset}.csv", args.dataset)

    print("Begin Generating Queries Set...")
    rng = np.random.RandomState(args.seed)
    query_set = [
        GenerateQuery(table, args.min_conditions, args.max_conditions + 1, rng, args.dataset)
        for _ in tqdm(range(args.query_size))
    ]
    print("Done.\n")

    table_size = table.data.shape
    n_row = table_size[0]
    n_column = table_size[1]

    print("Begin Intervalization ...")
    unique_intervals = dictionary_column_interval(table_size, query_set)
    column_interval_number = count_column_unique_interval(unique_intervals)
    print("Done.\n")
    print(column_interval_number)

    # 修改 x = [sys.maxsize] * n_column     # 这里使用每个col_unique_interval的最后一个元素即可
    # 如果使用两个input的话，一个修改为最大，一个修改为最小
    train_X = []
    train_Y = []
    for query in query_set:
        x = [sys.maxsize] * n_column  # 这里使用每个col_unique_interval的最后一个元素即可
        _, idxs, _, vals, sel = query
        for i in range(len(idxs)):
            x[idxs[i]] = vals[i][0]
        train_X.append(x)
        train_Y.append(sel)

    train_X = np.array(train_X).astype(np.float32)
    train_Y = np.array(train_Y).astype(np.float32).reshape(-1, 1)

    # make train set unique
    # train = np.concatenate((train_X, train_Y), axis=1)
    # train = np.unique(train, axis=0)
    # train_X, train_Y = np.hsplit(train, [-1])

    m = PWLLattice(
        modelPath,
        table_size,
        unique_intervals,
        pwl_keypoints=None,
        lattice_size=args.lattice_size,
    )

    m.fit(train_X, train_Y, lr=args.lr, bs=args.bs, epochs=args.epochs, loss=args.loss)

    # use interval grid to generate
    values = [v for v in unique_intervals.values()]
    mesh = np.meshgrid(*values)  # 所有 unique interval 的笛卡尔积网格
    grid = np.array(mesh).T.reshape(-1, len(values)).astype(np.float32)

    m.load()
    grid_pred = m.predict(grid)

    dataNew = m.generate(grid, grid_pred)
    dataNew = pd.DataFrame(dataNew, columns=[f"col_{i}" for i in range(n_column)])

    Q_error = calculate_Q_error_2(dataNew, query_set, table_size)
    print_Q_error(Q_error, args)

    # Plottings:
    pred = grid_pred
    # plot-1
    plt.figure(figsize=(20, 8))
    plt.plot(pred, "bo")
    plt.show()
    # plot-2
    fig1 = plt.figure(figsize=(15, 8))
    ax1 = plt.axes(projection="3d")
    # xx = unique_intervals[1]
    # yy = unique_intervals[0]
    # X, Y = np.meshgrid(xx, yy)
    X = grid[:, 1].reshape(column_interval_number[0], column_interval_number[1])  # 这样也可以
    Y = grid[:, 0].reshape(column_interval_number[0], column_interval_number[1])
    Z = pred.reshape(column_interval_number[0], column_interval_number[1])
    ax1.plot_surface(X, Y, Z, cmap="viridis")
    plt.show()
    # plot-3
    fig2 = plt.figure(figsize=(10, 8))
    ax2 = fig2.add_subplot(111)
    cs = ax2.contourf(X, Y, Z, cmap="viridis")
    # Alternatively, you can manually set the levels
    # and the norm:
    # lev_exp = np.arange(np.floor(np.log10(z.min())-1),
    #                    np.ceil(np.log10(z.max())+1))
    # levs = np.power(10, lev_exp)
    # cs = ax.contourf(X, Y, z, levs, norm=colors.LogNorm())
    cbar = fig2.colorbar(cs)
    plt.show()
    # plot-4
    # query 对网格的覆盖率 散点图
    fig4 = plt.figure(figsize=(10, 10))
    xtick = unique_intervals[0]
    ytick = unique_intervals[1]
    plt.scatter(train_X[:, 0], train_X[:, 1], c="b")
    plt.vlines(xtick, min(ytick), max(ytick), colors="green")
    plt.hlines(ytick, min(xtick), max(xtick), colors="green")
    plt.show()

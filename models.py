import argparse
import itertools
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_lattice as tfl
from matplotlib import pyplot as plt
from tqdm import tqdm

import common
import datasets
import estimators as estimators_lib
from query_func import *


class LatticeCDF:
    def __init__(
        self,
        name,
        table_shape,
        unique_intervals,
        pwl_keypoints=None,  # also can input table unique values
        lattice_size=2,
    ):
        self.name = name
        self.n_row = table_shape[0]
        self.n_column = table_shape[1]
        self.dim = len(unique_intervals.keys())
        self.lattice_size = lattice_size
        self.unique_intervals = unique_intervals
        self.pwl_calibration_input_keypoints = (
            unique_intervals if pwl_keypoints is None else pwl_keypoints
        )
        self.model_path = f"saved_models/models/{self.name}_model"
        self.weight_path = f"saved_models/weights/{self.name}_weight"
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
        loss="KL",
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

        # self.model.compile(loss=lossFunc, optimizer="adam", metrics=["accuracy"])
        self.model.compile(loss=lossFunc, optimizer=tf.keras.optimizers.Adam(lr))

        earlyStopping = tf.keras.callbacks.EarlyStopping(
            monitor="loss", patience=100, verbose=verbose, mode="min", restore_best_weights=True
        )
        mcp_save = tf.keras.callbacks.ModelCheckpoint(
            filepath=f"{self.weight_path}.hdf5",
            save_best_only=True,
            monitor="loss",
            mode="min",
            save_weights_only=True,
            verbose=1,
        )
        reduce_lr_loss = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="loss",
            factor=reduceLR_factor,
            patience=reduceLR_patience,
            verbose=verbose,
            epsilon=1e-10,
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
        self.model.save(f"{self.model_path}.h5")

    def load(self, name=None, summary=False):
        if name is None:
            name = self.name  # 这里要改,还要改self.model_path、self.weight_path
        # self.model = tf.keras.models.load_model(f"{name}.hdf5")
        self.model.load_weights(f"{name}.h5")
        if summary:
            print(self.model.summary())

    # 在代码块外（main.py中）调用
    # m.load("my_model_final")
    # print(m.model.summary())

    def predict(self, grid):
        assert grid.shape[1] == self.dim
        pred = self.model.predict(np.hsplit(grid, self.dim))
        return pred

    def generate(self, grid, pred=None):
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
        # 使用 common.CsvTable / cal_true_card / dataNew.iloc
        # 比第一种慢
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
        # 使用 common.CsvTable / cal_true_card / pd.concat
        # 比前两种快
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

                    # dataNew = dataNew.append(df3, ignore_index = True)
                    dataNew = pd.concat([dataNew, df3], ignore_index=True)
                    break
                else:
                    df3 = pd.DataFrame(
                        {f"col_{k}": np.tile(grid_value[k], card) for k in range(self.n_column)}
                    )
                    # dataNew = dataNew.append(df3, ignore_index = True)
                    dataNew = pd.concat([dataNew, df3], ignore_index=True)
                    count += card
        else:
            print("Finished table generation")
        # dataNew.dropna(axis=0, how="all", inplace=True)
        return dataNew

    def generate_4(self, grid, pred=None):
        # 使用 df / calculate_query_cardinality / np.concatenate
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
            card = pred[i, 0] - calculate_query_cardinality(dataNew, ops, vals)

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
        # 使用 numpy / calculate_query_cardinality_numpy / np.concatenate
        if pred is None:
            pred = self.predict(grid)
        assert pred.shape[0] == grid.shape[0]
        # generate by row, one query may generate several rows
        count = 0
        ArrayNew = None
        ops = ["<="] * self.n_column
        pred = (pred * self.n_row).astype(int)  # Case 1: change 0.8 to 0, 1.8 to 1
        column_names = [f"col_{i}" for i in range(self.n_column)]
        for i in tqdm(range(grid.shape[0])):
            vals = grid[i]
            card = pred[i, 0] - calculate_query_cardinality_numpy(ArrayNew, ops, vals)
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
            print("Completed table generation")
            # if count < n_row:
            #     print(
            #         f"Reached table max row length({n_row}) in the last row of grid, stop generation."
            #     )
            #     # 如果不足需要补 系统最大值
            #     # dataNew = pd.DataFrame(ArrayNew, columns=column_names)
            return pd.DataFrame(ArrayNew, columns=column_names)
        return pd.DataFrame(ArrayNew, columns=column_names).iloc[: self.n_row, :]

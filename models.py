import itertools

import numpy as np
import tensorflow as tf
import tensorflow_lattice as tfl
from tqdm import tqdm

from query_func import *


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
        assert grid.shape[1] == self.dim
        pred = self.model.predict(np.hsplit(grid, self.dim))
        return pred

    def generate(self, unique_intervals, batch_size=10000):
        print(f"\nBegin Generating Table from Batches ({batch_size=}) ...")
        values = [v for v in unique_intervals.values()]
        total_combinations = np.prod([len(v) for v in values])
        ArrayNew = None
        for grid_batch in tqdm(
            self._generate_grid_batches(values, batch_size),
            total=(total_combinations // batch_size) + 1,
        ):
            pred_batch = self.predict(grid_batch)
            ArrayNew = self.generate_from_batches(grid_batch, pred_batch, ArrayNew)
        if ArrayNew.shape[0] < self.n_row:
            print(
                f"Generated table row length({ArrayNew.shape[0]}) is less than the original table row length({self.n_row})."
            )
        print("Done.\n")
        return ArrayNew

    def _generate_grid_batches(self, values, batch_size):
        iterator = itertools.product(*values)
        while True:
            batch = list(itertools.islice(iterator, batch_size))
            if not batch:
                break
            yield np.array(batch).astype(np.float32)

    def generate_from_batches(self, grid_batch, pred_batch, ArrayNew=None):
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

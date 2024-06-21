import logging
import os
import sys

import numpy as np
import tensorflow as tf
import tensorflow_lattice as tfl
from tqdm import tqdm

logging.disable(sys.maxsize)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tf.config.threading.set_intra_op_parallelism_threads(1)


def generate_query(data, num):
    unique_vals = []
    for i in range(data.shape[1]):
        unique_vals.append(np.unique(data[:, i]))

    query = []
    for i in range(num):
        for j in range(data.shape[1]):
            if j == 0:
                query_i = np.sort(np.random.choice(unique_vals[j], 2))
            else:
                query_i = np.append(query_i, np.sort(np.random.choice(unique_vals[j], 2)))
        query.append(query_i)

    return np.array(query)


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


def compute_quantiles(features, num_keypoints=10, clip_min=None, clip_max=None, missing_value=None):
    # Clip min and max if desired.
    if clip_min is not None:
        features = np.maximum(features, clip_min)
        features = np.append(features, clip_min)
    if clip_max is not None:
        features = np.minimum(features, clip_max)
        features = np.append(features, clip_max)
    # Make features unique.
    unique_features = np.unique(features)
    # Remove missing values if specified.
    if missing_value is not None:
        unique_features = np.delete(unique_features, np.where(unique_features == missing_value))

    # Compute and return quantiles over unique non-missing feature values.
    return np.unique(
        np.quantile(
            unique_features, np.linspace(0.0, 1.0, num=num_keypoints), interpolation="nearest"
        )
    ).astype(float)


class LatticeCDF:
    def __init__(
        self,
        name,
        lattice_size,
        feat_mins,
        feat_maxs,
        tb_rows,
        pwl_calibration_input_keypoints,
        pwl_calibration_num_keypoints=200,
        l2=1e-6,
    ):
        assert len(feat_mins) == len(feat_maxs)
        self.l2 = l2
        self.name = name
        self.tb_rows = tb_rows
        self.dim = len(feat_mins)
        self.feat_mins = feat_mins
        self.feat_maxs = feat_maxs
        self.lattice_size = lattice_size
        self.pwl_calibration_num_keypoints = pwl_calibration_num_keypoints
        self.pwl_calibration_input_keypoints = pwl_calibration_input_keypoints
        self.sample_feat = None

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
                    input_keypoints=np.array(pwl_calibration_input_keypoints[i]),
                    # input_keypoints=np.linspace(
                    #     feat_mins[i],
                    #     feat_maxs[i],
                    #     num=pwl_calibration_num_keypoints),
                    dtype=tf.float32,
                    output_min=0.0,
                    output_max=lattice_size - 1.0,
                    monotonicity="increasing",
                )
            )

        self.lattice = tfl.layers.Lattice(
            lattice_sizes=[lattice_size] * self.dim,  # (self.dim * 2),
            interpolation="simplex",
            monotonicities=["increasing"] * self.dim,  # (self.dim * 2),
            output_min=0.0,
            output_max=1.0,
            name="lattice",
        )

        # self.output1 = tfl.layers.PWLCalibration(
        #     input_keypoints=np.linspace(0.0,
        #                                 np.log(tb_rows),
        #                                 num=pwl_calibration_num_keypoints),
        #     dtype=tf.float32,
        #     output_min=0.0,
        #     output_max=np.log(tb_rows),
        #     name='output1_calib',
        # )

        # self.output2 = tfl.layers.PWLCalibration(
        #     input_keypoints=np.linspace(0.0,
        #                                 tb_rows,
        #                                 num=pwl_calibration_num_keypoints),
        #     dtype=tf.float32,
        #     output_min=0.0,
        #     output_max=tb_rows,
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
        bs=16,
        epochs=3000,
        reduceLR_factor=0.5,
        reduceLR_patience=20,
        verbose=1,
        loss="MSLE",
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

        l = tf.keras.losses.mean_squared_logarithmic_error
        if loss == "MAE":
            l = tf.keras.losses.mean_absolute_error
        if loss == "MSE":
            l = tf.keras.losses.mean_squared_error
        if loss == "MAPE":
            l = tf.keras.losses.mean_absolute_percentage_error

        self.model.compile(loss=l, optimizer=tf.keras.optimizers.Adamax(lr))

        # earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=500, verbose=verbose, mode='min')
        mcp_save = tf.keras.callbacks.ModelCheckpoint(
            "%s.hdf5" % self.name,
            save_best_only=True,
            monitor="loss",
            mode="min",
            save_weights_only=True,
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
            callbacks=[mcp_save, reduce_lr_loss],
        )
        self.model.load_weights("%s.hdf5" % self.name)

    def load(self, name):
        self.model.load_weights("%s.hdf5" % self.name)

    def generate(self, grid, table_size):

        return df

    def sample(self, unique_vals, n):
        assert self.dim == len(unique_vals)

        if self.sample_feat is None:
            self.sample_feat = []
            for i in range(self.dim):
                features = []
                for j in range(self.dim):
                    features.append(np.array([-1e8] * len(unique_vals[i]), dtype=np.float32))
                    if i == j:
                        features.append(np.array(unique_vals[j], dtype=np.float32))
                    else:
                        features.append(
                            np.array([unique_vals[i][-1]] * len(unique_vals[i]), dtype=np.float32)
                        )
                self.sample_feat.append(features)

        s = np.zeros([n, self.dim])
        pred = self.model.predict(self.sample_feat[0], batch_size=10240).ravel() / self.tb_rows
        pred /= pred[-1]
        prob = np.array([pred[0]])
        prob = np.append(prob, pred[1:] - pred[:-1])
        prob[prob < 0] = 0
        prob[-1] += abs(1 - prob.sum())
        prob /= prob.sum()
        s0 = np.sort(np.random.choice(unique_vals[0], size=n, p=prob))
        # r = np.random.uniform(low=pred[0], high=pred[-1], size=n)
        # index = np.searchsorted(pred, r, side='right') - 1
        # s0 = unique_vals[0][index]
        # s0 = np.sort(s0)
        s[:, 0] = s0
        unique0, counts0 = np.unique(s0, return_counts=True)

        if self.dim > 1:
            s1 = np.array([])
            for i in tqdm(range(len(unique0))):
                if i == 0:
                    self.sample_feat[1][0] = np.array(
                        [-1e8] * len(self.sample_feat[1][0]), dtype=np.float32
                    )
                else:
                    self.sample_feat[1][0] = np.array(
                        [unique0[i - 1]] * len(self.sample_feat[1][0]), dtype=np.float32
                    )
                self.sample_feat[1][1] = np.array(
                    [unique0[i]] * len(self.sample_feat[1][0]), dtype=np.float32
                )
                pred = (
                    self.model.predict(self.sample_feat[1], batch_size=10240).ravel() / self.tb_rows
                )
                pred /= pred[-1]
                prob = np.array([pred[0]])
                prob = np.append(prob, pred[1:] - pred[:-1])
                prob[prob < 0] = 0
                prob[-1] += abs(1 - prob.sum())
                prob /= prob.sum()
                s1 = np.append(
                    s1, np.sort(np.random.choice(unique_vals[1], size=counts0[i], p=prob))
                )
                # r = np.random.uniform(low=pred[0], high=pred[-1], size=counts0[i])
                # index = np.searchsorted(pred, r, side='right') - 1
                # s1 = np.append(s1, np.sort(unique_vals[1][index]))
            s[:, 1] = s1
            unique1, counts1 = np.unique(s[:, [0, 1]], axis=0, return_counts=True)

            for j in range(2, self.dim):
                sj = np.array([])
                for i in tqdm(range(len(unique1))):
                    for k in range(1, j + 1):
                        idx = np.searchsorted(unique_vals[k - 1], unique1[i][k - 1])
                        if idx == 0:
                            self.sample_feat[j][(k - 1) * 2] = np.array(
                                [-1e8] * len(self.sample_feat[j][0]), dtype=np.float32
                            )
                        else:
                            self.sample_feat[j][(k - 1) * 2] = np.array(
                                [unique_vals[k - 1][idx - 1]] * len(self.sample_feat[j][0]),
                                dtype=np.float32,
                            )
                        self.sample_feat[j][(k - 1) * 2 + 1] = np.array(
                            [unique1[i][k - 1]] * len(self.sample_feat[j][0]), dtype=np.float32
                        )
                    pred = (
                        self.model.predict(self.sample_feat[j], batch_size=10240).ravel()
                        / self.tb_rows
                    )
                    pred /= pred[-1]
                    prob = np.array([pred[0]])
                    prob = np.append(prob, pred[1:] - pred[:-1])
                    prob[prob < 0] = 0
                    prob[-1] += abs(1 - prob.sum())
                    prob /= prob.sum()
                    sj = np.append(
                        sj, np.sort(np.random.choice(unique_vals[j], size=counts1[i], p=prob))
                    )
                    # r = np.random.uniform(low=pred[0], high=pred[-1], size=counts1[i])
                    # index = np.searchsorted(pred, r, side='right') - 1
                    # sj = np.append(sj, np.sort(unique_vals[j][index]))
                s[:, j] = sj
                if j < self.dim - 1:
                    unique1, counts1 = np.unique(s[:, : j + 1], axis=0, return_counts=True)

        return s


if __name__ == "__main__":
    # dataset = 'cover3'
    # dataset = 'dmv3'
    dataset = "wine3"
    n_conditions = 1
    n_query = 10000
    seed = 1234
    np.random.seed(seed)

    data = np.genfromtxt("datasets/%s.csv" % dataset, delimiter=",")
    data = data[:, :n_conditions]

    unique_vals = []
    for i in range(data.shape[1]):
        unique_vals.append(np.unique(data[:, i]))

    # train_X = np.load('/data1/xiao.hx/data/selectivity_experiments/random_5000_attr_059_features.npy').astype(np.float32)[:2000]
    # train_X = train_X[:, :4]

    train_X = generate_query(data, n_query).astype(np.float32)
    y = []
    for i in tqdm(range(len(train_X))):
        y.append(execute_query(data, train_X[i]))
    train_Y = np.array(y).astype(np.float32)

    train_X = train_X[train_Y != 0]
    train_Y = train_Y[train_Y != 0]

    # train_X = train_X[:10]
    # train_Y = train_Y[:10]

    # ceil = []
    # for i in range(data.shape[1]):
    #     ceil.append(unique_vals[i][0]-1)
    #     ceil.append(unique_vals[i][-1])
    # ceil = np.array(ceil)
    # train_X = np.vstack([train_X, ceil])
    # train_Y = np.append(train_Y, data.shape[0])

    train_col = []
    for i in range(data.shape[1]):
        train_col.append(train_X[:, i * 2 : (i + 1) * 2])

    feat_mins = [x.min() for x in train_col]
    feat_maxs = [x.max() for x in train_col]

    m = LatticeCDF(dataset, 30, feat_mins, feat_maxs, data.shape[0])
    m.fit(train_X, train_Y, bs=1024, epochs=1000, loss="MSE")
    # m.load(dataset)

    unique_vals = []
    for i in range(data.shape[1]):
        unique_vals.append(np.unique(data[:, i]))

    dataNew = m.sample(unique_vals, data.shape[0])
    y_pred = []
    for i in tqdm(range(len(train_X))):
        y_pred.append(execute_query(dataNew, train_X[i]))

    Q_err = []
    for i in range(train_Y.shape[0]):
        if train_Y[i] == 0 and y_pred[i] == 0:
            Q_err.append(1)
        elif train_Y[i] == 0:
            Q_err.append(y_pred[i])
        elif y_pred[i] == 0:
            Q_err.append(train_Y[i])
        else:
            Q_err.append(max(train_Y[i], y_pred[i]) / min(train_Y[i], y_pred[i]))
    print(
        np.median(Q_err),
        np.percentile(Q_err, 90),
        np.percentile(Q_err, 95),
        np.percentile(Q_err, 99),
        np.percentile(Q_err, 100),
    )

    # ip.embed()

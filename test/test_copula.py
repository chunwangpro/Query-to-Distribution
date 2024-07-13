import warnings

import numpy as np

from models import *
from query_func import *

np.random.seed(42)
warnings.filterwarnings("ignore")


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
            grad = tf.gradients(grad, x[i])
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

    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            y_pred, lattice_inputs, lattice_grad = self(x, training=True)
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
    def __init__(self, name, dim, pwl_keypoints=None, lattice_size=2):
        super().__init__(dim=dim, lattice_size=lattice_size)
        self.model_path = "./models/Lattice/model/" + name
        self.weight_path = "./models/Lattice/weight/" + name
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

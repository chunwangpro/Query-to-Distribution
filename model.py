import tensorflow as tf
import tensorflow_lattice as tfl
from tensorflow.keras import layers

from preprocessing import *


class ModelTypeError(ValueError):
    def __init__(self, message="Invalid model type. Please use '1-input' or '2-input'."):
        super().__init__(message)


def PWL(input_keypoints, col_idx, PWL_idx, monotonicity, suffix=""):
    # single PieceWise Linear Calibration Layer
    return tfl.layers.PWLCalibration(
        input_keypoints=np.array(input_keypoints),
        units=1,  # output dimension
        dtype=tf.float32,
        output_min=0.0,
        output_max=1.0,
        monotonicity=monotonicity,
        # "increasing" or "decreasing" or "none"
        kernel_initializer="equal_heights",
        # "equal_heights" or "equal_slopes"
        kernel_regularizer=None,
        # ("laplacian", 0.5, 0.5),  # ("laplacian", l1, l2)
        num_projection_iterations=8,
        input_keypoints_type="fixed",
        # "fixed" or "learned_interior"
        name=f"col_{col_idx}{suffix}_PWL_{PWL_idx}",
    )


def column_PWL(input_layer, pwl_keypoints, dim_idx, pwl_n, model_type, activation=False):
    # Assemble multiple PWL layers and optional activation layers for single column.
    layer_input = tf.keras.layers.Lambda(
        lambda x: tf.expand_dims(x[:, dim_idx], axis=-1), name=f"lambda_dim_{dim_idx}"
    )(input_layer)
    x = layer_input

    if model_type == "1-input":
        monotonicity = "increasing"
        suffix = ""
        col_idx = dim_idx
    elif model_type == "2-input":
        monotonicity = "increasing" if dim_idx % 2 else "decreasing"
        suffix = "_sup" if dim_idx % 2 else "_inf"
        col_idx = dim_idx // 2
    else:
        raise ModelTypeError()

    for j in range(pwl_n):
        if j == 0:
            # first PWL layer
            keypoints = pwl_keypoints[col_idx]
        else:
            # other PWL layers if pwl_n > 1
            keypoints = np.linspace(0, 1, num=len(pwl_keypoints[col_idx]))
        x = PWL(keypoints, col_idx, j, monotonicity, suffix)(x)
        if activation and j < pwl_n - 1:
            # add activation after PWL layer except the last PWL layer, if activation is True
            x = tf.keras.layers.Activation("tanh", name=f"col_{col_idx}{suffix}_tanh_{j}")(x)
    return x


def Lattice(lattice_size, monotonicities, col_idx, interpolation="simplex"):
    return tfl.layers.Lattice(
        lattice_sizes=lattice_size,
        units=1,  # output dimension
        interpolation=interpolation,
        # "simplex", "hypercube"
        monotonicities=monotonicities,
        # "increasing", "none"
        output_min=0.0,
        output_max=1.0,
        num_projection_iterations=10,
        monotonic_at_every_step=True,
        name=f"lattice_col_{col_idx}",
    )


class BaseModel:
    # Basic Trainer Class
    def __init__(
        self,
        args,
        path,
        table_size,
        unique_intervals,
        pwl_keypoints=None,
    ):
        self.name = "BaseModel"
        self.args = args
        self.path = path

        self.n_row, self.n_column = table_size
        self.dim = self.n_column
        self.unique_intervals = unique_intervals
        self.col_lattice_size = 2 if args.lattice_size == 0 else args.lattice_size

        self.pwl_keypoints = unique_intervals if pwl_keypoints is None else pwl_keypoints
        # also can be unique values of each column distribution
        self.pwl_n = args.pwl_n
        self.pwl_tanh = args.pwl_tanh

    def show_all_attributes(self):
        for key, value in self.__dict__.items():
            print(f"{key}: {value}")

    def build_model(self, use_CDF="lattice"):
        self.model_inputs = tf.keras.layers.Input(shape=[self.dim], name="input_layer")

        self.All_Column_PWLs = [
            column_PWL(
                self.model_inputs, self.pwl_keypoints, i, self.pwl_n, self.args.model, self.pwl_tanh
            )
            for i in range(self.dim)
        ]

        if self.args.model == "1-input":
            self.column_cdf = self.All_Column_PWLs
        elif self.args.model == "2-input":
            self.column_cdf = [
                Lattice(
                    lattice_size=[self.col_lattice_size] * 2,
                    monotonicities=["increasing"] * 2,
                    col_idx=i // 2,
                )([self.All_Column_PWLs[i], self.All_Column_PWLs[i + 1]])
                for i in range(0, self.dim, 2)
            ]  # 2-input lattice layers for each column
        else:
            raise ModelTypeError()

        # here use lattice layer as the last layer to learn the joint CDF
        # can be replaced by other layers, e.g., AutoRegressive Model
        if use_CDF == "lattice":
            last_lattice_size = (
                self.args.lattice_size
                if self.args.last_lattice_size == 0
                else self.args.last_lattice_size
            )
            self.Joint_CDF = Lattice(
                lattice_size=[last_lattice_size] * self.n_column,
                monotonicities=["increasing"] * self.n_column,
                col_idx="Joint-CDF",
            )(self.column_cdf)
        elif use_CDF == "res":
            concatenated_output = tf.keras.layers.Concatenate(axis=-1, name="concat_cdf")(
                self.column_cdf
            )
            self.Joint_CDF = ResNetCDFLayer(
                input_layer=concatenated_output, num_residual_blocks=4, hidden_dim=64, output_dim=1
            )

        if self.args.use_last_pwl:
            input_keypoints = np.linspace(0, 1, num=1000)
            self.model_output = PWL(input_keypoints, "last", "last", "increasing")(self.Joint_CDF)
        else:
            self.model_output = self.Joint_CDF

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


def ResNetCDFLayer(input_layer, num_residual_blocks=4, hidden_dim=10, output_dim=1):
    """
    A ResNet-style layer with residual connections.

    Parameters
    ----------
    input_layer : tf.Tensor
        Input tensor for the layer.
    num_residual_blocks : int
        Number of residual blocks to stack.
    hidden_dim : int
        Number of hidden units in each Dense layer.
    output_dim : int
        Number of output units (e.g., 1 for a CDF value).

    Returns
    -------
    tf.Tensor
        Output tensor with shape `(batch_size, output_dim)`.
    """

    def residual_block(x, hidden_dim):
        residual = x
        x = layers.Dense(hidden_dim, activation="relu")(x)
        x = layers.Dense(hidden_dim, activation=None)(x)  # No activation in the second layer
        x = layers.Add()([x, residual])  # Add skip connection
        x = layers.Activation("relu")(x)  # Apply activation after addition
        return x

    # Initial dense layer
    x = layers.Dense(hidden_dim, activation="relu")(input_layer)

    # Stack residual blocks
    for _ in range(num_residual_blocks):
        x = residual_block(x, hidden_dim)

    # Final dense layer for output
    output = layers.Dense(output_dim, activation="sigmoid")(x)
    return output

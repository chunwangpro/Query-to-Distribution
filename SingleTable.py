# rewrite by numpy to achieve faster speed
# PWL - Lattice model is an 1-input model
# PWL-Lattice-Copula model is an 2-input model

# Query Phase:
## calculate_query_cardinality: numpy version

# Generation Phase:
## generate_by_row / generate_row_batch_table / np.concatenate

# no Plottings
import warnings

warnings.filterwarnings("ignore")
import argparse
import os

import numpy as np
from tqdm import tqdm

from models import *
from query_func import *

np.random.seed(42)

# class LatticeCDFLayer(tf.keras.Model):
#     def __init__(self, dim, lattice_size=2):
#         super().__init__()
#         self.dim = dim
#         self.lattice_size = lattice_size

#         self.copula_lattice = tfl.layers.Lattice(
#             lattice_sizes=[self.lattice_size] * self.dim,
#             interpolation="hypercube",  # simplex
#             monotonicities=["increasing"] * self.dim,
#             output_min=0.0,
#             output_max=1.0,
#             name="lattice",
#         )

#     def call(self, x):
#         y = self.copula_lattice(x)
#         grad = y
#         for i in range(self.dim):
#             grad = tf.gradients(grad, x[i])  # , stop_gradients=[a, b])
#         return y, x, grad


# class CopulaModel(LatticeCDFLayer):
#     def __init__(self, dim, lattice_size=2, pwl_keypoints=None):
#         super().__init__(dim=dim, lattice_size=lattice_size)
#         self.pwl_calibration_input_keypoints = (
#             unique_intervals if pwl_keypoints is None else pwl_keypoints
#         )

#         self.model_inputs = []
#         for i in range(self.dim):
#             self.model_inputs.append(tf.keras.layers.Input(shape=[1], name="col_%s_inf" % i))
#             self.model_inputs.append(tf.keras.layers.Input(shape=[1], name="col_%s_sup" % i))

#         self.calibrators = []
#         for i in range(self.dim):
#             self.calibrators.append(
#                 tfl.layers.PWLCalibration(
#                     input_keypoints=np.array(self.pwl_calibration_input_keypoints[i]),
#                     dtype=tf.float32,
#                     output_min=0.0,
#                     output_max=1.0,
#                     clamp_min=True,
#                     clamp_max=True,
#                     monotonicity="decreasing",
#                     name="col_%s_inf_pwl" % i,
#                 )(self.model_inputs[2 * i])
#             )
#             self.calibrators.append(
#                 tfl.layers.PWLCalibration(
#                     input_keypoints=np.array(self.pwl_calibration_input_keypoints[i]),
#                     # input_keypoints=np.linspace(
#                     #     feat_mins[i],
#                     #     feat_maxs[i],
#                     #     num=pwl_calibration_num_keypoints),
#                     dtype=tf.float32,
#                     output_min=0.0,
#                     output_max=1.0,
#                     clamp_min=True,
#                     clamp_max=True,
#                     monotonicity="increasing",
#                     name="col_%s_sup_pwl" % i,
#                 )(self.model_inputs[2 * i + 1])
#             )

#         self.lattice_cdf = []
#         for i in range(self.dim):
#             self.lattice_cdf.append(
#                 tfl.layers.Lattice(
#                     lattice_sizes=[lattice_size] * 2,
#                     interpolation="hypercube",  # simplex
#                     monotonicities=["increasing"] * 2,
#                     output_min=0.0,
#                     output_max=1.0,
#                     name="lattice_col_%s" % i,
#                 )([self.calibrators[2 * i], self.calibrators[2 * i + 1]])
#             )

#         self.model = tf.keras.models.Model(
#             inputs=self.model_inputs, outputs=self.copula_lattice(self.lattice_cdf)
#         )

#     def train_step(self, data):
#         x, y = data
#         with tf.GradientTape() as tape:
#             y_pred, lattice_inputs, lattice_grad = self(x, training=True)  # , training=True)
#             loss1 = self.compiled_loss(y, y_pred)
#             loss2 = 100000  # min(x)
#             loss3 = 1  # max(sum(x)-self.dim+1, 0)
#             loss4 = 1
#             loss = loss1 + loss2 + loss3 + loss4
#         trainable_vars = self.trainable_variables
#         gradients = tape.gradient(loss, trainable_vars)
#         self.optimizer.apply_gradients(zip(gradients, trainable_vars))
#         self.compiled_metrics.update_state(y, y_pred)
#         return {m.name: m.result() for m in self.metrics}


# # loss 可以在complie里传入，但是不能通过重写fit方法来载入


# class Trainer_Lattice(CopulaModel):
#     def __init__(self, name, dim, pwl_keypoints=None, lattice_size=2):
#         super().__init__(dim=dim, lattice_size=lattice_size)
#         self.model_path = "./models/Lattice/model/" + name
#         self.weight_path = "./models/Lattice/weight/" + name
#         self.model.summary()

#     def fit(
#         self,
#         X,
#         y,
#         lr=0.01,
#         bs=1000,
#         epochs=1000,
#         reduceLR_factor=0.1,
#         reduceLR_patience=10,
#         ESt_patience=20,
#         verbose=1,
#         loss="MSE",
#         opt="Adam",
#     ):

#         X = X.astype(np.float32)
#         y = y.astype(np.float32)

#         features = [X[:, i] for i in range(X.shape[1])]
#         target = y

#         Loss = {
#             "MAE": tf.keras.losses.mean_absolute_error,
#             "MSE": tf.keras.losses.mean_squared_error,
#             "MAPE": tf.keras.losses.mean_absolute_percentage_error,
#         }

#         Opt = {
#             "Adam": tf.keras.optimizers.Adam(lr),
#             "Nadam": tf.keras.optimizers.Nadam(),
#             "Adagrad": tf.keras.optimizers.Adagrad(),
#             "Adadelta": tf.keras.optimizers.Adadelta(),
#             "Adamax": tf.keras.optimizers.Adamax(),
#             "RMSprop": tf.keras.optimizers.RMSprop(),
#         }

#         self.model.compile(loss=Loss[loss], optimizer=Opt[opt])

#         earlyStopping = tf.keras.callbacks.EarlyStopping(
#             restore_best_weights=True,
#             monitor="loss",
#             mode="min",
#             patience=ESt_patience,
#             verbose=verbose,
#         )
#         mcp_save = tf.keras.callbacks.ModelCheckpoint(
#             filepath=f"{self.weight_path}.h5",
#             save_weights_only=True,
#             save_best_only=True,
#             monitor="loss",
#             mode="min",
#             verbose=verbose,
#         )
#         reduce_lr_loss = tf.keras.callbacks.ReduceLROnPlateau(
#             monitor="loss",
#             factor=reduceLR_factor,
#             patience=reduceLR_patience,
#             verbose=verbose,
#             min_delta=1e-8,
#             mode="min",
#         )

#         self.model.fit(
#             features,
#             target,
#             epochs=epochs,
#             batch_size=bs,
#             verbose=1,
#             callbacks=[earlyStopping, mcp_save, reduce_lr_loss],
#         )

#     def load(self):
#         self.model = tf.keras.models.load_model("%s.hdf5" % self.model_path)
#         self.model.load_weights("%s.hdf5" % self.weight_path)

#     def inference(self, grid):
#         # assert grid.shape[1] == self.dim * 2
#         pred = self.model.predict(np.hsplit(grid, self.dim * 2))
#         return pred


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="2-input", help="model type")
parser.add_argument("--dataset", type=str, default="wine3", help="Dataset.")
parser.add_argument("--query-size", type=int, default=10000, help="query size")
parser.add_argument("--min-conditions", type=int, default=1, help="min num of query conditions")
parser.add_argument("--max-conditions", type=int, default=3, help="max num of query conditions")
parser.add_argument("--lattice-size", type=int, default=2, help="Lattice size for each column.")
parser.add_argument(
    "--last-lattice-size", type=int, default=2, help="Lattice size for Joint CDF model."
)
parser.add_argument(
    "--boundary", type=bool, default=False, help="whether add boundary point to train set."
)
parser.add_argument(
    "--unique-train", type=bool, default=False, help="whether make train set unique."
)
parser.add_argument("--pwl-n", type=int, default=1, help="Number of PWL layer for each column.")
parser.add_argument(
    "--pwl-tanh", type=bool, default=False, help="whether add tanh activation layer after pwl."
)
parser.add_argument("--epochs", type=int, default=2000, help="Number of train epochs.")
parser.add_argument("--bs", type=int, default=1000, help="Batch size.")
parser.add_argument("--loss", type=str, default="MSE", help="Loss.")
parser.add_argument("--opt", type=str, default="adamax", help="Optimizer.")
parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")


try:
    args = parser.parse_args()
except:
    # args = parser.parse_args([])
    args, unknown = parser.parse_known_args()


def make_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


FilePath = (
    f"{args.dataset}_{args.query_size}_{args.min_conditions}_{args.max_conditions}_{args.loss}"
)
resultsPath = f"results/{FilePath}"
modelPath = f"models/{FilePath}"
make_directory(resultsPath)
make_directory(modelPath)


OPS = {
    ">": np.greater,
    "<": np.less,
    ">=": np.greater_equal,
    "<=": np.less_equal,
    "=": np.equal,
}


print("\nBegin Loading Data ...")
table = np.loadtxt(f"datasets/{args.dataset}.csv", delimiter=",")
table = table.reshape(len(table), -1)
table_size = table.shape[0], table.shape[1]
np.savetxt(f"{resultsPath}/original_table.csv", table, delimiter=",")
print(f"{args.dataset}.csv,    shape: {table_size}")
print("Done.\n")


print("Begin Generating Queries Set ...")
rng = np.random.RandomState(42)
query_set = [generate_random_query(table, args, rng) for _ in tqdm(range(args.query_size))]
print("Done.\n")


print("Begin Intervalization ...")
unique_intervals = column_intervalization(query_set, table_size)
column_interval_number = count_column_unique_interval(unique_intervals)
print(f"{column_interval_number=}")
print("Done.\n")


print("Begin Building Train set and Model ...")
X, y, m, values = setup_train_set_and_model(
    args, query_set, unique_intervals, modelPath, table_size
)
# m.show_all_attributes()


# here use lattice layer as the last layer to learn the joint CDF
# can be replaced by other layers, e.g., AutoRegressive Model
last_lattice_size = args.lattice_size if args.last_lattice_size == 0 else args.last_lattice_size

JointCDFodel = Lattice(
    lattice_size=[last_lattice_size] * table_size[1],
    monotonicities=["increasing"] * table_size[1],
    col_idx="Joint-CDF",
)

m.build_model(JointCDFodel)
print("Done.\n")


m.fit(X, y, args)
m.load()


# Full-Factorial net of unique intervals
#     values = [v for v in unique_intervals.values()]
#     mesh = np.meshgrid(*values)
#     grid = np.array(mesh).T.reshape(-1, len(values)).astype(np.float32)

# Latin Hypercube sampling
#     lb = np.array([v[1] for v in unique_intervals.values()])
#     ub = np.array([v[-1] for v in unique_intervals.values()])
#     lhs_sample = lhs(n_column, samples=10000, criterion='center')
#     sample_df = pd.DataFrame(lb + (ub-lb)*lhs_sample, columns=[f'col_{i}' for i in range(n_column)])
#     grid = np.array(sample_df.sort_values(by=list(sample_df.columns)))

# Random sampling
# lb = np.array([1] * n_column)
# ub = np.array(column_interval_number) - 1
# lhs_sample = lb + (ub - lb) * lhs(n_column, samples=lhs_n, criterion="center")
# index = np.round(lhs_sample).astype(int)
# grid_mesh = np.empty_like(index, dtype=np.float32)
# for i in range(lhs_n):
#     idx = index[i, :]
#     grid_mesh[i] = [unique_intervals[j][idx[j]] for j in range(n_column)]
# sample_df = pd.DataFrame(grid_mesh, columns=[f"col_{i}" for i in range(n_column)])
# grid_a = np.array(sample_df.sort_values(by=list(sample_df.columns)))
# greatest = np.array([v[-1] for v in unique_intervals.values()]).reshape(1, -1)
# grid = np.concatenate([grid_a, greatest], axis=0)


Table_Generated = m.generate_table_by_row(values, batch_size=10000)
np.savetxt(f"{resultsPath}/generated_table.csv", Table_Generated, delimiter=",")


Q_error = calculate_Q_error(Table_Generated, query_set)
print_Q_error(Q_error, args, resultsPath)
print(f"\n Original table shape : {table_size}")
print(f"Generated table shape : {Table_Generated.shape}")

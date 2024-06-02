import time
import common
import argparse
import datasets
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from matplotlib import pyplot as plt
import estimators as estimators_lib
import tensorflow as tf
import tensorflow_lattice as tfl


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
    num_filters = rng.randint(max_num_filters - 1, max_num_filters)
    cols, idxs, ops, vals = SampleTupleThenRandom(table, num_filters, rng,
                                                  dataset)
    sel = cal_true_card(
        (cols, idxs, ops, vals), table) / float(table.cardinality)
    return cols, idxs, ops, vals, sel


def SampleTupleThenRandom(table, num_filters, rng, dataset):
    idxs = rng.choice(len(table.columns), replace=False, size=num_filters)
    cols = np.take(table.columns, idxs)

    # If dom size >= 10, okay to place a range filter.
    # Otherwise, low domain size columns should be queried with equality.
    # ops = rng.choice(['<=', '>=', '='], size=num_filters)
    # ops = rng.choice(['<=', '>'], size=num_filters)
    ops = rng.choice(['<='], size=num_filters)

    #     ops_all_eqs = ['='] * num_filters
    #     sensible_to_do_range = [c.DistributionSize() >= 10 for c in cols]
    #     ops = np.where(sensible_to_do_range, ops, ops_all_eqs)

    # if num_filters == len(table.columns):
    #     return table.columns,np.arange(len(table.columns)), ops, vals

    vals = []
    for i in range(num_filters):
        s = table.data.iloc[rng.randint(0, table.data.shape[0])]
        vals.append(s.values[idxs][i])

    vals = np.array(vals)

    op_a = []
    val_a = []
    for i in range(len(vals)):
        val_a.append([vals[i]])
        op_a.append([ops[i]])

    return cols, idxs, pd.DataFrame(op_a).values, pd.DataFrame(val_a).values


def dictionary_column_interval(table_size, query_set):
    # Traverse all queries to apply the intervalization skill for each column
    n_column = table_size[1]
    column_interval = {}
    for i in range(n_column):
        column_interval[i] = set(
        )  # use set([0, sys.maxsize]) to adapt '>' and '<'.
    for query in query_set:
        _, col_idxs, _, vals, _ = query
        for i in range(len(col_idxs)):
            column_interval[col_idxs[i]].add(vals[i][0])
    for k, v in column_interval.items():
        if not v:
            column_interval[k] = [
                0
            ]  # use [0] to represent empty column interval
        else:
            column_interval[k] = sorted(list(v))
    return column_interval


def count_column_unique_interval(unique_intervals):
    # count unique query interval in each column
    return [len(v) for v in unique_intervals.values()]


def execute_query(dataNew, query_set):
    diff = []
    for query in tqdm(query_set):
        sentence = ''
        for i in range(len(query[0])):
            if i != 0:
                sentence += ' and '
            sentence += f'col_{query[1][i]}'
            if query[2][i][0] == '=':
                sentence += '=='
            else:
                sentence += query[2][i][0]
            sentence += f'{query[3][i][0]}'
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


def print_error(diff, name):
    print(f"\n\n Q-error of Lattice ({name}):\n")
    print(f"min:    {np.min(diff)}")
    print(f"10:     {np.percentile(diff, 10)}")
    print(f"20:     {np.percentile(diff, 20)}")
    print(f"30:     {np.percentile(diff, 30)}")
    print(f"40:     {np.percentile(diff, 40)}")
    print(f"median: {np.median(diff)}")
    print(f"60:     {np.percentile(diff, 60)}")
    print(f"70:     {np.percentile(diff, 70)}")
    print(f"80:     {np.percentile(diff, 80)}")
    print(f"90:     {np.percentile(diff, 90)}")
    print(f"95:     {np.percentile(diff, 95)}")
    print(f"max:    {np.max(diff)}")
    print(f"mean:   {np.mean(diff)}")


class LatticeCDF:
    def __init__(
            self,
            name,
            n_row,
            unique_intervals,
            pwl_keypoints=None,  # also can input table unique values
            lattice_size=2,
            l2=1e-6):

        self.l2 = l2
        self.model_path = './models/Lattice/model/' + name
        self.weight_path = './models/Lattice/weight/' + name
        self.n_row = n_row
        self.dim = len(unique_intervals.keys())
        self.lattice_size = lattice_size
        self.unique_intervals = unique_intervals
        self.pwl_calibration_input_keypoints = unique_intervals if pwl_keypoints is None else pwl_keypoints

        self.model_inputs = []
        for i in range(self.dim):
            self.model_inputs.append(
                tf.keras.layers.Input(shape=[1], name='col_%s' % i))
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
                    input_keypoints=np.array(
                        self.pwl_calibration_input_keypoints[i]),
                    # input_keypoints=np.linspace(
                    #     feat_mins[i],
                    #     feat_maxs[i],
                    #     num=pwl_calibration_num_keypoints),
                    dtype=tf.float32,
                    output_min=0.0,
                    output_max=1.0,
                    monotonicity='increasing',
                ))

        self.lattice = tfl.layers.Lattice(
            lattice_sizes=[lattice_size] * self.dim,  # (self.dim * 2),
            # lattice_sizes=[len(v) for v in self.unique_intervals.values()],
            interpolation='simplex',  # 可以尝试别的插值类型
            monotonicities=['increasing'] * self.dim,  # (self.dim * 2),
            output_min=0,  # 1 / self.n_row,
            output_max=1.0,
            name='lattice',
        )

        # self.output1 = tfl.layers.PWLCalibration(
        #     input_keypoints=np.linspace(0.0,
        #                                 np.log(n_row),
        #                                 num=pwl_calibration_num_keypoints),
        #     dtype=tf.float32,
        #     output_min=0.0,
        #     output_max=np.log(n_row),
        #     name='output1_calib',
        # )

        # self.output2 = tfl.layers.PWLCalibration(
        #     input_keypoints=np.linspace(0.0,
        #                                 n_row,
        #                                 num=pwl_calibration_num_keypoints),
        #     dtype=tf.float32,
        #     output_min=0.0,
        #     output_max=n_row,
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
            self.lattice_inputs.append(self.calibrators[i](
                self.model_inputs[i]))
        self.model_output = self.lattice(self.lattice_inputs)

        self.model = tf.keras.models.Model(
            inputs=self.model_inputs,
            outputs=self.model_output,
        )
        self.model.save('%s.hdf5' % self.model_path)
        self.model.summary()

    def fit(self,
            X,
            y,
            lr=0.01,
            bs=16,
            epochs=3000,
            reduceLR_factor=0.5,
            reduceLR_patience=20,
            verbose=1,
            loss='MSE',
            opt='Adam'):
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

        Loss = {
            'MAE': tf.keras.losses.mean_absolute_error,
            'MSE': tf.keras.losses.mean_squared_error,
            'MAPE': tf.keras.losses.mean_absolute_percentage_error
        }

        Opt = {
            'Adam': tf.keras.optimizers.Adam(),
            'Nadam': tf.keras.optimizers.Nadam(),
            'Adagrad': tf.keras.optimizers.Adagrad(),
            'Adadelta': tf.keras.optimizers.Adadelta(),
            'Adamax': tf.keras.optimizers.Adamax(),
            'RMSprop': tf.keras.optimizers.RMSprop(),
        }
        self.model.compile(loss=Loss[loss], optimizer=Opt[opt])
        # self.model.compile(loss=Loss[loss], optimizer=tf.keras.optimizers.Adamax(lr))

        earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                         patience=500,
                                                         verbose=verbose,
                                                         mode='min')
        mcp_save = tf.keras.callbacks.ModelCheckpoint('%s.hdf5' %
                                                      self.weight_path,
                                                      save_best_only=True,
                                                      monitor='loss',
                                                      mode='min',
                                                      save_weights_only=True)
        reduce_lr_loss = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='loss',
            factor=reduceLR_factor,
            patience=reduceLR_patience,
            verbose=verbose,
            epsilon=1e-10,
            mode='min')

        self.model.fit(features,
                       target,
                       epochs=epochs,
                       batch_size=bs,
                       verbose=1,
                       callbacks=[earlyStopping, mcp_save, reduce_lr_loss])
        self.model.load_weights('%s.hdf5' % self.weight_path)

    def load(self):
        self.model = tf.keras.models.load_model('%s.hdf5' % self.model_path)
        self.model.load_weights('%s.hdf5' % self.weight_path)

    def inference(self, grid):
        # predict and generate table
        assert grid.shape[1] == self.dim
        # df_grid = pd.DataFrame(cart_2, columns=[f'col_{i}' for i in range(n_column)])
        pred = m.model.predict(np.hsplit(grid, self.dim))
        return pred

    def plotting(self, grid, pred):
        Xshape = len(np.unique(grid[:, 0]))
        Yshape = len(np.unique(grid[:, 1]))
        X = grid[:, 1].reshape(Xshape, Yshape)
        Y = grid[:, 0].reshape(Xshape, Yshape)
        Z = pred.reshape(Xshape, Yshape)

        fig0 = plt.figure(figsize=(20, 8))
        ax0 = fig0.add_subplot(111)
        ax0.plot(pred, 'bo')
        plt.show()

        fig1 = plt.figure(figsize=(15, 8))
        ax1 = plt.axes(projection='3d')
        ax1.plot_surface(X, Y, Z, cmap='viridis')
        plt.show()

        fig2 = plt.figure(figsize=(10, 8))
        ax2 = fig2.add_subplot(111)
        cs = ax2.contourf(X, Y, Z, cmap='viridis')
        # Alternatively, you can manually set the levels
        # and the norm:
        # lev_exp = np.arange(np.floor(np.log10(z.min())-1),
        #                    np.ceil(np.log10(z.max())+1))
        # levs = np.power(10, lev_exp)
        # cs = ax.contourf(X, Y, z, levs, norm=colors.LogNorm())    # 这个是啥
        cbar = fig2.colorbar(cs)  # 让colorbar细粒度更高一点
        plt.show()

        # query 对网格的覆盖率 散点图
        fig3 = plt.figure(figsize=(10, 10))
        xtick = unique_intervals[0]
        ytick = unique_intervals[1]
        plt.scatter(train_X[:, 0], train_X[:, 1], c='b')
        plt.vlines(xtick, min(ytick), max(ytick), colors="green")
        plt.hlines(ytick, min(xtick), max(xtick), colors="green")
        plt.show()

    def generate(self, grid):
        assert grid.shape[1] == self.dim
        pred = m.model.predict(np.hsplit(grid, self.dim))
        dataNew = pd.DataFrame(
            columns=[f'col_{i}' for i in range(n_column)],
            index=[i for i in range(np.prod(column_interval_number))])

        # generate 一条再query一条，再generate下一条
        count = 0
        for i in trange(grid.shape[0]):
            sentence = ''
            for j in range(grid.shape[1]):
                if j != 0:
                    sentence += ' and '
                sentence += f'col_{j}'  # 这里要改
                sentence += " <= "  # 这里也要改
                sentence += f'{grid[i][j]:f}'

            xi = pred[i][0] * n_row - dataNew.query(sentence).shape[0]

            if int(xi) > 0:
                floor = np.floor(xi)
                ceil = np.ceil(xi)
                if floor == ceil:
                    card = int(xi)
                else:
                    card = int(xi)


#                     card = np.random.choice([floor, ceil],
#                                             p=[xi - floor,
#                                                ceil - xi]).astype("int")
                dataNew.iloc[count:count + card, :] = grid[i]
                count += card
        dataNew.dropna(axis=0, how='all', inplace=True)
        return dataNew


def calc_time(tic, toc):
    total_time = toc - tic
    m, s = divmod(total_time, 60)
    h, m = divmod(m, 60)
    return f"{h:0>2.0f}:{m:0>2.0f}:{s:0>2.0f}"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        type=str,
                        default='wine2',
                        help='Dataset.')
    parser.add_argument('--loss', type=str, default='MSE', help='Loss.')
    parser.add_argument('--opt', type=str, default='Adam', help='Optimizer.')
    parser.add_argument('--query-size',
                        type=int,
                        default=10000,
                        help='query size')
    parser.add_argument('--num-conditions',
                        type=int,
                        default=1,
                        help='num of conditions')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--bs', type=int, default=1000, help='Batch size.')
    parser.add_argument('--lattice', type=int, default=2, help='Lattice size.')
    parser.add_argument('--seed', type=int, default=4321, help='Random seed')
    parser.add_argument('--sample',
                        type=int,
                        default=0,
                        help='reload trained mode')
    parser.add_argument('--epochs',
                        type=int,
                        default=10000,
                        help='Number of epochs to train for.')
    # args = parser.parse_args()
    args, unknown = parser.parse_known_args()
    bs = int(args.bs)
    lr = float(args.lr)
    epochs = int(args.epochs)
    lattice = int(args.lattice)
    sample = int(args.sample)

    if args.dataset == 'dmv-tiny':
        table = datasets.LoadDmv('dmv-tiny.csv')
    elif args.dataset == 'dmv':
        table = datasets.LoadDmv()
    else:
        type_casts = {}
        if args.dataset in ['orders1']:
            type_casts = {4: np.datetime64, 5: np.float}
        elif args.dataset in ['orders']:
            type_casts = {4: np.datetime64}
        elif args.dataset == 'lineitem':
            type_casts = {
                10: np.datetime64,
                11: np.datetime64,
                12: np.datetime64
            }
        elif args.dataset == 'order_line':
            type_casts = {6: np.datetime64}

        table = datasets.LoadDataset(args.dataset + '.csv',
                                     args.dataset,
                                     type_casts=type_casts)
    table_train = table
    train_data = common.TableDataset(table_train)
    query_set = None

    print('Begin Generating Queries ...')
    time0 = time.time()
    rng = np.random.RandomState(args.seed)
    query_set = [
        GenerateQuery(table, 2, args.num_conditions + 1, rng, args.dataset)
        for i in trange(args.query_size)
    ]
    print('Complete Generating Queries.')

    # Lattice
    print("\n\nBuilding Lattice...")
    time1 = time.time()
    table_size = table.data.shape
    n_row = table_size[0]
    n_column = table_size[1]
    unique_intervals = dictionary_column_interval(table_size, query_set)
    column_interval_number = count_column_unique_interval(unique_intervals)
    train_X = []
    train_Y = []
    for query in query_set:
        x = [unique_intervals[i][-1] + 1
             for i in range(n_column)]  # 这里使用每个col_unique_interval的最后一个元素即可
        _, col_idxs, _, vals, sel = query
        for i in range(len(col_idxs)):
            x[col_idxs[i]] = vals[i][0]
        train_X.append(x)
        train_Y.append(sel)
    train_X = np.array(train_X).astype(np.float32)
    train_Y = np.array(train_Y).astype(np.float32).reshape(-1, 1)
    train = np.concatenate((train_X, train_Y), axis=1)
    train = np.unique(train, axis=0)
    train_X, train_Y = np.hsplit(train, [-1])
    name = f"{args.dataset}_{args.query_size}query_{args.num_conditions}column_{args.epochs}epoch"
    m = LatticeCDF(name, n_row, unique_intervals, pwl_keypoints=None)

    print("\n\nLattice is already built, begin training...\n")
    time2 = time.time()
    m.fit(train_X,
          train_Y,
          lr=lr,
          bs=bs,
          epochs=epochs,
          loss=args.loss,
          opt=args.opt)

    print("\nFinish training, begin generate table...")
    time3 = time.time()
    values = [v for v in unique_intervals.values()]
    mesh = np.meshgrid(*values)  # 所有unique interval 的笛卡尔积网格
    grid = np.array(mesh).T.reshape(-1, len(values)).astype(np.float32)
    dataNew = m.generate(grid)

    print("\nFinish generate table, begin calculate Q-error on new table...")
    time4 = time.time()
    diff = execute_query(dataNew, query_set)
    print_error(diff, name)

    print(f"\noriginal table shape: {table_size}")
    print(f"  Our table shape   : {dataNew.shape}")
    time5 = time.time()

    print("\nTime passed:")
    print(" Generate Query  :  ", calc_time(time0, time1))
    print(" Build  Lattice  :  ", calc_time(time1, time2))
    print("   Training      :  ", calc_time(time2, time3))
    print("Generate  Table  :  ", calc_time(time3, time4))
    print("Calculate Q-error:  ", calc_time(time4, time5))

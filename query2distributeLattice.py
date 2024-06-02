# import pdb

import common
import argparse
import datasets
import numpy as np
import pandas as pd
import IPython as ip

import estimators as estimators_lib

from LatticeCDF import LatticeCDF, execute_query


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
    # print (235)
    cols, idxs, ops, vals = SampleTupleThenRandom(table, num_filters, rng,
                                                  dataset)
    # print (vals)
    sel = cal_true_card(
        (cols, idxs, ops, vals), table) / float(table.cardinality)
    return cols, idxs, ops, vals, sel


def SampleTupleThenRandom(table, num_filters, rng, dataset):
    vals = []
    # new_table = table.data.dropna(axis=0,how='any')
    new_table = table.data
    # print (248)
    s = new_table.iloc[rng.randint(0, new_table.shape[0])]
    vals = s.values
    # print (251)
    if dataset in ['dmv', 'dmv-tiny', 'order_line']:
        vals[6] = vals[6].to_datetime64()
    elif dataset in ['orders1', 'orders']:
        vals[4] = vals[4].to_datetime64()
    elif dataset == 'lineitem':
        vals[10] = vals[10].to_datetime64()
        vals[11] = vals[11].to_datetime64()
        vals[12] = vals[12].to_datetime64()
    # print (260)
    idxs = rng.choice(len(table.columns), replace=False, size=num_filters)
    # idxs = [12]
    cols = np.take(table.columns, idxs)
    # print (264)
    # print (cols)
    # If dom size >= 10, okay to place a range filter.
    # Otherwise, low domain size columns should be queried with equality.
    # ops = rng.choice(['<=', '>=', '='], size=num_filters)
    ops = rng.choice(['<=', '>='], size=num_filters)
    # ops = rng.choice(['<='], size=num_filters)
    ops_all_eqs = ['='] * num_filters
    sensible_to_do_range = [c.DistributionSize() >= 10 for c in cols]
    # print (271)
    ops = np.where(sensible_to_do_range, ops, ops_all_eqs)
    # print (273)
    # if num_filters == len(table.columns):
    #     return table.columns,np.arange(len(table.columns)), ops, vals
    # print (276)
    vals = vals[idxs]
    op_a = []
    val_a = []
    for i in range(len(vals)):
        val_a.append([vals[i]])
        op_a.append([ops[i]])
    # print (283)
    return cols, idxs, pd.DataFrame(op_a).values, pd.DataFrame(val_a).values


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        type=str,
                        default='wine3',
                        help='Dataset.')
    parser.add_argument('--loss', type=str, default='MSE', help='Loss.')
    parser.add_argument('--query-size',
                        type=int,
                        default=10000,
                        help='query size')
    parser.add_argument('--num-conditions',
                        type=int,
                        default=2,
                        help='num of conditions')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--bs', type=int, default=1024, help='Batch size.')
    parser.add_argument('--lattice', type=int, default=3, help='Lattice size.')
    parser.add_argument('--seed', type=int, default=4321, help='Random seed')
    parser.add_argument('--sample',
                        type=int,
                        default=0,
                        help='reload trained mode')
    parser.add_argument('--epochs',
                        type=int,
                        default=200,
                        help='Number of epochs to train for.')
    args = parser.parse_args()

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
    rng = np.random.RandomState(args.seed)
    query_set = [
        GenerateQuery(table, 2, args.num_conditions + 1, rng, args.dataset)
        for i in range(args.query_size)
    ]
    print('Complete Generating Queries ...')

    data = table.data.to_numpy()

    unique_vals = []
    for i in range(data.shape[1]):
        unique_vals.append(np.unique(data[:, i]))

    train_X = []
    train_Y = []

    for query in query_set:
        cols, idxs, ops, vals, sel = query
        x = []
        for i in range(data.shape[1]):
            if unique_vals[i][0] > 1:
                x.append(unique_vals[i][0] - 1)
            else:
                x.append(unique_vals[i][0] / 2)
            x.append(unique_vals[i][-1])
        for i in range(len(idxs)):
            if ops[i] == '<=':
                x[2 * idxs[i] + 1] = vals[i]
            if ops[i] == '>=':
                index = np.searchsorted(unique_vals[idxs[i]], vals[i])
                if index > 0:
                    x[2 * idxs[i]] = unique_vals[idxs[i]][index - 1]
                else:
                    if unique_vals[idxs[i]][0] > 1:
                        x[2 * idxs[i]] = unique_vals[idxs[i]][0] - 1
                    else:
                        x[2 * idxs[i]] = unique_vals[idxs[i]][0] / 2
            if ops[i] == '=':
                x[2 * idxs[i] + 1] = vals[i]
                index = np.searchsorted(unique_vals[idxs[i]], vals[i])
                if index > 0:
                    x[2 * idxs[i]] = unique_vals[idxs[i]][index - 1]
                else:
                    if unique_vals[idxs[i]][0] > 1:
                        x[2 * idxs[i]] = unique_vals[idxs[i]][0] - 1
                    else:
                        x[2 * idxs[i]] = unique_vals[idxs[i]][0] / 2
        train_X.append(x)
        train_Y.append(data.shape[0] * sel)

    train_X = np.array(train_X).astype(np.float32)
    train_Y = np.array(train_Y).astype(np.float32)

    train_X = train_X[train_Y != 0]
    train_Y = train_Y[train_Y != 0]

    train_col = []
    for i in range(data.shape[1]):
        train_col.append(train_X[:, i * 2:(i + 1) * 2])

    feat_mins = [x.min() for x in train_col]
    feat_maxs = [x.max() for x in train_col]
    m = LatticeCDF(args.dataset + '_' + args.loss, lattice, feat_mins,
                   feat_maxs, data.shape[0])
    if sample == 1:
        m.load(args.dataset)
        dataNew = m.sample(unique_vals, data.shape[0])

        # np.savetxt('datasets/%s_lattice.csv' % args.dataset, dataNew, delimiter=',')
        dataNew = np.loadtxt('datasets/%s_lattice.csv' % args.dataset,
                             delimiter=',')

        y_pred = []
        for x in train_X:
            y_pred.append(execute_query(dataNew, x))

        Q_err = []
        for i in range(train_Y.shape[0]):
            if train_Y[i] == 0 and y_pred[i] == 0:
                Q_err.append(1)
            elif train_Y[i] == 0:
                Q_err.append(y_pred[i])
            elif y_pred[i] == 0:
                Q_err.append(train_Y[i])
            else:
                Q_err.append(
                    max(train_Y[i], y_pred[i]) / min(train_Y[i], y_pred[i]))
        print(np.median(Q_err), np.percentile(Q_err, 90),
              np.percentile(Q_err, 95), np.percentile(Q_err, 99),
              np.percentile(Q_err, 100))

        ip.embed()
    else:
        m.fit(train_X, train_Y, lr=lr, bs=bs, epochs=epochs, loss=args.loss)
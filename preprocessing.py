import numpy as np


def build_train_set_1_input(query_set, unique_intervals, args, table_size):
    X = []
    for query in query_set:
        x = [v[-3] for v in unique_intervals.values()]
        idxs, _, vals, _ = query
        for i, v in zip(idxs, vals):
            x[i] = v
        X.append(x)
    X = np.array(X, dtype=np.float32)
    y = np.array([query[-1] for query in query_set], dtype=np.float32).reshape(-1, 1)
    y /= table_size[0]
    train = np.hstack((X, y))

    # make train set unique
    if args.unique_train:
        train = np.unique(train, axis=0)

    # add boundary
    if args.boundary:
        train = add_boundary_1_input(train, unique_intervals, args.boundary)

    # shuffle and split
    np.random.shuffle(train)
    X, y = np.hsplit(train, [-1])
    return X, y


def add_boundary_1_input(train, unique_intervals, alpha=0.1):
    # create boundary set
    min_x = [v[0] for v in unique_intervals.values()]
    max_x = [v[-3] for v in unique_intervals.values()]
    border_x = np.array([min_x, max_x])
    border_y = np.array([[0], [1]])
    border = np.hstack((border_x, border_y))

    # repeat boundary to raise weight
    k = int(train.shape[0] / border.shape[0] * alpha)
    repeated_border = np.tile(border, (k, 1))
    train = np.vstack((train, repeated_border))

    return train


def build_train_set_2_input(query_set, unique_intervals, args, table_size):

    def process_op_lt(x, idx, val):
        x[idx * 2 + 1] = val

    def process_op_le(x, idx, val):
        ind = unique_intervals[idx].index(val) + 1
        x[idx * 2 + 1] = unique_intervals[idx][ind]

    def process_op_ge(x, idx, val):
        x[idx * 2] = val

    def process_op_gt(x, idx, val):
        ind = unique_intervals[idx].index(val) + 1
        x[idx * 2] = unique_intervals[idx][ind]

    def process_op_eq(x, idx, val):
        ind = unique_intervals[idx].index(val) + 1
        x[idx * 2] = val
        x[idx * 2 + 1] = unique_intervals[idx][ind]

    op_functions = {
        "<": process_op_lt,
        "<=": process_op_le,
        ">=": process_op_ge,
        ">": process_op_gt,
        "=": process_op_eq,
    }

    X = []
    origin = [[v[0], v[-1]] for v in unique_intervals.values()]
    for query in query_set:
        x = np.array(origin).ravel()
        idxs, ops, vals, _ = query
        for i, o, v in zip(idxs, ops, vals):
            op_functions[o](x, i, v)
        X.append(x)
    X = np.array(X, dtype=np.float32)
    y = np.array([query[-1] for query in query_set], dtype=np.float32).reshape(-1, 1)
    y /= table_size[0]
    train = np.hstack((X, y))

    # make train set unique
    if args.unique_train:
        train = np.unique(train, axis=0)

    # add boundary
    if args.boundary:
        train = add_boundary_2_input(train, unique_intervals, args.boundary)

    # shuffle and split
    np.random.shuffle(train)
    X, y = np.hsplit(train, [-1])
    return X, y


def add_boundary_2_input(train, unique_intervals, alpha=0.1):
    # add total k = int(train.shape[0] * alpha) boundary points
    # 1/4 for one point, 1/4 for two zero points, 1/2 for other zero points
    # create boundary set

    # 1. one point
    one = np.array([[v[0], v[-1]] for v in unique_intervals.values()]).ravel()
    one = np.append(one, 1)
    k = int(train.shape[0] * alpha / 4)
    repeated_one = np.tile(one, (k, 1))

    # 2. two zero points
    zero_0 = np.array([[v[0]] * 2 for v in unique_intervals.values()]).ravel()
    zero_1 = np.array([[v[-1]] * 2 for v in unique_intervals.values()]).ravel()
    zero = np.vstack((zero_0, zero_1))
    zero_y = np.zeros((2, 1))
    zero = np.hstack((zero, zero_y))
    k = int(train.shape[0] * alpha / 8)
    repeated_zero = np.tile(zero, (k, 1))

    # 3. other zero points
    k = int(train.shape[0] * alpha / 2)
    other_zero = [[[np.random.choice(v)] * 2 for v in unique_intervals.values()] for _ in range(k)]
    other_zero = [np.array(v).ravel() for v in other_zero]
    other_zero = np.hstack((np.array(other_zero), np.zeros((k, 1))))

    train = np.vstack((train, repeated_one, repeated_zero, other_zero))
    return train

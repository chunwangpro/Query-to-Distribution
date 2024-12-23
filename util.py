import numpy as np
import pandas as pd
from tqdm import tqdm

OPS = {
    ">": np.greater,
    "<": np.less,
    ">=": np.greater_equal,
    "<=": np.less_equal,
    "=": np.equal,
}


def generate_random_query(table, args, rng):
    """Generate a random query."""
    conditions = rng.randint(args.min_conditions, args.max_conditions + 1)
    if args.model == "1-input":
        ops = rng.choice(["<="], replace=True, size=conditions)
    elif args.model == "2-input":
        # ops = rng.choice(["<", "<=", ">", ">=", "="], replace=True, size=conditions)
        ops = rng.choice(["<="], replace=True, size=conditions)
        # ops = rng.choice(["="], replace=True, size=conditions)
    idxs = rng.choice(table.shape[1], replace=False, size=conditions)
    idxs = np.sort(idxs)
    cols = table[:, idxs]
    vals = table[rng.randint(0, table.shape[0]), idxs]
    card = calculate_query_cardinality(cols, ops, vals)
    return idxs, ops, vals, card


def column_intervalization(query_set, table_size):
    column_interval = {i: set() for i in range(table_size[1])}
    for query in query_set:
        idxs, _, vals, _ = query
        for i in range(len(idxs)):
            column_interval[idxs[i]].add(vals[i])
    # Apply the column_interval to <, <=, >, >=, =
    for k, v in column_interval.items():
        if not v:
            # use [0] for empty column interval
            column_interval[k] = [0, 1]
        else:
            interval_list = sorted(list(v))
            add_small = 2 * interval_list[0] - interval_list[1]
            add_big_1 = 2 * interval_list[-1] - interval_list[-2]
            add_big_2 = 3 * interval_list[-1] - 2 * interval_list[-2]
            column_interval[k] = [add_small] + interval_list + [add_big_1, add_big_2]
    return column_interval


def count_column_unique_interval(unique_intervals):
    # count unique query interval number for each column
    return [len(v) for v in unique_intervals.values()]


def calculate_query_cardinality(data, ops, vals):
    """
    Use ops and vals as queries to find the number of rows in data that meet the conditions.

    Parameters:
    data (2D-array): The subset of table columns involved in the query. Table columns not involved in the query are not included in data.
    ops (1D-array): A list of operators, support operators: '>', '>=', '<', '<=', '='.
    vals (1D-array): A list of values.

    Returns:
    int: The cardinality (number of rows) that satisfy the query.

    Example:
    for empty table, use np.empty((n_row, n_col), dtype=np.float32), return 0.

    for non-empty table:
    table = np.array([[1, 2, 3, 4, 5],
                      [10, 20, 30, 40, 50],
                      [10, 20, 30, 40, 50]]).T
    data = table[:, [1, 2]]
    ops = [">=", ">="]
    vals = [20, 20]
    result = calculate_query_cardinality(data, ops, vals)
    print(result)
    """

    # assert data.shape[1] == len(ops) == len(vals)
    bools = np.ones(data.shape[0], dtype=bool)
    for i, (o, v) in enumerate(zip(ops, vals)):
        bools &= OPS[o](data[:, i], v)
    return bools.sum()


def calculate_Q_error(dataNew, query_set, table_size=None):
    print("\nBegin Calculating Q-error ...")
    Q_error = []
    for query in tqdm(query_set):
        idxs, ops, vals, card_true = query
        card_pred = calculate_query_cardinality(dataNew[:, idxs], ops, vals)
        if card_pred == 0 and card_true == 0:
            Q_error.append(1)
        elif card_pred == 0:
            Q_error.append(card_true)
        elif card_true == 0:
            Q_error.append(card_pred)
        else:
            Q_error.append(max(card_pred / card_true, card_true / card_pred))
    print("Done.\n")
    return Q_error


def print_Q_error(Q_error, args, resultsPath):
    print("Summary of Q-error:")
    print(args)
    statistics = {
        "min": np.min(Q_error),
        "50": np.median(Q_error),
        "75": np.percentile(Q_error, 75),
        "90": np.percentile(Q_error, 90),
        "99": np.percentile(Q_error, 99),
        "max": np.max(Q_error),
        "mean": np.mean(Q_error),
    }
    df = pd.DataFrame.from_dict(statistics, orient="index", columns=["Q-error"])
    df.index.name = None
    df.to_csv(f"{resultsPath}/Q_error.csv", index=True, header=False)
    print(df)

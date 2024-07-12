# Description: This script reads a CSV file and converts all floats to integers by multiplying them by 10^d, where d is the maximum number of decimal places in the dataset.

import pandas as pd


def find_decimal_places(x):
    if isinstance(x, float):
        decimal_part = str(x).split(".")[1]
        return len(decimal_part.rstrip("0"))
    return 0


def change_floats_to_ints_in_csv(input_file):
    df = pd.read_csv(f"{input_file}.csv", header=None)

    max_decimal_places = df.map(find_decimal_places).max().max()
    print(f"Max decimal places: {max_decimal_places}")

    def int_to_max_decimal(x):
        return int(x * 10**max_decimal_places)

    df = df.map(int_to_max_decimal)
    df.to_csv(f"{input_file}_int.csv", index=False, header=False)


if __name__ == "__main__":
    input_file = "datasets/wine"
    change_floats_to_ints_in_csv(input_file)
    print("\nDone.")

import pandas as pd
import describe_helpers as dh
import sys


def compute_column_stats(df_column):
    """Function calculating the statistics of a dataframe's column."""
    column = df_column.dropna()
    count = len(column)
    mean = dh.mean_(column)
    qrtl1, qrtl2 = dh.quartile_(column)
    std = dh.std_(column)
    min = dh.min_(column)
    max = dh.max_(column)
    median = dh.median_(column)
    return [count, mean, std, min, qrtl1, median, qrtl2, max]


def main():
    """Program providing statistical information about dataframe's numerical columns."""
    try:
        if len(sys.argv) != 2:
            raise AssertionError("Need one argument.")
        else:
            tab = {}
            df = pd.read_csv(sys.argv[1])
            for column in df.columns :
                if column == "Index":
                    continue
                if pd.api.types.is_numeric_dtype(df[column]):
                    tab[column] = compute_column_stats(df[column])
            df_result = pd.DataFrame(tab)
            df_result.index = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
            print(df_result)
    except Exception as e:
        print(f"{type(e).__name__} : {e}")


if __name__ == "__main__":
    main()

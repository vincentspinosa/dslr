import pandas as pd
import numpy as np
import library as lib
import sys

def calcul(df_column):
    column = df_column.dropna()
    count = len(column)
    mean = lib.mean(column)
    qrtl1, qrtl2 = lib.quartile(column)
    std = lib.std(column)
    min = lib.min(column)
    max = lib.max(column)
    return [count, std, min, qrtl1, mean, qrtl2, max]

def main():
    """Programme providing statistics informations
    about dataframe's numerical columns.
    """
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
                    tab[column] = calcul(df[column])
            df_result = pd.DataFrame(tab)
            df_result.index = ["count", "std", "min", "25%", "50%", "75%", "max"]
            print(df_result)
    except Exception as e:
        print(f"{type(e).__name__} : {e}")


if __name__ == "__main__":
    main()






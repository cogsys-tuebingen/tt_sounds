"""
Split the full dataset into train and test datasets
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def get_statistics(df):
    # print(df.groupby(["spin-magnitude"])["racket-type"].value_counts())
    # for racket_id in range(1, 11):
    #     print(f"{racket_id:02d}")
    #     df_tmp = df[df["racket-type"] == f"{racket_id:02d}"]
    #     print(df_tmp.groupby(["spin-direction"]).count())
    #     print()
    # Group by 'surface' and 'spin-direction' and count occurrences
    df_grouped = (
        df.groupby(["racket-type", "spin-direction"]).size().unstack(fill_value=0)
    )
    print(df_grouped)

    # Add the 'Total' column
    df_grouped["Total"] = df_grouped.sum(axis=1)

    # Print LaTeX table rows
    for racket_type, row in df_grouped.iterrows():
        back_count = row.get("back", 0)
        flat_count = row.get("none", 0)
        top_count = row.get("top", 0)
        total = row["Total"]
        print(
            f"{racket_type} & {back_count} & {flat_count} & {top_count} & {total} \\\\"
        )

    print(df.groupby(["surface"]).count())


if __name__ == "__main__":
    metadata_file = Path("../data/full.csv")
    df = pd.read_csv(metadata_file, sep=";")
    df = df[~((df["racket-type"] == "none") & (df["surface"] == "racket"))]
    print(df)

    # Assuming you want to stratify based on all columns except 'bounce-id', 'original-file', and 'timestamp'
    # Create a stratification column by concatenating the values of the attributes
    df["stratify_col"] = df[
        ["surface", "racket-type", "spin-magnitude", "spin-direction"]
    ].apply(lambda x: "_".join(x), axis=1)

    # Perform the stratified split
    df_train, df_test = train_test_split(
        df, test_size=0.2, stratify=df["stratify_col"], random_state=42
    )

    # Drop the auxiliary stratification column
    df_train = df_train.drop(columns=["stratify_col"])
    df_test = df_test.drop(columns=["stratify_col"])

    # Print the results
    print("80% split:")
    print(df_train)
    print("\n20% split:")
    print(df_test)

    get_statistics(df)
    get_statistics(df_train)
    get_statistics(df_test)

    df_train.to_csv("../data/train.csv", index=False)
    df_test.to_csv("../data/test.csv", index=False)

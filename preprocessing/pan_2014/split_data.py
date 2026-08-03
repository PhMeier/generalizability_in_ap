import csv
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

"""
Creates a 60/20/20 split for PAN 2014, since the offical test set is not public. 

"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a stratified train/test split."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input CSV file"
    )
    parser.add_argument(
        "--data",
        required=True,
        help="datasplit"
    )
    args = parser.parse_args()

    # Load data
    dataset = args.data
    df = pd.read_csv(args.input, delimiter='\t', quotechar='"', escapechar='\\', quoting=csv.QUOTE_ALL,
                           lineterminator="\n", index_col=False)
    df = df.drop("Unnamed: 0", errors="ignore")
    print(df.shape)
    train_df, val_test_df = train_test_split(
        df,
        test_size=0.4,
        random_state=42,
        stratify=df["label"]
    )
    print(val_test_df.shape)
    val_df, test_df = train_test_split(
        val_test_df,
        test_size=0.5,
        random_state=42,
        stratify=val_test_df["label"]
    )


    print(f"Train size: {len(train_df)}")
    print(f"Val size: {len(val_df)}")
    print(f"Test size: {len(test_df)}")

    print("\nTrain:")
    print(train_df["label"].value_counts(normalize=True))

    print("\nVal:")
    print(val_df["label"].value_counts(normalize=True))

    print("\nTest:")
    print(test_df["label"].value_counts(normalize=True))

    train_df.to_csv(f"./stratified/train_{dataset}_stratified.tsv", sep="\t", encoding="utf-8",index=False,quotechar='"',
                           escapechar='\\', quoting=csv.QUOTE_ALL, lineterminator="\n")
    val_df.to_csv(f"./stratified/validation_{dataset}_stratified.tsv", sep="\t",encoding="utf-8",index=False,quotechar='"',
                  escapechar='\\', quoting=csv.QUOTE_ALL, lineterminator="\n")
    test_df.to_csv(f"./stratified/test_{dataset}_stratified.tsv", sep="\t",encoding="utf-8",index=False,quotechar='"', # blogs_lm
                  escapechar='\\', quoting=csv.QUOTE_ALL, lineterminator="\n")

    print(train_df.columns)
    train_ids = train_df[["author_id", "label"]].copy()
    val_ids = val_df[["author_id", "label"]].copy()
    test_ids = test_df[["author_id", "label"]].copy()


    train_ids.to_csv(f"./stratified/train_{dataset}_ids.tsv", sep="\t", index=False)
    val_ids.to_csv(f"./stratified/val_{dataset}_ids.tsv", sep="\t", index=False)
    test_ids.to_csv(f"./stratified/test_{dataset}_ids.tsv", sep="\t", index=False)
import csv
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
"""
Creates a 80% Training and 20% validation Split from the training data.
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
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )

    print(f"Train size: {len(train_df)}")
    print(f"Test size: {len(val_df)}")


    print("\nTrain:")
    print(train_df["label"].value_counts(normalize=True))

    print("\nVal:")
    print(val_df["label"].value_counts(normalize=True))
    train_df.to_csv(f"./stratified/train_{dataset}_stratified.tsv", sep="\t", encoding="utf-8",index=False,quotechar='"',
                           escapechar='\\', quoting=csv.QUOTE_ALL, lineterminator="\n")
    val_df.to_csv(f"./stratified/validation_{dataset}_stratified.tsv", sep="\t",encoding="utf-8",index=False,quotechar='"',
                  escapechar='\\', quoting=csv.QUOTE_ALL, lineterminator="\n")

    train_ids = train_df["author_id"]
    val_ids = val_df["author_id"]

    train_ids.to_csv(f"./stratified/train_{dataset}_ids.tsv", sep="\t", index=False)
    val_ids.to_csv(f"./stratified/val_{dataset}_ids.tsv", sep="\t", index=False)
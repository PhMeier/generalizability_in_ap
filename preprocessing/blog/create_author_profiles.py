import csv
import sys
import re
import pandas as pd
import sys

URL_EXTRACT_PATTERN_14 = re.compile(r'https?://(?:[/\-\w.]|(?:%[\da-fA-F]{2}))+')

"""
After Downsampling the data, create the representation U of an author by concatenating the texts of the author.
Creates Author profiles for BERT, RoBERTa, XLNET, the feature pipeline, the TF-IDF classifiers (baseline) and the LLM
(baseline)
"""


model_to_sep = {
    "bert": "[SEP]",
    "roberta": "</s>",
    "xlnet": "<sep>",
    "features": " ",
    "baseline": " ",
}


def group_dataframe(df):
    grouped = (df.groupby("author_id").agg(
        document=("document", list),
        document_count=("document", "size"),
        label=("label", "first")
    )
    .reset_index()
               )
    grouped = grouped.sort_values("document_count", ascending=False)
    return grouped

def create_profiles_features(df):
    profiles = (
        df.groupby(["author_id", "label"], as_index=False)
        .agg(
            document=("document", list),
            document_count=("document", "size")
        )
    )
    return profiles

def create_profiles_lm(df, sep_token):
    profiles = (
        df.groupby(["author_id", "label"], as_index=False)
        .agg(
            document=("document", f" {sep_token} ".join),
            document_count=("document", "size")
        )
    )
    return profiles

def create_profiles_baseline(df, sep_token):
    profiles = (
        df.groupby(["author_id", "label"], as_index=False)
        .agg(
            document=("document", f" ".join),
            document_count=("document", "size")
        )
    )
    return profiles

#"blog_sampled_1999.tsv"


def routine(inputfile, model_name):
    sep_token = model_to_sep[model_name]
    output = inputfile.split(".tsv")[0] + f"_{model_name}_ap.tsv"

    df = pd.read_csv(inputfile, sep="\t", encoding="utf-8", index_col=False, #reddit_train_male_ap
                              quotechar='"', escapechar='\\', quoting=csv.QUOTE_ALL, lineterminator="\n")
    if model_name == "baseline":
        df_grouped = create_profiles_lm(df, sep_token)
    elif model_name != "features":
        df_grouped = create_profiles_lm(df, sep_token)
    else:
        df_grouped = create_profiles_features(df)
    print(df_grouped)
    df_grouped.to_csv(output, sep="\t", encoding="utf-8", index=False, #reddit_train_male_ap
                              quotechar='"', escapechar='\\', quoting=csv.QUOTE_ALL, lineterminator="\n")

if __name__ == "__main__":
    filenames = ["blog_sampled_30000000_train.tsv", "blog_sampled_10000000_val.tsv", "blog_sampled_10000000_test.tsv"]
    model_name = sys.argv[1]
    assert model_name in model_to_sep
    print(f"Creating Author Profile for {model_name}")
    for fname in filenames:
        print("FILENAME : ", fname)
        routine(fname, model_name)


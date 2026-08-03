import csv
import sys
import re
import pandas as pd
import ast


URL_EXTRACT_PATTERN_14 = re.compile(r'https?://(?:[/\-\w.]|(?:%[\da-fA-F]{2}))+')

model_to_sep = {
    "bert": "[SEP]",
    "roberta": "</s>",
    "xlnet": "<sep>",
    "features": " ",
    "baseline": " "
}


def create_profiles(df, sep_token):
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


def create_profiles_features(df):
    profiles = (
        df.groupby(["author_id", "label"], as_index=False)
        .agg(
            document=("document", list),
            document_count=("document", "size")
        )
    )
    return profiles


def training_set_routine(model_name):
    sep_token = model_to_sep[model_name]
    df_male = pd.read_csv("./pan_13_train_male_preprocessed.tsv", sep="\t", encoding="utf-8", index_col=False,
                          quotechar='"', escapechar='\\', quoting=csv.QUOTE_ALL, lineterminator="\n")
    # "./pan_13_train_female_preprocessed.tsv"
    df_female = pd.read_csv("./pan_13_train_female_preprocessed.tsv", sep="\t", encoding="utf-8",
                            index_col=False, quotechar='"', escapechar='\\', quoting=csv.QUOTE_ALL, lineterminator="\n")
    df_male["label"] = "0"
    df_female["label"] = "1"
    df_male["document"] = df_male["document"].fillna("").astype(str)
    df_female["document"] = df_female["document"].fillna("").astype(str)
    # df_male["document"] = df_male["document"].str.replace(URL_EXTRACT_PATTERN_14, "[URL]", regex=True)
    # df_female["document"] = df_female["document"].str.replace(URL_EXTRACT_PATTERN_14, "[URL]", regex=True)
    if model_name != "features":
        df_male_ap = create_profiles(df_male, sep_token)
        df_female_ap = create_profiles(df_female, sep_token)
    else:
        df_male_ap = create_profiles_features(df_male)
        df_female_ap = create_profiles_features(df_female)

    df_male_ap["label"] = "0"
    df_female_ap["label"] = "1"


    df_merged = pd.concat([df_male_ap, df_female_ap], ignore_index=True)

    df_merged.to_csv(f"./pan_13_train_preprocessed_masked_url_{model_name}.tsv", sep="\t", encoding="utf-8",
                     index=False,
                     quotechar='"', escapechar='\\', quoting=csv.QUOTE_ALL, lineterminator="\n")

    # df_male.to_csv(f"./pan_13_train_male_preprocessed_masked_url_{model_name}.tsv", sep="\t", encoding="utf-8", index=False,
    #               quotechar='"', escapechar='\\', quoting=csv.QUOTE_ALL, lineterminator="\n")

    # df_female.to_csv(f"./pan_13_train_female_preprocessed_masked_url_{model_name}.tsv", sep="\t", encoding="utf-8", index=False,
    #                 quotechar='"', escapechar='\\', quoting=csv.QUOTE_ALL, lineterminator="\n")


def routine_for_extracted_data(model_name):
    sep_token = model_to_sep[model_name]
    df_train = pd.read_csv("pan13_sampled_no_sort_30000000_train.tsv", sep="\t", encoding="utf-8", index_col=False,
                           quotechar='"', escapechar='\\', quoting=csv.QUOTE_ALL, lineterminator="\n")
    # "./pan_13_train_female_preprocessed.tsv"
    df_val = pd.read_csv("pan13_sampled_no_sort_6000000_val.tsv", sep="\t", encoding="utf-8",
                         index_col=False, quotechar='"', escapechar='\\', quoting=csv.QUOTE_ALL, lineterminator="\n")

    df_train["document"] = df_train["document"].fillna("").astype(str)
    df_val["document"] = df_val["document"].fillna("").astype(str)
    # df_male["document"] = df_male["document"].str.replace(URL_EXTRACT_PATTERN_14, "[URL]", regex=True)
    # df_female["document"] = df_female["document"].str.replace(URL_EXTRACT_PATTERN_14, "[URL]", regex=True)
    if model_name != "features":
        df_train_ap = create_profiles(df_train, sep_token)
        df_val_ap = create_profiles(df_val, sep_token)
    elif model_name == "baseline":
        df_train_ap = create_profiles_baseline(df_train, sep_token)
        df_val_ap = create_profiles_baseline(df_val, sep_token)
    else:
        df_train_ap = create_profiles_features(df_train)
        df_val_ap = create_profiles_features(df_val)

    # df_male_ap["label"] = 0
    # print(df_male_ap.columns)
    # df_male_ap = df_male_ap[["author_id", "document", 'document_count', "label"]]
    # print(df_male_ap.columns)

    df_train_ap.to_csv(f"./sampled_documents_train_no_sort_30000000_{model_name}.tsv", sep="\t", encoding="utf-8",
                       index=False,
                       quotechar='"', escapechar='\\', quoting=csv.QUOTE_ALL, lineterminator="\n")

    df_val_ap.to_csv(f"./sampled_documents_val_no_sort_6000000_{model_name}.tsv", sep="\t", encoding="utf-8",
                     index=False,
                     quotechar='"', escapechar='\\', quoting=csv.QUOTE_ALL, lineterminator="\n")


def routine_for_test_data(model_name):
    sep_token = model_to_sep[model_name]
    df_test = pd.read_csv("pan_13_test_preprocessed_URL_masked.tsv", sep="\t", encoding="utf-8", index_col=False,
                          quotechar='"', escapechar='\\', quoting=csv.QUOTE_ALL, lineterminator="\n")

    df_test["document"] = df_test["document"].fillna("").astype(str)
    # df_male["document"] = df_male["document"].str.replace(URL_EXTRACT_PATTERN_14, "[URL]", regex=True)
    # df_female["document"] = df_female["document"].str.replace(URL_EXTRACT_PATTERN_14, "[URL]", regex=True)
    if model_name != "features":
        df_test_ap = create_profiles(df_test, sep_token)

    elif model_name == "baseline":
        df_test_ap = create_profiles_baseline(df_test, sep_token)

    else:
        df_test_ap = create_profiles_features(df_test)

    # df_male_ap["label"] = 0
    # print(df_male_ap.columns)
    # df_male_ap = df_male_ap[["author_id", "document", 'document_count', "label"]]
    # print(df_male_ap.columns)

    df_test_ap.to_csv(f"./pan_13_test_preprocessed_URL_masked_{model_name}.tsv", sep="\t", encoding="utf-8",
                      index=False,
                      quotechar='"', escapechar='\\', quoting=csv.QUOTE_ALL, lineterminator="\n")


if __name__ == "__main__":
    model_name = sys.argv[1]  # "xlnet"
    routine_for_test_data(model_name)


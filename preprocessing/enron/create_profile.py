import csv
import ast
import json
import argparse
import pandas as pd
import re
surrogate_pattern = re.compile(r"[\ud800-\udfff]")

URL_EXTRACT_PATTERN_14 = re.compile(r'https?://(?:[/\-\w.]|(?:%[\da-fA-F]{2}))+')

parser = argparse.ArgumentParser(
                    prog='Add separator token',
                    description='Merge data and create a representation with separator token',
                    epilog='Text at the bottom of help')

parser.add_argument(
    "--input",
    required=True,
    help="Path to the input file"
)
parser.add_argument("--model", "--model", help="Name of the model")


sep_tokens = {
    "BERT": "[SEP]",
    "XL": "<sep>",
    "ROBERTA": "</s>",
    "BASELINE": ""
}

file_ending = {
    "XL": "_xl_author_profiles.tsv",
    "BERT":"_BERT_author_profiles.tsv",
    "ROBERTA": "_ROBERTA_author_profiles.tsv",
    "BASELINE": "_BASELINE_author_profiles.tsv"
}

gender_to_number = {
    "male": 0,
    "female": 1
}

"""
Add the corresponding separator token for the masked columns

"""

if __name__ == "__main__":
    filename = "/enron_preprocessed_ap_between_50_and_1000.tsv"
    model = "BASELINE" #sys.argv[2]
    #args = parser.parse_args()
    #filename = args.input
    #model = args.model
    model = model.upper()

    print(filename)
    df = pd.read_csv(filename, sep="\t", index_col=False, encoding="utf-8", #reddit_train_male_ap
                              quotechar='"', escapechar='\\', quoting=csv.QUOTE_ALL, lineterminator="\n")

    # for enron
    df = pd.read_csv(filename,sep="\t",
        quotechar='"',
        quoting=csv.QUOTE_ALL,
        doublequote=True,
        encoding="utf-8",)

    outputfile_grande = filename.split(".tsv")[0] + file_ending[model] #"_xl_grande_final.tsv"


    token = sep_tokens[model]
    #assert model in filename

    print(df.columns)


    #df["document"] = df["document"].apply(ast.literal_eval)
    df["document"] = df["document"].apply(json.loads)
    df["document"] = df["document"].astype(str)

    df["document"] = df["document"].str.replace(
        URL_EXTRACT_PATTERN_14,
        "[URL]",
        regex=True
    )

    df["document"] = df["document"].apply(lambda docs: f" {token} ".join(map(str, docs)))
    df["label"] = df["label"].map(gender_to_number)

    df.to_csv(outputfile_grande, encoding="utf-8", sep='\t', quotechar='"', escapechar='\\', quoting=csv.QUOTE_ALL, lineterminator="\n")

    #df["document"] = df["document"].apply(ast.literal_eval)

import csv
import ast
import json
import argparse
import pandas as pd
import re
surrogate_pattern = re.compile(r"[\ud800-\udfff]")
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
    "ROBERTA": "</s>"
}

file_ending = {
    "XL": ["_xl_grande_final.tsv", "_xl_lite_final.tsv"],
    "BERT":["_BERT_grande_final.tsv", "_BERT_lite_final.tsv"],
    "ROBERTA": ["_ROBERTA_grande_final.tsv", "_ROBERTA_lite_final.tsv"]
}

"""
Add the corresponding separator token for the masked columns

"""

if __name__ == "__main__":

    args = parser.parse_args()
    filename = args.input
    model = args.model
    model = model.upper()

    print(filename)
    df = pd.read_csv(filename, sep="\t", index_col=False, #reddit_train_male_ap
                              quotechar='"', escapechar='\\', quoting=csv.QUOTE_ALL, lineterminator="\n")
    df_lite = df.copy()
    df_grande = df.copy()
    outputfile_grande = filename.split(".tsv")[0] + file_ending[model][0] #"_xl_grande_final.tsv"
    outputfile_lite = filename.split(".tsv")[0] + file_ending[model][1] #"_xl_lite_final.tsv"


    token = sep_tokens[model]
    assert model in filename

    print(df.columns)
    #if pan17
    #df["masked_lite"] = df["masked_lite"].apply(json.loads) #(ast.literal_eval)
    #df["masked_grande"] = df["masked_grande"].apply(json.loads) #(ast.literal_eval)

    df["masked_lite"] = df["masked_lite"].apply(ast.literal_eval)
    df["masked_grande"] = df["masked_grande"].apply(ast.literal_eval)

    df_lite["masked_text"] = df["masked_lite"].apply(lambda docs: f" {token} ".join(docs))
    df_grande["masked_text"] = df["masked_grande"].apply(lambda docs: f" {token} ".join(docs))

    df_lite["masked_text"] = (
        df_lite["masked_text"]
        .fillna("")
        .astype(str)
        .map(lambda text: surrogate_pattern.sub("\uFFFD", text))
    )

    df_grande["masked_text"] = (
        df_grande["masked_text"]
        .fillna("")
        .astype(str)
        .map(lambda text: surrogate_pattern.sub("\uFFFD", text))
    )



    #print(df)
    print("DF GRANDE")
    print(df_grande["masked_text"])

    df_grande.to_csv(outputfile_grande, encoding="utf-8", sep='\t', quotechar='"', escapechar='\\', quoting=csv.QUOTE_ALL, lineterminator="\n")
    df_lite.to_csv(outputfile_lite, encoding="utf-8", sep='\t', quotechar='"', escapechar='\\', quoting=csv.QUOTE_ALL, lineterminator="\n")


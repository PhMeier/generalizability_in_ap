import re
import sys
import csv
import pandas as pd


URL_EXTRACT_PATTERN_14 = re.compile(r'https?://(?:[/\-\w.]|(?:%[\da-fA-F]{2}))+')

"""
Takes in the dataframe of PAN 2014 Hotel Reviews and replaces URLs with [URL]
"""

from collections import Counter

if __name__ == "__main__":
    filename_male = sys.argv[1]
    filename_female = sys.argv[1]
    outputfile = "pan_2014_reviews_prepro.tsv"
    df_male = pd.read_csv(filename_male, sep="\t", encoding="utf-8", index_col=False,
                          quotechar='"', escapechar='\\', quoting=csv.QUOTE_ALL, lineterminator="\n")
    df_female = pd.read_csv(filename_female, sep="\t", encoding="utf-8", index_col=False,
                          quotechar='"', escapechar='\\', quoting=csv.QUOTE_ALL, lineterminator="\n")
    df = pd.concat([df_male, df_female])
    print(Counter(df["label"].to_list()))
    df["document"] = df["document"].str.replace(URL_EXTRACT_PATTERN_14, "[URL]", regex=True)
    print(df["document"])

    output = outputfile #filename.split(".tsv")[0] + "_prepro.tsv"
    df.to_csv(output, sep="\t",
        index=False,
        encoding="utf-8",
        quotechar='"',
        escapechar='\\',
        quoting=csv.QUOTE_ALL,
        lineterminator="\n")

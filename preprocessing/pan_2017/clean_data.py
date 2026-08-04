import re
import sys
import csv
import pandas as pd
# for 17, pan, blog
#URL_EXTRACT_PATTERN = re.compile(r"(?:https?:\/\/)?(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&\/=]*)")

URL_EXTRACT_PATTERN_14 = re.compile(r'https?://(?:[/\-\w.]|(?:%[\da-fA-F]{2}))+')



from collections import Counter

if __name__ == "__main__":
    filename_male = sys.argv[1]
    filename_female = sys.argv[2]
    partition = sys.argv[3]
    outputfile = f"pan_2017_{partition}_prepro.tsv"
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

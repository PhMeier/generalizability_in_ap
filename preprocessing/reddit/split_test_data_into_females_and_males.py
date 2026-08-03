import pandas as pd
import csv

input_file = "reddit_test/pandora_author_info_to_comment.tsv"

df = pd.read_csv(
    input_file,
    sep="\t",
    quotechar='"',
    escapechar="\\",
    quoting=csv.QUOTE_ALL,
    lineterminator="\n",
    encoding="utf-8",
)

df_male = df[df["gender"] == "male"]
df_female = df[df["gender"] == "female"]

df_male.to_csv(
    "reddit_pandora_male_test.tsv",
    sep="\t",
    index=False,
    quotechar='"',
    escapechar="\\",
    quoting=csv.QUOTE_ALL,
    lineterminator="\n",
)

df_female.to_csv(
    "reddit_pandora_female_test.tsv",
    sep="\t",
    index=False,
    quotechar='"',
    escapechar="\\",
    quoting=csv.QUOTE_ALL,
    lineterminator="\n",
)

print(f"Male authors: {len(df_male)}")
print(f"Female authors: {len(df_female)}")
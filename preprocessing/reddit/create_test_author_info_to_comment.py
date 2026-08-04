import csv
import pandas as pd


"""
Merge the test set author with the comments

"""

f = "pandora_baseline/res/is_female-LR-N/preds.csv"
df = pd.read_csv(
    f,
    sep=","
)
df["gender"] = df["true"].map({
    0.0: "male",
    1.0: "female"
})


df_comments = pd.read_csv("../data/reddit/mf_comments_all_noq.csv", sep=",")

merged = pd.merge(df, df_comments, on="author", how="inner")

output ="reddit_test/pandora_author_info_to_comment.tsv"

merged.to_csv(output,
                   sep="\t",
                   quotechar='"',
                   escapechar="\\",
                   quoting=csv.QUOTE_ALL,
                   lineterminator="\n",
                   encoding="utf-8",
                   )


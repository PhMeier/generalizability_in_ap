import csv
import re
from bs4 import BeautifulSoup
import pandas as pd
URL_EXTRACT_PATTERN = re.compile(r"(?:https?:\/\/)?(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&\/=]*)")
URL_EXTRACT_PATTERN_14 = re.compile(r'https?://(?:[/\-\w.]|(?:%[\da-fA-F]{2}))+')
"""
Remove the HTML tags from PAN 13 corpora to compute statistics.

"""
def group_dataframe(df):
    """
    Groups the dataframe for feature and baselines
    """
    grouped = (df.groupby("author_id").agg(
        document=("document", list),
        document_count=("document", "size"),
        label=("label", "first")
    )
    .reset_index()
               )
    grouped = grouped.sort_values("document_count", ascending=False)
    return grouped


def parse_html(html):
    text = BeautifulSoup(html, "html.parser").get_text(separator=" ")
    # do only remove texts which are sourrounded by whitespace! These semicolons are leftovers from the HTML tags
    text = re.sub(r'\s+(?:;\s+)+', ' ', text)
    text = re.sub("urlLink", "[URL]", text)
    result = re.sub(URL_EXTRACT_PATTERN_14, "[URL]", text)
    return result



def train_routine():
    df_male = pd.read_csv("./pan_13_train_male.tsv",
                          sep="\t", encoding="utf-8", index_col=False,
                          quotechar='"', escapechar='\\', quoting=csv.QUOTE_ALL, lineterminator="\n")
    df_female = pd.read_csv("./pan_2013_train_female.tsv",
                            sep="\t", encoding="utf-8",
                            index_col=False,quotechar='"', escapechar='\\', quoting=csv.QUOTE_ALL, lineterminator="\n")
    df_male["label"] = "0"
    df_female["label"] = "1"


    gr = group_dataframe(df_male)
    print(gr)

    gr = group_dataframe(df_female)
    print(gr)


    df_male["document"] = df_male["document"].apply(parse_html)
    df_female["document"] = df_female["document"].apply(parse_html)



    df_male.to_csv(
        "./pan_13_train_male_preprocessed.tsv",
        sep="\t",
        encoding="utf-8",
        index=False,
        quotechar='"',
        escapechar='\\',
        quoting=csv.QUOTE_ALL,
        lineterminator="\n"
    )

    df_female.to_csv(
        "./pan_13_train_female_preprocessed.tsv",
        sep="\t",
        encoding="utf-8",
        index=False,
        quotechar='"',
        escapechar='\\',
        quoting=csv.QUOTE_ALL,
        lineterminator="\n"
    )


def test_routine():

    df_male = pd.read_csv("/pan13/pan_2013_test_male.tsv",
                          sep="\t", encoding="utf-8", index_col=False,
                          quotechar='"', escapechar='\\', quoting=csv.QUOTE_ALL, lineterminator="\n")
    df_female = pd.read_csv("/pan13/pan_2013_test_female.tsv",
                            sep="\t", encoding="utf-8",
                            index_col=False,quotechar='"', escapechar='\\', quoting=csv.QUOTE_ALL, lineterminator="\n")




    df_male["document"] = df_male["document"].apply(parse_html)
    df_female["document"] = df_female["document"].apply(parse_html)
    df_male["label"] = "0"
    df_female["label"] = "1"


    #df_male = group_dataframe_for_lm(df_male)
    print(df_male)

    #df_female = group_dataframe_for_lm(df_female)
    print(df_female)

    df_concat = pd.concat([df_male, df_female])

    df_concat.to_csv(
        "./pan_13_test_preprocessed_URL_masked.tsv",
        sep="\t",
        encoding="utf-8",
        index=False,
        quotechar='"',
        escapechar='\\',
        quoting=csv.QUOTE_ALL,
        lineterminator="\n"
    )



if __name__ == "__main__":


    """
    df_male.to_csv(
        "./pan_13_test_male_preprocessed_URL_masked_lm.tsv",
        sep="\t",
        encoding="utf-8",
        index=False,
        quotechar='"',
        escapechar='\\',
        quoting=csv.QUOTE_ALL,
        lineterminator="\n"
    )

    df_female.to_csv(
        "./pan_13_test_female_preprocessed_URL_masked_lm.tsv",
        sep="\t",
        encoding="utf-8",
        index=False,
        quotechar='"',
        escapechar='\\',
        quoting=csv.QUOTE_ALL,
        lineterminator="\n"
    )


    df_male = pd.read_csv("/home/philipp.meier/author_profiling_generalizability/data/pan_2013_male.tsv",
                          sep="\t", encoding="utf-8", index_col=False,
                          quotechar='"', escapechar='\\', quoting=csv.QUOTE_ALL, lineterminator="\n")
    df_female = pd.read_csv("/home/philipp.meier/author_profiling_generalizability/data/pan_2013_female.tsv",
                            sep="\t", encoding="utf-8",
                            index_col=False,quotechar='"', escapechar='\\', quoting=csv.QUOTE_ALL, lineterminator="\n")
    df_male["label"] = "0"
    df_female["label"] = "1"


    gr = group_dataframe(df_male)
    print(gr)

    gr = group_dataframe(df_female)
    print(gr)


    df_male["document"] = df_male["document"].apply(parse_html)
    df_female["document"] = df_female["document"].apply(parse_html)



    df_male.to_csv(
        "./pan_13_train_male_preprocessed.tsv",
        sep="\t",
        encoding="utf-8",
        index=False,
        quotechar='"',
        escapechar='\\',
        quoting=csv.QUOTE_ALL,
        lineterminator="\n"
    )

    df_female.to_csv(
        "./pan_13_train_female_preprocessed.tsv",
        sep="\t",
        encoding="utf-8",
        index=False,
        quotechar='"',
        escapechar='\\',
        quoting=csv.QUOTE_ALL,
        lineterminator="\n"
    )
    """
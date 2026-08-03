import re
import csv
import json
import pandas as pd
from talon.signature.bruteforce import extract_signature
import talon
# don't forget to init the library first
# it loads machine learning classifiers
talon.init()
from talon import signature

"""
- Remove Forwarded Mails
- Length Constraints
    - Remove Mails with less than 50 words (do we need to do that?)
    - Remove Mails with more than 1000 words (do we need to do that?)

- only e-mails with more than 50 words and less than 1000 words were used for our analysis

The body of each e-mail was then
parsed by removing the header, reply texts (if present) and
signatures. All duplicated or carbon-copied e-mails were
removed
"""


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


CLOSINGS = [
    r"many thanks",
    r"Many Thanks",
    r"many Thanks",
    r"Many thanks",
    r"thanks,",
    r"Thanks,",
    r".thanks,",
    r".Thanks,",
    r"Good luck,",
    r"good Luck,",
    r"good luck,",
    r"thank you,",
    r"Thank you,",
    r"thank You,",
    r"Thank You,",
    r"thank you,",
    r"Thank you",
    r"thank You",
    r"Thank You",
    r"Best",
    r"Best",
    r"kind regards",
    r"Kind regards",
    r"kind Regards",
    r"Kind Regards",
    r"best regards",
    r"Best regards",
    r"best Regards",
    r"Best Regards",
    r"regards",
    r"Regards",
    r"cheers",
    r"Cheers",
    r"happy holidays",
    r"Happy holidays",
    r"happy Holidays",
    r"Happy Holidays",
    r"merry christmas",
    r"Merry Christmas",
    r"merry Christmas",
    r"Merry christmas",
    r"Fondly,",
    r"fondly,",
    r"Happy New Year,",
    r"happy new year,",
    r"Happy New year,",
    r"Happy new Year,",
    r"happy new Year,",
    r"happy New Year,",
    r"happy New year,",
]

closing_pattern = "|".join(
    re.escape(closing)
    for closing in sorted(CLOSINGS, key=len, reverse=True)
)

pattern = re.compile(
    rf"(?i)(?:^|(?<=[.!?…]))\s*({closing_pattern})\s*[,!.-]*",
    re.MULTILINE,)

URL_EXTRACT_PATTERN_14 = re.compile(r'https?://(?:[/\-\w.]|(?:%[\da-fA-F]{2}))+')

def group_dataframe_after_authors(df, sep_token):
    grouped = (
        df.groupby("name")
        .agg(
            label=("gender", "first"),
            document=("data", lambda docs: f" {sep_token} ".join(map(str, docs))),
            document_count=("data", "count"),
        )
        .reset_index()
    )
    return grouped

# --------------------------Sent from my BlackBerry Wireless Handheld
# --------------------------Sent from my BlackBerry Wireless Handheld (www.BlackBerry.net)
def remove_signature(text):
    text =str(text)
    #print(text, type(text))
    m = pattern.search(text)
    #print(text)
    #print("m ", m)
    if m:
        return text[:m.start()].rstrip(), text[m.start():].strip() # keep everything before the signature
    return text

if __name__ == "__main__":
    model = "BASELINE"
    f_ending = file_ending[model]
    sep_tok = sep_tokens[model]
    df_gender = pd.read_csv("enron_preprocessed_ap.csv", sep='\t')
    df = pd.read_csv("enron_all_sent_mail.tsv", sep='\t')
    print(df)
    df = df[~df["data"].str.contains(
        r"Forwarded by",
        case=False,
        na=False
    )]
    print(df)

    df = df[~df["data"].str.contains(
        r"Original Message",
        case=False,
        na=False
    )]

    df["data"] = df["data"].apply(remove_signature)


    df = df[~df["data"].str.split().str.len().between(50, 1000)]
    df["data"] = df["data"].str.replace(
        URL_EXTRACT_PATTERN_14,
        "[URL]",
        regex=True
    )



    print(df)

    grouped = group_dataframe_after_authors(df, sep_tok)
    grouped["label"] = grouped["label"].map(gender_to_number)
    #grouped["document"] = grouped["document"].apply(json.dumps)

    grouped = (
        grouped.drop(columns="label")
        .merge(
            df_gender[["name", "label"]],
            on="name",
            how="left"
        )
    )

    grouped["document"] = grouped["document"].apply(json.dumps)

    print(grouped)
    grouped["label"] = grouped["label"].map(gender_to_number)
    print(grouped)
    grouped.to_csv(f"enron_preprocessed_ap_between_50_and_1000_{f_ending}", sep="\t",index=False,
        quotechar='"',
        quoting=csv.QUOTE_ALL,
        doublequote=True,
        lineterminator="\n",
        encoding="utf-8")
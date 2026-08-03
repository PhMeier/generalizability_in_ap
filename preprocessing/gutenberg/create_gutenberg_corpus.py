import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import re
import glob
import json

"""
def group_dataframe_after_authors(df):
    grouped = (
        df.groupby("author_id")
        .agg(
            label=("gender_x", "first"),
            document=("document", list),
            document_count=("document", "count"),
        )
        .reset_index()
    )
    return grouped
"""

def group_dataframe_after_authors(df):
    grouped = (
        df.groupby("author_id")
        .agg(
            label=("gender_x", "first"),
            document=("document",
                      lambda docs: [doc[0] if isinstance(doc, list) else doc
                                    for doc in docs]),
            document_count=("document", "count"),
        )
        .reset_index()
    )
    return grouped

chapter_pattern = re.compile(
    r"([IVXLCDM]+)\n",
    re.IGNORECASE | re.MULTILINE
)
first_chapter_pattern = re.compile("([I\.?]{1}\n)")
first_chapter_with_dots = re.compile("(I\.?\n)")
first_chapter_with_chapter = re.compile("(CHAPTER* I\.?.*)")
chapter_pattern = re.compile("^\s*(?:CHAPTER\s+)?I(?:\.|\b)(?:\s+.*)?$", re.MULTILINE)
def plot_author_distribution_with_words(df, sex:str, outputfile:str):
    # 1. Create word counts if not already present
    if "word_count" not in df.columns:
        df["word_count"] = df["body"].fillna("").str.split().str.len()

    # 2. Author-level statistics
    author_stats = (
        df.groupby("author")
        .agg(
            document_count=("author_id", "size"),
            total_words=("word_count", "sum")
        )
        .reset_index()
    )

    # 3. Distribution curve: document_count -> number of authors
    distribution = author_stats["document_count"].value_counts().sort_index()
    smoothed = distribution.rolling(window=5, min_periods=1).mean()

    # 4. Buckets for word contribution
    bins = [1, 5, 10, 20, 50, 100, np.inf]
    labels = ["1-5", "6-10", "11-20", "21-50", "51-100", "101+"]

    author_stats["doc_bucket"] = pd.cut(
        author_stats["document_count"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    bucket_stats = (
        author_stats.groupby("doc_bucket", observed=False)
        .agg(
            author_count=("document_count", "size"),
            total_words=("total_words", "sum")
        )
        .reset_index()
    )

    bucket_stats["word_percentage"] = (
                                              bucket_stats["total_words"] / bucket_stats["total_words"].sum()
                                      ) * 100

    # 5. Plot
    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(11, 9),
        gridspec_kw={"height_ratios": [2, 1]}
    )

    ax1, ax2 = axes

    # Top plot: author document distribution
    ax1.plot(distribution.index, smoothed, linewidth=2)

    ax1.set_xscale("log")
    ax1.set_yscale("log")

    x_ticks = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 4000, 10000] # 1,2,5*10^k
    y_ticks = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 4000, 10000]

    x_ticks = [t for t in x_ticks if distribution.index.min() <= t <= distribution.index.max()]
    #y_ticks = [t for t in y_ticks if smoothed.min() <= t <= smoothed.max()]

    ax1.set_xticks(x_ticks)
    ax1.set_yticks(y_ticks)

    ax1.xaxis.set_major_formatter(ScalarFormatter())
    ax1.yaxis.set_major_formatter(ScalarFormatter())

    ax1.grid(which="major", linestyle="--", linewidth=1.0, alpha=0.8)
    ax1.grid(which="minor", linestyle="--", linewidth=0.4, alpha=0.4)

    ax1.set_ylabel("Number of authors")
    ax1.set_title(f"{sex} Author Document Distribution and Word Contribution Gutenberg - Log-Scaled")

    # Bottom plot: total word contribution by document-count bucket
    ax2.bar(
        bucket_stats["doc_bucket"].astype(str),
        bucket_stats["word_percentage"]
    )

    bars = ax2.bar(
        bucket_stats["doc_bucket"].astype(str),
        bucket_stats["word_percentage"]
    )

    # Add labels above bars
    for bar, authors, words in zip(
            bars,
            bucket_stats["author_count"],
            bucket_stats["total_words"]
    ):
        height = bar.get_height()

        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{authors:,} / {words:,}",
            ha="center",
            va="bottom",
            fontsize=9
        )
    ax2.set_ylim(0, bucket_stats["word_percentage"].max() * 1.25)
    ax2.set_xlabel("Documents per author")
    ax2.set_ylabel("Corpus words (%)")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    #plt.savefig(outputfile)
    plt.show()

    return author_stats, bucket_stats


def plot_author_distribution_with_paragraphs(df, sex:str, outputfile:str):
    # 1. Create word counts if not already present
    if "word_count" not in df.columns:
        df["word_count"] = df["body"].fillna("").str.split().str.len()

    # 2. Author-level statistics
    author_stats = (
        df.groupby("author")
        .agg(
            document_count=("author_id", "size"),
            total_words=("word_count", "sum"),
            paragraph_count=("paragraph_count", "sum")
        )
        .reset_index()
    )

    # 3. Distribution curve: document_count -> number of authors
    distribution = author_stats["document_count"].value_counts().sort_index()
    smoothed = distribution.rolling(window=5, min_periods=1).mean()

    # 4. Buckets for word contribution
    bins = [1, 5, 10, 20, 50, 100, np.inf]
    labels = ["1-5", "6-10", "11-20", "21-50", "51-100", "101+"]

    author_stats["doc_bucket"] = pd.cut(
        author_stats["document_count"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    bucket_stats = (
        author_stats.groupby("doc_bucket", observed=False)
        .agg(
            author_count=("document_count", "size"),
            total_words=("total_words", "sum"),
            paragraph_count=("paragraph_count", "sum")
        )
        .reset_index()
    )

    bucket_stats["paragraph_percentage"] = (
                                              bucket_stats["paragraph_count"] / bucket_stats["paragraph_count"].sum()
                                      ) * 100


def downsample_bucket(group, gender_col="gender", target_gender="female", random_state=42):
    ...



def create_dict(files):
    d = {}
    for file in files:
        file_name = file.split("\\")[1]
        book_id = file_name.split("_")[0]
        file_name = file.replace("\\","/")
        d[book_id] = file_name
    return d

def create_stats_for_gender(path, df):
    filenames = df["filename"].to_list()#[:2]
    d = {}
    maximum = 0
    minimum = 9999999999999
    for fname in filenames:
        d[fname] = []
        with open(fname, "r", encoding="utf-8") as f:
            content = f.read()
            first_chapter_start = chapter_pattern.search(content)
            if first_chapter_start:
                first_chapter_start = chapter_pattern.search(content).start()
                content = content[first_chapter_start:]
                amount_of_words = len(content.split())
                maximum = max(maximum, amount_of_words)
                minimum = min(minimum, amount_of_words)
                d[fname].append(amount_of_words)
            else:
                amount_of_words = len(content.split())
                d[fname].append(amount_of_words)
                maximum = max(maximum, amount_of_words)
                minimum = min(minimum, amount_of_words)
            paragraphs = re.split(r"\n\s*\n", content)
            paragraphs = [p.strip() for p in paragraphs if p.strip()]
            # paragraphs = [p for p in paragraphs if len(p.split()) > 5]
            paragraph_count = len(paragraphs)
            d[fname].append(paragraph_count)
            paragraph_len_in_words = [len(p) for p in paragraphs]
            if len(paragraph_len_in_words) > 0:
                avg = sum(paragraph_len_in_words) / len(paragraph_len_in_words)
                d[fname].append(avg)
            else:
                avg = 0
                d[fname].append(avg)
    stats_dict = {
        k: {
            "word_count": v[0],
            "paragraph_count": v[1],
            "average_paragraph_length": v[2]
        }
        for k,v in d.items()
    }
    stats_df = pd.DataFrame.from_dict(stats_dict, orient="index").reset_index()
    stats_df = stats_df.rename(columns={"index": "filename"})
    df = df.merge(stats_df, on="filename", how="left")
    print("Maximum: ", maximum)
    print("Minimum: ", minimum)
    return df



def cut_book_content(path, df):
    filenames = df["filename"].to_list()#[:2]
    d = {}
    author_to_books = {}
    maximum = 0
    minimum = 9999999999999
    for fname in filenames:
        d[fname] = []
        author_to_books[fname] = []
        with open(fname, "r", encoding="utf-8") as f:
            content = f.read()
            first_chapter_start = chapter_pattern.search(content)
            if first_chapter_start:
                first_chapter_start = chapter_pattern.search(content).start()
                content = content[first_chapter_start:]
                paragraphs = re.split(r"\n\s*\n", content)
                paragraphs = [p.strip() for p in paragraphs if p.strip()]
                ten_percent_paragraphs = int((len(paragraphs)/100)*10)
                paragraph_without_first_and_last_ten_percent = paragraphs[ten_percent_paragraphs:-ten_percent_paragraphs]
                content = " ".join(paragraph_without_first_and_last_ten_percent)
                amount_of_words = len(content.split())
                maximum = max(maximum, amount_of_words)
                minimum = min(minimum, amount_of_words)
                d[fname].append(amount_of_words)
                author_to_books[fname].append(content)
            else:
                paragraphs = re.split(r"\n\s*\n", content)
                paragraphs = [p.strip() for p in paragraphs if p.strip()]
                ten_percent_paragraphs = int((len(paragraphs)/100)*10)
                paragraph_without_first_and_last_ten_percent = paragraphs[ten_percent_paragraphs:-ten_percent_paragraphs]
                content = " ".join(paragraph_without_first_and_last_ten_percent)
                author_to_books[fname].append(content)
                amount_of_words = len(content.split())
                d[fname].append(amount_of_words)
                maximum = max(maximum, amount_of_words)
                minimum = min(minimum, amount_of_words)
            paragraphs = re.split(r"\n\s*\n", content)
            paragraphs = [p.strip() for p in paragraphs if p.strip()]
            # paragraphs = [p for p in paragraphs if len(p.split()) > 5]
            paragraph_count = len(paragraphs)
            d[fname].append(paragraph_count)
            paragraph_len_in_words = [len(p) for p in paragraphs]
            if len(paragraph_len_in_words) > 0:
                avg = sum(paragraph_len_in_words) / len(paragraph_len_in_words)
                d[fname].append(avg)
                author_to_books[fname].append(content)
            else:
                avg = 0
                d[fname].append(avg)
                author_to_books[fname].append(content)
    stats_dict = {
        k: {
            "word_count": v[0],
            "paragraph_count": v[1],
            "average_paragraph_length": v[2]
        }
        for k,v in d.items()
    }
    for key, value in author_to_books.items():
        stats_dict[key]["document"] = value


    stats_df = pd.DataFrame.from_dict(stats_dict, orient="index").reset_index()
    stats_df = stats_df.rename(columns={"index": "filename"})
    df = df.merge(stats_df, on="filename", how="left")
    print("Maximum: ", maximum)
    print("Minimum: ", minimum)
    return df



def downsample_with_word_constraint(df, TARGET_AUTHORS=269):
    df = df.copy()
    # Ensure word count exists
    if "word_count" not in df.columns:
        df["word_count"] = df["document"].fillna("").str.split().str.len()

    # Author-level stats
    author_stats = (
        df.groupby(["author_id", "gender_x"], as_index=False)
        .agg(
            document_count=("author_id", "size"),
            total_words=("word_count", "sum"),
        )
    )

    # Bucket author contribution
    doc_bins = [0, 5, 10, 20, 50, 100, np.inf]
    doc_labels = ["1-5", "6-10", "11-20", "21-50", "51-100", "101+"]

    # cut for sorting a dataframe into bins.
    author_stats["doc_bucket"] = pd.cut(
        author_stats["document_count"],
        bins=doc_bins,
        labels=doc_labels,
        include_lowest=True
    )

    # Bucket paragraph/document length
    length_bins = [0, 50, 100, 200, 500, 1000, np.inf]
    length_labels = ["1-50", "51-100", "101-200", "201-500", "501-1000", "1001+"]

    female_authors = author_stats[author_stats["gender_x"]=="female"]
    male_authors = author_stats[author_stats["gender_x"]=="male"]
    print("FEMALES")
    print(female_authors)
    print("MALES")
    print(male_authors)
    # get the percentages of documents per class
    female_dist = (
        female_authors["doc_bucket"]
        .value_counts(normalize=True)
        .sort_index()
    )
    male_dist = (
        male_authors["doc_bucket"]
        .value_counts(normalize=True)
        .sort_index()
    )

    target_distribution = female_dist
    bucket_author_targets = (
            target_distribution * TARGET_AUTHORS
    ).round().astype(int)

    print(bucket_author_targets)


    def sample_authors_by_count(authors, n_authors, random_state=42):
        n = min(len(authors), n_authors)

        return authors.sample(
            n=n,
            random_state=random_state
        )

    sampled_authors = []
    for label_value, label_authors in [(0, male_authors), (1, female_authors)]:
        for bucket, n_authors in bucket_author_targets.items():
            bucket_authors = label_authors[
                label_authors["doc_bucket"] == bucket
                ]
            if bucket_authors.empty or n_authors == 0:
                continue
            sampled = sample_authors_by_count(
                bucket_authors,
                n_authors=n_authors,
                random_state=42
            )
            sampled_authors.append(sampled)
    sampled_authors = pd.concat(sampled_authors, ignore_index=True)

    print(sampled_authors.to_csv(sep="\t", index=False))
    sampled_df = df.merge(
        sampled_authors[["author_id", "gender_x"]],
        on=["author_id", "gender_x"],
        how="inner"
    )

    report = (
        sampled_authors
        .groupby(["gender_x", "doc_bucket"], observed=True)
        .agg(
            author_count=("author_id", "nunique"),
            document_count=("document_count", "sum"),
            total_words=("total_words", "sum"),
        )
        .reset_index()
    )
    print("REPORT")
    print(report.to_csv(sep="\t", index=False))
    report.to_csv(f"report_{TARGET_AUTHORS}.tsv", sep="\t",index=False)

    s2 = sampled_authors.groupby("gender_x").agg(
            author_count=("author_id", "nunique"),
            document_count=("document_count", "sum"),
            total_words=("total_words", "sum"),
        )

    print(s2)
    sampled_authors_with_documents = df.merge(
        sampled_authors[["author_id", "gender_x"]],
        on=["author_id", "gender_x"],
        how="inner"
    )

    sampled_authors_with_documents.to_csv(f"sampled_gutenberg_authors_{TARGET_AUTHORS}_authors.tsv",sep="\t", encoding="utf-8", # reddit_train_female_ap
                            index=False, quotechar='"', escapechar='\\', quoting=csv.QUOTE_ALL, lineterminator="\n")

    print(s2.to_csv(sep="\t", index=False))
    s2.to_csv(f"stats_{TARGET_AUTHORS}.tsv", sep="\t",index=False)





if __name__ == "__main__":

    # TODO: Cut einführen, start und beginn minus 10%

    #df = pd.read_csv("C:/Users/phMei/Documents/PhD/Korpora/Project_Gutenberg/nlp_proj/nlp_proj_share/dataset_complete.csv", dtype={"book_id":str, "author_id":str}, index_col=False)
    df = pd.read_csv("dataset_complete_wikidata_gender_compared_final.csv", dtype={"book_id":str, "author_id":str}, index_col=False)
    path = "/nlp_proj/nlp_proj_share/whole_dataset/"
    df["author_x"] = df["author_x"].apply(
        lambda x: " ".join(x.split(", ")[::-1]) if "," in x else x
    )
    files  = glob.glob(path+"*")
    sum_avg = 0
    amount_of_words = 0
    d = create_dict(files)
    df = df[df["author_x"] == df["author_y"]]
    df["filename"] = df["book_id"].map(d)
    print(df)
    df = df[df["gender_y"] == df["gender_x"]]
    # Name Matching!
    # split bei komma und dann drehe den namen um

    #df = df.drop("gender_y")
    print(df)
    df_male = df[df["gender_x"] == "male"]
    df_female = df[df["gender_y"] == "female"]

    print("Male DF")
    print(df_male)

    print("Female DF")
    print(df_female)
    print(df_female.columns)
    df_male = cut_book_content(path, df_male) # create_stats_for_gender
    df_female = cut_book_content(path, df_female)
    print(df_female["word_count"].sum())
    g_f = df_male["author_id"].value_counts()
    g_f_total = g_f.sum()
    print("Average Posts per Author, Female")
    print(g_f_total/g_f.shape[0])

    print("Male Word Count Mean: ", df_male["word_count"].mean())

    print("Female Word Count Mean: ", df_female["word_count"].mean())

    print("Male Word Count Sum: ", df_male["word_count"].sum())

    print("Female Word Count Sum: ", df_female["word_count"].sum())

    print(len(set(df_male["author_x"].to_list())))
    print(len(set(df_female["author_x"].to_list())))


    df_concat = pd.concat([df_male, df_female], axis=0)

    #print(df_concat)
    grouped = group_dataframe_after_authors(df_concat)
    grouped["document"] = grouped["document"].apply(json.dumps)
    print(grouped)
    grouped = grouped[["author_id", "label", "document", "document_count"]]


    grouped.to_csv("concatenated_checked_gutenberg.tsv", sep="\t", index=False, encoding="utf-8",
                     quotechar='"', escapechar='\\', quoting=csv.QUOTE_ALL, lineterminator="\n")

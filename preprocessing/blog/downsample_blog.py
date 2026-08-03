
import csv
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatter

import numpy as np
from matplotlib.ticker import ScalarFormatter


RANDOM_STATE = 42

def group_dataframe(df):
    grouped = (df.groupby("author_id", "label").agg(
        document=("document", list),
        document_count=("document", "size")
    )
    .reset_index()
               )
    grouped = grouped.sort_values("document_count", ascending=False)
    return grouped

def create_author_stats(df):
    author_stats = (
        df.groupby("author_id")
          .agg(
              document_count=("document", "size"),
              total_words=("word_count", "sum")
          )
          .reset_index()
    )
    return author_stats


def downsample_with_word_constraint_close_to_females_repeated_sampling(df, partition, TARGET_WORDS=30_000_000, previously_used_males = None,
                                                     previously_used_females=None, AUTHORS_PER_GENDER=None):
    """
    Sample 50 Million words and split afterwards?
    :param df:
    :param partition:
    :param TARGET_WORDS:
    :param previously_used_males:
    :param previously_used_females:
    :return:
    """
    print(previously_used_females)
    if previously_used_females is None:
        previously_used_females = {}
    if previously_used_males is None:
        previously_used_males = {}
    df = df.copy()
    # Ensure word count exists
    if "word_count" not in df.columns:
        df["word_count"] = df["document"].fillna("").str.split().str.len()

    # Author-level stats
    author_stats = (
        df.groupby(["author_id", "label"], as_index=False)
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

    female_authors = author_stats[author_stats["label"]==1]
    male_authors = author_stats[author_stats["label"]==0]

    # exclude the salready sampled males and females to prohibit they are sampled again

    female_authors = female_authors[
        ~female_authors["author_id"].isin(previously_used_females)
    ].copy()

    male_authors = male_authors[
        ~male_authors["author_id"].isin(previously_used_males)
    ].copy()

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



    sampled_authors = []
    # 1. Sample females by bucket
    TARGET_WORDS_PER_GENDER = TARGET_WORDS//2
    sampled_females = []
    used_female_authors = set()
    female_word_sum = 0

    bucket_order = target_distribution.index.repeat(
        (target_distribution*100).astype(int)
    ).tolist()
    rng = np.random.default_rng(42)

    # use repeated sampling:
    BEST_SAMPLE = None
    BEST_DIFF = float("inf")
    count_of_female_authors = 0

    for seed in range(100):
        rng = np.random.default_rng(seed)

        sampled_females = []
        used_female_authors = previously_used_females #set()
        female_word_sum = 0
        while count_of_female_authors < AUTHORS_PER_GENDER:
            bucket = rng.choice(target_distribution.index,p=target_distribution.values)

            female_bucket = female_authors[
                (female_authors["doc_bucket"] == bucket)
                & (~female_authors["author_id"].isin(used_female_authors))
            ]

            if female_bucket.empty:
                continue

            sampled = female_bucket.sample(
                n=1,
                random_state=int(rng.integers(0, 1_000_000))
            )

            selected_author = sampled.iloc[0]

            sampled_females.append(sampled)
            used_female_authors.add(selected_author["author_id"])
            female_word_sum += selected_author["total_words"]
            count_of_female_authors += 1


        diff = abs(female_word_sum - TARGET_WORDS_PER_GENDER)
        # get the sample with the lowest difference to 5 million words
        if diff < BEST_DIFF:
            BEST_DIFF = diff
            BEST_SAMPLE = sampled_females

    sampled_females = pd.concat(BEST_SAMPLE, ignore_index=True)



    print("female words:", sampled_females["total_words"].sum())
    print("female authors:", sampled_females["author_id"].nunique())



    # 2. Match males to sampled females
    male_pool = male_authors.copy()
    matched_males = []
    used_male_authors = set()

    for _, female in sampled_females.iterrows():
        candidates = male_pool[
            male_pool["doc_bucket"] == female["doc_bucket"]
            ].copy()

        if candidates.empty:
            candidates = male_pool.copy()

        # define the distance between a male and a female authors as the difference between the amount of words.
        candidates["distance"] = (
                candidates["total_words"]
                - female["total_words"]
        ).abs()

        best_match = candidates.sort_values("distance").iloc[0]

        matched_males.append(best_match)
        used_male_authors.add(best_match["author_id"])

        male_pool = male_pool[
            male_pool["author_id"] != best_match["author_id"]
            ]

    matched_males = pd.DataFrame(matched_males)

    sampled_authors = pd.concat(
        [sampled_females, matched_males],
        ignore_index=True
    )

    sampled_documents = df.merge(
        sampled_authors[["author_id", "label"]],
        on=["author_id", "label"],
        how="inner"
    )
    #print(sampled_authors.to_csv(sep="\t", index=False))

    sampled_authors.to_csv(f"sampled_authors_{TARGET_WORDS}_{partition}.tsv", sep="\t")
    sampled_documents.to_csv(
        f"sampled_documents_{TARGET_WORDS}.tsv",
        sep="\t", encoding="utf-8",index=False, quotechar='"', escapechar='\\',
        quoting=csv.QUOTE_ALL, lineterminator="\n"
    )

    sampled_df = df.merge(
        sampled_authors[["author_id", "label"]],
        on=["author_id", "label"],
        how="inner"
    )
    sampled_df.to_csv(f"blog_sampled_{TARGET_WORDS}_{partition}.tsv", sep="\t", encoding="utf-8", index=False,
                          quotechar='"', escapechar='\\', quoting=csv.QUOTE_ALL, lineterminator="\n")
    report = (
        sampled_authors
        .groupby(["label", "doc_bucket"], observed=True)
        .agg(
            author_count=("author_id", "nunique"),
            document_count=("document_count", "sum"),
            total_words=("total_words", "sum"),
        )
        .reset_index()
    )
    print("REPORT")
    print(report.to_csv(sep="\t", index=False))
    report.to_csv(f"blog_report_{TARGET_WORDS}_{partition}.tsv", sep="\t")

    s2 = sampled_authors.groupby("label").agg(
            author_count=("author_id", "nunique"),
            document_count=("document_count", "sum"),
            total_words=("total_words", "sum"),
        )

    print(s2.to_csv(sep="\t", index=False))

    used_female_authors.update(previously_used_females)
    used_male_authors.update(previously_used_males)

    return used_female_authors, used_male_authors





def downsample_with_word_constraint_close_to_females(df, partition, TARGET_WORDS=30_000_000, previously_used_males = None,
                                                     previously_used_females=None):
    """
    Sample 50 Million words and split afterwards?
    :param df:
    :param partition:
    :param TARGET_WORDS:
    :param previously_used_males:
    :param previously_used_females:
    :return:
    """
    if previously_used_females is None:
        previously_used_females = {}
    if previously_used_males is None:
        previously_used_males = {}
    df = df.copy()
    # Ensure word count exists
    if "word_count" not in df.columns:
        df["word_count"] = df["document"].fillna("").str.split().str.len()

    # Author-level stats
    author_stats = (
        df.groupby(["author_id", "label"], as_index=False)
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

    female_authors = author_stats[author_stats["label"]==1]
    male_authors = author_stats[author_stats["label"]==0]

    # exclude the salready sampled males and females to prohibit they are sampled again

    female_authors = female_authors[
        ~female_authors["author_id"].isin(previously_used_females)
    ].copy()

    male_authors = male_authors[
        ~male_authors["author_id"].isin(previously_used_males)
    ].copy()

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



    sampled_authors = []
    # 1. Sample females by bucket
    TARGET_WORDS_PER_GENDER = TARGET_WORDS//2
    sampled_females = []
    used_female_authors = set()
    female_word_sum = 0

    bucket_order = target_distribution.index.repeat(
        (target_distribution*100).astype(int)
    ).tolist()
    rng = np.random.default_rng(42)

    while female_word_sum < TARGET_WORDS_PER_GENDER:
        if len(used_female_authors) >= len(female_authors):
            break

        bucket = rng.choice(
            target_distribution.index,
            p=target_distribution.values
        )

        female_bucket = female_authors[
            (female_authors["doc_bucket"] == bucket)
            & (~female_authors["author_id"].isin(used_female_authors))] # do not the already seen female authors

        if female_bucket.empty:
            continue

        sampled = female_bucket.sample(n=1,random_state=int(rng.integers(0, 1_00_000)))

        sampled_females.append(sampled)

        selected_author = sampled.iloc[0]
        used_female_authors.add(selected_author["author_id"])
        female_word_sum += selected_author["total_words"]

    sampled_females = pd.concat(sampled_females, ignore_index=True)
    print("female words:", sampled_females["total_words"].sum())
    print("female authors:", sampled_females["author_id"].nunique())



    # 2. Match males to sampled females
    male_pool = male_authors.copy()
    matched_males = []
    used_male_authors = set()

    for _, female in sampled_females.iterrows():
        candidates = male_pool[
            male_pool["doc_bucket"] == female["doc_bucket"]
            ].copy()

        if candidates.empty:
            candidates = male_pool.copy()

        # define the distance between a male and a female authors as the difference between the amount of words.
        candidates["distance"] = (
                candidates["total_words"]
                - female["total_words"]
        ).abs()

        best_match = candidates.sort_values("distance").iloc[0]

        matched_males.append(best_match)
        used_male_authors.add(best_match["author_id"])

        male_pool = male_pool[
            male_pool["author_id"] != best_match["author_id"]
            ]

    matched_males = pd.DataFrame(matched_males)

    sampled_authors = pd.concat(
        [sampled_females, matched_males],
        ignore_index=True
    )

    sampled_documents = df.merge(
        sampled_authors[["author_id", "label"]],
        on=["author_id", "label"],
        how="inner"
    )
    #print(sampled_authors.to_csv(sep="\t", index=False))

    sampled_authors.to_csv(f"sampled_authors_{TARGET_WORDS}_{partition}.tsv", sep="\t")
    sampled_documents.to_csv(
        f"sampled_documents_{TARGET_WORDS}.tsv",
        sep="\t", encoding="utf-8",index=False, quotechar='"', escapechar='\\',
        quoting=csv.QUOTE_ALL, lineterminator="\n"
    )

    sampled_df = df.merge(
        sampled_authors[["author_id", "label"]],
        on=["author_id", "label"],
        how="inner"
    )
    sampled_df.to_csv(f"blog_sampled_{TARGET_WORDS}_{partition}.tsv", sep="\t", encoding="utf-8", index=False,
                          quotechar='"', escapechar='\\', quoting=csv.QUOTE_ALL, lineterminator="\n")
    report = (
        sampled_authors
        .groupby(["label", "doc_bucket"], observed=True)
        .agg(
            author_count=("author_id", "nunique"),
            document_count=("document_count", "sum"),
            total_words=("total_words", "sum"),
        )
        .reset_index()
    )
    print("REPORT")
    print(report.to_csv(sep="\t", index=False))
    report.to_csv(f"blog_report_{TARGET_WORDS}_{partition}.tsv", sep="\t")

    s2 = sampled_authors.groupby("label").agg(
            author_count=("author_id", "nunique"),
            document_count=("document_count", "sum"),
            total_words=("total_words", "sum"),
        )

    print(s2.to_csv(sep="\t", index=False))

    s2.to_csv("short_report_{TARGET_WORDS}_{partition}.tsv", sep="\t")

    return used_female_authors, used_male_authors







if __name__ == "__main__":
    """
    Takes in the cleaned blog posts from clean_blog.py and creates a training, validation and test set
    """

    df_male = pd.read_csv("../all_male_blog_posts_cleaned.tsv", sep="\t", encoding="utf-8", index_col=False,
                          quotechar='"', escapechar='\\', quoting=csv.QUOTE_ALL, lineterminator="\n")
    df_female = pd.read_csv("../all_female_blog_posts_cleaned.tsv", sep="\t", encoding="utf-8",
                            index_col=False,quotechar='"', escapechar='\\', quoting=csv.QUOTE_ALL, lineterminator="\n")


    # LABELS: Male=0, Female=1
    print(df_male)
    df_male["label"] = 0
    # columns: author, document, label


    print(df_female)
    df_female["label"] = 1

    df_concat = pd.concat([df_male, df_female])
    print(df_concat)


    #TARGET_AUTHORS = 30_000_000#2200
    #TARGET_WORDS = 50_000_000
    #downsample_with_word_constraint(df_concat, TARGET_AUTHORS)

    TARGET_WORDS = 30_000_000
    female_ids, male_ids = downsample_with_word_constraint_close_to_females(df_concat, "train", TARGET_WORDS)
    train_size = len(female_ids) + len(male_ids)
    val_size = int((len(female_ids)/100)*20)

    # 423,2 20% of train ids per gender

    TARGET_WORDS = 10_000_000
    female_ids_dev, male_ids_dev = downsample_with_word_constraint_close_to_females_repeated_sampling(df_concat, "val", TARGET_WORDS, male_ids,
                                                                                    female_ids, val_size)
    female_ids.update(female_ids_dev)
    male_ids.update(male_ids_dev)

    TARGET_WORDS = 10_000_000
    female_ids_test, male_ids_test = downsample_with_word_constraint_close_to_females_repeated_sampling(df_concat, "test", TARGET_WORDS, male_ids,
                                                                                    female_ids, val_size)

    print(len(female_ids_dev))
    #female_ids_test, male_ids_test = downsample_with_word_constraint_close_to_females(df_concat, "test", TARGET_WORDS)


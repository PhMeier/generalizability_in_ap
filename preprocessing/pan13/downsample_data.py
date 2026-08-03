import csv
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatter
import pickle
import numpy as np
from matplotlib.ticker import ScalarFormatter

"""
Downsamples the training set and creates a training and validation set

"""


def downsample_with_word_constraint_close_to_females(df, partition, TARGET_WORDS=30_000_000, previously_used_males = None,
                                                     previously_used_females=None, percentage=0.8):
    """
    Logic:
    Take the buckets 6-10, 11-20, 21-50 of females and take every author in there.
    Fill then up with authors from 1-5.
    For males: Take the most similar male authors from the same buckets as the females (drops out 51-100 and 101+

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
    # Remove authors which have a word count of zero
    author_stats = author_stats[author_stats["total_words"] > 0]

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
    rng = np.random.default_rng(42)


    base_buckets = ["6-10", "11-20", "21-50",]

    for bucket in base_buckets:
        female_bucket = female_authors[
            female_authors["doc_bucket"] == bucket
        ].copy()

        #n_take = int(len(female_bucket)) # take all and split later, do not use a percentage
        n_take = int(len(female_bucket) * percentage)
        print("N TAKE ", n_take)
        print("Bucket ", bucket)

        female_bucket_sampled = female_bucket.sample(n=n_take, random_state=42)

        sampled_females.append(female_bucket_sampled)
        used_female_authors.update(female_bucket_sampled["author_id"])
        female_word_sum += female_bucket_sampled["total_words"].sum()

    last_bucket = "1-5" # females do not have documents in 51-100 and 101+

    # get the female authors from the last bucket
    female_1_to_5 = (
        female_authors[
            (female_authors["doc_bucket"] == last_bucket)
            & (~female_authors["author_id"].isin(used_female_authors))
            ]
        #.sort_values("total_words", ascending=True) # ascneding or not?
        .copy()
    )
    # fill up until we reach the word limit
    for _, author in female_1_to_5.iterrows():
        candidate_words = author["total_words"]

        if female_word_sum + candidate_words <= TARGET_WORDS_PER_GENDER:
            sampled_females.append(author.to_frame().T)
            used_female_authors.add(author["author_id"])
            female_word_sum += candidate_words

    sampled_females = pd.concat(sampled_females, ignore_index=True)
    print("female words:", sampled_females["total_words"].sum())
    print("female authors:", sampled_females["author_id"].nunique())



    # 2. Match males to sampled females
    male_pool = male_authors.copy()
    matched_males = []
    used_male_authors = set()

    for _, female in sampled_females.iterrows():
        candidates = male_pool[
            male_pool["doc_bucket"] == female["doc_bucket"] # only from the buckets where the females are from
            ].copy()

        if candidates.empty:
            candidates = male_pool.copy()

        # define the distance between a male and a female authors as the difference between the amount of words.
        candidates["distance"] = ( # find the most similar male
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

    sampled_authors.to_csv(f"pan13_no_sort_sampled_authors_{TARGET_WORDS}_{partition}.tsv", sep="\t")
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
    sampled_df.to_csv(f"pan13_sampled_no_sort_{TARGET_WORDS}_{partition}.tsv", sep="\t", encoding="utf-8", index=False,
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
    report.to_csv(f"pan13_report_{TARGET_WORDS}_{partition}.tsv", sep="\t")

    s2 = sampled_authors.groupby("label").agg(
            author_count=("author_id", "nunique"),
            document_count=("document_count", "sum"),
            total_words=("total_words", "sum"),
        )

    print(s2.to_csv(sep="\t", index=False))

    return used_female_authors, used_male_authors

def sample_validation_set(df, previously_used_males, previously_used_females, AUTHORS_PER_GENDER, TARGET_WORDS=6_000_000, partition="val"):
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
    author_stats = author_stats[author_stats["total_words"] > 0]


    female_authors = author_stats[author_stats["label"] == 1]
    male_authors = author_stats[author_stats["label"] == 0]

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
    TARGET_WORDS_PER_GENDER = TARGET_WORDS // 2
    sampled_females = []
    used_female_authors = set()
    female_word_sum = 0

    # sample from the base buckets
    base_buckets = ["6-10", "11-20", "21-50",]
    n_females = 0

    for bucket in base_buckets:
        female_bucket = female_authors[
            female_authors["doc_bucket"] == bucket
        ].copy()

        n_take = int(len(female_bucket)) # take all and split later, do not use a percentage

        female_bucket_sampled = female_bucket.sample(n=n_take, random_state=42)
        n_females += n_take
        sampled_females.append(female_bucket_sampled)
        used_female_authors.update(female_bucket_sampled["author_id"])
        female_word_sum += female_bucket_sampled["total_words"].sum()

    print("NUM FEMALES: ", n_females)
    last_bucket = "1-5" # females do not have documents in 51-100 and 101+

    # get the female authors from the last bucket
    female_1_to_5 = (
        female_authors[
            (female_authors["doc_bucket"] == last_bucket)
            & (~female_authors["author_id"].isin(used_female_authors))
            ]
        #.sort_values("total_words", ascending=True)
        .copy()
    )

    for _, author in female_1_to_5.iterrows():
        candidate_words = author["total_words"]

        #if female_word_sum + candidate_words <= TARGET_WORDS_PER_GENDER:
        if n_females < authors_per_gender:
            sampled_females.append(author.to_frame().T)
            used_female_authors.add(author["author_id"])
            female_word_sum += candidate_words
            n_females += 1
        else:
            break

    sampled_females = pd.concat(sampled_females, ignore_index=True)
    print("female words:", sampled_females["total_words"].sum())
    print("female authors:", sampled_females["author_id"].nunique())
    print("LEN SAMPLED FEMALES ", len(sampled_females))

    """
    # fill up until we reach the word limit
    for _, author in female_1_to_5.iterrows():
        candidate_words = author["total_words"]

        if female_word_sum + candidate_words <= TARGET_WORDS_PER_GENDER:
            sampled_females.append(author.to_frame().T)
            used_female_authors.add(author["author_id"])
            female_word_sum += candidate_words

    sampled_females = pd.concat(sampled_females, ignore_index=True)
    print("female words:", sampled_females["total_words"].sum())
    print("female authors:", sampled_females["author_id"].nunique())
    """

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
    # print(sampled_authors.to_csv(sep="\t", index=False))

    sampled_authors.to_csv(f"sampled_authors_no_sort_{TARGET_WORDS}_{partition}.tsv", sep="\t")
    sampled_documents.to_csv(
        f"sampled_documents_{TARGET_WORDS}.tsv",
        sep="\t", encoding="utf-8", index=False, quotechar='"', escapechar='\\',
        quoting=csv.QUOTE_ALL, lineterminator="\n"
    )

    sampled_df = df.merge(
        sampled_authors[["author_id", "label"]],
        on=["author_id", "label"],
        how="inner"
    )
    sampled_df.to_csv(f"pan13_sampled_no_sort_{TARGET_WORDS}_{partition}.tsv", sep="\t", encoding="utf-8", index=False,
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
    report.to_csv(f"pan13_report_no_sort_{TARGET_WORDS}_{partition}.tsv", sep="\t")

    s2 = sampled_authors.groupby("label").agg(
        author_count=("author_id", "nunique"),
        document_count=("document_count", "sum"),
        total_words=("total_words", "sum"),
    )

    print(s2.to_csv(sep="\t", index=False))

    return used_female_authors, used_male_authors


if __name__ == "__main__":
    """
    Take the input files created from html_parsing.py
    """
    df_male = pd.read_csv("../pan13/pan_13_train_male_preprocessed_masked_url.tsv", sep="\t", encoding="utf-8", index_col=False,
                          quotechar='"', escapechar='\\', quoting=csv.QUOTE_ALL, lineterminator="\n")
    df_female = pd.read_csv("../pan13/pan_13_train_female_preprocessed_masked_url.tsv", sep="\t", encoding="utf-8",
                            index_col=False,quotechar='"', escapechar='\\', quoting=csv.QUOTE_ALL, lineterminator="\n")

    df_male["label"] = 0
    df_female["label"] = 1

    df_concat = pd.concat([df_male, df_female])

    TARGET_AUTHORS = 30_000_000#2200
    TARGET_WORDS = 30_000_000
    #downsample_with_word_constraint(df_concat, TARGET_AUTHORS)
    #TARGET_WORDS = 6_000_000
    #female_ids_dev, c = downsample_with_word_constraint_close_to_females(df_concat, "val", TARGET_WORDS)

    # make a complete one and then separate later

    TARGET_WORDS = 30_000_000
    female_ids, male_ids = downsample_with_word_constraint_close_to_females(df_concat, "train", TARGET_WORDS)

    #val_size = int(((len(female_ids)/100))*20)
    """
    with open("female_ids_no_sort.pickle", "wb") as f:
        pickle.dump(female_ids, f)
    with open("male_ids_no_sort.pickle", "wb") as f:
        pickle.dump(male_ids, f)


    with open("female_ids.pickle", "rb") as f:
        female_ids = pickle.load(f)
    with open("male_ids.pickle", "rb") as f:
        male_ids=pickle.load(f)
    """
    print(len(male_ids))
    TARGET_WORDS = 6_000_000
    authors_per_gender = (len(male_ids)/100)*20
    # downsample it after the amount of authors

    sample_validation_set(df_concat, male_ids, female_ids, authors_per_gender, TARGET_WORDS)

import csv
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatter
import pickle
import numpy as np
from matplotlib.ticker import ScalarFormatter


def downsample_with_word_constraint_close_to_females_train(df, partition, minimum_authors, TARGET_WORDS=30_000_000,
                                                           previously_used_males=None,
                                                           previously_used_females=None):
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
        df.groupby(["author", "document_count", "label"], as_index=False)
        .agg(
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

    female_authors = author_stats[author_stats["label"] == 1]
    male_authors = author_stats[author_stats["label"] == 0]

    # exclude the salready sampled males and females to prohibit they are sampled again
    female_authors = female_authors[
        ~female_authors["author"].isin(previously_used_females)
    ].copy()

    male_authors = male_authors[
        ~male_authors["author"].isin(previously_used_males)
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

    # target_distribution = pd.Series(
    #    TARGET_BUCKET_DIST_DICT_NEW,
    #    dtype=float
    # )

    target_distribution = female_dist
    target_distribution = target_distribution.reindex(doc_labels)

    sampled_authors = []
    # 1. Sample females by bucket
    TARGET_WORDS_PER_GENDER = TARGET_WORDS // 2
    sampled_females = []
    used_female_authors = set()
    female_word_sum = 0

    rng = np.random.default_rng(42)
    # Take every female author from these buckets
    base_buckets = ["1-5", "6-10", "11-20", "21-50", "51-100"]

    for bucket in base_buckets:
        female_bucket = female_authors[
            female_authors["doc_bucket"] == bucket
            ].copy()
        # 80 percent of this bucket
        n_take = int(len(female_bucket) * 0.85)
        print("BUCKET: ", bucket)
        print("Train - n_take: ", n_take)

        female_bucket_sampled = female_bucket.sample(n=n_take, random_state=42)

        sampled_females.append(female_bucket_sampled)
        used_female_authors.update(female_bucket_sampled["author"])
        female_word_sum += female_bucket_sampled["total_words"].sum()

    last_bucket = "101+"

    female_101 = (
        female_authors[
            (female_authors["doc_bucket"] == last_bucket)
            & (~female_authors["author"].isin(used_female_authors))
            ]
        # .sort_values("total_words", ascending=True)
        .copy()

    )
    print("101_female")
    print(female_101)
    # check that we have at least minimum_authors

    N_RUNS = 200

    best_sampled_females = None
    best_used_female_authors = None
    best_word_sum = None
    best_score = float("inf")

    for _, author in female_101.iterrows():
        candidate_words = author["total_words"]

        if female_word_sum + candidate_words <= TARGET_WORDS_PER_GENDER:
            sampled_females.append(author.to_frame().T)
            used_female_authors.add(author["author"])
            female_word_sum += candidate_words


    sampled_females = pd.concat(sampled_females, ignore_index=True)

    print("female words:", sampled_females["total_words"].sum())
    print("female authors:", sampled_females["author"].nunique())
    print(sampled_females["doc_bucket"].value_counts().sort_index())

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
        used_male_authors.add(best_match["author"])

        male_pool = male_pool[
            male_pool["author"] != best_match["author"]
            ]

    matched_males = pd.DataFrame(matched_males)

    print("male words:", matched_males["total_words"].sum())
    print("male authors:", matched_males["author"].nunique())
    print(sampled_females["doc_bucket"].value_counts().sort_index())

    sampled_authors = pd.concat(
        [sampled_females, matched_males],
        ignore_index=True
    )

    sampled_documents = df.merge(
        sampled_authors[["author", "label"]],
        on=["author", "label"],
        how="inner"
    )
    #print(sampled_authors.to_csv(sep="\t", index=False))

    sampled_authors.to_csv(f"reddit_sampled_authors_no_sort_{TARGET_WORDS}_{partition}.tsv", sep="\t")
    sampled_documents.to_csv(
        f"sampled_documents_{TARGET_WORDS}.tsv",
        sep="\t", encoding="utf-8", index=False, quotechar='"', escapechar='\\',
        quoting=csv.QUOTE_ALL, lineterminator="\n"
    )

    sampled_df = df.merge(
        sampled_authors[["author", "label"]],
        on=["author", "label"],
        how="inner"
    )
    sampled_df.to_csv(f"reddit_sampled_no_sort_{TARGET_WORDS}_{partition}.tsv", sep="\t", encoding="utf-8", index=False,
                      quotechar='"', escapechar='\\', quoting=csv.QUOTE_ALL, lineterminator="\n")
    report = (
        sampled_authors
        .groupby(["label", "doc_bucket"], observed=True)
        .agg(
            author_count=("author", "nunique"),
            document_count=("document_count", "sum"),
            total_words=("total_words", "sum"),
        )
        .reset_index()
    )
    print("REPORT")
    print(report.to_csv(sep="\t", index=False))
    report.to_csv(f"reddit_report_no_sort_{TARGET_WORDS}_{partition}.tsv", sep="\t")

    s2 = sampled_authors.groupby("label").agg(
        author_count=("author", "nunique"),
        document_count=("document_count", "sum"),
        total_words=("total_words", "sum"),
    )

    print(s2.to_csv(sep="\t", index=False))

    return used_female_authors, used_male_authors


def downsample_with_word_constraint_close_to_females(df, partition, authors_per_gender, TARGET_WORDS=30_000_000,
                                                     previously_used_males=None,
                                                     previously_used_females=None):
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
        df.groupby(["author", "document_count", "label"], as_index=False)
        .agg(
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

    female_authors = author_stats[author_stats["label"] == 1]
    male_authors = author_stats[author_stats["label"] == 0]

    # exclude the salready sampled males and females to prohibit they are sampled again
    female_authors = female_authors[
        ~female_authors["author"].isin(previously_used_females)
    ].copy()

    male_authors = male_authors[
        ~male_authors["author"].isin(previously_used_males)
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

    # target_distribution = pd.Series(
    #    TARGET_BUCKET_DIST_DICT_NEW,
    #    dtype=float
    # )

    target_distribution = female_dist
    target_distribution = target_distribution.reindex(doc_labels)

    sampled_authors = []
    # 1. Sample females by bucket
    TARGET_WORDS_PER_GENDER = TARGET_WORDS // 2
    sampled_females = []
    used_female_authors = set()
    female_word_sum = 0

    rng = np.random.default_rng(42)
    # Take every female author from these buckets
    base_buckets = ["1-5", "6-10", "11-20", "21-50", "51-100"]
    female_author_count = 0
    for bucket in base_buckets:
        female_bucket = female_authors[
            female_authors["doc_bucket"] == bucket
            ].copy()

        n_take = int(len(female_bucket) * 1)
        print("BUCKET ", bucket)
        print("n take ", n_take)

        female_bucket_sampled = female_bucket.sample(n=n_take, random_state=42)
        female_author_count += n_take

        sampled_females.append(female_bucket_sampled)
        used_female_authors.update(female_bucket_sampled["author"])
        female_word_sum += female_bucket_sampled["total_words"].sum()

    print("After base buckets")
    # print("female words:", sampled_females["total_words"].sum())
    # print("female authors:", sampled_females["author"].nunique())
    print("LEN SAMPLED FEMALES ", len(sampled_females))
    print("FEMALE AUTHOR COUNTS ", female_author_count)
    print("authors per gender: ", authors_per_gender)
    last_bucket = "101+"
    N_RUNS = 200

    best_extra_authors = None
    best_word_sum = None
    best_difference = float("inf")

    # Values already accumulated from the lower buckets
    base_author_count = female_author_count
    base_word_sum = female_word_sum
    base_used_authors = used_female_authors.copy()
    authors_needed = authors_per_gender - base_author_count

    if authors_needed < 0:
        raise ValueError(
            f"The base buckets already contain {base_author_count} authors, "
            f"which exceeds the target of {authors_per_gender}."
        )
    female_101_pool = female_authors[
        (female_authors["doc_bucket"] == last_bucket)
        & (~female_authors["author"].isin(base_used_authors))
    ].copy()

    if len(female_101_pool) < authors_needed:
        raise ValueError(
            f"Need {authors_needed} additional female authors, but only "
            f"{len(female_101_pool)} are available in {last_bucket}."
        )
    for seed in range(N_RUNS):
        # Select exactly the required number of authors in this run
        sampled_101_run = female_101_pool.sample(
            n=authors_needed,
            random_state=seed
        )

        word_sum_run = (
                base_word_sum
                + sampled_101_run["total_words"].sum()
        )

        difference = abs(
            TARGET_WORDS_PER_GENDER - word_sum_run
        )

        if difference < best_difference:
            best_difference = difference
            best_extra_authors = sampled_101_run.copy()
            best_word_sum = word_sum_run

    print("Best word sum ", best_word_sum)
    if best_extra_authors is None:
        raise RuntimeError("No valid sample was produced.")

    sampled_females.append(best_extra_authors)

    used_female_authors.update(
        best_extra_authors["author"]
    )

    female_author_count += len(best_extra_authors)
    female_word_sum = best_word_sum
    sampled_females = pd.concat(
        sampled_females,
        ignore_index=True
    )

    print("Female authors:", female_author_count)
    print("Female words:", female_word_sum)
    print("Difference from target:", best_difference)


    #sampled_females = pd.concat(sampled_females, ignore_index=True)

    print("female words:", sampled_females["total_words"].sum())
    print("female authors:", sampled_females["author"].nunique())
    print(sampled_females["doc_bucket"].value_counts().sort_index())
    print("FEMALE AUTHOR COUNTS ", female_author_count)

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
        used_male_authors.add(best_match["author"])

        male_pool = male_pool[
            male_pool["author"] != best_match["author"]
            ]

    matched_males = pd.DataFrame(matched_males)

    print("male words:", matched_males["total_words"].sum())
    print("male authors:", matched_males["author"].nunique())
    #print(sampled_females["doc_bucket"].value_counts().sort_index())

    sampled_authors = pd.concat(
        [sampled_females, matched_males],
        ignore_index=True
    )

    sampled_documents = df.merge(
        sampled_authors[["author", "label"]],
        on=["author", "label"],
        how="inner"
    )
    #print(sampled_authors.to_csv(sep="\t", index=False))

    sampled_authors.to_csv(f"reddit_sampled_authors_no_sort_{TARGET_WORDS}_{partition}.tsv", sep="\t")
    sampled_documents.to_csv(
        f"sampled_documents_no_sort_{TARGET_WORDS}.tsv",
        sep="\t", encoding="utf-8", index=False, quotechar='"', escapechar='\\',
        quoting=csv.QUOTE_ALL, lineterminator="\n"
    )

    sampled_df = df.merge(
        sampled_authors[["author", "label"]],
        on=["author", "label"],
        how="inner"
    )
    sampled_df.to_csv(f"reddit_sampled_no_sort_{TARGET_WORDS}_{partition}.tsv", sep="\t", encoding="utf-8", index=False,
                      quotechar='"', escapechar='\\', quoting=csv.QUOTE_ALL, lineterminator="\n")
    report = (
        sampled_authors
        .groupby(["label", "doc_bucket"], observed=True)
        .agg(
            author_count=("author", "nunique"),
            document_count=("document_count", "sum"),
            total_words=("total_words", "sum"),
        )
        .reset_index()
    )
    print("REPORT")
    print(report.to_csv(sep="\t", index=False))
    report.to_csv(f"reddit_report_no_sort_{TARGET_WORDS}_{partition}.tsv", sep="\t")

    s2 = sampled_authors.groupby("label").agg(
        author_count=("author", "nunique"),
        document_count=("document_count", "sum"),
        total_words=("total_words", "sum"),
    )

    print(s2.to_csv(sep="\t", index=False))
    used_female_authors.update(previously_used_females)
    used_male_authors.update(previously_used_males)

    return used_female_authors, used_male_authors



def downsample_test_data_with_word_constraint_close_to_females(df, df_test, partition, authors_per_gender, TARGET_WORDS=30_000_000,
                                                     previously_used_males=None,
                                                     previously_used_females=None):
    """
    Uses the buckets smaller than 101+ to sample test data. Uses the bucket 101+ from training to fill up the
    :param df:
    :param partition:
    :param authors_per_gender:
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
    df_test = df_test.copy()
    # Ensure word count exists
    if "word_count" not in df.columns:
        df["word_count"] = df["document"].fillna("").str.split().str.len()
    if "word_count" not in df.columns:
        df_test["word_count"] = df_test["document"].fillna("").str.split().str.len()

    # Author-level stats
    author_stats = (
        df.groupby(["author", "document_count", "label"], as_index=False)
        .agg(
            total_words=("word_count", "sum"),
        )
    )
    df_test = df_test.rename(columns={"author_id": "author"})
    author_stats_test = (
        df_test.groupby(["author", "label"], as_index=False)
        .agg(
            total_words=("word_count", "sum"),
            document_count=("author", "size"),
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

    author_stats_test["doc_bucket"] = pd.cut(
        author_stats_test["document_count"],
        bins=doc_bins,
        labels=doc_labels,
        include_lowest=True
    )

    # Bucket paragraph/document length

    female_authors = author_stats[author_stats["label"] == 1]
    male_authors = author_stats[author_stats["label"] == 0]

    female_authors_test = author_stats_test[author_stats_test["label"] == 1]
    male_authors_test = author_stats_test[author_stats_test["label"] == 0]

    # exclude the already sampled males and females to prohibit they are sampled again
    female_authors = female_authors[
        ~female_authors["author"].isin(previously_used_females)
    ].copy()

    female_authors_test = female_authors_test[
        ~female_authors_test["author"].isin(previously_used_females)
    ].copy()


    male_authors = male_authors[
        ~male_authors["author"].isin(previously_used_males)
    ].copy()

    male_authors_test = male_authors_test[
        ~male_authors_test["author"].isin(previously_used_males)
    ].copy()


    # replace the 101+ rows in test with the ones from training
    test_without_101 = male_authors_test[
        male_authors_test["doc_bucket"] != "101+"
        ]

    # Sample the same number of 101+ authors from the source dataframe
    training_101 = male_authors[
        male_authors["doc_bucket"] == "101+"
        ]

    male_authors_test = pd.concat(
        [test_without_101, training_101],
        ignore_index=True
    )

    # also for the females
    # delete rows with 101+ in test
    test_without_101_female = female_authors_test[
        female_authors_test["doc_bucket"] != "101+"
    ]

    # Sample the same number of 101+ authors from the source dataframe
    training_101_fem = female_authors[
        female_authors["doc_bucket"] == "101+"
        ]

    # combine the test data with the training 101+
    female_authors_test = pd.concat(
        [test_without_101_female, training_101_fem ],
        ignore_index=True
    )

    # get the percentages of documents per class
    female_dist = (
        female_authors["doc_bucket"]
        .value_counts(normalize=True)
        .sort_index()
    )

    female_dist_test = (
        female_authors_test["doc_bucket"]
        .value_counts(normalize=True)
        .sort_index()
    )


    # target_distribution = pd.Series(
    #    TARGET_BUCKET_DIST_DICT_NEW,
    #    dtype=float
    # )

    target_distribution = female_dist
    target_distribution = target_distribution.reindex(doc_labels)

    sampled_authors = []
    # 1. Sample females by bucket
    TARGET_WORDS_PER_GENDER = TARGET_WORDS // 2
    sampled_females = []
    used_female_authors = set()
    female_word_sum = 0

    rng = np.random.default_rng(42)
    # Take every female author from these buckets
    base_buckets = ["1-5", "6-10", "11-20", "21-50", "51-100"]
    female_author_count = 0
    for bucket in base_buckets:
        # changed to test
        female_bucket = female_authors_test[
            female_authors_test["doc_bucket"] == bucket
            ].copy()

        n_take = int(len(female_bucket) * 1)
        print("BUCKET ", bucket)
        print("n take ", n_take)

        female_bucket_sampled = female_bucket.sample(n=n_take, random_state=42)
        female_author_count += n_take

        sampled_females.append(female_bucket_sampled)
        used_female_authors.update(female_bucket_sampled["author"])
        female_word_sum += female_bucket_sampled["total_words"].sum()
    #print("FEM AUTHOR COUNT: ", female_author_count)
    print("After base buckets")
    # print("female words:", sampled_females["total_words"].sum())
    # print("female authors:", sampled_females["author"].nunique())
    print("LEN SAMPLED FEMALES ", len(sampled_females))
    print("FEMALE AUTHOR COUNTS ", female_author_count)


    last_bucket = "101+"
    # replaced 101+ previously with instances from train
    female_101 = (
        female_authors_test[
            (female_authors_test["doc_bucket"] == last_bucket)
            & (~female_authors_test["author"].isin(used_female_authors))
            ]
        .sort_values("total_words", ascending=True)
        .copy()

    )
    for _, author in female_101.iterrows():
        candidate_words = author["total_words"]

        # if female_word_sum + candidate_words <= TARGET_WORDS_PER_GENDER:
        if female_author_count < authors_per_gender:
            sampled_females.append(author.to_frame().T)
            used_female_authors.add(author["author"])
            female_word_sum += candidate_words
            female_author_count += 1

    sampled_females = pd.concat(sampled_females, ignore_index=True)


    print("female words:", sampled_females["total_words"].sum())
    print("female authors:", sampled_females["author"].nunique())
    #print(sampled_females["doc_bucket"].value_counts().sort_index())
    print("FEMALE AUTHOR COUNTS ", female_author_count)

    # 2. Match males to sampled females
    male_pool = male_authors_test.copy()
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
        used_male_authors.add(best_match["author"])

        male_pool = male_pool[
            male_pool["author"] != best_match["author"]
            ]

    matched_males = pd.DataFrame(matched_males)

    print("male words:", matched_males["total_words"].sum())
    print("male authors:", matched_males["author"].nunique())
    print(sampled_females["doc_bucket"].value_counts().sort_index())

    sampled_authors = pd.concat(
        [sampled_females, matched_males],
        ignore_index=True
    )

    sampled_documents = df.merge(
        sampled_authors[["author", "label"]],
        on=["author", "label"],
        how="inner"
    )
    #print(sampled_authors.to_csv(sep="\t", index=False))

    sampled_authors.to_csv(f"reddit_sampled_authors_no_sort_{TARGET_WORDS}_{partition}.tsv", sep="\t")
    sampled_documents.to_csv(
        f"sampled_documents_no_sort_{TARGET_WORDS}.tsv",
        sep="\t", encoding="utf-8", index=False, quotechar='"', escapechar='\\',
        quoting=csv.QUOTE_ALL, lineterminator="\n"
    )

    sampled_df = df.merge(
        sampled_authors[["author", "label"]],
        on=["author", "label"],
        how="inner"
    )
    sampled_df.to_csv(f"reddit_sampled_no_sort_{TARGET_WORDS}_{partition}.tsv", sep="\t", encoding="utf-8", index=False,
                      quotechar='"', escapechar='\\', quoting=csv.QUOTE_ALL, lineterminator="\n")
    report = (
        sampled_authors
        .groupby(["label", "doc_bucket"], observed=True)
        .agg(
            author_count=("author", "nunique"),
            document_count=("document_count", "sum"),
            total_words=("total_words", "sum"),
        )
        .reset_index()
    )
    print("REPORT")
    print(report.to_csv(sep="\t", index=False))
    report.to_csv(f"reddit_report_no_sort_{TARGET_WORDS}_{partition}.tsv", sep="\t")

    s2 = sampled_authors.groupby("label").agg(
        author_count=("author", "nunique"),
        document_count=("document_count", "sum"),
        total_words=("total_words", "sum"),
    )

    print(s2.to_csv(sep="\t", index=False))

    return used_female_authors, used_male_authors



if __name__ == "__main__":

    # reddit_train_male_ap
    # reddit_train_female_ap
    # test_male_ap
    # test_female_ap
    df_male = pd.read_csv("./reddit_train_male_ap.tsv", sep="\t", encoding="utf-8", index_col=False,
                          # reddit_train_male_ap
                          quotechar='"', escapechar='\\', quoting=csv.QUOTE_ALL, lineterminator="\n")
    df_female = pd.read_csv("./reddit_train_female_ap.tsv", sep="\t", encoding="utf-8",  # reddit_train_female_ap
                            index_col=False, quotechar='"', escapechar='\\', quoting=csv.QUOTE_ALL, lineterminator="\n")

    # df_male = df_male.head()
    # df_female = df_female.head()

    df_concat = pd.concat([df_male, df_female])

    print(df_concat)
    print(df_concat.columns)

    print("############################################################\n")
    print("Target 30_000_000")

    TARGET_WORDS = 30_000_000  # 42_000_000
    #female_ids, male_ids = downsample_with_word_constraint_close_to_females_train(df_concat, "train", TARGET_WORDS)


    #with open("female_ids.pickle", "wb") as f:
    #    pickle.dump(female_ids, f)
    #with open("male_ids.pickle", "wb") as f:
    #    pickle.dump(male_ids, f)

    with open("female_ids.pickle", "rb") as f:
        female_ids = pickle.load(f)

    with open("male_ids.pickle", "rb") as f:
         male_ids = pickle.load(f)

    authors_per_gender = int((len(female_ids) / 100) * 20)
    TARGET_WORDS = 2_000_000

    # change it so it samples the amount of authors!
    # we need 20% of the authors contained in the training data
    female_ids_val, male_ids_val =  downsample_with_word_constraint_close_to_females(df_concat, "val", authors_per_gender, TARGET_WORDS, male_ids,
                                                     female_ids)
    male_ids.update(male_ids_val)
    female_ids.update(female_ids_val)

    # sample test data
    # Take the buckets smaller than 101+ and use the 101+ from the train distribution
    df_male = pd.read_csv("/reddit_pandora_male_test.tsv", sep="\t", encoding="utf-8", index_col=False,
                          quotechar='"', escapechar='\\', quoting=csv.QUOTE_ALL, lineterminator="\n")
    df_female = pd.read_csv("/reddit_pandora_female_test.tsv", sep="\t", encoding="utf-8",
                            index_col=False,quotechar='"', escapechar='\\', quoting=csv.QUOTE_ALL, lineterminator="\n")

    print(df_male.columns)
    print(df_female.columns)
    print(df_male.head)
    print(df_female.head)
    df_male = df_male.rename(columns={"gender": "label"})
    df_female = df_female.rename(columns={"gender": "label"})
    df_concat_test = pd.concat([df_male, df_female])
    df_concat_test["label"] = df_concat_test["label"].map({
        "m": 0,
        "f": 1
    })

    downsample_test_data_with_word_constraint_close_to_females(df_concat, df_concat_test, "test", authors_per_gender,
                                                               TARGET_WORDS,
                                                               male_ids, female_ids)

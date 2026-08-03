import csv
import pandas as pd

def group_after_author(df):
    grouped = (
        df.groupby("author")
        .agg({
            "body":list,
        })
        .reset_index()
    )
    return grouped


def group_after_author_with_document_counts(df):
    grouped = (
        df.groupby("author", as_index=False)
        .agg(
            document=("body", " ".join),
            document_count=("body", "size"),
        )
    )
    return grouped

def create_word_count(data):
    length = []
    maximum = 0
    minimum = 99999
    for line in data:
        if type(line) == float:
            #print(line)
            continue
        words = line.strip().split()
        maximum = max(maximum, len(words))
        minimum = min(minimum, len(words))
        length.append(len(words))
    print("Average length: ", sum(length)/len(length))
    print("Maximum: ", maximum)
    print("Minimum: ", minimum)
    print("Total amount of words: ", sum(length))


def count_posts_per_author(df_male, df_female, author_col="author", doc_col="body"):
    g_m = df_male[author_col].value_counts()
    #print(g_m)
    print("Total Amount of posts male: ", g_m.sum())


    #g_m.to_csv("pan_2014_blog_posts_counts_male.csv")
    g_f = df_female[author_col].value_counts()
    #print(g_f)
    print("Total amount of posts female:", g_f.sum())
    #g_f.to_csv("pan_2014_blog_posts_counts_female.csv")

    # Average posts per Author
    print("Average Posts per Author, Male")
    g_m = df_male[author_col].value_counts()
    g_m_total = g_m.sum()
    print(g_m_total/g_m.shape[0])
    print("Total amount of Posts: ", g_m_total)

    #g_m.to_csv("blog_post_counts_male.csv")
    g_f = df_female[author_col].value_counts()
    g_f_total = g_f.sum()
    print("Average Posts per Author, Female")
    print(g_f_total/g_f.shape[0])
    print("Total amount of Posts: ", g_f_total)

    male_text = df_male[doc_col].to_list()
    female_text = df_female[doc_col].to_list()
    #print(male_text)
    create_word_count(male_text)
    create_word_count(female_text)



if __name__ == "__main__":
    df_authors = pd.read_csv("../data/reddit/mf_gender_label.csv", sep=",")
    df_comments = pd.read_csv("../data/reddit/mf_comments_all_noq.csv", sep=",")
    print(df_authors)
    df_authors_non_ambigous = df_authors[df_authors["is_ambiguous_gender"] == False] # do not take ambigous gender into account
    print(df_authors_non_ambigous)
    df_male = df_authors_non_ambigous[df_authors_non_ambigous["is_male"] == True]

    df_female = df_authors_non_ambigous[df_authors_non_ambigous["is_female"] == True]
    print(df_male) # 21510
    print(df_female) #13080
    merged_male = pd.merge(df_male, df_comments, on="author", how="inner")
    merged_female = pd.merge(df_female, df_comments, on="author", how="inner")
    #merged_male = merged_male.head()
    #merged_female = merged_female.head()
    merged_male_grouped = group_after_author_with_document_counts(merged_male)
    merged_female_grouped = group_after_author_with_document_counts(merged_female)

    merged_male_grouped.to_csv("reddit_train_male_ap.tsv", sep="\t",quotechar='"', escapechar='\\', quoting=csv.QUOTE_ALL,
                            lineterminator="\n")

    merged_female_grouped.to_csv("reddit_train_female_ap.tsv", sep="\t",quotechar='"', escapechar='\\', quoting=csv.QUOTE_ALL,
                            lineterminator="\n")

    merged = pd.concat([merged_male, merged_female])
    merged.to_csv("pandora_train_authors_to_comments.csv", sep="\t",quotechar='"', escapechar='\\', quoting=csv.QUOTE_ALL,
                            lineterminator="\n")



    # For statistics
    #count_posts_per_author(merged_male, merged_female)
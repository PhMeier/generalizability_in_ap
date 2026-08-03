import csv
import pandas as pd


"""
Match the extracted authors with their posts to insert separator tokens for later

"""



if __name__ == "__main__":
    # author id, label, document (we need author id and label)
    filenames = ["reddit_sampled_no_sort_2000000_test.tsv"]#["reddit_sampled_no_sort_30000000_train.tsv", "reddit_sampled_no_sort_2000000_val.tsv"] #, "reddit_sampled_no_sort_2000000_test.tsv"]
    # author id, comment
    male_tsv = "reddit_train_male.tsv"
    female_tsv = "reddit_train_female.tsv"
    test_male = "reddit/test/reddit_pandora_male_test.tsv"
    test_female = "/test/reddit_pandora_female_test.tsv"
    df_male = pd.read_csv(male_tsv, sep="\t", encoding="utf-8", index_col=False, #reddit_train_male_ap
                              quotechar='"', escapechar='\\', quoting=csv.QUOTE_ALL, lineterminator="\n")
    df_female = pd.read_csv(female_tsv, sep="\t", encoding="utf-8", index_col=False, #reddit_train_male_ap
                              quotechar='"', escapechar='\\', quoting=csv.QUOTE_ALL, lineterminator="\n")
    df_test_m = pd.read_csv(test_male, sep="\t", encoding="utf-8", index_col=False, #reddit_train_male_ap
                              quotechar='"', escapechar='\\', quoting=csv.QUOTE_ALL, lineterminator="\n")
    df_test_f = pd.read_csv(test_female, sep="\t", encoding="utf-8", index_col=False, #reddit_train_male_ap
                              quotechar='"', escapechar='\\', quoting=csv.QUOTE_ALL, lineterminator="\n")
    df_concat = pd.concat([df_male, df_female])
    for fname in filenames:
        outfile = fname.split(".tsv")[0] + "_doc_per_line.tsv"
        df = pd.read_csv(fname, sep="\t", encoding="utf-8", index_col=False,  # reddit_train_male_ap
                                quotechar='"', escapechar='\\', quoting=csv.QUOTE_ALL, lineterminator="\n")
        author_names = df["author"].to_list()
        df_merged = df[["author", "label"]].merge(
            df_concat[["author", "body"]],
            on = "author",
            how="left"
        )
        df_merged.to_csv(outfile, sep="\t",encoding="utf-8", index=False,  # reddit_train_male_ap
                                quotechar='"', escapechar='\\', quoting=csv.QUOTE_ALL, lineterminator="\n")



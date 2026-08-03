import pandas as pd



if __name__ == "__main__":
    df_complete =pd.read_csv("../dataset_complete_wikidata_gender.csv", index_col=False)
    print(df_complete.columns)
    df_complete["gender_x"] = df_complete["gender_x"].map({
        "m": "male",
        "f": "female"
    })
    correct_gender = df_complete[df_complete["gender_x"] == df_complete["gender_y"]]
    print(correct_gender["gender_y"].value_counts())
    authors_gender = (
        correct_gender[["author_id", "author_x", "gender_y"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    print(authors_gender)
    print(authors_gender["gender_y"].value_counts())
    correct_gender.to_csv("../dataset_complete_wikidata_gender_compared_final.csv", index=False)
import requests
import pandas as pd
import time

GENDER_MAP = {
    "Q6581097": "male",
    "Q6581072": "female",
    "Q48270": "non-binary",
    "Q1052281": "trans woman",
    "Q2449503": "trans man",
}
USER_AGENT = "author-gender-project/1.0"

def chunks(items, size=50):
    """
    Divide the author list into chunks
    :param items:
    :param size:
    :return:
    """
    for i in range(0, len(items), size):
        yield items[i:i+size]


def get_json(url, params):
    """
    Send the json request to Wikidata
    :param url:
    :param params:
    :return:
    """
    r = requests.get(url, params=params, headers={"User-Agent": USER_AGENT})
    if r.status_code == 429:
        time.sleep(int(r.headers.get("Retry-After", 30)))
        return get_json(url, params)
    r.raise_for_status()
    return r.json()

def get_qids(authors):
    rows = []
    for author, id in authors:
        data = get_json("https://en.wikipedia.org/w/api.php", {
            "action": "query",
            "titles": author,
            "prop": "pageprops",
            "redirects": 1,
            "format": "json",
        })
        page = list(data["query"]["pages"].values())[0]

        rows.append({
            "author": page.get("title"),
            "wikidata_id": page.get("pageprops", {}).get("wikibase_item"),
            "gutenberg_id": id
        })

        time.sleep(1)
    return pd.DataFrame(rows)

def get_genders(qids):
    rows = []
    for batch in chunks(qids, 1):
        data = get_json("https://www.wikidata.org/w/api.php", {
            "action": "wbgetentities",
            "ids": "|".join(batch),
            "props": "claims",
            "format": "json",
        })

        for qid, entity in data["entities"].items():
            claims = entity.get("claims", {})
            gender = "unknown"

            if "P21" in claims:
                gender_qid = claims["P21"][0]["mainsnak"]["datavalue"]["value"]["id"]
                gender = GENDER_MAP.get(gender_qid, f"other:{gender_qid}")

            rows.append({
                "wikidata_id": qid,
                "gender": gender,

            })

        time.sleep(1)

    return pd.DataFrame(rows)


def read_in_authors(author_df):
    df = pd.read_csv(author_df, index_col=False)
    names = df["author"].to_list()
    author_gutenberg_id = df["author_id"].to_list()
    names_in_correct_order = []
    missed = []
    for n,id_ in zip(names, author_gutenberg_id):
        if "," in n:
            name_ = n.split(",")
            new_name = name_[1].strip() +" "+ name_[0].strip()
            names_in_correct_order.append((new_name ,id_))
        else:
            missed.append(n)

    return list(set(names_in_correct_order)), set(missed)


if __name__ == "__main__":
    authors, missed = read_in_authors("../dataset_complete.csv")
    print(len(authors))
    print("Missed ", len(missed))
    # First, get the ids of the authors in wikidata
    qid_df = get_qids(authors)
    # after you get the ids, get the gender
    gender_df = get_genders(qid_df["wikidata_id"].dropna().tolist())

    result = qid_df.merge(gender_df, on="wikidata_id", how="left")
    result.to_csv("authors_with_gender.csv", index=False)

    print(result)

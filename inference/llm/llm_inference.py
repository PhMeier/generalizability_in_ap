import re
import csv
import json
import sys
import requests
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report



API_URL = "PATH-TO-API"
# neuer key, rotation am 20.05
API_KEY = "YOUR API KEY"  # <-- keep secret!
DEFAULT_MODEL = "GPT OSS 120B"# any model that can follow the prompt
DEFAULT_CONFIDENCE_THRESHOLD = 0.7        # change to whatever you need
MAX_RETRIES = 5
BASE_BACKOFF = 2
# ----------------------------------------------------------------------
# 3️⃣  PROMPT (system + user)
# ----------------------------------------------------------------------
SYSTEM_PROMPT = """
Given the following text, decide whether the author is more likely to be male
or female. Answer only with one label: 0 for male and 1 for female.
Return a json with the following fields:
{
  "gender":      0 | 1,
  "confidence":  0.0 … 1.0,

}
"""

def label_documents_with_retry(document, max_retries=3, sleep_seconds=2):
    last_error = None
    for attempt in range(max_retries):
        try:
            gender, answer = call_llm(document)  # your requests.post(...) function
            return gender, answer, None

        except Exception as e:
            last_error = str(e)
            time.sleep(sleep_seconds * (attempt + 1))

    return None, None, last_error



MAXIMUM_CONTEXT_LENGTH = 131072
def call_llm(document):
    document = document[:MAXIMUM_CONTEXT_LENGTH]
    x = requests.post(
        "https://PATH-TO-API/api/v1/chat/completions",
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer [INSERT-YOUR-TOKEN]"
        },
        json={
            "messages": [
                {
                    "role": "user",
                    "content": f"""
                    Given the following concatenated documents of an author, decide whether the author is more likely to be male
                    or female. Answer only with one label: 0 for male and 1 for female.
                    Return a raw json with the following fields:
                    {{"gender":      0 | 1}}
                    and the input documents:
                    {document}
                     """
                }
            ],
            "model": "GPT OSS 120B",
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
            "seed":42,
            "top_p": 0.01 + 0.0
        }
    )
    try:
        s = x.json()
    except Exception as e:
        raise ValueError(f"Response is not JSON: {e}")
    if "choices" not in s:
        raise ValueError(f"No choices in response: {s}")
    try:
        content = s["choices"][0]["message"]["content"]
    except Exception as e:
        raise ValueError(f"Unexpected response structure: {s}") from e

    content = content.strip()
    content = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        content.strip(),
        flags=re.MULTILINE
    ).strip()

    try:
        answer = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not parse content as JSON: {content}") from e

    if "gender" not in answer:
        raise ValueError(f"No gender field in answer: {answer}")
    gender = int(answer["gender"])
    if gender not in [0, 1]:
        raise ValueError(f"Invalid gender value: {gender}")

    return int(gender), answer


if __name__ == "__main__":

    f = sys.argv[1]
    data = sys.argv[2]
    df = pd.read_csv(f, sep="\t",quotechar='"', escapechar='\\', quoting=csv.QUOTE_ALL,lineterminator="\n",
                     index_col=False)

    ids = df["author_id"].to_list()
    document = df["document"].to_list()
    gold_labels = df["label"].to_list()
    prediction = []
    confidence = []

    for i in range(3):
        prediction = []
        confidence = []
        errors = []
        for id_, doc, glabl in zip(ids, document, gold_labels):
            output_file = "./" + f.split("/")[-1].split(".tsv")[
                0] + f"{data}_predictions_run_{i}.tsv"  # f"test_pan_14_stratified_run_{i}.tsv"
            # print(id_, doc)
            # added for pan 13:
            pred, resp, last_error = label_documents_with_retry(doc)
            if last_error is not None:
                errors.append({
                    "Author_id": id_,
                    "Gold_label": glabl,
                    "Document": doc,
                    "Error": last_error
                })
                prediction.append(None)
            else:
                prediction.append(pred)
            # confidence.append(resp["confidence"])
            # print(f"Gold: {glabl} |  Prediction: {pred}")
        df_errors = pd.DataFrame(errors)
        df_errors.to_csv(f"{data}_failed_instances_{i}.csv", sep=";", index=False, quoting=csv.QUOTE_ALL)
        df_output = pd.DataFrame({"Prediction": prediction, "Gold_label": gold_labels,
                                  "Author_id": ids, "Document": document})
        df_output.to_csv(output_file, sep=";", index=False, quoting=csv.QUOTE_ALL, lineterminator="\n")
        if None not in prediction:
            print("###### RUN ########")
            print("Accuracy: ", accuracy_score(gold_labels, prediction))
            print("Precision: ", precision_score(gold_labels, prediction, average=None))
            print("Recall: ", recall_score(gold_labels, prediction, average=None))
            print("F1 Score: ", f1_score(gold_labels, prediction, average=None))
            print(classification_report(gold_labels, prediction))
            print(confusion_matrix(gold_labels, prediction))


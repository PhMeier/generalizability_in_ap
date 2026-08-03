import argparse
import pandas as pd
import stanza
import sys
import csv
import ast
import json

stanza.download("en")
"""
03/06
Mask the words in the same manner as https://aclanthology.org/2023.tacl-1.80.pdf
for ROberta and XL NET:
https://huggingface.co/docs/transformers/model_doc/roberta
https://huggingface.co/docs/transformers/model_doc/xlnet

"""
# Mask all Nouns, main verbs, adjectives and adverbs
PERT_LE_GRANDE_SCHEME = ["NOUN", "VERB", "ADV", "ADJ"]  # greedy

# In contrast, in our PertLE Lite schema we mask only nouns, which are most likely to carr
# PROPN?
PERT_LE_LITE_SCHEME = ["NOUN"]

PUNCTUATION = [".", ";", "!", "?"]
PLACEHOLDER_MAP = {
    "[URL]": "URLTOKEN",
}
REVERSE_MAP = {v: k for k, v in PLACEHOLDER_MAP.items()}

def protect_placeholders(documents):
    protected = []

    for doc in documents:
        for original, replacement in PLACEHOLDER_MAP.items():
            doc = doc.replace(original, replacement)
        protected.append(doc)

    return protected

def restore_placeholders(documents):
    restored = []

    for doc in documents:
        for replacement, original in REVERSE_MAP.items():
            doc = doc.replace(replacement, original)
        restored.append(doc)

    return restored


def apply_scheme(doc):
    sentences_lite = []
    sentences_grande = []
    for sentence in doc.sentences:
        lite_words = []
        grande_words = []
        # print("Sentence:")
        for word in sentence.words:
            text = word.text
            upos = word.upos
            lite_words.append(
                "<mask>" if upos in PERT_LE_LITE_SCHEME else text
            )
            grande_words.append(
                "<mask>" if upos in PERT_LE_GRANDE_SCHEME else text
            )
        sentences_lite.append(" ".join(lite_words).strip())
        sentences_grande.append(" ".join(grande_words).strip())
    return (
        " ".join(sentences_lite),
        " ".join(sentences_grande)
    )

def process_document_list(documents):
    lite_documents = []
    grande_documents = []

    for text in documents:
        doc = nlp(text)
        lite, grande = apply_scheme(doc)

        lite_documents.append(lite)
        grande_documents.append(grande)

    return pd.Series({
        "masked_lite": lite_documents,
        "masked_grande": grande_documents
    })


if __name__ == "__main__":
    filename = sys.argv[1]
    output = filename.split(".tsv")[0] + "_ROBERTA_XL_masked.tsv"
    nlp = stanza.Pipeline(
        lang="en",
        processors="tokenize,pos"
    )
    df = pd.read_csv(filename,  # train.tsv",
                     delimiter='\t', quotechar='"', escapechar='\\', quoting=csv.QUOTE_ALL,
                     lineterminator="\n")
    #df.head()

    df["document"] = df["document"].apply(ast.literal_eval)
    df["document"] = df["document"].apply(protect_placeholders)
    masked = df["document"].apply(process_document_list)

    df[["masked_lite", "masked_grande"]] = masked
    df["masked_lite"] = df["masked_lite"].apply(restore_placeholders)
    df["masked_grande"] = df["masked_grande"].apply(restore_placeholders)

    df["masked_lite"] = df["masked_lite"].apply(json.dumps)
    df["masked_grande"] = df["masked_grande"].apply(json.dumps)

    df.to_csv(output, sep='\t', quotechar='"', escapechar='\\', quoting=csv.QUOTE_ALL, lineterminator="\n")
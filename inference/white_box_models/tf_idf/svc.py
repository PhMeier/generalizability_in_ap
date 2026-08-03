import sys
import ast
import json
import sklearn
import json
import argparse
from numpy import mean
from numpy import std
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report, \
    confusion_matrix

from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
import csv
import random
from sklearn.inspection import permutation_importance
from sklearn.base import clone

random.seed(42)
np.random.seed(0)

FEATURE_INDICES = {
    "all": [1, -1],
    "only_kw": [-27, -1],
    "without_kw": [1, -27]

}


def identify_classifications(y_true: list, y_pred: list) -> list:
    """
    Compare the predictions against the ground truth and retrieve the classifcation type
    :param y_true:
    :param y_pred:
    :return classifications:
    """
    classifications = []
    if len(y_true) != len(y_pred):
        raise ValueError("Lists do not have an equal length!")
    for t, p in zip(y_true, y_pred):
        if t == 1 and p == 1:
            classifications.append("TP")
        elif t == 0 and p == 1:
            classifications.append("FP")
        elif t == 1 and p == 0:
            classifications.append("FN")
        else:
            classifications.append("TN")
    return classifications


def slice_after_feature_indices(df):
    X = df["document"].to_list()
    y = df["label"].to_list()
    ids = df["author_id"]
    return X, y, ids

def inference(train, test, best_parameters, seed=42):

    """
    Read in the training and test folds.
    Initialize the models, get their stored hyperparameters
    Run Training, Predict, save predictions and calculate current binary metrics, confusion matrix and the
    classification report.
    """
    print("START FOR: ", dataset_key)
    print("Training: ", train)
    print("Test: ", test)

    training = pd.read_csv(train,
                           delimiter='\t', quotechar='"', escapechar='\\', quoting=csv.QUOTE_ALL,
                           lineterminator="\n", index_col=False)
    test = pd.read_csv(test,
                       delimiter='\t', quotechar='"', escapechar='\\', quoting=csv.QUOTE_ALL,
                       lineterminator="\n", index_col=False)

    training = training.drop(columns=["Unnamed: 0"], errors="ignore")
    test = test.drop(columns=["Unnamed: 0"], errors="ignore")

    X_train, y_train, train_ids = slice_after_feature_indices(training)
    X_test, y_test, test_ids = slice_after_feature_indices(test)

    print("First ten Test labels: ", y_test)

    print("Current Model: SVM")

    pipe = Pipeline([
        ("vect", TfidfVectorizer()),
        ("svc", LinearSVC(random_state=42))
    ])

    print("Parameters: ", best_parameters)
    with open(best_parameters, "r") as f:
        result = json.load(f)
    best_params = result["best_params"]
    if isinstance(best_params, str):
        best_params = ast.literal_eval(best_params)


    best_params.pop("vect", None)
    if isinstance(best_params.get("vect__ngram_range"), str):
        best_params["vect__ngram_range"] = ast.literal_eval(
            best_params["vect__ngram_range"]
        )

    print(best_params)
    pipe.set_params(**best_params)
    pipe.fit(X_train, y_train)

    print("TRAIN PRED")

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_train)
    y_pred = y_pred.tolist()
    acc = accuracy_score(y_train, y_pred)
    prec = precision_score(y_train, y_pred, average=None)
    rec = recall_score(y_train, y_pred, average=None)
    f1_sc = f1_score(y_train, y_pred, average=None)
    print("\n")
    print(classification_report(y_train, y_pred))
    print("\n")
    print(confusion_matrix(y_train, y_pred))

    print("Accuracy: ", acc, round(acc, 4))
    print("Precision: ", prec)
    print("Recall: ", rec)
    print("F1: ", f1_sc)




    print("TEST")
    # predict
    y_pred = pipe.predict(X_test)
    y_pred = y_pred.tolist()
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average=None)
    rec = recall_score(y_test, y_pred, average=None)
    f1_sc = f1_score(y_test, y_pred, average=None)
    print("\n")
    print(classification_report(y_test, y_pred))
    print("\n")
    print(confusion_matrix(y_test, y_pred))

    print("Accuracy: ", acc, round(acc, 4))
    print("Precision Binary: ", prec)
    print("Recall Binary: ", rec)
    print("F1 Binary: ", f1_sc)

    classifications = identify_classifications(y_test, y_pred)

    predictions = pd.DataFrame({
        "author_ids": test_ids,
        "y_true": y_test,
        "y_pred": y_pred,
        "classifications": classifications
    })

    predictions.to_csv(f"./svm_{dataset_key}_{seed}_predictions.csv", index=False)

    print("#######################\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='svm.py',
        description='Inference for SVM',
        epilog='Produces prediction results and outputs the predictions')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--train_file', type=str)
    parser.add_argument('--test_file', type=str)
    parser.add_argument('--parameters', type=str)
    args = parser.parse_args()
    dataset_key = args.dataset
    seed = args.seed
    print("SEED: ", seed)
    train_set = args.train_file
    test_set = args.test_file
    parameters = args.parameters

    inference(train_set, test_set, parameters, seed)

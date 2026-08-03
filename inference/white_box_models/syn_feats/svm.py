
import sys
import json
import sklearn
import json
import argparse
from numpy import mean
from numpy import std
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report, \
    confusion_matrix

from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
import csv
import random
from sklearn.inspection import permutation_importance
from sklearn.base import clone

random.seed(42)
np.random.seed(0)



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


def slice_after_feature_indices(df, upper, lower):
    cols = df.columns.to_list()
    lower_idx = cols.index("flesch_kincaid_easiness") #1
    upper_idx = cols.index("label")
    print(lower_idx, upper_idx)
    features = cols[lower_idx:upper_idx]
    label = cols[-1]
    print("features: ", features)
    X = df[features]
    y = df[label].to_list()
    ids = df["author_id"]
    return X, y, ids


data_set_indices = {
    "blog": [2, -1],
    "reddit": [1, -1]
}

def inference(train, test, best_parameters, SEED):

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

    print(training[training.isna().any(axis=1)])


    X_test = X_test[X_train.columns]
    print("First ten Test labels: ", y_test)

    print("Current Model: SVM")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(random_state=SEED, C=1e-06, kernel="linear", class_weight="balanced"))
    ])

    print("Parameters: ", best_parameters)
    with open(best_parameters, "r") as f:
        result = json.load(f)


    best_params = result["best_params"]
    best_params["svc__C"] = float(best_params["svc__C"])
    try:
        best_params["svc__gamma"] = float(best_params["svc__gamma"])
    except ValueError:
        best_params["svc__gamma"] = best_params["svc__gamma"]
    pipe.set_params(**best_params)
    pipe.fit(X_train, y_train)

    print("TRAIN PRED")
    train_pred = pipe.predict(X_train)

    print(classification_report(y_train, train_pred))


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
    print("Precision: ", prec, prec, 4)
    print("Recall: ", rec, rec, 4)
    print("F1: ", f1_sc, f1_sc)

    classifications = identify_classifications(y_test, y_pred)

    predictions = pd.DataFrame({
        "author_ids": test_ids,
        "y_true": y_test,
        "y_pred": y_pred,
        "classifications": classifications
    })

    predictions.to_csv(f"./svm_{dataset_key}_{SEED}_predictions.csv", index=False)

    print("#######################\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='svm.py',
        description='Inference for SVM',
        epilog='Produces prediction results and outputs the predictions')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--seed', type=str)
    parser.add_argument('--train_file', type=str)
    parser.add_argument('--test_file', type=str)
    parser.add_argument('--parameters', type=str)
    args = parser.parse_args()
    SEED = int(args.seed)
    dataset_key = args.dataset
    train_set = args.train_file
    test_set = args.test_file
    parameters = args.parameters
    assert dataset_key in train_set, f"Dataset key {dataset_key} not in train set path {train_set}"
    assert dataset_key in test_set, f"Dataset key {dataset_key} not in test set path {test_set}"
    inference(train_set, test_set, parameters, SEED)

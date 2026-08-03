
import csv
import json
import random
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report, \
    confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline



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

def inference(train, test, best_parameters, SEED):

    """
    Read in the training and test folds.
    Initialize the models, get their stored hyperparameters
    Run Training, Predict, save predictions and calculate current binary metrics, confusion matrix and the
    classification report.
    """
    print("SEED: ", SEED)
    print("START FOR: ", dataset_key)
    print("Training: ", train)
    print("Test: ", test)
    print("Best parameters: ", best_parameters)


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
    X_test = X_test[X_train.columns]
    print("First ten Test labels: ", y_test)

    print("Current Model: RF")

    pipe = Pipeline([
        ("rf", RandomForestClassifier(
            random_state=SEED, n_estimators=100, min_samples_leaf = 200, max_features=0.1, max_depth=2, min_samples_split = 50
        ))
    ])

    print("Parameters: ", best_parameters)
    with open(best_parameters, "r") as f:
        result = json.load(f)
    result["best_params"]
    best_params = {
        f"rf__{k}": v
        for k, v in result["best_params"].items()
    }

    pipe.set_params(**best_params)
    pipe.fit(X_train, y_train)

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

    print("Accuracy: ", acc)
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

    predictions.to_csv(f"./rf_{dataset_key}_{SEED}_predictions.csv", index=False)

    print("##############################################\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='rf.py',
        description='Inference for Random Forest',
        epilog='Produces prediction results and outputs the predictions')
    parser.add_argument('--dataset', type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument('--train_file', type=str)
    parser.add_argument('--test_file', type=str)
    parser.add_argument('--parameters', type=str)
    args = parser.parse_args()
    SEED = args.seed
    dataset_key = args.dataset
    train_set = args.train_file
    test_set = args.test_file
    parameters = args.parameters
    assert dataset_key in train_set, f"Dataset key {dataset_key} not in train set path {train_set}"
    assert dataset_key in test_set, f"Dataset key {dataset_key} not in test set path {test_set}"
    inference(train_set, test_set, parameters, SEED)

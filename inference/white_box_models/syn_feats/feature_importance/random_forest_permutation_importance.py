
import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report, confusion_matrix
import argparse
from collections import defaultdict
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from spacy.tokens.doc import defaultdict
import random
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


random.seed(42)
np.random.seed(0)


def random_forest_analysis(train_set, test_set, best_parameters, seed, TOP_N=20):

    pipeline = Pipeline([
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
    pipeline.set_params(**best_params)

    scores = defaultdict(list)
    model_feature_importance = defaultdict(list)
    cols = ""
    print("ITEM")
    df_train = pd.read_csv(train_set, sep="\t", index_col=False)
    #df_train = df_train.drop(columns=['Unnamed: 0'])
    df_test = pd.read_csv(test_set, sep="\t", index_col=False)
    #df_test = df_test.drop(columns=['Unnamed: 0'])
    df_train.fillna("")
    df_test.fillna("")
    # df_train = df_train.head()
    # df_test = df_test.head()

    cols = df_train.columns.to_list()
    lower, upper = cols.index("flesch_kincaid_easiness"), -1
    train_features = cols[lower:upper]

    print("TRAIN FEATURES: ")
    print(train_features)
    X_train = df_train[train_features]
    y_train = df_train["label"]
    X_test = df_test[train_features]
    y_test = df_test["label"]

    print("Initial Accuracy")
    pipeline.fit(X_train, y_train)
    acc = accuracy_score(y_test, pipeline.predict(X_test))
    print(acc)

    for i in range(10):
        print("FOLD")
        print(i)
        pipeline.fit(X_train, y_train)
        acc = accuracy_score(y_test, pipeline.predict(X_test))
        for feature, importance in zip(train_features, pipeline["rf"].feature_importances_):
            model_feature_importance[feature].append(importance)
        for column in X_train.columns: #s huffle the auccraucy scores in the dataset
            X_t = X_test.copy()
            X_t[column] = np.random.permutation(X_t[column].values)
            shuff_acc = accuracy_score(y_test, pipeline.predict(X_t))
            scores[column].append((acc-shuff_acc)/acc)
            print("Shuffled ACC: ", shuff_acc)

    print(type(train_features))
    print(type(scores))
    print(type(model_feature_importance))
    len(train_features)
    len(pipeline.named_steps["rf"].feature_importances_)
    df = pd.DataFrame({
        "feature":train_features,
        "Accuracy decrease": [np.mean(scores[column]) for column in train_features],
        "Gini decrease": [model_feature_importance[f] for f in train_features],
    })
    df = df.sort_values("Accuracy decrease")
    df.to_csv(f"rf_feature_importance_{dataset_key}_{seed}.csv")
    df_top = df.nlargest(TOP_N, "Accuracy decrease")
    fig, ax = plt.subplots(figsize=(12, 12))
    df_top.plot(
        kind="barh",
        x="feature",
        y="Accuracy decrease",
        legend=False,
        ax=ax,
        color="#1f77b4"
    )
    ax.set_ylabel("")
    ax.set_xlabel("Relative accuracy drop", fontsize=18)# (higher = more important)")
    ax.invert_yaxis()  # optional: biggest bar on top
    ax.tick_params(axis='both', labelsize=20)
    plt.tight_layout()
    #plt.show()
    plt.savefig(f"Random_Forest_Feature_analysis_{dataset_key}_t20.png")
    #plt.show()

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
    args = parser.parse_args()
    SEED = args.seed
    dataset_key = args.dataset
    train_set = args.train_file
    test_set = args.test_file
    parameters = args.parameters
    assert dataset_key in train_set, f"Dataset key {dataset_key} not in train set path {train_set}"
    assert dataset_key in test_set, f"Dataset key {dataset_key} not in test set path {test_set}"

    random_forest_analysis(train_set, test_set, parameters, SEED)
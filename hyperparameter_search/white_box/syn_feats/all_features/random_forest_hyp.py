import sys
import csv
import numpy as np
import json
from datetime import datetime
import pandas as pd
from joblib import parallel_backend
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report, confusion_matrix

data_with_60_20_20_split = [""]
data_with_80_20_split = [""]

def read_in_files(train, val):
    cols_t = train.columns.to_list()
    cols_v = val.columns.to_list()
    start = cols_t.index("flesch_kincaid_easiness")
    end = -1

    features = cols_t[start:end]
    label = cols_t[end]
    print("Used Features")
    print(features)
    assert features[0] == "flesch_kincaid_easiness"
    X_train = train[features]
    y_train = train[label]
    X_val = val[features]
    y_val = val[label]
    return  X_train, X_val, y_train, y_val

if __name__ == "__main__":
    n_jobs = int(sys.argv[1])
    filename = sys.argv[2]
    val_set = sys.argv[3]
    data_name = sys.argv[4]
    outputfile = f"rf_str_{data_name}.json"

    print("FILENAME TRAIN\n")
    print(filename)

    print("FILENAME VAL\n")
    print(filename)

    #outputfile = f"rf_{data_name}.json"


    dataset = pd.read_csv(filename,
                          delimiter='\t',quotechar='"', escapechar='\\', quoting=csv.QUOTE_ALL,
                          lineterminator="\n", index_col=False)
    val_set = pd.read_csv(val_set,
                          delimiter='\t',quotechar='"', escapechar='\\', quoting=csv.QUOTE_ALL,
                          lineterminator="\n", index_col=False)

    dataset = dataset.fillna(0)
    val_set = val_set.fillna(0)
    dataset = dataset.drop("Unnamed: 0", errors="ignore")
    val_set = val_set.drop("Unnamed: 0", errors="ignore")
    X_train, X_val, y_train, y_val = read_in_files(dataset, val_set)



    X = np.concatenate([X_train, X_val])
    y = np.concatenate([y_train, y_val])

    test_fold = [-1] * len(X_train) + [0] * len(X_val)
    #X,y = create_80_20_split(dataset)
    ps = PredefinedSplit(test_fold)

    param_grid = [
        {
            "bootstrap": [True],
            "max_samples": [None, 0.8],
            "n_estimators": [200, 500],
            "max_depth": [None, 20, 50],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", 0.5],
            "criterion": ["gini", "entropy"],
        }
    ]

    rf = RandomForestClassifier(random_state=42, n_jobs=1)
    grid = GridSearchCV(
        rf,
        param_grid,
        scoring="accuracy",
        cv=ps,
        n_jobs=n_jobs,
        verbose=3 # the higher, the more messages, range from 1-3
    )
    with parallel_backend("loky"): # default is loky, single-host, process-based parallelism
        grid.fit(X,y)
        print(sorted(grid.cv_results_.keys()))
        print("Best parameters found: ", grid.best_params_)
        print("Best cross-validation score: ", grid.best_score_)

        best_params = {
            k: str(v)
            for k, v in grid.best_params_.items()
        }

        result = {
            "timestamp": datetime.now().isoformat(),
            "data": filename,
            "best_params": best_params,
            "best_score": grid.best_score_,
            "scoring": str(grid.scoring),
            "best_estimator": str(grid.best_estimator_),
        }
        with open(outputfile, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4)

    print("\n")






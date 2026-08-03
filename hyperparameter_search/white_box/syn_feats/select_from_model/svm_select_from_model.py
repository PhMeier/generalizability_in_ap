import sys
import csv
import json
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from joblib import parallel_backend
from sklearn.svm import SVC, LinearSVC


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
    print("FILENAME: ", filename)
    data_name = sys.argv[4]

    #outputfile = f"log_{data_name}.json"
    outputfile = f"svm_str_{data_name}.json"
    output_list = f"svm_{data_name}_features.pkl"

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

    selector_svm = LinearSVC(
        penalty="l1",
        loss="squared_hinge",
        dual=False,
        C=1.0,
        max_iter=10000,
        random_state=42
    )

    pipeline = Pipeline([
        ("selector1", VarianceThreshold()),
        ("scaler", StandardScaler()),
        (
            "selector",
            SelectFromModel(
                estimator=selector_svm,
                threshold=1e-8
            )
        ),
        (
            "svc",
            SVC(
                max_iter=-1,
                cache_size=2000
            )
        )
    ])

    selector_C_values = [0.001, 0.01, 0.1, 1, 10]

    classifier_C_values = [0.001, 0.01, 0.1, 1, 10]

    param_grid = [
        # =========================================================
        # No feature selection: RBF SVM
        # =========================================================
        {
            "selector": ["passthrough"],

            "svc__kernel": ["rbf"],
            "svc__C": [0.1, 1, 10, 100],
            "svc__gamma": ["scale", 0.01, 0.001, 0.0001],
        },

        # No feature selection: linear SVM
        {
            "selector": ["passthrough"],

            "svc__kernel": ["linear"],
            "svc__C": [0.1, 1, 10, 100],
        },

        # No feature selection: polynomial SVM
        {
            "selector": ["passthrough"],

            "svc__kernel": ["poly"],
            "svc__C": [0.1, 1, 10, 100],
            "svc__gamma": ["scale", 0.01, 0.001],
            "svc__degree": [2, 3],
            "svc__coef0": [0, 1],
        },

        # =========================================================
        # L1 LinearSVC selection + RBF SVM
        # =========================================================
        {
            "selector__estimator__C": selector_C_values,
            "selector__threshold": [
                1e-8,
                "0.5*mean",
                "mean",
                "1.5*mean",
            ],

            "svc__kernel": ["rbf"],
            "svc__C": [0.1, 1, 10, 100],
            "svc__gamma": ["scale", 0.01, 0.001, 0.0001],
        },

        # L1 LinearSVC selection + linear SVM
        {
            "selector__estimator__C": selector_C_values,
            "selector__threshold": [
                1e-8,
                "0.5*mean",
                "mean",
                "1.5*mean",
            ],

            "svc__kernel": ["linear"],
            "svc__C": [0.1, 1, 10, 100],
        },

        # L1 LinearSVC selection + polynomial SVM
        {
            "selector__estimator__C": selector_C_values,
            "selector__threshold": [
                1e-8,
                "0.5*mean",
                "mean",
                "1.5*mean",
            ],

            "svc__kernel": ["poly"],
            "svc__C": [0.1, 1, 10, 100],
            "svc__gamma": ["scale", 0.01, 0.001],
            "svc__degree": [2, 3],
            "svc__coef0": [0, 1],
        },
    ]


    inner_cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="accuracy",  # or balanced_accuracy / roc_auc
        cv=inner_cv,
        n_jobs=n_jobs,
        refit=True,
        return_train_score=True
    )



    with parallel_backend("loky"):  # default is loky, single-host, process-based parallelism
        search.fit(X_train, y_train)
        print("Best parameters:", search.best_params_)
        print("Best CV score:", search.best_score_)

        best_model = search.best_estimator_

        val_predictions = best_model.predict(X_val)
        print("Val Predictions: ", val_predictions)

        print("Best parameters found: ", search.best_params_)
        print("Best cross-validation score: ", search.best_score_)
        feature_names = X_train.columns.to_numpy()
        feature_names = np.asarray(feature_names)

        selector = best_model.named_steps["selector"]

        if selector == "passthrough":
            print("No feature selection was used.")
            selected_features = feature_names.tolist()
        else:
            selector1 = best_model.named_steps["selector1"]
            selector = best_model.named_steps["selector"]

            feature_names = np.asarray(X_train.columns)

            features_after_selector1 = feature_names[
                selector1.get_support()
            ]

            # Features actually passed to the final SVC
            selected_features = features_after_selector1[
                selector.get_support()
            ]


        print(f"Selected {len(selected_features)} features:")
        print(selected_features)

        best_params = {
            k: str(v)
            for k, v in search.best_params_.items()
        }

        result = {
            "timestamp": datetime.now().isoformat(),
            "data": filename,
            "best_params": search.best_params_,
            "best_score": search.best_score_,
            "scoring": str(search.scoring),
            "best_estimator": str(search.best_estimator_),
        }

        with open(output_list, "wb") as f:
            pickle.dump(selected_features, f)

        with open(outputfile, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4)

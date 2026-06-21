
import os
import json
import joblib
import pandas as pd

from huggingface_hub import hf_hub_download, upload_folder, create_repo

from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

HF_USERNAME = "deepakpandit08"
DATASET_REPO = f"{HF_USERNAME}/visit-with-us-tourism-data"
MODEL_REPO = f"{HF_USERNAME}/visit-with-us-wellness-model"

os.makedirs("model", exist_ok=True)

train_file = hf_hub_download(
    repo_id=DATASET_REPO,
    filename="train.csv",
    repo_type="dataset"
)

test_file = hf_hub_download(
    repo_id=DATASET_REPO,
    filename="test.csv",
    repo_type="dataset"
)

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

X_train = train_df.drop(columns=["ProdTaken"])
y_train = train_df["ProdTaken"]

X_test = test_df.drop(columns=["ProdTaken"])
y_test = test_df["ProdTaken"]

numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

models = {
    "Decision Tree": {
        "model": DecisionTreeClassifier(random_state=42, class_weight="balanced"),
        "params": {
            "classifier__max_depth": [3, 5, 8, 10, None],
            "classifier__min_samples_split": [2, 5, 10]
        }
    },

    "Random Forest": {
        "model": RandomForestClassifier(random_state=42, class_weight="balanced"),
        "params": {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [5, 10, None],
            "classifier__min_samples_split": [2, 5]
        }
    },

    "Gradient Boosting": {
        "model": GradientBoostingClassifier(random_state=42),
        "params": {
            "classifier__n_estimators": [100, 200],
            "classifier__learning_rate": [0.05, 0.1],
            "classifier__max_depth": [3, 5]
        }
    },

    "AdaBoost": {
        "model": AdaBoostClassifier(random_state=42),
        "params": {
            "classifier__n_estimators": [50, 100, 200],
            "classifier__learning_rate": [0.05, 0.1, 1.0]
        }
    }
}

experiment_results = []

best_model = None
best_auc = -1
best_params = None
best_model_name = None

for model_name, model_info in models.items():

    print("Training:", model_name)

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model_info["model"])
    ])

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=model_info["params"],
        cv=5,
        scoring="roc_auc",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    y_pred = grid.predict(X_test)
    y_proba = grid.predict_proba(X_test)[:, 1]

    result = {
        "model_name": model_name,
        "best_params": grid.best_params_,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba)
    }

    experiment_results.append(result)

    if result["roc_auc"] > best_auc:
        best_auc = result["roc_auc"]
        best_model = grid.best_estimator_
        best_params = grid.best_params_
        best_model_name = model_name

results_df = pd.DataFrame(experiment_results)
results_df = results_df.sort_values(by="roc_auc", ascending=False)

results_df.to_csv("model/experiment_results.csv", index=False)

with open("model/best_params.json", "w") as f:
    json.dump(best_params, f, indent=4)

model_summary = {
    "best_model_name": best_model_name,
    "best_roc_auc": best_auc,
    "best_params": best_params
}

with open("model/model_summary.json", "w") as f:
    json.dump(model_summary, f, indent=4)

joblib.dump(best_model, "model/best_model.joblib")

create_repo(
    repo_id=MODEL_REPO,
    repo_type="model",
    exist_ok=True
)

upload_folder(
    folder_path="model",
    repo_id=MODEL_REPO,
    repo_type="model"
)

print("Model training completed successfully.")
print("Best model:", best_model_name)
print("Best ROC-AUC:", best_auc)

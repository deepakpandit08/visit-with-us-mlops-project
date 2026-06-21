
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from huggingface_hub import hf_hub_download, upload_file

HF_USERNAME = "deepakpandit08"
DATASET_REPO = f"{HF_USERNAME}/visit-with-us-tourism-data"

os.makedirs("data", exist_ok=True)

raw_file = hf_hub_download(
    repo_id=DATASET_REPO,
    filename="tourism_raw.csv",
    repo_type="dataset"
)

df = pd.read_csv(raw_file)

if "CustomerID" in df.columns:
    df = df.drop(columns=["CustomerID"])

for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].astype(str).str.strip()

if "Gender" in df.columns:
    df["Gender"] = df["Gender"].replace({
        "Fe Male": "Female",
        "FeMale": "Female"
    })

df = df.replace(["nan", "NaN", "None", ""], np.nan)

X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

train_df = X_train.copy()
train_df["ProdTaken"] = y_train

test_df = X_test.copy()
test_df["ProdTaken"] = y_test

train_df.to_csv("data/train.csv", index=False)
test_df.to_csv("data/test.csv", index=False)

upload_file(
    path_or_fileobj="data/train.csv",
    path_in_repo="train.csv",
    repo_id=DATASET_REPO,
    repo_type="dataset"
)

upload_file(
    path_or_fileobj="data/test.csv",
    path_in_repo="test.csv",
    repo_id=DATASET_REPO,
    repo_type="dataset"
)

print("Data preparation completed successfully.")
print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

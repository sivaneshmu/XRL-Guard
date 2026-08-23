import pandas as pd
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

TRAIN_FILE = DATA_DIR / "KDDTrain+.txt"
TEST_FILE = DATA_DIR / "KDDTest+.txt"

# NSL-KDD column names
columns = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "label",
    "difficulty"
]

# Load datasets
train_data = pd.read_csv(
    TRAIN_FILE,
    header=None,
    names=columns
)

test_data = pd.read_csv(
    TEST_FILE,
    header=None,
    names=columns
)

print("=" * 60)
print("XRL-GUARD DATASET INSPECTOR")
print("=" * 60)

print("\nTraining dataset shape:")
print(train_data.shape)

print("\nTesting dataset shape:")
print(test_data.shape)

print("\nColumn names:")
print(train_data.columns.tolist())

print("\nAttack labels in training dataset:")
print(train_data["label"].value_counts())

print("\nAttack labels in testing dataset:")
print(test_data["label"].value_counts())

print("\nFirst 5 training records:")
print(train_data.head())

print("\nMissing values:")
print(train_data.isnull().sum().sum())

print("\nDataset inspection completed.")
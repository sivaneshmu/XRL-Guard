import pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

PROCESSED_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------
# NSL-KDD COLUMN NAMES
# ---------------------------------------------------------

COLUMNS = [
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


# ---------------------------------------------------------
# ATTACK CATEGORY MAPPING
# ---------------------------------------------------------

ATTACK_MAPPING = {

    # DoS
    "back": "dos",
    "land": "dos",
    "neptune": "dos",
    "pod": "dos",
    "smurf": "dos",
    "teardrop": "dos",
    "apache2": "dos",
    "mailbomb": "dos",
    "processtable": "dos",
    "udpstorm": "dos",
    "worm": "dos",

    # Probe
    "ipsweep": "probe",
    "nmap": "probe",
    "portsweep": "probe",
    "satan": "probe",
    "mscan": "probe",
    "saint": "probe",

    # R2L
    "ftp_write": "r2l",
    "guess_passwd": "r2l",
    "imap": "r2l",
    "multihop": "r2l",
    "phf": "r2l",
    "spy": "r2l",
    "warezclient": "r2l",
    "warezmaster": "r2l",
    "httptunnel": "r2l",
    "named": "r2l",
    "sendmail": "r2l",
    "snmpgetattack": "r2l",
    "snmpguess": "r2l",
    "xlock": "r2l",
    "xsnoop": "r2l",

    # U2R
    "buffer_overflow": "u2r",
    "loadmodule": "u2r",
    "perl": "u2r",
    "rootkit": "u2r",
    "ps": "u2r",
    "sqlattack": "u2r",
    "xterm": "u2r",

    # Normal
    "normal": "normal"
}


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

def load_dataset(file_path):

    df = pd.read_csv(
        file_path,
        header=None,
        names=COLUMNS
    )

    return df


# ---------------------------------------------------------
# CONVERT ATTACK LABELS
# ---------------------------------------------------------

def create_categories(df):

    df["attack_category"] = (
        df["label"]
        .str.lower()
        .map(ATTACK_MAPPING)
        .fillna("other")
    )

    return df


# ---------------------------------------------------------
# MAIN PREPROCESSING
# ---------------------------------------------------------

print("=" * 60)
print("XRL-GUARD DATA PREPROCESSING")
print("=" * 60)


train_path = DATA_DIR / "KDDTrain+.txt"
test_path = DATA_DIR / "KDDTest+.txt"


print("\nLoading training data...")
train_df = load_dataset(train_path)

print("Loading testing data...")
test_df = load_dataset(test_path)


# ---------------------------------------------------------
# CREATE ATTACK CATEGORIES
# ---------------------------------------------------------

train_df = create_categories(train_df)
test_df = create_categories(test_df)


print("\nTraining attack categories:")
print(train_df["attack_category"].value_counts())

print("\nTesting attack categories:")
print(test_df["attack_category"].value_counts())


# ---------------------------------------------------------
# SEPARATE FEATURES
# ---------------------------------------------------------

categorical_columns = [
    "protocol_type",
    "service",
    "flag"
]

drop_columns = [
    "label",
    "difficulty",
    "attack_category"
]

X_train = train_df.drop(columns=drop_columns)
X_test = test_df.drop(columns=drop_columns)

y_train = train_df["attack_category"]
y_test = test_df["attack_category"]


# ---------------------------------------------------------
# ONE-HOT ENCODE CATEGORICAL FEATURES
# ---------------------------------------------------------

print("\nApplying one-hot encoding...")

combined = pd.concat(
    [X_train, X_test],
    axis=0,
    ignore_index=True
)

combined = pd.get_dummies(
    combined,
    columns=categorical_columns,
    dtype=float
)

X_train = combined.iloc[:len(X_train)].copy()
X_test = combined.iloc[len(X_train):].copy()


# ---------------------------------------------------------
# SCALE FEATURES
# ---------------------------------------------------------

print("Scaling numerical features...")

scaler = StandardScaler()

X_train = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns
)

X_test = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns
)


# ---------------------------------------------------------
# SAVE PROCESSED DATA
# ---------------------------------------------------------

X_train.to_csv(
    PROCESSED_DIR / "X_train.csv",
    index=False
)

X_test.to_csv(
    PROCESSED_DIR / "X_test.csv",
    index=False
)

y_train.to_csv(
    PROCESSED_DIR / "category_train.csv",
    index=False
)

y_test.to_csv(
    PROCESSED_DIR / "category_test.csv",
    index=False
)


# ---------------------------------------------------------
# FINAL INFORMATION
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETED")
print("=" * 60)

print("\nProcessed training shape :", X_train.shape)
print("Processed testing shape  :", X_test.shape)

print("\nTraining categories:")
print(y_train.value_counts())

print("\nTesting categories:")
print(y_test.value_counts())

print("\nProcessed files saved in:")
print(PROCESSED_DIR)

print("=" * 60)
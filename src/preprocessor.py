import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pathlib import Path


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

PROCESSED_DIR.mkdir(exist_ok=True)


# NSL-KDD column names
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


def load_dataset(file_path):
    """Load an NSL-KDD dataset."""

    df = pd.read_csv(
        file_path,
        header=None,
        names=COLUMNS
    )

    return df


def convert_attack_labels(df):
    """
    Convert individual attack names into broader cybersecurity categories.
    """

    attack_mapping = {
        # Denial of Service
        "back": "dos",
        "land": "dos",
        "neptune": "dos",
        "pod": "dos",
        "smurf": "dos",
        "teardrop": "dos",

        # Probe
        "ipsweep": "probe",
        "nmap": "probe",
        "portsweep": "probe",
        "satan": "probe",

        # Remote to Local
        "ftp_write": "r2l",
        "guess_passwd": "r2l",
        "imap": "r2l",
        "multihop": "r2l",
        "phf": "r2l",
        "spy": "r2l",
        "warezclient": "r2l",
        "warezmaster": "r2l",

        # User to Root
        "buffer_overflow": "u2r",
        "loadmodule": "u2r",
        "perl": "u2r",
        "rootkit": "u2r"
    }

    df["attack_category"] = df["label"].map(
        lambda x: "normal" if x == "normal"
        else attack_mapping.get(x, "other")
    )

    return df


def preprocess_data(df, encoder=None, scaler=None, training=True):
    """Encode categorical features and scale numerical features."""

    df = df.copy()

    # Remove difficulty because it is not a network feature
    df = df.drop(columns=["difficulty"])

    # Separate target information
    labels = df["label"]
    attack_categories = df["attack_category"]

    features = df.drop(
        columns=["label", "attack_category"]
    )

    categorical_columns = [
        "protocol_type",
        "service",
        "flag"
    ]

    numerical_columns = [
        column
        for column in features.columns
        if column not in categorical_columns
    ]

    # Encode categorical columns
    if training:
        encoders = {}

        for column in categorical_columns:
            encoder_column = LabelEncoder()
            features[column] = encoder_column.fit_transform(
                features[column].astype(str)
            )
            encoders[column] = encoder_column

    else:
        encoders = encoder

        for column in categorical_columns:
            features[column] = encoders[column].transform(
                features[column].astype(str)
            )

    # Scale numerical features
    if training:
        scaler = StandardScaler()
        features[numerical_columns] = scaler.fit_transform(
            features[numerical_columns]
        )
    else:
        features[numerical_columns] = scaler.transform(
            features[numerical_columns]
        )

    return (
        features,
        labels,
        attack_categories,
        encoders,
        scaler
    )


def main():

    print("=" * 60)
    print("XRL-GUARD DATASET PREPROCESSOR")
    print("=" * 60)

    train_path = DATA_DIR / "KDDTrain+.txt"
    test_path = DATA_DIR / "KDDTest+.txt"

    print("\nLoading datasets...")

    train_df = load_dataset(train_path)
    test_df = load_dataset(test_path)

    print(f"Training records: {len(train_df)}")
    print(f"Testing records : {len(test_df)}")

    print("\nConverting attack labels...")

    train_df = convert_attack_labels(train_df)
    test_df = convert_attack_labels(test_df)

    print("\nTraining attack categories:")
    print(train_df["attack_category"].value_counts())

    print("\nTesting attack categories:")
    print(test_df["attack_category"].value_counts())

    print("\nPreprocessing training data...")

    (
        X_train,
        y_train,
        category_train,
        encoders,
        scaler
    ) = preprocess_data(
        train_df,
        training=True
    )

    print("Training preprocessing completed.")

    print("\nPreprocessing testing data...")

    (
        X_test,
        y_test,
        category_test,
        _,
        _
    ) = preprocess_data(
        test_df,
        encoder=encoders,
        scaler=scaler,
        training=False
    )

    print("Testing preprocessing completed.")

    # Save processed feature datasets
    X_train.to_csv(
        PROCESSED_DIR / "X_train.csv",
        index=False
    )

    X_test.to_csv(
        PROCESSED_DIR / "X_test.csv",
        index=False
    )

    y_train.to_csv(
        PROCESSED_DIR / "y_train.csv",
        index=False
    )

    y_test.to_csv(
        PROCESSED_DIR / "y_test.csv",
        index=False
    )

    category_train.to_csv(
        PROCESSED_DIR / "category_train.csv",
        index=False
    )

    category_test.to_csv(
        PROCESSED_DIR / "category_test.csv",
        index=False
    )

    print("\nProcessed files saved to:")
    print(PROCESSED_DIR)

    print("\nProcessed training shape:")
    print(X_train.shape)

    print("\nProcessed testing shape:")
    print(X_test.shape)

    print("\nFeature data preview:")
    print(X_train.head())

    print("\n" + "=" * 60)
    print("DATASET PREPROCESSING COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
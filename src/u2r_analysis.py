import pandas as pd


# -------------------------------------------------
# FILE PATHS
# -------------------------------------------------

TRAIN_PATH = "data/KDDTrain+.txt"
TEST_PATH = "data/KDDTest+.txt"


# -------------------------------------------------
# NSL-KDD COLUMN NAMES
# -------------------------------------------------

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


# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------

print("=" * 65)
print("XRL-GUARD U2R ATTACK ANALYSIS")
print("=" * 65)

train = pd.read_csv(
    TRAIN_PATH,
    header=None,
    names=columns
)

test = pd.read_csv(
    TEST_PATH,
    header=None,
    names=columns
)


# -------------------------------------------------
# U2R ATTACK LIST
# -------------------------------------------------

u2r_attacks = [
    "buffer_overflow",
    "loadmodule",
    "perl",
    "rootkit",
    "ps",
    "sqlattack",
    "xterm"
]


# -------------------------------------------------
# FILTER U2R
# -------------------------------------------------

train_u2r = train[
    train["label"].isin(u2r_attacks)
]

test_u2r = test[
    test["label"].isin(u2r_attacks)
]


# -------------------------------------------------
# CREATE COMPARISON TABLE
# -------------------------------------------------

train_counts = train_u2r["label"].value_counts()

test_counts = test_u2r["label"].value_counts()

comparison = pd.DataFrame({
    "Train": train_counts,
    "Test": test_counts
}).fillna(0).astype(int)

comparison["Train"] = comparison["Train"].astype(int)
comparison["Test"] = comparison["Test"].astype(int)

comparison = comparison.sort_index()


# -------------------------------------------------
# DISPLAY
# -------------------------------------------------

print("\nU2R ATTACK DISTRIBUTION")
print("-" * 65)

print(
    f"{'Attack Type':<20}"
    f"{'Train':>10}"
    f"{'Test':>10}"
)

print("-" * 65)

for attack in comparison.index:

    print(
        f"{attack:<20}"
        f"{comparison.loc[attack, 'Train']:>10}"
        f"{comparison.loc[attack, 'Test']:>10}"
    )

print("-" * 65)

print(
    f"{'TOTAL':<20}"
    f"{comparison['Train'].sum():>10}"
    f"{comparison['Test'].sum():>10}"
)


# -------------------------------------------------
# UNSEEN U2R ATTACKS
# -------------------------------------------------

print("\n" + "=" * 65)
print("U2R ATTACKS NOT PRESENT IN TRAINING")
print("=" * 65)

for attack in comparison.index:

    if comparison.loc[attack, "Train"] == 0:

        print(
            f"{attack:<20}"
            f"Test samples: "
            f"{comparison.loc[attack, 'Test']}"
        )


# -------------------------------------------------
# U2R ATTACKS PRESENT IN BOTH
# -------------------------------------------------

print("\n" + "=" * 65)
print("U2R ATTACKS PRESENT IN BOTH TRAIN AND TEST")
print("=" * 65)

for attack in comparison.index:

    if comparison.loc[attack, "Train"] > 0:

        print(
            f"{attack:<20}"
            f"Train: {comparison.loc[attack, 'Train']:<6}"
            f"Test: {comparison.loc[attack, 'Test']}"
        )

print("\nAnalysis completed.")
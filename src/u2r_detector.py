import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


# -------------------------------------------------
# FILE PATHS
# -------------------------------------------------

TRAIN_X = "data/processed/X_train.csv"
TEST_X = "data/processed/X_test.csv"

TRAIN_Y = "data/processed/category_train.csv"
TEST_Y = "data/processed/category_test.csv"


# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------

print("=" * 60)
print("XRL-GUARD U2R THRESHOLD EXPERIMENT")
print("=" * 60)

X_train = pd.read_csv(TRAIN_X)
X_test = pd.read_csv(TEST_X)

y_train = pd.read_csv(TRAIN_Y).iloc[:, 0]
y_test = pd.read_csv(TEST_Y).iloc[:, 0]

print("\nTraining records :", len(X_train))
print("Testing records  :", len(X_test))
print("Features         :", X_train.shape[1])


# -------------------------------------------------
# CREATE BINARY U2R LABEL
# -------------------------------------------------

y_train_u2r = (y_train == "u2r").astype(int)
y_test_u2r = (y_test == "u2r").astype(int)

print("\nU2R training samples :", y_train_u2r.sum())
print("U2R testing samples  :", y_test_u2r.sum())


# -------------------------------------------------
# TRAIN RANDOM FOREST
# -------------------------------------------------

print("\nTraining U2R detector...")

model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train_u2r)

print("Training completed.")


# -------------------------------------------------
# U2R PROBABILITY
# -------------------------------------------------

u2r_probability = model.predict_proba(X_test)[:, 1]


# -------------------------------------------------
# THRESHOLD EXPERIMENT
# -------------------------------------------------

thresholds = [
    0.50,
    0.30,
    0.20,
    0.10,
    0.05,
    0.01
]

print("\n" + "=" * 60)
print("U2R THRESHOLD RESULTS")
print("=" * 60)

print(
    f"{'Threshold':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1':<12}"
    f"{'Detected':<12}"
)

for threshold in thresholds:

    predictions = (
        u2r_probability >= threshold
    ).astype(int)

    precision = precision_score(
        y_test_u2r,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test_u2r,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test_u2r,
        predictions,
        zero_division=0
    )

    detected = predictions.sum()

    print(
        f"{threshold:<12.2f}"
        f"{precision:<12.4f}"
        f"{recall:<12.4f}"
        f"{f1:<12.4f}"
        f"{detected:<12}"
    )


# -------------------------------------------------
# DETAILED RESULT FOR BEST F1 THRESHOLD
# -------------------------------------------------

results = []

for threshold in thresholds:

    predictions = (
        u2r_probability >= threshold
    ).astype(int)

    precision = precision_score(
        y_test_u2r,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test_u2r,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test_u2r,
        predictions,
        zero_division=0
    )

    results.append(
        (threshold, precision, recall, f1)
    )


best_threshold, best_precision, best_recall, best_f1 = max(
    results,
    key=lambda x: x[3]
)

best_predictions = (
    u2r_probability >= best_threshold
).astype(int)


print("\n" + "=" * 60)
print("BEST THRESHOLD")
print("=" * 60)

print("Threshold :", best_threshold)
print("Precision :", f"{best_precision:.4f}")
print("Recall    :", f"{best_recall:.4f}")
print("F1 Score  :", f"{best_f1:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(
    y_test_u2r,
    best_predictions
))

print("\nU2R threshold experiment completed.")
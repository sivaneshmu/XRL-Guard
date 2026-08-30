import numpy as np
import pandas as pd
from explainer import XRLGuardExplainer


MODEL_PATH = "models/xrl_guard_ppo"
FEATURE_PATH = "data/processed/X_test.csv"
LABEL_PATH = "data/processed/category_test.csv"


print("=" * 60)
print("XRL-GUARD XAI CATEGORY TEST")
print("=" * 60)

print("\nLoading test data...")

X = pd.read_csv(FEATURE_PATH)
y = pd.read_csv(LABEL_PATH)

if y.shape[1] == 1:
    y = y.iloc[:, 0]
else:
    y = y["attack_category"]

print("Testing records :", len(X))
print("Features        :", X.shape[1])

print("\nLoading XAI explainer...")

explainer = XRLGuardExplainer(
    MODEL_PATH
)

print("Explainer loaded successfully.")


categories = [
    "normal",
    "dos",
    "probe",
    "other",
    "r2l",
    "u2r"
]


print("\n" + "=" * 60)
print("CATEGORY-WISE EXPLANATIONS")
print("=" * 60)


for category in categories:

    indexes = np.where(
        y.values == category
    )[0]

    if len(indexes) == 0:
        print("\n" + category.upper())
        print("No records found.")
        continue

    index = indexes[0]

    observation = X.iloc[index].values.astype(
        np.float32
    )

    result = explainer.explain(
        observation
    )

    print("\n" + "-" * 60)
    print("Category          :", category.upper())
    print("Record index      :", index)
    print("Predicted action  :", result["action_name"])
    print(
        "Confidence        :",
        f"{result['confidence']:.4f}"
    )

    print("\nImportant features:")

    for item in result["important_features"][:5]:

        print(
            f"  {item['feature']:32s} "
            f"Impact={item['importance']:+.6f} "
            f"{item['effect']}"
        )


print("\n" + "=" * 60)
print("XRL-GUARD XAI CATEGORY TEST COMPLETED")
print("=" * 60)
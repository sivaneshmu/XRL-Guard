import numpy as np
import pandas as pd

from explainer import XRLGuardExplainer


MODEL_PATH = "models/xrl_guard_ppo"
FEATURE_PATH = "data/processed/X_test.csv"
LABEL_PATH = "data/processed/category_test.csv"


ACTION_NAMES = {
    0: "Allow",
    1: "Monitor",
    2: "Block",
    3: "Quarantine"
}


SEVERITY = {
    "normal": "LOW",
    "dos": "HIGH",
    "probe": "HIGH",
    "other": "HIGH",
    "r2l": "CRITICAL",
    "u2r": "CRITICAL"
}


def print_report(
    record_index,
    category,
    result
):

    action = result["action_name"]
    confidence = result["confidence"]

    severity = SEVERITY.get(
        category,
        "UNKNOWN"
    )

    print("\n" + "=" * 70)
    print("XRL-GUARD SECURITY DECISION REPORT")
    print("=" * 70)

    print("\nRecord Information")
    print("-" * 70)

    print(
        f"Record index       : {record_index}"
    )

    print(
        f"Attack category    : {category.upper()}"
    )

    print(
        f"Threat severity    : {severity}"
    )

    print("\nAgent Decision")
    print("-" * 70)

    print(
        f"Action             : {action.upper()}"
    )

    print(
        f"Confidence         : {confidence:.2%}"
    )

    print("\nExplanation")
    print("-" * 70)

    if action == "Allow":

        print(
            "The PPO agent determined that the "
            "connection can be allowed."
        )

    elif action == "Monitor":

        print(
            "The PPO agent recommends monitoring "
            "the connection for further activity."
        )

    elif action == "Block":

        print(
            "The PPO agent recommends blocking "
            "the connection because the observed "
            "features support malicious activity."
        )

    elif action == "Quarantine":

        print(
            "The PPO agent recommends quarantining "
            "the connection because of the high "
            "security risk."
        )

    print("\nImportant Features")
    print("-" * 70)

    for number, item in enumerate(
        result["important_features"],
        start=1
    ):

        print(
            f"{number:2d}. "
            f"{item['feature']:35s} "
            f"Impact={item['importance']:+.6f} "
            f"({item['effect']})"
        )

    print("\nDecision Summary")
    print("-" * 70)

    print(
        f"The XRL-Guard agent selected "
        f"{action.upper()} with "
        f"{confidence:.2%} confidence."
    )

    print(
        f"The detected category is "
        f"{category.upper()} with "
        f"{severity} severity."
    )

    print("\n" + "=" * 70)


if __name__ == "__main__":

    print("=" * 70)
    print("XRL-GUARD DECISION REPORT TEST")
    print("=" * 70)

    print("\nLoading test data...")

    X = pd.read_csv(FEATURE_PATH)
    y = pd.read_csv(LABEL_PATH)

    if y.shape[1] == 1:
        y = y.iloc[:, 0]
    else:
        y = y["attack_category"]

    print(
        "Testing records :",
        len(X)
    )

    print(
        "Features        :",
        X.shape[1]
    )

    print("\nLoading XAI explainer...")

    explainer = XRLGuardExplainer(
        MODEL_PATH
    )

    print(
        "Explainer loaded successfully."
    )

    record_index = 0

    category = y.iloc[
        record_index
    ]

    observation = X.iloc[
        record_index
    ].values.astype(
        np.float32
    )

    result = explainer.explain(
        observation
    )

    print_report(
        record_index,
        category,
        result
    )

    print(
        "\nXRL-GUARD DECISION REPORT "
        "COMPLETED"
    )
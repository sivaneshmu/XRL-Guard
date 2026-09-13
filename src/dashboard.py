from flask import Flask, render_template
import numpy as np
import pandas as pd

from explainer import XRLGuardExplainer


# ============================================================
# XRL-GUARD WEB DASHBOARD
# ============================================================

app = Flask(
    __name__,
    template_folder="../dashboard"
)


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = "models/xrl_guard_ppo"
FEATURE_PATH = "data/processed/X_test.csv"
CATEGORY_PATH = "data/processed/category_test.csv"


# ============================================================
# ACTION INFORMATION
# ============================================================

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


# ============================================================
# EXPECTED ACTION
# ============================================================

EXPECTED_ACTION = {
    "normal": 0,
    "dos": 2,
    "probe": 2,
    "other": 2,
    "r2l": 3,
    "u2r": 3
}


# ============================================================
# LOAD DATA ONCE
# ============================================================

print("Loading XRL-Guard dashboard data...")

X_test = pd.read_csv(FEATURE_PATH)
category_data = pd.read_csv(CATEGORY_PATH)


if "category" in category_data.columns:
    categories = category_data["category"]
else:
    categories = category_data.iloc[:, 0]


categories = (
    categories
    .astype(str)
    .str.strip()
    .str.lower()
    .reset_index(drop=True)
)


print(f"Test records loaded: {len(X_test)}")
print(f"Features loaded: {X_test.shape[1]}")


# ============================================================
# LOAD PPO MODEL
# ============================================================

print("Loading PPO model...")

explainer = XRLGuardExplainer(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# FAST PPO PREDICTIONS
# ============================================================

def get_predictions():

    predictions = []

    for index in range(len(X_test)):

        observation = (
            X_test.iloc[index]
            .values
            .astype(np.float32)
        )

        try:

            action, _ = explainer.model.predict(
                observation,
                deterministic=True
            )

            predictions.append(int(action))

        except Exception:

            predictions.append(0)

    return np.array(predictions)


# ============================================================
# GENERATE PREDICTIONS ONLY ONCE
# ============================================================

print("Generating predictions...")

predicted_actions = get_predictions()

print("Predictions ready.")


# ============================================================
# EXPECTED ACTION ARRAY
# ============================================================

expected_actions = np.array([
    EXPECTED_ACTION.get(
        category,
        0
    )
    for category in categories
])


# ============================================================
# CATEGORY PERFORMANCE
# ============================================================

def get_category_results():

    results = []

    category_order = [
        "normal",
        "dos",
        "probe",
        "other",
        "r2l",
        "u2r"
    ]

    for category in category_order:

        mask = categories == category

        total = int(mask.sum())

        if total == 0:
            continue

        actual = predicted_actions[mask]
        expected = expected_actions[mask]

        correct = int(
            np.sum(actual == expected)
        )

        incorrect = total - correct

        accuracy = (
            correct / total * 100
            if total > 0
            else 0
        )

        expected_action_id = EXPECTED_ACTION.get(
            category,
            0
        )

        expected_action_name = ACTION_NAMES[
            expected_action_id
        ]

        action_class = (
            expected_action_name
            .lower()
        )

        results.append({
            "category": category,
            "total": total,
            "correct": correct,
            "incorrect": incorrect,
            "accuracy": round(
                accuracy,
                2
            ),
            "expected_action": expected_action_name,
            "action_class": action_class
        })

    return results


# ============================================================
# DASHBOARD ROUTE
# ============================================================

@app.route("/")
def dashboard():

    total_records = len(X_test)


    # --------------------------------------------------------
    # ACCURACY
    # --------------------------------------------------------

    correct = int(
        np.sum(
            predicted_actions == expected_actions
        )
    )

    incorrect = (
        total_records - correct
    )

    accuracy = (
        correct / total_records * 100
        if total_records > 0
        else 0
    )


    # --------------------------------------------------------
    # REWARD
    # --------------------------------------------------------

    rewards = np.where(
        predicted_actions == expected_actions,
        5,
        -5
    )

    avg_reward = (
        float(np.mean(rewards))
        if len(rewards) > 0
        else 0
    )


    # --------------------------------------------------------
    # ATTACK CATEGORY CHART
    # --------------------------------------------------------

    category_order = [
        "normal",
        "dos",
        "probe",
        "other",
        "r2l",
        "u2r"
    ]

    attack_labels = []
    attack_values = []

    for category in category_order:

        count = int(
            np.sum(
                categories == category
            )
        )

        if count > 0:

            attack_labels.append(
                category.upper()
            )

            attack_values.append(
                count
            )


    # --------------------------------------------------------
    # ACTION DISTRIBUTION
    # --------------------------------------------------------

    action_labels = [
        "Allow",
        "Monitor",
        "Block",
        "Quarantine"
    ]

    action_values = []

    for action_id in range(4):

        count = int(
            np.sum(
                predicted_actions == action_id
            )
        )

        action_values.append(count)


    # --------------------------------------------------------
    # ACTION DISTRIBUTION TABLE
    # --------------------------------------------------------

    action_distribution = {}

    for action_id, action_name in ACTION_NAMES.items():

        count = int(
            np.sum(
                predicted_actions == action_id
            )
        )

        percentage = (
            count / total_records * 100
            if total_records > 0
            else 0
        )

        key = action_name.lower()

        action_distribution[key] = count

        action_distribution[
            f"{key}_percent"
        ] = round(
            percentage,
            2
        )


    # --------------------------------------------------------
    # CATEGORY RESULTS
    # --------------------------------------------------------

    category_results = get_category_results()


    # --------------------------------------------------------
    # RENDER DASHBOARD
    # --------------------------------------------------------

    return render_template(
        "dashboard.html",

        # Summary cards
        total_records=total_records,
        accuracy=round(
            accuracy,
            2
        ),
        incorrect=incorrect,
        avg_reward=round(
            avg_reward,
            2
        ),

        # Attack chart
        attack_labels=attack_labels,
        attack_values=attack_values,

        # Action chart
        action_labels=action_labels,
        action_values=action_values,

        # Tables
        category_results=category_results,
        action_distribution=action_distribution
    )


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("XRL-GUARD DASHBOARD")
    print("=" * 60)
    print("HTTPS: https://127.0.0.1:5000")
    print("=" * 60)
    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        ssl_context=(
            "cert/localhost.crt",
            "cert/localhost.key"
        )
    )
    
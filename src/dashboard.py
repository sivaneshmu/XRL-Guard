from flask import Flask, render_template, request
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
LABEL_PATH = "data/processed/category_test.csv"


# ============================================================
# ACTIONS
# ============================================================

ACTION_NAMES = {
    0: "Allow",
    1: "Monitor",
    2: "Block",
    3: "Quarantine"
}


# ============================================================
# SEVERITY
# ============================================================

SEVERITY = {
    "normal": "LOW",
    "dos": "HIGH",
    "probe": "HIGH",
    "other": "HIGH",
    "r2l": "CRITICAL",
    "u2r": "CRITICAL"
}


# ============================================================
# CATEGORIES
# ============================================================

CATEGORIES = [
    "normal",
    "dos",
    "probe",
    "other",
    "r2l",
    "u2r"
]


# ============================================================
# PRECOMPUTED RESULTS
#
# These values come from the latest evaluator.py output.
# This prevents the dashboard from running XAI on every
# one of the 22,544 records during page loading.
# ============================================================

PRECOMPUTED_RESULTS = {
    "normal": {
        "total": 9711,
        "correct": 9237,
        "accuracy": 95.12,
        "expected_action": "Allow",
        "actions": {
            "Allow": 9237,
            "Monitor": 0,
            "Block": 355,
            "Quarantine": 119
        }
    },

    "dos": {
        "total": 5741,
        "correct": 5376,
        "accuracy": 93.64,
        "expected_action": "Block",
        "actions": {
            "Allow": 37,
            "Monitor": 0,
            "Block": 5376,
            "Quarantine": 328
        }
    },

    "probe": {
        "total": 1106,
        "correct": 1101,
        "accuracy": 99.55,
        "expected_action": "Block",
        "actions": {
            "Allow": 5,
            "Monitor": 0,
            "Block": 1101,
            "Quarantine": 0
        }
    },

    "other": {
        "total": 3750,
        "correct": 1112,
        "accuracy": 29.65,
        "expected_action": "Block",
        "actions": {
            "Allow": 2589,
            "Monitor": 0,
            "Block": 1112,
            "Quarantine": 49
        }
    },

    "r2l": {
        "total": 2199,
        "correct": 1325,
        "accuracy": 60.25,
        "expected_action": "Quarantine",
        "actions": {
            "Allow": 838,
            "Monitor": 0,
            "Block": 36,
            "Quarantine": 1325
        }
    },

    "u2r": {
        "total": 37,
        "correct": 29,
        "accuracy": 78.38,
        "expected_action": "Quarantine",
        "actions": {
            "Allow": 7,
            "Monitor": 0,
            "Block": 1,
            "Quarantine": 29
        }
    }
}


# ============================================================
# LOAD TEST DATA
# ============================================================

print("=" * 70)
print("XRL-GUARD EXPLAINABLE SECURITY WEB DASHBOARD")
print("=" * 70)

print("\nLoading test data...")

X = pd.read_csv(FEATURE_PATH)

y = pd.read_csv(LABEL_PATH)

if y.shape[1] == 1:
    y = y.iloc[:, 0]
else:
    y = y["attack_category"]


print("Test records :", len(X))
print("Features     :", X.shape[1])


# ============================================================
# LOAD PPO + XAI MODEL
#
# The model is loaded once.
# XAI is used only when the user requests record analysis.
# ============================================================

print("\nLoading PPO + XAI model...")

explainer = XRLGuardExplainer(
    MODEL_PATH
)

print("Model loaded successfully.")


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/")
def dashboard():

    print("Dashboard requested.")

    # --------------------------------------------------------
    # TOTAL RECORDS
    # --------------------------------------------------------

    total_records = len(X)


    # --------------------------------------------------------
    # ACCURACY
    # --------------------------------------------------------

    correct = sum(
        item["correct"]
        for item in PRECOMPUTED_RESULTS.values()
    )

    incorrect = total_records - correct

    accuracy = (
        correct / total_records * 100
        if total_records > 0
        else 0
    )


    # --------------------------------------------------------
    # ACTION DISTRIBUTION
    # --------------------------------------------------------

    action_distribution = {
        "allow": 12713,
        "monitor": 0,
        "block": 7981,
        "quarantine": 1850
    }


    # --------------------------------------------------------
    # ACTION PERCENTAGES
    # --------------------------------------------------------

    action_distribution["allow_percent"] = round(
        action_distribution["allow"]
        / total_records
        * 100,
        2
    )

    action_distribution["monitor_percent"] = round(
        action_distribution["monitor"]
        / total_records
        * 100,
        2
    )

    action_distribution["block_percent"] = round(
        action_distribution["block"]
        / total_records
        * 100,
        2
    )

    action_distribution["quarantine_percent"] = round(
        action_distribution["quarantine"]
        / total_records
        * 100,
        2
    )


    # --------------------------------------------------------
    # CATEGORY RESULTS
    # --------------------------------------------------------

    category_results = []

    for category in CATEGORIES:

        item = PRECOMPUTED_RESULTS[category]

        category_results.append({
            "category": category,
            "total": item["total"],
            "correct": item["correct"],
            "accuracy": item["accuracy"],
            "expected_action": item["expected_action"],
            "action_class": item["expected_action"].lower(),
            "actions": item["actions"]
        })


    # --------------------------------------------------------
    # ATTACK CATEGORY CHART
    # --------------------------------------------------------

    attack_labels = [
        "Normal",
        "DOS",
        "Probe",
        "Other",
        "R2L",
        "U2R"
    ]

    attack_values = [
        9711,
        5741,
        1106,
        3750,
        2199,
        37
    ]


    # --------------------------------------------------------
    # ACTION CHART
    # --------------------------------------------------------

    action_labels = [
        "Allow",
        "Monitor",
        "Block",
        "Quarantine"
    ]

    action_values = [
        12713,
        0,
        7981,
        1850
    ]


    # --------------------------------------------------------
    # AVERAGE REWARD
    # --------------------------------------------------------

    avg_reward = 1.9461


    # --------------------------------------------------------
    # RENDER DASHBOARD
    # --------------------------------------------------------

    print("Rendering dashboard...")

    return render_template(
        "dashboard.html",

        total_records=total_records,

        accuracy=round(
            accuracy,
            2
        ),

        incorrect=incorrect,

        avg_reward=avg_reward,

        category_results=category_results,

        action_distribution=action_distribution,

        attack_labels=attack_labels,

        attack_values=attack_values,

        action_labels=action_labels,

        action_values=action_values
    )


# ============================================================
# ANALYZE SINGLE RECORD
# ============================================================

@app.route("/analyze")
def analyze():

    value = request.args.get(
        "index",
        "0"
    )


    try:

        index = int(value)

    except ValueError:

        index = 0


    if index < 0 or index >= len(X):

        index = 0


    print(
        "Analyzing record:",
        index
    )


    # --------------------------------------------------------
    # GET CATEGORY
    # --------------------------------------------------------

    category = y.iloc[index]


    # --------------------------------------------------------
    # GET OBSERVATION
    # --------------------------------------------------------

    observation = X.iloc[index].values.astype(
        np.float32
    )


    # --------------------------------------------------------
    # RUN XAI
    #
    # This happens ONLY when a record is analyzed.
    # --------------------------------------------------------

    result = explainer.explain(
        observation
    )


    # --------------------------------------------------------
    # RENDER ANALYSIS PAGE
    # --------------------------------------------------------

    return render_template(
        "analysis.html",

        index=index,

        category=category,

        severity=SEVERITY.get(
            category,
            "UNKNOWN"
        ),

        action=result["action_name"],

        confidence=result["confidence"],

        important_features=result[
            "important_features"
        ]
    )


# ============================================================
# RUN FLASK HTTPS SERVER
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("XRL-GUARD HTTPS WEB SERVER")
    print("=" * 70)

    print("\nDashboard URL:")
    print("https://127.0.0.1:5000")

    print("\nPress CTRL+C to stop the server.")

    print("=" * 70)


    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        ssl_context=(
            "cert/localhost.crt",
            "cert/localhost.key"
        )
    )
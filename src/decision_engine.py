import numpy as np
import pandas as pd
from explainer import XRLGuardExplainer


MODEL_PATH = "models/xrl_guard_ppo"
DATA_PATH = "data/processed/X_test.csv"


ACTION_NAMES = {
    0: "Allow",
    1: "Monitor",
    2: "Block",
    3: "Quarantine"
}


class XRLGuardDecisionEngine:

    def __init__(self, model_path):

        self.explainer = XRLGuardExplainer(
            model_path
        )

    def get_decision(self, observation):

        observation = np.asarray(
            observation,
            dtype=np.float32
        )

        # ---------------------------------------------
        # GET XAI EXPLANATION
        # ---------------------------------------------

        explanation = self.explainer.explain(
            observation
        )

        # ---------------------------------------------
        # GET ACTION PROBABILITIES
        # ---------------------------------------------

        probabilities = (
            self.explainer.get_action_probabilities(
                observation
            )
        )

        # ---------------------------------------------
        # RANK ALTERNATIVE ACTIONS
        # ---------------------------------------------

        ranked_actions = np.argsort(
            probabilities
        )[::-1]

        alternatives = []

        for action in ranked_actions:

            action = int(action)

            if action != explanation["action"]:

                alternatives.append({
                    "action": ACTION_NAMES[action],
                    "confidence": float(
                        probabilities[action]
                    )
                })

        # ---------------------------------------------
        # DECISION REASON
        # ---------------------------------------------

        action = explanation["action"]

        if action == 0:

            reason = (
                "The PPO agent recommends allowing "
                "the activity because the observed "
                "network characteristics support "
                "normal operation."
            )

        elif action == 1:

            reason = (
                "The PPO agent recommends monitoring "
                "the activity because the observed "
                "network characteristics may require "
                "further observation."
            )

        elif action == 2:

            reason = (
                "The PPO agent recommends blocking "
                "the activity because the observed "
                "network characteristics indicate "
                "potentially harmful behavior."
            )

        else:

            reason = (
                "The PPO agent recommends quarantine "
                "because the observed network "
                "characteristics indicate suspicious "
                "or potentially harmful behavior."
            )

        # ---------------------------------------------
        # SUPPORTING / OPPOSING FACTORS
        # ---------------------------------------------

        supporting_features = (
            explanation.get(
                "supporting_features",
                []
            )
        )

        opposing_features = (
            explanation.get(
                "opposing_features",
                []
            )
        )

        return {
            "action": explanation["action_name"],

            # This is PPO action probability,
            # NOT prediction accuracy.
            "confidence": explanation["confidence"],

            "reason": reason,

            "important_features": (
                explanation["important_features"]
            ),

            "supporting_features": (
                supporting_features
            ),

            "opposing_features": (
                opposing_features
            ),

            "alternatives": alternatives[:3]
        }


if __name__ == "__main__":

    print("=" * 60)
    print("XRL-GUARD DECISION ENGINE TEST")
    print("=" * 60)

    print("\nLoading test data...")

    data = pd.read_csv(
        DATA_PATH
    )

    print(
        "Testing records :",
        len(data)
    )

    print(
        "Features        :",
        data.shape[1]
    )

    print("\nLoading decision engine...")

    engine = XRLGuardDecisionEngine(
        MODEL_PATH
    )

    print(
        "Decision engine loaded successfully."
    )

    # Use first test record
    observation = data.iloc[0].values.astype(
        np.float32
    )

    result = engine.get_decision(
        observation
    )

    print("\n" + "=" * 60)
    print("FINAL SECURITY DECISION")
    print("=" * 60)

    print(
        "\nRecommended Action :",
        result["action"]
    )

    print(
        "Model Confidence   :",
        f"{result['confidence']:.2%}"
    )

    print(
        "\nReason:"
    )

    print(
        result["reason"]
    )

    print("\nSupporting Factors:")

    if result["supporting_features"]:

        for feature in result["supporting_features"]:

            print(
                " -",
                feature
            )

    else:

        print(
            " - None identified"
        )

    print("\nOpposing Factors:")

    if result["opposing_features"]:

        for feature in result["opposing_features"]:

            print(
                " -",
                feature
            )

    else:

        print(
            " - None identified"
        )

    print("\nImportant Features:")

    for item in result["important_features"]:

        print(
            f"  {item['feature']:40s} "
            f"Impact={item['importance']:+.6f} "
            f"{item['effect']}"
        )

    print("\nAlternative Actions:")

    for item in result["alternatives"]:

        print(
            f"  {item['action']:12s} "
            f"{item['confidence']:.2%}"
        )

    print("\n" + "=" * 60)
    print(
        "XRL-GUARD DECISION ENGINE TEST COMPLETED"
    )
    print("=" * 60)
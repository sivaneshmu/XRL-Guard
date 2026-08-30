import numpy as np
from explainer import XRLGuardExplainer

MODEL_PATH = "models/xrl_guard_ppo"

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

        explanation = self.explainer.explain(
            observation
        )

        probabilities = (
            self.explainer.get_action_probabilities(
                observation
            )
        )

        ranked_actions = np.argsort(
            probabilities
        )[::-1]

        alternatives = []

        for action in ranked_actions:

            if action != explanation["action"]:

                alternatives.append({
                    "action": ACTION_NAMES[int(action)],
                    "confidence": float(
                        probabilities[action]
                    )
                })

        if explanation["action"] == 0:

            reason = (
                "The agent considers the observed "
                "network activity suitable for normal "
                "operation."
            )

        elif explanation["action"] == 1:

            reason = (
                "The agent recommends monitoring "
                "the activity because it may require "
                "further observation."
            )

        elif explanation["action"] == 2:

            reason = (
                "The agent recommends blocking the "
                "activity because the observed features "
                "indicate potentially harmful behavior."
            )

        else:

            reason = (
                "The agent recommends quarantine because "
                "the observed features indicate suspicious "
                "or potentially harmful activity."
            )

        return {
            "action": explanation["action_name"],
            "confidence": explanation["confidence"],
            "reason": reason,
            "important_features": explanation[
                "important_features"
            ],
            "alternatives": alternatives[:3]
        }


if __name__ == "__main__":

    print("=" * 60)
    print("XRL-GUARD DECISION ENGINE TEST")
    print("=" * 60)

    import pandas as pd

    data_path = "data/processed/X_test.csv"

    print("\nLoading test data...")

    data = pd.read_csv(data_path)

    print("Testing records :", len(data))
    print("Features        :", data.shape[1])

    engine = XRLGuardDecisionEngine(
        MODEL_PATH
    )

    observation = data.iloc[0].values.astype(
        np.float32
    )

    result = engine.get_decision(
        observation
    )

    print("\nFINAL SECURITY DECISION")
    print("-----------------------")

    print(
        "Action       :",
        result["action"]
    )

    print(
        "Confidence   :",
        f"{result['confidence']:.2%}"
    )

    print(
        "\nReason       :",
        result["reason"]
    )

    print("\nImportant features:")

    for item in result["important_features"]:

        print(
            f"  {item['feature']:35s}"
            f" {item['importance']:+.6f}"
        )

    print("\nAlternative actions:")

    for item in result["alternatives"]:

        print(
            f"  {item['action']:12s}"
            f" {item['confidence']:.2%}"
        )

    print("\n" + "=" * 60)
    print("XRL-GUARD DECISION ENGINE TEST COMPLETED")
    print("=" * 60)
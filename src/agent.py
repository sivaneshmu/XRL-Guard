import numpy as np
import pandas as pd
from decision_engine import XRLGuardDecisionEngine

MODEL_PATH = "models/xrl_guard_ppo"


class XRLGuardAgent:

    def __init__(self, model_path):

        self.engine = XRLGuardDecisionEngine(
            model_path
        )

    def analyze(self, observation):

        observation = np.asarray(
            observation,
            dtype=np.float32
        )

        result = self.engine.get_decision(
            observation
        )

        confidence = result["confidence"]

        if confidence >= 0.70:
            risk_level = "HIGH"
        elif confidence >= 0.40:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        if result["action"] == "Allow":
            recommendation = (
                "Allow the activity and continue monitoring."
            )

        elif result["action"] == "Monitor":
            recommendation = (
                "Monitor the activity for further suspicious behavior."
            )

        elif result["action"] == "Block":
            recommendation = (
                "Block the activity to prevent possible harm."
            )

        else:
            recommendation = (
                "Quarantine the activity for further investigation."
            )

        return {
            "action": result["action"],
            "confidence": confidence,
            "risk_level": risk_level,
            "recommendation": recommendation,
            "reason": result["reason"],
            "important_features": result[
                "important_features"
            ],
            "alternatives": result[
                "alternatives"
            ]
        }


if __name__ == "__main__":

    print("=" * 60)
    print("XRL-GUARD AGENTIC WORKFLOW TEST")
    print("=" * 60)

    data_path = "data/processed/X_test.csv"

    print("\nLoading test data...")

    data = pd.read_csv(data_path)

    print("Testing records :", len(data))
    print("Features        :", data.shape[1])

    agent = XRLGuardAgent(
        MODEL_PATH
    )

    observation = data.iloc[0].values.astype(
        np.float32
    )

    result = agent.analyze(
        observation
    )

    print("\nAGENT ANALYSIS")
    print("-----------------------")

    print(
        "Final action  :",
        result["action"]
    )

    print(
        "Confidence    :",
        f"{result['confidence']:.2%}"
    )

    print(
        "Risk level    :",
        result["risk_level"]
    )

    print(
        "\nRecommendation :",
        result["recommendation"]
    )

    print(
        "\nReason        :",
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
    print("XRL-GUARD AGENTIC WORKFLOW COMPLETED")
    print("=" * 60)
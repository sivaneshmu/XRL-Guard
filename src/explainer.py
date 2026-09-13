import numpy as np
import pandas as pd
from stable_baselines3 import PPO


MODEL_PATH = "models/xrl_guard_ppo"
DATA_PATH = "data/processed/X_test.csv"


ACTION_NAMES = {
    0: "Allow",
    1: "Monitor",
    2: "Block",
    3: "Quarantine"
}


class XRLGuardExplainer:

    def __init__(self, model_path):

        self.model = PPO.load(model_path)

        # Get the actual feature names used by the processed dataset
        self.feature_names = pd.read_csv(
            DATA_PATH,
            nrows=0
        ).columns.tolist()

    def predict_action(self, observation):

        observation = np.asarray(
            observation,
            dtype=np.float32
        )

        action, _ = self.model.predict(
            observation,
            deterministic=True
        )

        return int(action)

    def get_action_probabilities(self, observation):

        observation = np.asarray(
            observation,
            dtype=np.float32
        )

        observation_tensor, _ = self.model.policy.obs_to_tensor(
            observation.reshape(1, -1)
        )

        self.model.policy.set_training_mode(False)

        distribution = self.model.policy.get_distribution(
            observation_tensor
        )

        probabilities = (
            distribution.distribution.probs
            .detach()
            .cpu()
            .numpy()[0]
        )

        return probabilities

    def explain(self, observation):

        observation = np.asarray(
            observation,
            dtype=np.float32
        )

        # ---------------------------------------------
        # ORIGINAL PREDICTION
        # ---------------------------------------------

        action = self.predict_action(
            observation
        )

        probabilities = self.get_action_probabilities(
            observation
        )

        confidence = probabilities[action]

        # ---------------------------------------------
        # BASELINE
        # ---------------------------------------------
        # Zero represents the scaled baseline for
        # numerical features and the inactive state
        # for one-hot features.

        baseline = np.zeros_like(
            observation,
            dtype=np.float32
        )

        # ---------------------------------------------
        # FEATURE IMPORTANCE
        # ---------------------------------------------

        importance = []

        for index in range(
            len(observation)
        ):

            modified_observation = observation.copy()

            modified_observation[index] = baseline[index]

            modified_probabilities = (
                self.get_action_probabilities(
                    modified_observation
                )
            )

            contribution = (
                confidence -
                modified_probabilities[action]
            )

            importance.append(
                contribution
            )

        importance = np.asarray(
            importance
        )

        # ---------------------------------------------
        # RANK FEATURES
        # ---------------------------------------------

        ranked_indices = np.argsort(
            np.abs(importance)
        )[::-1]

        important_features = []

        for index in ranked_indices[:10]:

            if index < len(self.feature_names):

                feature_name = (
                    self.feature_names[index]
                )

            else:

                feature_name = (
                    f"feature_{index}"
                )

            if importance[index] > 0:

                effect = "supports"

            elif importance[index] < 0:

                effect = "opposes"

            else:

                effect = "neutral"

            important_features.append({
                "feature": feature_name,
                "value": float(
                    observation[index]
                ),
                "importance": float(
                    importance[index]
                ),
                "effect": effect
            })

        # ---------------------------------------------
        # ACTION EXPLANATION
        # ---------------------------------------------

        positive_features = [
            item["feature"]
            for item in important_features
            if item["effect"] == "supports"
        ]

        negative_features = [
            item["feature"]
            for item in important_features
            if item["effect"] == "opposes"
        ]

        if positive_features:

            support_text = ", ".join(
                positive_features[:3]
            )

        else:

            support_text = "the observed traffic features"

        if negative_features:

            oppose_text = ", ".join(
                negative_features[:3]
            )

        else:

            oppose_text = "no major opposing features"

        if action == 0:

            reason = (
                "The PPO agent recommends Allow. "
                "The decision is mainly supported by "
                f"{support_text}."
            )

        elif action == 1:

            reason = (
                "The PPO agent recommends Monitor. "
                "The decision is mainly supported by "
                f"{support_text}."
            )

        elif action == 2:

            reason = (
                "The PPO agent recommends Block. "
                "The decision is mainly supported by "
                f"{support_text}."
            )

        else:

            reason = (
                "The PPO agent recommends Quarantine. "
                "The decision is mainly supported by "
                f"{support_text}."
            )

        return {
            "action": action,
            "action_name": ACTION_NAMES[action],
            "confidence": float(confidence),
            "reason": reason,
            "important_features": important_features,
            "supporting_features": positive_features[:5],
            "opposing_features": negative_features[:5]
        }


if __name__ == "__main__":

    print("=" * 60)
    print("XRL-GUARD EXPLAINABLE AI TEST")
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

    print("\nLoading PPO model...")

    explainer = XRLGuardExplainer(
        MODEL_PATH
    )

    print(
        "Model loaded successfully."
    )

    observation = data.iloc[0].values.astype(
        np.float32
    )

    result = explainer.explain(
        observation
    )

    print("\n" + "=" * 60)
    print("XRL-GUARD DECISION")
    print("=" * 60)

    print(
        "\nRecommended action :",
        result["action_name"]
    )

    print(
        "Model confidence   :",
        f"{result['confidence']:.2%}"
    )

    print(
        "\nReason:",
        result["reason"]
    )

    print("\nImportant features:")

    for item in result["important_features"]:

        print(
            f"{item['feature']:40s} "
            f"Value={item['value']:10.4f} "
            f"Impact={item['importance']:+.6f} "
            f"{item['effect']}"
        )

    print("\nSupporting features:")

    for feature in result["supporting_features"]:

        print(
            " -",
            feature
        )

    print("\nOpposing features:")

    for feature in result["opposing_features"]:

        print(
            " -",
            feature
        )

    print("\n" + "=" * 60)
    print("XRL-GUARD XAI TEST COMPLETED")
    print("=" * 60)
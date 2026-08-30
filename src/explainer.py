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


FEATURE_NAMES = [
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
    "dst_host_srv_rerror_rate"
]


class XRLGuardExplainer:

    def __init__(self, model_path):

        self.model = PPO.load(model_path)

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

        baseline = np.zeros_like(
            observation,
            dtype=np.float32
        )

        # ---------------------------------------------
        # FEATURE IMPORTANCE
        # ---------------------------------------------

        importance = []

        for index in range(len(observation)):

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

            if importance[index] > 0:
                effect = "supports"

            elif importance[index] < 0:
                effect = "opposes"

            else:
                effect = "neutral"

            important_features.append({
                "feature": FEATURE_NAMES[index],
                "value": float(
                    observation[index]
                ),
                "importance": float(
                    importance[index]
                ),
                "effect": effect
            })

        return {
            "action": action,
            "action_name": ACTION_NAMES[action],
            "confidence": float(confidence),
            "important_features": important_features
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
        "\nPredicted action :",
        result["action_name"]
    )

    print(
        "Confidence       :",
        f"{result['confidence']:.4f}"
    )

    print("\nImportant features:")

    for item in result["important_features"]:

        print(
            f"{item['feature']:35s} "
            f"Value={item['value']:10.4f} "
            f"Impact={item['importance']:+.6f} "
            f"{item['effect']}"
        )

    print("\n" + "=" * 60)
    print("XRL-GUARD XAI TEST COMPLETED")
    print("=" * 60)
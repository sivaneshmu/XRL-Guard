import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


class XRLGuardEnv(gym.Env):

    def __init__(self, feature_path, label_path):
        super().__init__()

        # -------------------------------------------------
        # LOAD DATA
        # -------------------------------------------------

        self.X = pd.read_csv(feature_path)
        self.y = pd.read_csv(label_path)

        if self.y.shape[1] == 1:
            self.y = self.y.iloc[:, 0]
        else:
            self.y = self.y["attack_category"]

        self.n_features = self.X.shape[1]

        # -------------------------------------------------
        # OBSERVATION SPACE
        # -------------------------------------------------

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_features,),
            dtype=np.float32
        )

        # -------------------------------------------------
        # ACTION SPACE
        # -------------------------------------------------
        #
        # 0 = Allow
        # 1 = Monitor
        # 2 = Block
        # 3 = Quarantine
        #

        self.action_space = spaces.Discrete(4)

        # -------------------------------------------------
        # CATEGORY INDEXES
        # -------------------------------------------------

        self.category_indexes = {}

        for category in self.y.unique():

            self.category_indexes[category] = np.where(
                self.y.values == category
            )[0]

        self.current_index = 0

    # =====================================================
    # RESET
    # =====================================================

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        categories = [
            "normal",
            "dos",
            "probe",
            "r2l",
            "u2r",
            "other"
        ]

        available_categories = [
            category
            for category in categories
            if category in self.category_indexes
            and len(self.category_indexes[category]) > 0
        ]

        # -------------------------------------------------
        # BALANCED CATEGORY SAMPLING
        # -------------------------------------------------

        category = self.np_random.choice(
            available_categories
        )

        self.current_index = self.np_random.choice(
            self.category_indexes[category]
        )

        observation = self.X.iloc[
            self.current_index
        ].values.astype(np.float32)

        return observation, {}

    # =====================================================
    # STEP
    # =====================================================

    def step(self, action):

        category = self.y.iloc[
            self.current_index
        ]

        action = int(action)

        # -------------------------------------------------
        # REWARD FUNCTION
        # -------------------------------------------------

        if category == "normal":

            rewards = {
                0: 5,      # Allow - correct
                1: 2,      # Monitor - acceptable
                2: -5,     # Block - false positive
                3: -6      # Quarantine - false positive
            }

        elif category == "dos":

            rewards = {
                0: -6,     # Allow - dangerous
                1: -2,     # Monitor
                2: 6,      # Block - correct
                3: 3       # Quarantine
            }

        elif category == "probe":

            rewards = {
                0: -5,     # Allow
                1: 2,      # Monitor
                2: 6,      # Block - correct
                3: 3       # Quarantine
            }

        elif category == "r2l":

            rewards = {
                0: -6,     # Allow
                1: 1,      # Monitor
                2: 3,      # Block
                3: 6       # Quarantine - correct
            }

        elif category == "u2r":

            rewards = {
                0: -6,     # Allow
                1: 1,      # Monitor
                2: 3,      # Block
                3: 6       # Quarantine - correct
            }

        else:

            # -------------------------------------------------
            # OTHER ATTACK CATEGORY
            # -------------------------------------------------

            rewards = {
                0: -6,     # Allow
                1: 1,      # Monitor
                2: 6,      # Block - correct
                3: 3       # Quarantine
            }

        reward = rewards[action]

        # -------------------------------------------------
        # ONE-STEP ENVIRONMENT
        # -------------------------------------------------

        terminated = True
        truncated = False

        next_observation = np.zeros(
            self.n_features,
            dtype=np.float32
        )

        info = {
            "attack_category": category,
            "action": action,
            "reward": reward
        }

        return (
            next_observation,
            reward,
            terminated,
            truncated,
            info
        )


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("XRL-GUARD RL ENVIRONMENT TEST")
    print("=" * 60)

    feature_path = "data/processed/X_train.csv"
    label_path = "data/processed/category_train.csv"

    print("\nLoading processed training data...")

    env = XRLGuardEnv(
        feature_path,
        label_path
    )

    print("Environment created successfully.")

    print("\nTraining records :", len(env.X))
    print("Number of features :", env.n_features)
    print("Number of actions  :", env.action_space.n)

    print("\nTraining category distribution:")
    print(env.y.value_counts())

    print("\nActions:")
    print("0 - Allow")
    print("1 - Monitor")
    print("2 - Block")
    print("3 - Quarantine")

    # -------------------------------------------------
    # TEST RESET
    # -------------------------------------------------

    observation, info = env.reset()

    print(
        "\nInitial observation shape:",
        observation.shape
    )

    # -------------------------------------------------
    # TEST ACTION
    # -------------------------------------------------

    action = env.action_space.sample()

    (
        observation,
        reward,
        terminated,
        truncated,
        info
    ) = env.step(action)

    print("Sample action          :", action)
    print("Sample reward          :", reward)
    print("Attack category        :", info["attack_category"])

    print("\n" + "=" * 60)
    print("RL ENVIRONMENT TEST COMPLETED")
    print("=" * 60)
import numpy as np
from stable_baselines3 import PPO
from rl_environment import XRLGuardEnv


feature_path = "data/processed/X_test.csv"
label_path = "data/processed/category_test.csv"

env = XRLGuardEnv(
    feature_path,
    label_path
)

model = PPO.load("models/xrl_guard_ppo")

action_names = {
    0: "Allow",
    1: "Monitor",
    2: "Block",
    3: "Quarantine"
}

print("=" * 60)
print("XRL-GUARD PPO AGENT TEST")
print("=" * 60)

print("\nTesting records :", len(env.X))
print("Features        :", env.n_features)

observation, info = env.reset()

total_reward = 0
action_counts = np.zeros(4, dtype=int)

for step in range(1000):

    action, _ = model.predict(
        observation,
        deterministic=True
    )

    action = int(action)

    observation, reward, terminated, truncated, info = env.step(action)

    total_reward += reward
    action_counts[action] += 1

    if step < 10:
        print(
            f"\nRecord {step + 1}"
            f"\nAttack category : {info['attack_category']}"
            f"\nAction          : {action_names[action]}"
            f"\nReward          : {reward}"
        )

    if terminated or truncated:
        break

print("\n" + "=" * 60)
print("TEST SUMMARY")
print("=" * 60)

print("\nRecords tested :", step + 1)
print("Total reward   :", total_reward)

print("\nAction counts:")

for action in range(4):
    print(
        f"{action} - {action_names[action]} : "
        f"{action_counts[action]}"
    )

print("\n" + "=" * 60)
print("PPO AGENT TEST COMPLETED")
print("=" * 60)
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

MODEL_PATH = "models/xrl_guard_ppo"

ACTION_NAMES = {
    0: "Allow",
    1: "Monitor",
    2: "Block",
    3: "Quarantine"
}

EXPECTED_ACTION = {
    "normal": 0,
    "dos": 2,
    "probe": 2,
    "other": 2,
    "r2l": 3,
    "u2r": 3
}

print("=" * 60)
print("XRL-GUARD AGENT EVALUATION")
print("=" * 60)

feature_path = "data/processed/X_test.csv"
label_path = "data/processed/category_test.csv"

print("\nLoading test data...")

X = pd.read_csv(feature_path)
y = pd.read_csv(label_path)

if y.shape[1] == 1:
    y = y.iloc[:, 0]
else:
    y = y["attack_category"]

print("Testing records :", len(X))
print("Features        :", X.shape[1])

print("\nLoading trained PPO model...")

model = PPO.load(MODEL_PATH)

print("Model loaded successfully.")

predicted_actions = []
expected_actions = []
rewards = []

for i in range(len(X)):

    observation = X.iloc[i].values.astype(np.float32)

    action, _ = model.predict(
        observation,
        deterministic=True
    )

    action = int(action)

    category = y.iloc[i]

    expected = EXPECTED_ACTION.get(category, 0)

    if category == "normal":

        if action == 0:
            reward = 3
        elif action == 1:
            reward = 1
        elif action == 2:
            reward = -2
        else:
            reward = -3

    elif category in ["dos", "probe", "other"]:

        if action == 0:
            reward = -3
        elif action == 1:
            reward = -1
        elif action == 2:
            reward = 3
        else:
            reward = 2

    elif category in ["r2l", "u2r"]:

        if action == 0:
            reward = -3
        elif action == 1:
            reward = -1
        elif action == 2:
            reward = 2
        else:
            reward = 3

    else:
        reward = -3

    predicted_actions.append(action)
    expected_actions.append(expected)
    rewards.append(reward)


accuracy = accuracy_score(
    expected_actions,
    predicted_actions
)

precision = precision_score(
    expected_actions,
    predicted_actions,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    expected_actions,
    predicted_actions,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    expected_actions,
    predicted_actions,
    average="weighted",
    zero_division=0
)

total_reward = sum(rewards)
average_reward = np.mean(rewards)


print("\n" + "=" * 60)
print("EVALUATION RESULTS")
print("=" * 60)

correct = sum(
    p == e
    for p, e in zip(predicted_actions, expected_actions)
)

incorrect = sum(
    p != e
    for p, e in zip(predicted_actions, expected_actions)
)

print("\nTotal records      :", len(X))
print("Correct decisions  :", correct)
print("Incorrect decisions:", incorrect)

print("Accuracy           :", f"{accuracy:.2%}")
print("Precision          :", f"{precision:.2%}")
print("Recall             :", f"{recall:.2%}")
print("F1 Score           :", f"{f1:.2%}")
print("Total reward       :", total_reward)
print("Average reward     :", f"{average_reward:.4f}")


print("\nAction distribution:")

for action in range(4):

    count = predicted_actions.count(action)
    percentage = count / len(X)

    print(
        f"{action} - "
        f"{ACTION_NAMES[action]:11s} : "
        f"{count:5d} "
        f"({percentage:.2%})"
    )


print("\nExpected action distribution:")

for action in range(4):

    count = expected_actions.count(action)
    percentage = count / len(X)

    print(
        f"{action} - "
        f"{ACTION_NAMES[action]:11s} : "
        f"{count:5d} "
        f"({percentage:.2%})"
    )


print("\n" + "=" * 60)
print("PER-ATTACK-CATEGORY RESULTS")
print("=" * 60)

categories = sorted(y.unique())

for category in categories:

    indexes = [
        i
        for i in range(len(y))
        if y.iloc[i] == category
    ]

    correct = sum(
        predicted_actions[i] == expected_actions[i]
        for i in indexes
    )

    total = len(indexes)

    print(
        f"{category:10s}: "
        f"{correct:5d}/{total:5d} "
        f"({correct / total:.2%})"
    )


print("\n" + "=" * 60)
print("PER-ATTACK-CATEGORY ACTION DISTRIBUTION")
print("=" * 60)

for category in categories:

    indexes = [
        i
        for i in range(len(y))
        if y.iloc[i] == category
    ]

    total = len(indexes)

    print("\n" + category.upper())

    print(
        "Expected action :",
        ACTION_NAMES[EXPECTED_ACTION.get(category, 0)]
    )

    for action in range(4):

        count = sum(
            predicted_actions[i] == action
            for i in indexes
        )

        percentage = count / total if total > 0 else 0

        print(
            f"  {ACTION_NAMES[action]:11s}: "
            f"{count:5d} "
            f"({percentage:.2%})"
        )


print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

matrix = confusion_matrix(
    expected_actions,
    predicted_actions,
    labels=[0, 1, 2, 3]
)

print("\nRows = Expected")
print("Columns = Predicted\n")

print(
    f"{'':18s}"
    f"{'Allow':>10s}"
    f"{'Monitor':>10s}"
    f"{'Block':>10s}"
    f"{'Quarantine':>13s}"
)

for i, row in enumerate(matrix):

    print(
        f"{ACTION_NAMES[i]:18s}"
        f"{row[0]:10d}"
        f"{row[1]:10d}"
        f"{row[2]:10d}"
        f"{row[3]:13d}"
    )


print("\n" + "=" * 60)
print("XRL-GUARD AGENT EVALUATION COMPLETED")
print("=" * 60)
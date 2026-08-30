import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from decision_engine import XRLGuardDecisionEngine
from agent import XRLGuardAgent


MODEL_PATH = "models/xrl_guard_ppo"
DATA_PATH = "data/processed/X_test.csv"


print("=" * 60)
print("XRL-GUARD COMPLETE WORKFLOW TEST")
print("=" * 60)

print("\nLoading test data...")

data = pd.read_csv(DATA_PATH)

print("Testing records :", len(data))
print("Features        :", data.shape[1])

observation = data.iloc[0].values.astype(
    np.float32
)

print("\nRunning XRL-Guard analysis...")

engine = XRLGuardDecisionEngine(
    MODEL_PATH
)

agent = XRLGuardAgent(
    MODEL_PATH
)

decision = engine.get_decision(
    observation
)

result = agent.analyze(
    observation
)

print("\n" + "=" * 60)
print("FINAL XRL-GUARD RESULT")
print("=" * 60)

print(
    "\nPredicted action :",
    decision["action"]
)

print(
    "Confidence       :",
    f"{decision['confidence']:.2%}"
)

print(
    "Risk level       :",
    result["risk_level"]
)

print(
    "\nRecommendation   :",
    result["recommendation"]
)

print(
    "\nReason           :",
    decision["reason"]
)

print("\nImportant features:")

for item in decision["important_features"]:

    print(
        f"  {item['feature']:35s}"
        f" {item['importance']:+.6f}"
    )

print("\nAlternative actions:")

for item in decision["alternatives"]:

    print(
        f"  {item['action']:12s}"
        f" {item['confidence']:.2%}"
    )

print("\n" + "=" * 60)
print("XRL-GUARD COMPLETE WORKFLOW COMPLETED")
print("=" * 60)
import os
import random
import uuid
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_PATH))

from agent import XRLGuardAgent


# ---------------------------------------------------------
# FILE PATHS
# ---------------------------------------------------------

MODEL_PATH = str(
    PROJECT_ROOT / "models" / "xrl_guard_ppo"
)

FEATURE_PATH = str(
    PROJECT_ROOT / "data" / "processed" / "X_test.csv"
)

CATEGORY_PATH = str(
    PROJECT_ROOT / "data" / "processed" / "category_test.csv"
)

LOG_PATH = str(
    PROJECT_ROOT / "data" / "live_incidents.csv"
)


# ---------------------------------------------------------
# ACTION NAMES
# ---------------------------------------------------------

ACTION_NAMES = {
    0: "Allow",
    1: "Monitor",
    2: "Block",
    3: "Quarantine"
}


# ---------------------------------------------------------
# SEVERITY
# ---------------------------------------------------------

SEVERITY = {
    "normal": "LOW",
    "dos": "HIGH",
    "probe": "HIGH",
    "other": "HIGH",
    "r2l": "CRITICAL",
    "u2r": "CRITICAL"
}


# ---------------------------------------------------------
# RUNTIME ENGINE
# ---------------------------------------------------------

class XRLGuardRuntimeEngine:

    def __init__(self):

        print("Loading XRL-Guard runtime engine...")

        # Load features
        self.features = pd.read_csv(
            FEATURE_PATH
        )

        # Load categories
        self.categories = (
            pd.read_csv(CATEGORY_PATH)
            .iloc[:, 0]
            .astype(str)
            .str.strip()
            .str.lower()
            .reset_index(drop=True)
        )

        # Check data consistency
        if len(self.features) != len(self.categories):

            raise ValueError(
                "Feature and category records do not match."
            )

        print(
            f"Runtime records available: "
            f"{len(self.features)}"
        )

        # Load trained XRL agent
        print("Loading XRL-Guard agent...")

        self.agent = XRLGuardAgent(
            MODEL_PATH
        )

        print("Agent loaded successfully.")

        # Prepare incident log
        self._prepare_log()

    # -----------------------------------------------------
    # PREPARE INCIDENT LOG
    # -----------------------------------------------------

    def _prepare_log(self):

        os.makedirs(
            os.path.dirname(LOG_PATH),
            exist_ok=True
        )

        if not os.path.exists(LOG_PATH):

            columns = [
                "incident_id",
                "timestamp",
                "record_id",
                "category",
                "severity",
                "suggested_action",
                "confidence",
                "risk_level",
                "recommendation",
                "reason",
                "user_action_status"
            ]

            pd.DataFrame(
                columns=columns
            ).to_csv(
                LOG_PATH,
                index=False
            )

    # -----------------------------------------------------
    # ANALYZE ONE NETWORK EVENT
    # -----------------------------------------------------

    def analyze_record(self, record_id):

        record_id = int(record_id)

        # Validate record ID
        if (
            record_id < 0
            or record_id >= len(self.features)
        ):

            raise ValueError(
                "Invalid record ID."
            )

        # Get network features
        observation = (
            self.features.iloc[record_id]
            .values
            .astype("float32")
        )

        # Ask XRL-Guard agent for recommendation
        result = self.agent.analyze(
            observation
        )

        # Get category
        category = self.categories.iloc[
            record_id
        ]

        # Get suggested action
        suggested_action = str(
            result["action"]
        )

        # Create unique incident ID
        incident_id = (
            f"INC-{uuid.uuid4().hex[:8].upper()}"
        )

        # Current timestamp
        timestamp = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        # Create incident
        incident = {

            "incident_id":
                incident_id,

            "timestamp":
                timestamp,

            "record_id":
                record_id,

            "category":
                category,

            "severity":
                SEVERITY.get(
                    category,
                    "UNKNOWN"
                ),

            "suggested_action":
                suggested_action,

            "confidence":
                round(
                    result["confidence"],
                    4
                ),

            "risk_level":
                result["risk_level"],

            "recommendation":
                result["recommendation"],

            "reason":
                result["reason"],

            "user_action_status":
                "PENDING"
        }

        return incident

    # -----------------------------------------------------
    # GENERATE LIVE INCIDENT
    # -----------------------------------------------------

    def generate_incident(self):

        record_id = random.randrange(
            len(self.features)
        )

        incident = self.analyze_record(
            record_id
        )

        self.save_incident(
            incident
        )

        return incident

    # -----------------------------------------------------
    # SAVE INCIDENT
    # -----------------------------------------------------

    def save_incident(self, incident):

        pd.DataFrame(
            [incident]
        ).to_csv(
            LOG_PATH,
            mode="a",
            header=False,
            index=False
        )

    # -----------------------------------------------------
    # GET RECENT INCIDENTS
    # -----------------------------------------------------

    def get_recent_incidents(
        self,
        limit=20
    ):

        if not os.path.exists(
            LOG_PATH
        ):

            return []

        data = pd.read_csv(
            LOG_PATH
        )

        if data.empty:

            return []

        data = data.tail(
            limit
        )

        data = data.fillna(
            ""
        )

        return data.to_dict(
            orient="records"
        )
        # -----------------------------------------------------
    # MARK USER ACTION AS COMPLETED
    # -----------------------------------------------------

    def complete_incident(self, incident_id):

        if not os.path.exists(LOG_PATH):
            return False

        data = pd.read_csv(LOG_PATH)

        if data.empty:
            return False

        matches = (
            data["incident_id"].astype(str)
            == str(incident_id)
        )

        if not matches.any():
            return False

        data.loc[
            matches,
            "user_action_status"
        ] = "COMPLETED"

        data.to_csv(
            LOG_PATH,
            index=False
        )

        return True


# ---------------------------------------------------------
# TEST RUNTIME ENGINE
# ---------------------------------------------------------

if __name__ == "__main__":

    print()

    print("=" * 60)

    print(
        "XRL-GUARD RUNTIME ENGINE TEST"
    )

    print("=" * 60)

    engine = XRLGuardRuntimeEngine()

    print()

    print(
        "Generating runtime incident..."
    )

    incident = engine.generate_incident()

    print()

    print(
        "LIVE SECURITY INCIDENT"
    )

    print(
        "-----------------------"
    )

    print(
        "Incident ID      :",
        incident["incident_id"]
    )

    print(
        "Timestamp        :",
        incident["timestamp"]
    )

    print(
        "Record ID        :",
        incident["record_id"]
    )

    print(
        "Category         :",
        incident["category"]
    )

    print(
        "Severity         :",
        incident["severity"]
    )

    print(
        "Suggested Action :",
        incident["suggested_action"]
    )

    print(
        "Confidence       :",
        f"{incident['confidence']:.2%}"
    )

    print(
        "Risk Level       :",
        incident["risk_level"]
    )

    print(
        "Recommendation   :",
        incident["recommendation"]
    )

    print(
        "Reason           :",
        incident["reason"]
    )

    print(
        "User Action      :",
        incident["user_action_status"]
    )

    print()

    print("=" * 60)

    print(
        "RUNTIME ENGINE TEST COMPLETED"
    )

    print("=" * 60)
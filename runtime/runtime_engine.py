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

from decision_engine import XRLGuardDecisionEngine


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
# SEVERITY
# ---------------------------------------------------------

SEVERITY = {
    "normal": "LOW",
    "dos": "HIGH",
    "probe": "HIGH",
    "r2l": "CRITICAL",
    "u2r": "CRITICAL",
    "other": "HIGH"
}


# ---------------------------------------------------------
# RUNTIME ENGINE
# ---------------------------------------------------------

class XRLGuardRuntimeEngine:

    def __init__(self):

        print("Loading XRL-GUARD runtime engine...")

        # -------------------------------------------------
        # LOAD PROCESSED TEST FEATURES
        # -------------------------------------------------

        self.features = pd.read_csv(
            FEATURE_PATH
        )

        # -------------------------------------------------
        # LOAD TEST CATEGORIES
        # -------------------------------------------------

        self.categories = (
            pd.read_csv(CATEGORY_PATH)
            .iloc[:, 0]
            .astype(str)
            .str.strip()
            .str.lower()
            .reset_index(drop=True)
        )

        # -------------------------------------------------
        # DATA CONSISTENCY CHECK
        # -------------------------------------------------

        if len(self.features) != len(self.categories):

            raise ValueError(
                "Feature and category records do not match."
            )

        print(
            f"Runtime records available: "
            f"{len(self.features)}"
        )

        # -------------------------------------------------
        # LOAD DECISION ENGINE
        # -------------------------------------------------

        print(
            "Loading XRL-GUARD decision engine..."
        )

        self.engine = XRLGuardDecisionEngine(
            MODEL_PATH
        )

        print(
            "Decision engine loaded successfully."
        )

        # -------------------------------------------------
        # PREPARE INCIDENT LOG
        # -------------------------------------------------

        self._prepare_log()


    # -----------------------------------------------------
    # PREPARE INCIDENT LOG
    # -----------------------------------------------------

    def _prepare_log(self):

        os.makedirs(
            os.path.dirname(LOG_PATH),
            exist_ok=True
        )

        columns = [
            "incident_id",
            "timestamp",
            "record_id",
            "category",
            "severity",
            "suggested_action",
            "confidence",
            "recommendation",
            "reason",
            "user_decision",
            "action_status",
            "deny_reason"
        ]

        # -------------------------------------------------
        # CREATE LOG IF IT DOES NOT EXIST
        # -------------------------------------------------

        if not os.path.exists(LOG_PATH):

            pd.DataFrame(
                {
                    column: pd.Series(dtype="object")
                    for column in columns
                }
            ).to_csv(
                LOG_PATH,
                index=False
            )

            return

        # -------------------------------------------------
        # READ EXISTING LOG
        # -------------------------------------------------

        try:

            data = pd.read_csv(
                LOG_PATH
            )

        except pd.errors.EmptyDataError:

            pd.DataFrame(
                {
                    column: pd.Series(dtype="object")
                    for column in columns
                }
            ).to_csv(
                LOG_PATH,
                index=False
            )

            return

        # -------------------------------------------------
        # ADD MISSING COLUMNS
        # -------------------------------------------------

        changed = False

        for column in columns:

            if column not in data.columns:

                if column == "user_decision":

                    data[column] = "PENDING"

                elif column == "action_status":

                    data[column] = "PENDING"

                else:

                    data[column] = ""

                changed = True

        # -------------------------------------------------
        # FIX TEXT COLUMN TYPES
        # -------------------------------------------------

        text_columns = [
            "incident_id",
            "timestamp",
            "record_id",
            "category",
            "severity",
            "suggested_action",
            "recommendation",
            "reason",
            "user_decision",
            "action_status",
            "deny_reason"
        ]

        for column in text_columns:

            if column in data.columns:

                data[column] = (
                    data[column]
                    .fillna("")
                    .astype(str)
                )

        # -------------------------------------------------
        # KEEP COLUMN ORDER
        # -------------------------------------------------

        data = data[columns]

        # -------------------------------------------------
        # SAVE UPDATED LOG
        # -------------------------------------------------

        if changed:

            data.to_csv(
                LOG_PATH,
                index=False
            )

    # -----------------------------------------------------
    # ANALYZE ONE NETWORK EVENT
    # -----------------------------------------------------

    def analyze_record(self, record_id):

        record_id = int(record_id)

        # -------------------------------------------------
        # VALIDATE RECORD ID
        # -------------------------------------------------

        if (
            record_id < 0
            or record_id >= len(self.features)
        ):

            raise ValueError(
                "Invalid record ID."
            )

        # -------------------------------------------------
        # GET NETWORK FEATURES
        # -------------------------------------------------

        observation = (
            self.features.iloc[record_id]
            .values
            .astype("float32")
        )

        # -------------------------------------------------
        # XRL-GUARD DECISION
        # -------------------------------------------------

        result = self.engine.get_decision(
            observation
        )

        # -------------------------------------------------
        # GET CATEGORY
        # -------------------------------------------------

        category = self.categories.iloc[
            record_id
        ]

        # -------------------------------------------------
        # CREATE INCIDENT ID
        # -------------------------------------------------

        incident_id = (
            f"INC-{uuid.uuid4().hex[:8].upper()}"
        )

        # -------------------------------------------------
        # CURRENT TIMESTAMP
        # -------------------------------------------------

        timestamp = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        # -------------------------------------------------
        # CREATE INCIDENT
        # -------------------------------------------------

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
                result["action"],

            "confidence":
                round(
                    result["confidence"],
                    4
                ),

            "recommendation":
                result["action"],

            "reason":
                result["reason"],

            "user_decision":
                "PENDING",

            "action_status":
                "PENDING",

            "deny_reason":
                ""
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

        try:

            data = pd.read_csv(
                LOG_PATH
            )

        except pd.errors.EmptyDataError:

            return []

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
    # ACCEPT USER DECISION
    # -----------------------------------------------------

    def accept_incident(self, incident_id):

        if not os.path.exists(LOG_PATH):
            return False

        try:
            data = pd.read_csv(LOG_PATH)

        except pd.errors.EmptyDataError:
            return False

        if data.empty:
            return False

        # Make sure text columns can safely store strings
        for column in [
            "user_decision",
            "action_status",
            "deny_reason"
        ]:

            if column not in data.columns:
                data[column] = ""

            data[column] = (
                data[column]
                .fillna("")
                .astype("object")
            )

        matches = data.index[
            data["incident_id"].astype(str) == str(incident_id)
        ]

        if len(matches) == 0:
            return False

        index = matches[0]

        data.loc[index, "user_decision"] = "ACCEPTED"
        data.loc[index, "action_status"] = "COMPLETED"
        data.loc[index, "deny_reason"] = ""

        data.to_csv(
            LOG_PATH,
            index=False
        )

        return True


    # -----------------------------------------------------
    # DENY USER DECISION
    # -----------------------------------------------------

    def deny_incident(self, incident_id, deny_reason=""):

        if not os.path.exists(LOG_PATH):
            return False

        try:
            data = pd.read_csv(LOG_PATH)

        except pd.errors.EmptyDataError:
            return False

        if data.empty:
            return False

        # Make sure required columns exist
        if "user_decision" not in data.columns:
            data["user_decision"] = ""

        if "action_status" not in data.columns:
            data["action_status"] = ""

        if "deny_reason" not in data.columns:
            data["deny_reason"] = ""

        # Force text columns to object/string-compatible dtype
        for column in [
            "user_decision",
            "action_status",
            "deny_reason"
        ]:

            data[column] = (
                data[column]
                .fillna("")
                .astype("object")
            )

        matches = data.index[
            data["incident_id"].astype(str) == str(incident_id)
        ]

        if len(matches) == 0:
            return False

        index = matches[0]

        data.loc[index, "user_decision"] = "DENIED"
        data.loc[index, "action_status"] = "NOT_PERFORMED"
        data.loc[index, "deny_reason"] = str(
            deny_reason
        ).strip()

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
        "Model Confidence :",
        f"{incident['confidence']:.2%}"
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
        "User Decision    :",
        incident["user_decision"]
    )

    print(
        "Action Status    :",
        incident["action_status"]
    )

    print()

    print("=" * 60)

    print(
        "RUNTIME ENGINE TEST COMPLETED"
    )

    print("=" * 60)

import sys
from pathlib import Path

from flask import Flask, jsonify
from flask_cors import CORS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_PATH))

from runtime_engine import XRLGuardRuntimeEngine


# ---------------------------------------------------------
# FLASK APPLICATION
# ---------------------------------------------------------

app = Flask(__name__)


# ---------------------------------------------------------
# CORS CONFIGURATION
# ---------------------------------------------------------

CORS(
    app,
    resources={
        r"/api/*": {
            "origins": [
                "https://127.0.0.1:5000",
                "https://localhost:5000"
            ]
        }
    }
)


# ---------------------------------------------------------
# RUNTIME ENGINE
# ---------------------------------------------------------

engine = XRLGuardRuntimeEngine()


# ---------------------------------------------------------
# HOME
# ---------------------------------------------------------

@app.route("/")
def home():

    return jsonify({
        "service": "XRL-Guard Runtime API",
        "status": "running",
        "endpoints": [
            "/api/status",
            "/api/incidents",
            "/api/generate",
            "/api/incidents/<incident_id>/complete"
        ]
    })


# ---------------------------------------------------------
# API STATUS
# ---------------------------------------------------------

@app.route(
    "/api/status",
    methods=["GET"]
)
def status():

    return jsonify({
        "service": "XRL-Guard Runtime API",
        "status": "running"
    })


# ---------------------------------------------------------
# GET RECENT INCIDENTS
# ---------------------------------------------------------

@app.route(
    "/api/incidents",
    methods=["GET"]
)
def incidents():

    data = engine.get_recent_incidents(
        limit=20
    )

    return jsonify({
        "count": len(data),
        "incidents": data
    })


# ---------------------------------------------------------
# GENERATE NEW LIVE INCIDENT
# ---------------------------------------------------------

@app.route(
    "/api/generate",
    methods=["POST"]
)
def generate():

    incident = engine.generate_incident()

    return jsonify({
        "success": True,
        "incident": incident
    })


# ---------------------------------------------------------
# MARK USER ACTION AS COMPLETED
# ---------------------------------------------------------

@app.route(
    "/api/incidents/<incident_id>/complete",
    methods=["POST"]
)
def complete_incident(incident_id):

    success = engine.complete_incident(
        incident_id
    )

    if not success:

        return jsonify({
            "success": False,
            "message": "Incident not found."
        }), 404

    return jsonify({
        "success": True,
        "message": "User action marked as completed.",
        "incident_id": incident_id,
        "user_action_status": "COMPLETED"
    })


# ---------------------------------------------------------
# START SERVER
# ---------------------------------------------------------

if __name__ == "__main__":

    print()

    print("=" * 60)

    print("XRL-GUARD RUNTIME API")

    print("=" * 60)

    print(
        "HTTPS: https://127.0.0.1:5001"
    )

    print("=" * 60)

    print()

    app.run(
        host="127.0.0.1",
        port=5001,
        debug=False,
        ssl_context=(
            str(
                PROJECT_ROOT
                / "cert"
                / "localhost.crt"
            ),
            str(
                PROJECT_ROOT
                / "cert"
                / "localhost.key"
            )
        )
    )
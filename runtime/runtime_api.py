import sys
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNTIME_PATH = Path(__file__).resolve().parent

sys.path.insert(0, str(RUNTIME_PATH))


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
            "/api/incidents/<incident_id>/accept",
            "/api/incidents/<incident_id>/deny"
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

    try:

        incident = engine.generate_incident()

        return jsonify({
            "success": True,
            "incident": incident
        })

    except Exception as error:

        return jsonify({
            "success": False,
            "message": str(error)
        }), 500


# ---------------------------------------------------------
# ACCEPT INCIDENT
# ---------------------------------------------------------

@app.route(
    "/api/incidents/<incident_id>/accept",
    methods=["POST"]
)
def accept_incident(incident_id):

    try:

        success = engine.accept_incident(
            incident_id
        )

        if not success:

            return jsonify({
                "success": False,
                "message": "Incident not found."
            }), 404

        return jsonify({
            "success": True,
            "message":
                "Recommendation accepted. "
                "User action marked as completed.",
            "incident_id":
                incident_id,
            "user_decision":
                "ACCEPTED",
            "action_status":
                "COMPLETED"
        })

    except Exception as error:

        return jsonify({
            "success": False,
            "message": str(error)
        }), 500


# ---------------------------------------------------------
# DENY INCIDENT
# ---------------------------------------------------------

@app.route(
    "/api/incidents/<incident_id>/deny",
    methods=["POST"]
)
def deny_incident(incident_id):

    try:

        data = request.get_json(
            silent=True
        )

        if data is None:

            data = {}

        deny_reason = data.get(
            "reason",
            ""
        )

        deny_reason = str(
            deny_reason
        ).strip()

        # -------------------------------------------------
        # REASON IS REQUIRED
        # -------------------------------------------------

        if not deny_reason:

            return jsonify({
                "success": False,
                "message":
                    "A reason is required when denying a recommendation."
            }), 400

        # -------------------------------------------------
        # DENY INCIDENT
        # -------------------------------------------------

        success = engine.deny_incident(
            incident_id,
            deny_reason
        )

        if not success:

            return jsonify({
                "success": False,
                "message": "Incident not found."
            }), 404

        return jsonify({
            "success": True,
            "message":
                "Recommendation denied. "
                "Recommended action was not performed.",
            "incident_id":
                incident_id,
            "user_decision":
                "DENIED",
            "action_status":
                "NOT_PERFORMED",
            "deny_reason":
                deny_reason
        })

    except Exception as error:

        return jsonify({
            "success": False,
            "message": str(error)
        }), 500


# ---------------------------------------------------------
# START SERVER
# ---------------------------------------------------------

if __name__ == "__main__":

    print()

    print("=" * 60)

    print(
        "XRL-GUARD RUNTIME API"
    )

    print("=" * 60)

    print(
        "HTTPS: https://127.0.0.1:5001"
    )

    print(
        "API Status:"
        " https://127.0.0.1:5001/api/status"
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

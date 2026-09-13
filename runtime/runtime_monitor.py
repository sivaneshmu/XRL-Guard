import time

from runtime_engine import XRLGuardRuntimeEngine


class XRLGuardRuntimeMonitor:

    def __init__(self):

        self.engine = XRLGuardRuntimeEngine()

        self.running = False

    def process_event(self):

        incident = self.engine.generate_incident()

        print()
        print("-" * 60)
        print("NEW NETWORK EVENT")
        print("-" * 60)

        print(
            "Incident ID      :",
            incident["incident_id"]
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
            "User Action      :",
            incident["user_action_status"]
        )

        return incident

    def start(self, interval=5):

        self.running = True

        print()
        print("=" * 60)
        print("XRL-GUARD RUNTIME MONITOR")
        print("=" * 60)

        print(
            f"Monitoring started. "
            f"New event every {interval} seconds."
        )

        print("Press CTRL+C to stop.")
        print("=" * 60)

        try:

            while self.running:

                self.process_event()

                time.sleep(interval)

        except KeyboardInterrupt:

            print()
            print("Runtime monitoring stopped.")

            self.running = False


if __name__ == "__main__":

    monitor = XRLGuardRuntimeMonitor()

    monitor.start(
        interval=5
    )
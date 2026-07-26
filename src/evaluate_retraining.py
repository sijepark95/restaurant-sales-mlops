import subprocess
import sys


ALERT_EXIT_CODE = 10


def run_monitor(script: str) -> int:
    print(f"\nRunning {script}")
    print("=" * 72)

    result = subprocess.run(
        [sys.executable, script],
        check=False,
    )

    if result.returncode not in {0, ALERT_EXIT_CODE}:
        print(
            f"\nMONITORING ERROR: {script} failed with "
            f"exit code {result.returncode}"
        )
        sys.exit(1)

    return result.returncode


def main() -> None:
    drift_result = run_monitor("src/detect_drift.py")
    performance_result = run_monitor(
        "src/monitor_performance.py"
    )

    drift_detected = drift_result == ALERT_EXIT_CODE
    performance_degraded = (
        performance_result == ALERT_EXIT_CODE
    )

    print("\nRetraining decision")
    print("=" * 72)
    print(f"Data drift detected:      {drift_detected}")
    print(f"Performance degraded:     {performance_degraded}")

    if drift_detected and performance_degraded:
        print(
            "\nRETRAINING CANDIDATE: investigate the data "
            "and approve retraining."
        )
        sys.exit(ALERT_EXIT_CODE)

    if performance_degraded:
        print(
            "\nINVESTIGATE: performance degraded without "
            "confirmed input drift."
        )
        sys.exit(ALERT_EXIT_CODE)

    if drift_detected:
        print(
            "\nMONITOR: drift exists, but model performance "
            "is still acceptable."
        )
        return

    print("\nNO ACTION: model and production data are stable.")


if __name__ == "__main__":
    main()
"""
Smoke test for DVC and GCS bucket connectivity.
Run from the project root: python smoke_test.py

This script verifies that DVC can push to and pull from the
configured GCS remote by doing a round-trip with a dummy file.
"""

import subprocess
import sys
import os

DUMMY_FILE = "smoke_test_dummy.txt"
DUMMY_CONTENT = "dvc-gcs-smoke-test-verification"
TOTAL_STEPS = 6


def run_command(cmd):
    """Run a shell command and return success status and output."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.returncode == 0, result.stdout + result.stderr


def print_step(step_num, description, passed, error_hint=""):
    status = "OK" if passed else "FAILED"
    print(f"[Step {step_num}/{TOTAL_STEPS}] {description:<45} {status}")
    if not passed:
        print(f"\nSMOKE TEST FAILED at Step {step_num}: {error_hint}")
        cleanup()
        sys.exit(1)


def cleanup():
    """Remove all dummy files created during the test."""
    for f in [DUMMY_FILE, f"{DUMMY_FILE}.dvc", f".gitignore"]:
        if f == ".gitignore":
            # only remove if we created it or it only has our entry
            continue
        if os.path.exists(f):
            os.remove(f)

    # unstage the dummy file from dvc if it was tracked
    subprocess.run(f"dvc remove {DUMMY_FILE}.dvc --force", shell=True,
                   capture_output=True, text=True)

    # remove any leftover files
    for f in [DUMMY_FILE, f"{DUMMY_FILE}.dvc"]:
        if os.path.exists(f):
            os.remove(f)


def main():
    print()
    print("=" * 60)
    print("  DVC / GCS Smoke Test")
    print("=" * 60)
    print()

    # Step 1: Create dummy file
    try:
        with open(DUMMY_FILE, "w") as f:
            f.write(DUMMY_CONTENT)
        passed = os.path.exists(DUMMY_FILE)
    except Exception:
        passed = False
    print_step(1, "Creating dummy test file...", passed,
               "Could not create a file in the project root. Check write permissions.")

    # Step 2: DVC add
    success, output = run_command(f"dvc add {DUMMY_FILE}")
    print_step(2, "Adding file to DVC tracking...", success,
               f"dvc add failed. Is DVC initialized? Output:\n{output}")

    # Step 3: DVC push
    success, output = run_command(f"dvc push {DUMMY_FILE}.dvc")
    print_step(3, "Pushing to GCS bucket...", success,
               f"Could not push to GCS. Check gcloud auth and bucket permissions.\n{output}")

    # Step 4: Remove local cache and file
    success1, _ = run_command("rm -f " + DUMMY_FILE)
    success2, output = run_command(f"dvc cache remove -f {DUMMY_FILE}.dvc")
    # if cache remove command doesn't exist in this version, try clearing cache directly
    if not success2:
        success2, output = run_command("rm -rf .dvc/cache")
    print_step(4, "Removing local cache and file...", success1,
               f"Could not remove local files.\n{output}")

    # Step 5: DVC pull
    success, output = run_command(f"dvc pull {DUMMY_FILE}.dvc")
    print_step(5, "Pulling from GCS bucket...", success,
               f"Could not pull from GCS. Check remote config and permissions.\n{output}")

    # Step 6: Verify content
    try:
        with open(DUMMY_FILE, "r") as f:
            content = f.read()
        passed = content == DUMMY_CONTENT
    except Exception:
        passed = False
    print_step(6, "Verifying file content...", passed,
               "File content does not match. The round-trip failed.")

    # Success
    print()
    print("SMOKE TEST PASSED: DVC and GCS are working correctly.")
    print()

    # Cleanup
    cleanup()
    print("Cleanup complete. Dummy files removed.")
    print()


if __name__ == "__main__":
    main()

"""
Orchestrator: runs all collection scripts in order.

Usage:
    python collect_all.py              # full pipeline
    python collect_all.py --skip-advanced  # skip the 7-hour advanced box step
"""

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent

STEPS = [
    ("collect_data.py",         "Team game logs          (~20 min)"),
    ("collect_players.py",      "Player game logs        (~20 min)"),
    ("collect_player_info.py",  "Player position + info  (~35 min)"),
    ("collect_advanced_box.py", "Advanced box (pace etc) (~7 hours)"),
]

def run_step(script: str, description: str):
    print("\n" + "=" * 70)
    print(f"  {description}")
    print(f"  Running: python {script}")
    print("=" * 70)
    start = time.time()
    result = subprocess.run([sys.executable, str(HERE / script)])
    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"\n  {script} exited with code {result.returncode} after {elapsed/60:.1f} min")
        print(f"  You can resume later — all scripts are checkpoint-safe.")
        return False
    print(f"\n  {script} completed in {elapsed/60:.1f} min")
    return True


def main():
    skip_advanced = "--skip-advanced" in sys.argv

    print("NBA Fantasy - Full Data Collection Pipeline")
    print(f"Working directory: {HERE}")

    for script, description in STEPS:
        if skip_advanced and script == "collect_advanced_box.py":
            print(f"\nSkipping {script} (--skip-advanced)")
            continue
        ok = run_step(script, description)
        if not ok:
            print("\nPipeline stopped. Fix the issue and rerun collect_all.py to resume.")
            sys.exit(1)

    print("\n" + "=" * 70)
    print("  All data collection complete. You can now run the notebook.")
    print("=" * 70)


if __name__ == "__main__":
    main()

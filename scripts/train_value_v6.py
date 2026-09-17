"""Train the fixed v6 input-view ablation matrix."""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True); p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda"); p.add_argument("--epochs", type=int, default=140)
    a = p.parse_args()
    for mode in ("none", "task", "top", "both"):
        for seed in (17, 29, 43):
            output = Path(a.out) / f"{mode}_{seed}"
            command = [sys.executable, "-m", "simbench.value.value_v6", "--data", a.data,
                       "--out", str(output), "--view-mode", mode, "--seed", str(seed),
                       "--epochs", str(a.epochs), "--device", a.device]
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

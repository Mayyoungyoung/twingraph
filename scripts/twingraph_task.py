#!/usr/bin/env python3
"""Panda workstation entry point; defaults to the adapter's dry-run command.

Install this file in ~/robot_panda/src/robot/scripts. The implementation stays
in the consolidated TwinGraph project so there is only one maintained copy.
"""
import os
from pathlib import Path
import sys


def main():
    root = Path(os.environ.get("TWINGRAPH_ROOT", "/home/jia/twingraph/code"))
    if not (root / "real_robot" / "panda_adapter.py").is_file():
        raise SystemExit("TwinGraph adapter not found at " + str(root))
    sys.path.insert(0, str(root))
    from real_robot.panda_adapter import main as adapter_main
    return adapter_main()


if __name__ == "__main__":
    raise SystemExit(main())

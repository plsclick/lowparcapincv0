#!/usr/bin/env python3
"""Run the reusable environment inspection and print its report."""

from pathlib import Path
import sys

project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from lowparllm.services.system_info import main


if __name__ == "__main__":
    raise SystemExit(main())

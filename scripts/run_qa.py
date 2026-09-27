"""Run tests with coverage, then enforce the project CRAP threshold."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    commands = (
        (sys.executable, "-m", "coverage", "run", "--branch", "-m", "pytest"),
        (sys.executable, "-m", "coverage", "json", "-o", "coverage.json"),
        (sys.executable, "quality/crap.py"),
    )
    for command in commands:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
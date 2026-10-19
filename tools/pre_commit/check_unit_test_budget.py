# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

"""Count unit Python source lines, including comments, blanks and shared fakes."""

from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[2] / "tests/unit"
    baseline = int((root / ".loc-baseline").read_text().strip())
    lines = sum(len(path.read_text().splitlines()) for path in root.rglob("*.py"))
    print(f"Unit test lines: {lines}; budget: {baseline}")
    if lines > baseline:
        print("Reduce test code or document an approved budget increase in the PR.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

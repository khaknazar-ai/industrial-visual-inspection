"""Launch the local LOCO splicing-connectors demo from a project checkout."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from industrial_inspection.core.paths import find_project_root


def main(argv: list[str] | None = None) -> int:
    try:
        root = find_project_root()
    except FileNotFoundError:
        root = Path(__file__).resolve().parents[2]
    runner = root / "scripts/run_loco_splicing_demo.py"
    if not runner.is_file():
        print(
            "The LOCO demo runner is available from a project source checkout; "
            f"missing: {runner}",
            file=sys.stderr,
        )
        return 2
    arguments = sys.argv[1:] if argv is None else argv
    return subprocess.run(
        [sys.executable, str(runner), *arguments],
        cwd=root,
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())

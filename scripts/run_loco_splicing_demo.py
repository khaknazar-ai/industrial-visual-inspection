"""Fit/reuse the local LOCO normal profile and inspect one image with an overlay."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path


def _absolute(path: Path, root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _profile_cache_path(root: Path, category_root: Path) -> Path:
    tracked_inputs = (
        root / "src/industrial_inspection/loco_components.py",
        root / "src/industrial_inspection/loco_appearance.py",
        root / "scripts/fit_loco_splicing_normal_profile.py",
        category_root.parent / "SOURCE.json",
    )
    digest = hashlib.sha256()
    for path in tracked_inputs:
        if not path.is_file():
            raise FileNotFoundError(f"Required LOCO demo input is missing: {path}")
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return root / ".tmp/loco-demo/profiles" / f"splicing-{digest.hexdigest()[:16]}.json"


def _run(command: list[str], root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="Image to inspect")
    parser.add_argument(
        "--category-root",
        type=Path,
        default=Path("data/external/mvtec_loco/splicing_connectors"),
        help="Local LOCO category directory used to fit the normal profile if needed",
    )
    parser.add_argument("--normal-profile", type=Path, help="Reuse an explicit profile file")
    parser.add_argument("--output-root", type=Path, default=Path(".tmp/loco-demo/runs"))
    args = parser.parse_args(argv)

    image_path = _absolute(args.image, root)
    category_root = _absolute(args.category_root, root)
    if not image_path.is_file():
        parser.error(f"Image does not exist: {image_path}")

    if args.normal_profile is not None:
        profile_path = _absolute(args.normal_profile, root)
        if not profile_path.is_file():
            parser.error(f"Normal profile does not exist: {profile_path}")
    else:
        profile_path = _profile_cache_path(root, category_root)
        if not profile_path.exists():
            fit = _run([
                sys.executable,
                str(root / "scripts/fit_loco_splicing_normal_profile.py"),
                "--category-root", str(category_root),
                "--output", str(profile_path),
            ], root)
            if fit.returncode:
                sys.stderr.write(fit.stderr or fit.stdout)
                return fit.returncode
            if fit.stdout:
                print(fit.stdout, end="")

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    run_dir = _absolute(args.output_root, root) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    report_path = run_dir / "report.json"
    overlay_path = run_dir / "overlay.png"
    inspect = _run([
        sys.executable,
        str(root / "scripts/inspect_loco_splicing_connectors.py"),
        "--image", str(image_path),
        "--normal-profile", str(profile_path),
        "--json-output", str(report_path),
        "--overlay-output", str(overlay_path),
    ], root)
    if inspect.returncode:
        sys.stderr.write(inspect.stderr or inspect.stdout)
        return inspect.returncode

    report = json.loads(report_path.read_text(encoding="utf-8"))
    prediction = report["prediction"]
    print(f"status={prediction['status']}")
    print(f"issues={','.join(prediction['issues']) or 'none'}")
    print(f"profile={profile_path}")
    print(f"report={report_path}")
    print(f"overlay={overlay_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Copy the existing static assets into Vercel's public/** directory.

Vercel serves static files from public/** on the CDN and explicitly warns not
to rely on Flask's ``static_folder``. The dashboard source of truth stays in
``static/``; this build step only creates the deployment copy.
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "static"
DESTINATION = ROOT / "public" / "static"


def main() -> None:
    if not SOURCE.is_dir():
        raise SystemExit(f"Missing static source directory: {SOURCE}")

    if DESTINATION.exists():
        shutil.rmtree(DESTINATION)
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(SOURCE, DESTINATION)
    print(f"Prepared {DESTINATION.relative_to(ROOT)} from static/")


if __name__ == "__main__":
    main()

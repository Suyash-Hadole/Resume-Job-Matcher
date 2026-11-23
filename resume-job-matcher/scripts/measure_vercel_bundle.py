"""Estimate the uncompressed Vercel function bundle for vercel_proxy.py.

A real `vercel build` must still be run in the Vercel project, but this script
gives a reproducible local check without deploying: it adds the source files
that survive .vercelignore to the installed size of the three packages listed
in requirements.txt (and their transitive dependencies).

Usage:
    .venv\\Scripts\\python.exe scripts\\measure_vercel_bundle.py
"""

from __future__ import annotations

import fnmatch
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MB = 1024 * 1024
LIMIT_MB = 500

EXCLUDED_DIRS = {
    ".venv", "venv", "env", ".git", ".vercel", "__pycache__",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", "tests", "scripts",
    "notebooks", "data", "models", "model-cache", "hf-cache", ".cache",
}
EXCLUDED_FILE_PATTERNS = [
    ".env", ".env.*", "*.pyc", "*.pyo", "*.log", "*.bin",
    "*.safetensors", "*.onnx", "*.gguf", "*.ipynb",
]
EXCLUDED_FILES = {
    "requirements-backend.txt", "requirements-dev.txt", "nlp_api.py",
    "Procfile", "BUILD_STATUS.md", "DEMO_GUIDE.md",
    "IMPLEMENTATION_REPORT.md", "RUBRIC_COVERAGE.md", "RUBRIC_MAPPING.md",
}

# Direct requirements plus their runtime transitive dependencies.
DEPENDENCY_DIRS = [
    "flask", "werkzeug", "jinja2", "itsdangerous", "click", "blinker",
    "markupsafe", "requests", "urllib3", "certifi", "charset_normalizer",
    "idna", "dotenv",
]


def dir_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def source_size() -> int:
    total = 0
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if any(part in EXCLUDED_DIRS for part in relative.parts):
            continue
        if path.name in EXCLUDED_FILES:
            continue
        if any(fnmatch.fnmatch(path.name, pattern) for pattern in EXCLUDED_FILE_PATTERNS):
            continue
        if path.is_file():
            total += path.stat().st_size
    return total


def dependency_size() -> tuple[int, list[str]]:
    site_packages = ROOT / ".venv" / "Lib" / "site-packages"
    if not site_packages.is_dir():
        return 0, ["[WARN] .venv not found; dependency size not measured"]

    by_lower_name = {path.name.lower(): path for path in site_packages.iterdir()}
    total = 0
    warnings: list[str] = []
    for name in DEPENDENCY_DIRS:
        path = by_lower_name.get(name)
        if path is None:
            warnings.append(f"[WARN] missing from local .venv: {name}")
            continue
        total += dir_size(path) if path.is_dir() else path.stat().st_size
    return total, warnings


def main() -> int:
    src = source_size()
    deps, warnings = dependency_size()
    total = src + deps

    print(f"Vercel-bound source files : {src / MB:8.2f} MB")
    print(f"Minimal runtime packages  : {deps / MB:8.2f} MB")
    print(f"Estimated uncompressed    : {total / MB:8.2f} MB")
    print(f"Vercel standard limit     : {LIMIT_MB:8d} MB")
    for warning in warnings:
        print(warning)

    if total >= LIMIT_MB * MB:
        print("FAIL: estimated bundle exceeds the standard Python function limit.")
        return 1
    print("PASS: comfortably below the standard Python function limit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

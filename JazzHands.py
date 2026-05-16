"""One-button launcher for Jazz Hands."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from jazzhands import app


ROOT_DIR = Path(__file__).resolve().parent
SETUP_SCRIPT = ROOT_DIR / "Test_Debug_Scripts" / "full_mocap_run.py"


def build_launcher_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch Jazz Hands, optionally running camera setup first.",
        add_help=False,
    )
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Run camera settings, calibration, and movement alignment before launching.",
    )
    parser.add_argument(
        "--no-setup",
        action="store_true",
        help="Skip the setup prompt and launch the app immediately.",
    )
    parser.add_argument(
        "--setup-ui",
        action="store_true",
        help="Open the setup workflow UI instead of running setup in the terminal.",
    )
    parser.add_argument("-h", "--help", action="store_true", help="Show Jazz Hands app help.")
    return parser


def should_run_setup(args: argparse.Namespace) -> bool:
    if args.setup:
        return True
    if args.no_setup:
        return False
    try:
        answer = input("Run camera setup/calibration first? [y/N]: ").strip().lower()
    except EOFError:
        return False
    return answer in {"y", "yes"}


def run_setup(use_ui: bool) -> int:
    command = [sys.executable, str(SETUP_SCRIPT)]
    if not use_ui:
        command.append("--no-ui")
    return subprocess.run(command, cwd=str(ROOT_DIR)).returncode


def main(argv: list[str] | None = None) -> int:
    raw_args = sys.argv[1:] if argv is None else argv
    launcher_args, app_args = build_launcher_parser().parse_known_args(raw_args)

    if launcher_args.help:
        return app.main(["--help"])

    if should_run_setup(launcher_args):
        setup_result = run_setup(launcher_args.setup_ui)
        if setup_result != 0:
            return setup_result

    return app.main(app_args)


if __name__ == "__main__":
    raise SystemExit(main())

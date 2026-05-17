"""One-button launcher for Jazz Hands."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from tkinter import Button, Label, Listbox, Scrollbar, StringVar, Tk, messagebox
from tkinter.constants import BOTH, END, LEFT, RIGHT, SINGLE, Y

from jazzhands import app


ROOT_DIR = Path(__file__).resolve().parent
SETUP_SCRIPT = ROOT_DIR / "Test_Debug_Scripts" / "full_mocap_run.py"
DEFAULT_SCALE = "chromatic"


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


def app_args_include_scale(app_args: list[str]) -> bool:
    return any(arg == "--scale" or arg.startswith("--scale=") for arg in app_args)


def app_args_are_info_only(app_args: list[str]) -> bool:
    return any(arg in {"--help", "-h", "--list-scales"} for arg in app_args)


def choose_scale_with_dialog(default_scale: str = DEFAULT_SCALE) -> str | None:
    scales = sorted(app.SCALE_INTERVALS)
    root = Tk()
    root.title("Jazz Hands Scale")
    root.geometry("320x420")
    root.resizable(False, True)

    selected_scale = StringVar(value=default_scale if default_scale in scales else scales[0])
    Label(root, text="Choose the scale to play").pack(padx=12, pady=(12, 6))

    scrollbar = Scrollbar(root)
    scrollbar.pack(side=RIGHT, fill=Y)
    listbox = Listbox(root, selectmode=SINGLE, yscrollcommand=scrollbar.set)
    listbox.pack(side=LEFT, fill=BOTH, expand=True, padx=(12, 4), pady=(0, 12))
    scrollbar.config(command=listbox.yview)

    for scale_name in scales:
        listbox.insert(END, scale_name)
    selected_index = scales.index(selected_scale.get())
    listbox.selection_set(selected_index)
    listbox.see(selected_index)

    def confirm() -> None:
        selection = listbox.curselection()
        if not selection:
            messagebox.showinfo("Jazz Hands Scale", "Pick a scale first.")
            return
        selected_scale.set(scales[int(selection[0])])
        root.destroy()

    def cancel() -> None:
        selected_scale.set(default_scale)
        root.destroy()

    Button(root, text="Use Scale", command=confirm).pack(pady=(0, 8))
    root.protocol("WM_DELETE_WINDOW", cancel)
    root.mainloop()
    return selected_scale.get()


def choose_scale_in_terminal(default_scale: str = DEFAULT_SCALE) -> str:
    scales = sorted(app.SCALE_INTERVALS)
    print("Choose the scale to play:")
    for index, scale_name in enumerate(scales, start=1):
        default_marker = " default" if scale_name == default_scale else ""
        print(f"  {index:2d}. {scale_name}{default_marker}")
    try:
        answer = input(f"Scale [{default_scale}]: ").strip()
    except EOFError:
        return default_scale
    if not answer:
        return default_scale
    if answer.isdigit():
        index = int(answer)
        if 1 <= index <= len(scales):
            return scales[index - 1]
    normalized = app.normalize_scale_name(answer)
    return normalized


def choose_scale(default_scale: str = DEFAULT_SCALE) -> str:
    try:
        selected = choose_scale_with_dialog(default_scale)
    except Exception as error:
        print(f"Could not open scale picker UI ({error}); using terminal prompt.")
        return choose_scale_in_terminal(default_scale)
    return selected or default_scale


def main(argv: list[str] | None = None) -> int:
    raw_args = sys.argv[1:] if argv is None else argv
    launcher_args, app_args = build_launcher_parser().parse_known_args(raw_args)

    if launcher_args.help:
        return app.main(["--help"])
    if app_args_are_info_only(app_args):
        return app.main(app_args)

    run_camera_setup = should_run_setup(launcher_args)
    if not app_args_include_scale(app_args):
        selected_scale = choose_scale(DEFAULT_SCALE)
        app_args = [*app_args, "--scale", selected_scale]

    if run_camera_setup:
        setup_result = run_setup(launcher_args.setup_ui)
        if setup_result != 0:
            return setup_result

    return app.main(app_args)


if __name__ == "__main__":
    raise SystemExit(main())

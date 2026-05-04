"""
One-click Jazz Hands mocap setup runner.

This launcher applies the saved UVC camera settings, runs the tetrahedral
camera calibration, then runs movement-based axis alignment. The actual camera
preview windows still come from the calibration/alignment scripts, so their
normal controls remain available.
"""

from __future__ import annotations

import argparse
import os
import queue
import shlex
import subprocess
import sys
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import ttk


CAMERA_TESTS_DIR = Path(__file__).resolve().parent
DEFAULT_SETTINGS_JSON = CAMERA_TESTS_DIR / "camera_uvc_settings_values.json"
DEFAULT_CALIBRATION_JSON = CAMERA_TESTS_DIR / "mocap_calibration.json"
DEFAULT_ALIGNED_JSON = CAMERA_TESTS_DIR / "mocap_calibration_aligned.json"
DEFAULT_CAMERA_IDS = "1,2"

UVC_SCRIPT = CAMERA_TESTS_DIR / "camera_uvc_settings.py"
CALIBRATION_SCRIPT = CAMERA_TESTS_DIR / "calibrate_mocap_cameras.py"
ALIGNMENT_SCRIPT = CAMERA_TESTS_DIR / "mocap_movement_alignment.py"


@dataclass(slots=True)
class WorkflowConfig:
    camera_ids: str = DEFAULT_CAMERA_IDS
    settings_json: Path = DEFAULT_SETTINGS_JSON
    calibration_output: Path = DEFAULT_CALIBRATION_JSON
    aligned_output: Path = DEFAULT_ALIGNED_JSON
    apply_uvc: bool = True
    threaded: bool = True
    calibration_extra: str = ""
    alignment_extra: str = ""


def python_command(script: Path) -> list[str]:
    return [sys.executable, "-u", str(script)]


def extra_args(text: str) -> list[str]:
    return shlex.split(text.strip()) if text.strip() else []


def build_uvc_command(config: WorkflowConfig) -> list[str]:
    return [
        *python_command(UVC_SCRIPT),
        "--apply",
        "--camera-ids",
        config.camera_ids,
        "--settings-json",
        str(config.settings_json),
    ]


def build_calibration_command(config: WorkflowConfig) -> list[str]:
    command = [
        *python_command(CALIBRATION_SCRIPT),
        "--cameras",
        config.camera_ids,
        "--output",
        str(config.calibration_output),
        "--exit-after-save",
    ]
    if config.threaded:
        command.append("--threaded")
    command.extend(extra_args(config.calibration_extra))
    return command


def build_alignment_command(config: WorkflowConfig) -> list[str]:
    command = [
        *python_command(ALIGNMENT_SCRIPT),
        "--cameras",
        config.camera_ids,
        "--calibration",
        str(config.calibration_output),
        "--output",
        str(config.aligned_output),
        "--exit-after-save",
    ]
    if config.threaded:
        command.append("--threaded")
    command.extend(extra_args(config.alignment_extra))
    return command


def command_for_step(step: str, config: WorkflowConfig) -> list[str]:
    if step == "uvc":
        return build_uvc_command(config)
    if step == "calibration":
        return build_calibration_command(config)
    if step == "alignment":
        return build_alignment_command(config)
    raise ValueError(f"Unknown step: {step}")


def display_command(command: list[str]) -> str:
    return " ".join(subprocess.list2cmdline([part]) for part in command)


class FullMocapRunApp:
    STEP_TITLES = {
        "uvc": "Apply Camera Settings",
        "calibration": "Camera Calibration",
        "alignment": "Movement Alignment",
    }

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Jazz Hands Full Mocap Run")
        self.root.geometry("980x720")
        self.root.minsize(880, 620)

        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.worker_thread: threading.Thread | None = None
        self.current_process: subprocess.Popen | None = None
        self.stop_requested = False

        self.camera_ids_var = tk.StringVar(value=DEFAULT_CAMERA_IDS)
        self.settings_json_var = tk.StringVar(value=str(DEFAULT_SETTINGS_JSON))
        self.calibration_output_var = tk.StringVar(value=str(DEFAULT_CALIBRATION_JSON))
        self.aligned_output_var = tk.StringVar(value=str(DEFAULT_ALIGNED_JSON))
        self.apply_uvc_var = tk.BooleanVar(value=True)
        self.threaded_var = tk.BooleanVar(value=True)
        self.calibration_extra_var = tk.StringVar(value="")
        self.alignment_extra_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Ready")
        self.step_status_vars = {
            step: tk.StringVar(value="waiting")
            for step in self.STEP_TITLES
        }

        self._build_style()
        self._build_widgets()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._poll_events()

    def _build_style(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Title.TLabel", font=("Segoe UI", 18, "bold"))
        style.configure("Header.TLabel", font=("Segoe UI", 11, "bold"))
        style.configure("Step.TLabel", font=("Segoe UI", 10))
        style.configure("Run.TButton", font=("Segoe UI", 11, "bold"))

    def _build_widgets(self) -> None:
        outer = ttk.Frame(self.root, padding=18)
        outer.pack(fill="both", expand=True)

        ttk.Label(outer, text="Jazz Hands Full Mocap Run", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            outer,
            text=(
                "Apply saved camera settings, calibrate camera poses, then align the world axes. "
                "Calibration: press c to capture, s to save and continue. "
                "Alignment: capture origin/up/forward, then press s to save and finish."
            ),
            wraplength=920,
        ).pack(anchor="w", pady=(4, 14))

        body = ttk.Frame(outer)
        body.pack(fill="both", expand=True)
        left = ttk.Frame(body)
        left.pack(side="left", fill="y", padx=(0, 16))
        right = ttk.Frame(body)
        right.pack(side="right", fill="both", expand=True)

        config_frame = ttk.LabelFrame(left, text="Run Settings", padding=12)
        config_frame.pack(fill="x")
        self._add_entry(config_frame, "Camera IDs", self.camera_ids_var)
        self._add_entry(config_frame, "UVC settings JSON", self.settings_json_var, width=52)
        self._add_entry(config_frame, "Calibration output", self.calibration_output_var, width=52)
        self._add_entry(config_frame, "Aligned output", self.aligned_output_var, width=52)
        self._add_entry(config_frame, "Calibration extra args", self.calibration_extra_var, width=52)
        self._add_entry(config_frame, "Alignment extra args", self.alignment_extra_var, width=52)
        ttk.Checkbutton(config_frame, text="Apply saved UVC camera settings first", variable=self.apply_uvc_var).pack(anchor="w", pady=(8, 0))
        ttk.Checkbutton(config_frame, text="Use threaded camera processing", variable=self.threaded_var).pack(anchor="w")

        step_frame = ttk.LabelFrame(left, text="Workflow", padding=12)
        step_frame.pack(fill="x", pady=(14, 0))
        for step, title in self.STEP_TITLES.items():
            row = ttk.Frame(step_frame)
            row.pack(fill="x", pady=3)
            ttk.Label(row, text=title, style="Step.TLabel", width=24).pack(side="left")
            ttk.Label(row, textvariable=self.step_status_vars[step], style="Step.TLabel").pack(side="left")

        buttons = ttk.Frame(left)
        buttons.pack(fill="x", pady=(14, 0))
        ttk.Button(buttons, text="Run Full Setup", style="Run.TButton", command=self.start_full_run).pack(fill="x")
        ttk.Button(buttons, text="Apply Settings Only", command=lambda: self.start_steps(["uvc"])).pack(fill="x", pady=(8, 0))
        ttk.Button(buttons, text="Calibration Only", command=lambda: self.start_steps(["calibration"])).pack(fill="x", pady=(8, 0))
        ttk.Button(buttons, text="Alignment Only", command=lambda: self.start_steps(["alignment"])).pack(fill="x", pady=(8, 0))
        ttk.Button(buttons, text="Stop Current Step", command=self.stop_current_step).pack(fill="x", pady=(18, 0))

        ttk.Label(right, textvariable=self.status_var, style="Header.TLabel").pack(anchor="w")
        log_frame = ttk.Frame(right)
        log_frame.pack(fill="both", expand=True, pady=(8, 0))
        self.log_text = tk.Text(log_frame, wrap="word", height=24, font=("Consolas", 9))
        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def _add_entry(self, parent: ttk.Frame, label: str, variable: tk.StringVar, width: int = 28) -> None:
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=4)
        ttk.Label(row, text=label, width=20).pack(side="left")
        ttk.Entry(row, textvariable=variable, width=width).pack(side="left", fill="x", expand=True)

    def current_config(self) -> WorkflowConfig:
        return WorkflowConfig(
            camera_ids=self.camera_ids_var.get().strip() or DEFAULT_CAMERA_IDS,
            settings_json=Path(self.settings_json_var.get()).expanduser(),
            calibration_output=Path(self.calibration_output_var.get()).expanduser(),
            aligned_output=Path(self.aligned_output_var.get()).expanduser(),
            apply_uvc=bool(self.apply_uvc_var.get()),
            threaded=bool(self.threaded_var.get()),
            calibration_extra=self.calibration_extra_var.get(),
            alignment_extra=self.alignment_extra_var.get(),
        )

    def start_full_run(self) -> None:
        steps = ["calibration", "alignment"]
        if self.apply_uvc_var.get():
            steps.insert(0, "uvc")
        self.start_steps(steps)

    def start_steps(self, steps: list[str]) -> None:
        if self.worker_thread is not None and self.worker_thread.is_alive():
            self._append_log("[full run] a workflow is already running\n")
            return

        config = self.current_config()
        self.stop_requested = False
        for step in self.STEP_TITLES:
            self.step_status_vars[step].set("waiting")
        self.status_var.set("Running")
        self.worker_thread = threading.Thread(
            target=self._run_steps,
            args=(steps, config),
            daemon=True,
        )
        self.worker_thread.start()

    def stop_current_step(self) -> None:
        self.stop_requested = True
        process = self.current_process
        if process is not None and process.poll() is None:
            self._append_log("[full run] stopping current step...\n")
            process.terminate()

    def _run_steps(self, steps: list[str], config: WorkflowConfig) -> None:
        try:
            for step in steps:
                if self.stop_requested:
                    self.events.put(("step", (step, "stopped")))
                    break
                if step == "alignment" and not config.calibration_output.exists():
                    self.events.put(("log", f"[full run] calibration file not found: {config.calibration_output}\n"))
                    self.events.put(("step", (step, "blocked")))
                    return

                self.events.put(("step", (step, "running")))
                rc = self._run_command(step, command_for_step(step, config))
                if rc != 0:
                    self.events.put(("step", (step, f"failed ({rc})")))
                    return
                self.events.put(("step", (step, "done")))

            self.events.put(("done", "Workflow finished"))
        except Exception as error:
            self.events.put(("log", f"[full run] error: {error}\n"))
            self.events.put(("done", "Workflow stopped with an error"))

    def _run_command(self, step: str, command: list[str]) -> int:
        self.events.put(("log", f"\n[{self.STEP_TITLES[step]}]\n{display_command(command)}\n"))
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(
            command,
            cwd=str(CAMERA_TESTS_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )
        self.current_process = process
        assert process.stdout is not None
        for line in process.stdout:
            self.events.put(("log", line))
        rc = process.wait()
        self.current_process = None
        self.events.put(("log", f"[full run] {self.STEP_TITLES[step]} exited with code {rc}\n"))
        return int(rc)

    def _poll_events(self) -> None:
        try:
            while True:
                event, payload = self.events.get_nowait()
                if event == "log":
                    self._append_log(str(payload))
                elif event == "step":
                    step, status = payload
                    self.step_status_vars[str(step)].set(str(status))
                    self.status_var.set(f"{self.STEP_TITLES[str(step)]}: {status}")
                elif event == "done":
                    self.status_var.set(str(payload))
        except queue.Empty:
            pass
        self.root.after(100, self._poll_events)

    def _append_log(self, text: str) -> None:
        self.log_text.insert("end", text)
        self.log_text.see("end")

    def _on_close(self) -> None:
        self.stop_current_step()
        self.root.destroy()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the full Jazz Hands mocap setup workflow.")
    parser.add_argument("--no-ui", action="store_true", help="Run from the terminal instead of opening the launcher UI.")
    parser.add_argument("--camera-ids", default=DEFAULT_CAMERA_IDS)
    parser.add_argument("--settings-json", type=Path, default=DEFAULT_SETTINGS_JSON)
    parser.add_argument("--calibration-output", type=Path, default=DEFAULT_CALIBRATION_JSON)
    parser.add_argument("--aligned-output", type=Path, default=DEFAULT_ALIGNED_JSON)
    parser.add_argument("--skip-uvc", action="store_true")
    parser.add_argument("--single-threaded", action="store_true")
    parser.add_argument("--calibration-extra", default="")
    parser.add_argument("--alignment-extra", default="")
    return parser


def run_console(args: argparse.Namespace) -> int:
    config = WorkflowConfig(
        camera_ids=args.camera_ids,
        settings_json=args.settings_json,
        calibration_output=args.calibration_output,
        aligned_output=args.aligned_output,
        apply_uvc=not args.skip_uvc,
        threaded=not args.single_threaded,
        calibration_extra=args.calibration_extra,
        alignment_extra=args.alignment_extra,
    )
    steps = ["calibration", "alignment"]
    if config.apply_uvc:
        steps.insert(0, "uvc")

    for step in steps:
        if step == "alignment" and not config.calibration_output.exists():
            print(f"[full run] calibration file not found: {config.calibration_output}")
            return 1
        command = command_for_step(step, config)
        print(f"\n[{FullMocapRunApp.STEP_TITLES[step]}]")
        print(display_command(command))
        completed = subprocess.run(command, cwd=str(CAMERA_TESTS_DIR))
        if completed.returncode != 0:
            return int(completed.returncode)
    return 0


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.no_ui:
        return run_console(args)

    root = tk.Tk()
    FullMocapRunApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

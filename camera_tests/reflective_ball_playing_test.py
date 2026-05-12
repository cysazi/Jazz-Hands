"""
Reflective ball playing test.

This is a stripped-down playing test for two reflective marker balls. It reuses
the mocap position tracking and FL Studio visualizer plane/MIDI logic, but does
not use glove rotation, serial IMU packets, or physical glove buttons.
"""

from __future__ import annotations

import argparse
import ctypes
from pathlib import Path
import sys
import time

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import fl_studio_debug_visualizer as fl_debug
import mocap_tracker as mocap
import mocap_tracker_combined_vispy as combined


DEFAULT_PREVIEW_HZ = 15.0
DEFAULT_VISUALIZER_HZ = 60.0
HAND_LABELS = ("LEFT", "RIGHT")
HAND_SIDE_AXIS = "y"
RIGHT_HAND_HIGHER_ON_SIDE_AXIS = False
CONTROLLED_HAND_LABEL = "RIGHT"
DEFAULT_CALIBRATION_PATH = combined.ALIGNED_MOVEMENT_CALIBRATION_PATH
MIN_OCTAVE_OFFSET = -4
MAX_OCTAVE_OFFSET = 4


class KeyboardPoller:
    KEY_CODES = {
        "SPACE": 0x20,
        "C": 0x43,
        "R": 0x52,
        "ESCAPE": 0x1B,
        "UP": 0x26,
        "DOWN": 0x28,
        "0": 0x30,
        "1": 0x31,
        "2": 0x32,
        "3": 0x33,
        "4": 0x34,
        "5": 0x35,
        "6": 0x36,
        "7": 0x37,
        "8": 0x38,
        "9": 0x39,
    }

    def __init__(self):
        self.enabled = sys.platform == "win32"
        self.previous_states = {name: False for name in self.KEY_CODES}
        self.user32 = None
        if self.enabled:
            try:
                self.user32 = ctypes.windll.user32
            except Exception:
                self.enabled = False

    def update(self) -> tuple[set[str], set[str]]:
        if not self.enabled or self.user32 is None:
            return set(), set()

        pressed: set[str] = set()
        released: set[str] = set()
        for name, code in self.KEY_CODES.items():
            is_down = bool(self.user32.GetAsyncKeyState(code) & 0x8000)
            was_down = self.previous_states.get(name, False)
            if is_down and not was_down:
                pressed.add(name)
            elif was_down and not is_down:
                released.add(name)
            self.previous_states[name] = is_down
        return pressed, released


class BallPlayingVisualizer(fl_debug.DualHandFLStudioVisualizer):
    def _update_left_controls(self) -> None:
        self.current_attack_value = self.midi_velocity


def build_arg_parser() -> argparse.ArgumentParser:
    parser = combined.build_arg_parser()
    parser.description = (
        "Track two reflective marker balls and feed their mocap positions into "
        "the FL Studio plane/MIDI visualizer. Space acts as the draw button."
    )
    parser.set_defaults(
        calibration=str(DEFAULT_CALIBRATION_PATH),
        tracked_point_count=2,
        update_hz=120.0,
    )
    parser.add_argument("--preview-hz", type=float, default=DEFAULT_PREVIEW_HZ)
    parser.add_argument("--visualizer-hz", type=float, default=DEFAULT_VISUALIZER_HZ)
    parser.add_argument(
        "--hand-side-axis",
        choices=("x", "y", "z"),
        default=HAND_SIDE_AXIS,
        help="Mocap axis used to decide left ball vs right ball.",
    )
    parser.add_argument(
        "--right-hand-higher-on-axis",
        action="store_true",
        dest="right_hand_higher_on_axis",
        default=RIGHT_HAND_HIGHER_ON_SIDE_AXIS,
        help="Treat the larger value on --hand-side-axis as the right hand.",
    )
    parser.add_argument(
        "--right-hand-lower-on-axis",
        action="store_false",
        dest="right_hand_higher_on_axis",
        help="Treat the smaller value on --hand-side-axis as the right hand.",
    )
    parser.add_argument(
        "--controlled-hand",
        choices=HAND_LABELS,
        default=CONTROLLED_HAND_LABEL,
        help="The hand/ball Space draws with and number keys retarget.",
    )
    parser.add_argument("--no-midi", action="store_true")
    parser.add_argument("--midi-output-hint", default=None)
    parser.add_argument("--midi-base-note", type=int, default=fl_debug.MIDI_BASE_NOTE)
    parser.add_argument("--midi-velocity", type=int, default=fl_debug.MIDI_VELOCITY)
    parser.add_argument("--left-midi-channel", type=int, default=fl_debug.HAND_MIDI_CHANNELS_1_BASED["LEFT"])
    parser.add_argument("--right-midi-channel", type=int, default=fl_debug.HAND_MIDI_CHANNELS_1_BASED["RIGHT"])
    parser.add_argument("--max-midi-channel", type=int, default=fl_debug.DEFAULT_MAX_MIDI_CHANNEL)
    parser.add_argument(
        "--require-inside-plane",
        action="store_true",
        default=False,
        help="Only play notes when the ball is inside the drawn rectangle.",
    )
    return parser


class ReflectiveBallPlayingApp:
    def __init__(
        self,
        args: argparse.Namespace,
        sources: list[mocap.CameraSource],
        calibrations: dict[int, mocap.CameraCalibration],
    ):
        self.args = args
        self.sources = sources
        self.calibrations = calibrations
        self.preview_camera_ids = [source.camera_id for source in sources[:2]]
        self.settings_by_camera = {
            source.camera_id: mocap.build_detection_settings(args, source.camera_id)
            for source in sources
        }
        self.detectors = {
            camera_id: mocap.ReflectiveMarkerDetector(settings)
            for camera_id, settings in self.settings_by_camera.items()
        }
        self.triangulator = mocap.MultiCameraTriangulator(
            calibrations,
            args.max_reprojection_error,
            args.cluster_distance,
            args.room_bounds,
        )
        self.tracker = mocap.MarkerTracker(
            max_match_distance_m=args.track_distance,
            max_missing_frames=args.max_missing_frames,
            min_confirmed_hits=args.track_confirmation_hits,
            max_tentative_missing_frames=args.tentative_max_missing_frames,
            duplicate_track_distance_m=args.duplicate_track_distance,
            velocity_damping=args.velocity_damping,
            stationary_distance_m=args.stationary_distance,
            max_prediction_dt=args.max_prediction_dt,
        )
        self.keyboard = KeyboardPoller()
        self.visualizer = BallPlayingVisualizer(
            keyboard_controlled=False,
            keyboard_buttons_enabled=True,
            enable_midi=not args.no_midi,
            midi_output_hint=args.midi_output_hint,
            midi_base_note=args.midi_base_note,
            midi_velocity=args.midi_velocity,
            midi_channels_1_based={
                "LEFT": args.left_midi_channel,
                "RIGHT": args.right_midi_channel,
            },
            max_midi_channel=args.max_midi_channel,
            enable_haptics=False,
            require_inside_plane_to_play=args.require_inside_plane,
            controlled_hand_label=args.controlled_hand,
            allow_hand_switching=False,
            update_hz=args.visualizer_hz,
            start_timer=False,
            show=True,
        )
        self.initial_midi_base_note = int(self.visualizer.midi_base_note)
        self.octave_offset = 0
        self.display_track_ids: list[int] = []
        self.track_id_to_hand_label: dict[int, str] = {}
        self.display_positions_by_track: dict[int, np.ndarray] = {}
        self.last_preview_draw_time = 0.0
        self.last_status_print_time = 0.0
        self.last_used_exclusive_pairing = False
        self.closed = False

        if not args.no_preview:
            mocap.cv2.namedWindow(combined.DEFAULT_COMBINED_WINDOW_NAME, mocap.cv2.WINDOW_NORMAL)
            mocap.cv2.resizeWindow(
                combined.DEFAULT_COMBINED_WINDOW_NAME,
                int(args.panel_width) * 2,
                int(args.panel_height) * 2,
            )

        self.visualizer.canvas.events.close.connect(self.close)
        self.timer = fl_debug.app.Timer(
            interval=max(1.0 / max(float(args.update_hz), 1.0), 0.001),
            connect=self.update,
            start=True,
        )

    def update(self, _event) -> None:
        if self.closed:
            return

        timestamp = time.time()
        self._handle_keyboard()
        frames, observations_by_camera = self._read_camera_observations(timestamp)
        mocap.lock_observations_to_existing_tracks(
            frames,
            observations_by_camera,
            self.tracker.tracks,
            self.settings_by_camera,
            self.args.track_memory_pixels,
            timestamp,
        )
        measurements = self._triangulate_measurements(observations_by_camera, timestamp)
        tracks = self.tracker.update(measurements, timestamp)
        live_tracks = [track for track in tracks if track.confirmed and track.missing_frames == 0]
        selected_tracks = self._select_display_tracks(live_tracks)
        assigned_tracks = self._assign_tracks_to_hands(selected_tracks)
        self._feed_tracks_to_visualizer(assigned_tracks, selected_tracks)
        self._update_status_lines(len(measurements), len(live_tracks), assigned_tracks)
        self.visualizer.update(None)

        if not self.args.no_preview:
            self._draw_preview_if_due(timestamp, frames, observations_by_camera, tracks)
            self._handle_cv2_key(mocap.cv2.waitKey(1) & 0xFF)

        self._print_status_if_due(timestamp, len(measurements), len(live_tracks), assigned_tracks)

    def _read_camera_observations(
        self,
        timestamp: float,
    ) -> tuple[dict[int, np.ndarray], dict[int, list[mocap.MarkerObservation]]]:
        frames: dict[int, np.ndarray] = {}
        observations_by_camera: dict[int, list[mocap.MarkerObservation]] = {}
        for source in self.sources:
            ok, frame = source.read()
            if not ok or frame is None:
                observations_by_camera[source.camera_id] = []
                continue
            frames[source.camera_id] = frame
            observations_by_camera[source.camera_id] = self.detectors[source.camera_id].detect(
                frame,
                source.camera_id,
                timestamp,
            )
        return frames, observations_by_camera

    def _triangulate_measurements(
        self,
        observations_by_camera: dict[int, list[mocap.MarkerObservation]],
        timestamp: float,
    ) -> list[mocap.MarkerMeasurement]:
        self.last_used_exclusive_pairing = False
        visible_calibrated_camera_ids = [
            source.camera_id
            for source in self.sources
            if source.camera_id in self.calibrations
            and observations_by_camera.get(source.camera_id)
        ]
        if self.args.exclusive_pairing and len(visible_calibrated_camera_ids) == 2:
            exclusive_measurements = combined.triangulate_exclusive_two_camera_pairs(
                observations_by_camera,
                self.calibrations,
                [source.camera_id for source in self.sources],
                self.args.room_bounds,
                self.args.max_reprojection_error,
                self._expected_measurement_count(observations_by_camera),
                self.args.min_measurement_separation,
                reference_positions=self._pairing_reference_positions(timestamp),
                track_bias_distance_m=self.args.pairing_track_bias_distance,
            )
            if exclusive_measurements:
                self.last_used_exclusive_pairing = True
                return exclusive_measurements

        measurements = self.triangulator.triangulate(observations_by_camera)
        if not self.args.exclusive_pairing:
            return measurements

        expected_count = self._expected_measurement_count(observations_by_camera)
        if len(measurements) >= expected_count:
            return measurements

        exclusive_measurements = combined.triangulate_exclusive_two_camera_pairs(
            observations_by_camera,
            self.calibrations,
            [source.camera_id for source in self.sources],
            self.args.room_bounds,
            self.args.max_reprojection_error,
            expected_count,
            self.args.min_measurement_separation,
            reference_positions=self._pairing_reference_positions(timestamp),
            track_bias_distance_m=self.args.pairing_track_bias_distance,
        )
        if len(exclusive_measurements) > len(measurements):
            self.last_used_exclusive_pairing = True
            return exclusive_measurements
        return measurements

    def _expected_measurement_count(
        self,
        observations_by_camera: dict[int, list[mocap.MarkerObservation]],
    ) -> int:
        calibrated_counts = [
            len(observations_by_camera.get(source.camera_id, []))
            for source in self.sources
            if source.camera_id in self.calibrations
        ]
        if len(calibrated_counts) < 2:
            return 1
        return max(1, min(int(self.args.tracked_point_count), *calibrated_counts[:2]))

    def _pairing_reference_positions(self, timestamp: float) -> list[np.ndarray]:
        max_tracks = max(min(int(self.args.tracked_point_count), len(HAND_LABELS)), 1)
        live_reference_tracks = [
            track
            for track in self.tracker.tracks
            if track.confirmed and track.missing_frames <= self.args.max_missing_frames
        ]
        tracks_by_id = {track.track_id: track for track in live_reference_tracks}
        ordered_tracks = [
            tracks_by_id[track_id]
            for track_id in self.display_track_ids
            if track_id in tracks_by_id
        ]
        ordered_track_ids = {track.track_id for track in ordered_tracks}
        ordered_tracks.extend(
            track
            for track in sorted(live_reference_tracks, key=lambda item: item.track_id)
            if track.track_id not in ordered_track_ids
        )
        return [
            self.tracker._predicted_position(track, timestamp)
            for track in ordered_tracks[:max_tracks]
        ]

    def _select_display_tracks(
        self,
        live_tracks: list[mocap.MarkerTrack],
    ) -> list[mocap.MarkerTrack]:
        max_tracks = max(min(int(self.args.tracked_point_count), len(HAND_LABELS)), 1)
        live_by_id = {track.track_id: track for track in live_tracks}
        selected_ids = [
            track_id
            for track_id in self.display_track_ids
            if track_id in live_by_id
        ][:max_tracks]

        if len(selected_ids) < max_tracks:
            for track in sorted(live_tracks, key=lambda item: item.track_id):
                if track.track_id in selected_ids:
                    continue
                selected_ids.append(track.track_id)
                if len(selected_ids) >= max_tracks:
                    break

        self.display_track_ids = selected_ids
        return [live_by_id[track_id] for track_id in selected_ids]

    def _assign_tracks_to_hands(
        self,
        selected_tracks: list[mocap.MarkerTrack],
    ) -> dict[str, mocap.MarkerTrack]:
        if not selected_tracks:
            self.track_id_to_hand_label = {}
            return {}

        axis_index = {"x": 0, "y": 1, "z": 2}[str(self.args.hand_side_axis).lower()]
        right_is_higher = bool(self.args.right_hand_higher_on_axis)
        assignments: dict[str, mocap.MarkerTrack] = {}

        if len(selected_tracks) >= 2:
            ordered = sorted(selected_tracks, key=lambda track: float(track.position[axis_index]))
            lower_track = ordered[0]
            higher_track = ordered[-1]
            if right_is_higher:
                assignments["RIGHT"] = higher_track
                assignments["LEFT"] = lower_track
            else:
                assignments["RIGHT"] = lower_track
                assignments["LEFT"] = higher_track
        else:
            track = selected_tracks[0]
            previous_label = self.track_id_to_hand_label.get(track.track_id)
            if previous_label in HAND_LABELS:
                assignments[previous_label] = track
            else:
                axis_value = float(track.position[axis_index])
                if (axis_value >= 0.0 and right_is_higher) or (
                    axis_value < 0.0 and not right_is_higher
                ):
                    assignments["RIGHT"] = track
                else:
                    assignments["LEFT"] = track

        self.track_id_to_hand_label = {
            track.track_id: label
            for label, track in assignments.items()
        }
        return assignments

    def _feed_tracks_to_visualizer(
        self,
        assigned_tracks: dict[str, mocap.MarkerTrack],
        selected_tracks: list[mocap.MarkerTrack],
    ) -> None:
        valid_track_ids = {track.track_id for track in selected_tracks}
        for track_id in list(self.display_positions_by_track):
            if track_id not in valid_track_ids:
                self.display_positions_by_track.pop(track_id, None)

        scale = np.array(
            [
                float(self.args.x_scaling_factor),
                float(self.args.y_scaling_factor),
                float(self.args.z_scaling_factor),
            ],
            dtype=np.float64,
        )
        for label in HAND_LABELS:
            track = assigned_tracks.get(label)
            if track is None:
                self.visualizer.set_hand_tracking_active(label, False)
                continue
            self.visualizer.set_hand_pose(
                label,
                self._smoothed_display_position(track) * scale,
                rotation_quaternion=None,
                button_pressed=None,
            )
            self.visualizer.set_hand_tracking_active(label, True)

    def _smoothed_display_position(self, track: mocap.MarkerTrack) -> np.ndarray:
        raw_position = np.asarray(track.position, dtype=np.float64)
        smoothing = float(np.clip(self.args.visual_smoothing, 0.0, 0.98))
        previous_position = self.display_positions_by_track.get(track.track_id)
        if previous_position is None or smoothing <= 0.0:
            display_position = raw_position.copy()
        else:
            display_position = smoothing * previous_position + (1.0 - smoothing) * raw_position
        self.display_positions_by_track[track.track_id] = display_position
        return display_position

    def _handle_keyboard(self) -> None:
        pressed, released = self.keyboard.update()
        if "ESCAPE" in pressed:
            self.visualizer.canvas.close()
            return
        if "C" in pressed or "R" in pressed:
            self.visualizer.clear_plane(self.visualizer.controlled_hand_label)
        if "SPACE" in pressed:
            self.visualizer.set_hand_button(self.visualizer.controlled_hand_label, True)
        if "SPACE" in released:
            self.visualizer.set_hand_button(self.visualizer.controlled_hand_label, False)
        if "UP" in pressed:
            self._adjust_octave(1)
        if "DOWN" in pressed:
            self._adjust_octave(-1)

        for key in sorted(pressed):
            if key.isdigit():
                channel = 10 if key == "0" else int(key)
                if 1 <= channel <= self.visualizer.max_midi_channel:
                    self.visualizer.set_hand_midi_channel(
                        self.visualizer.controlled_hand_label,
                        channel,
                    )

    def _handle_cv2_key(self, key: int) -> None:
        if key in (255, -1):
            return
        if key == 27:
            self.visualizer.canvas.close()
        elif key in (ord("c"), ord("r")):
            self.visualizer.clear_plane(self.visualizer.controlled_hand_label)
        elif key == ord(" ") and not self.keyboard.enabled:
            current = self.visualizer.hands[self.visualizer.controlled_hand_label].button_pressed
            self.visualizer.set_hand_button(self.visualizer.controlled_hand_label, not current)
        elif ord("0") <= key <= ord("9"):
            value = chr(key)
            channel = 10 if value == "0" else int(value)
            if 1 <= channel <= self.visualizer.max_midi_channel:
                self.visualizer.set_hand_midi_channel(
                    self.visualizer.controlled_hand_label,
                    channel,
                )

    def _adjust_octave(self, delta: int) -> None:
        next_offset = int(np.clip(self.octave_offset + int(delta), MIN_OCTAVE_OFFSET, MAX_OCTAVE_OFFSET))
        if next_offset == self.octave_offset:
            return
        self.visualizer._all_midi_notes_off()
        self.octave_offset = next_offset
        self.visualizer.midi_base_note = int(
            np.clip(self.initial_midi_base_note + self.octave_offset * 12, 0, 127)
        )
        print(
            f"[ball playing] octave offset {self.octave_offset:+d}; "
            f"base_note={self.visualizer.midi_base_note}"
        )

    def _update_status_lines(
        self,
        measurement_count: int,
        live_track_count: int,
        assigned_tracks: dict[str, mocap.MarkerTrack],
    ) -> None:
        assignments = " ".join(
            f"{label}=track{track.track_id}" for label, track in sorted(assigned_tracks.items())
        )
        if not assignments:
            assignments = "no tracked balls"
        self.visualizer.set_external_status(
            [
                "Ball controls: Space draw | 1-9/0 channel | Up/Down octave | C/R clear | Esc quit",
                f"octave={self.octave_offset:+d} base_note={self.visualizer.midi_base_note}",
                (
                    f"mocap measurements={measurement_count} live_tracks={live_track_count} "
                    f"exclusive_pairing={'yes' if self.last_used_exclusive_pairing else 'no'} | "
                    f"{assignments}"
                ),
            ]
        )

    def _draw_preview_if_due(
        self,
        timestamp: float,
        frames: dict[int, np.ndarray],
        observations_by_camera: dict[int, list[mocap.MarkerObservation]],
        tracks: list[mocap.MarkerTrack],
    ) -> None:
        interval = 1.0 / max(float(self.args.preview_hz), 1.0)
        if timestamp - self.last_preview_draw_time < interval:
            return
        self.last_preview_draw_time = timestamp
        preview = combined.build_combined_preview(
            self.preview_camera_ids,
            frames,
            observations_by_camera,
            tracks,
            self.settings_by_camera,
            self.args.track_memory_pixels,
            int(self.args.panel_width),
            int(self.args.panel_height),
        )
        mocap.cv2.imshow(combined.DEFAULT_COMBINED_WINDOW_NAME, preview)

    def _print_status_if_due(
        self,
        timestamp: float,
        measurement_count: int,
        live_track_count: int,
        assigned_tracks: dict[str, mocap.MarkerTrack],
    ) -> None:
        if timestamp - self.last_status_print_time < float(self.args.print_interval):
            return
        self.last_status_print_time = timestamp
        assignment_text = ", ".join(
            f"{label}=track{track.track_id}" for label, track in sorted(assigned_tracks.items())
        ) or "none"
        print(
            "[ball playing] "
            f"measurements={measurement_count} live_tracks={live_track_count} "
            f"assignments={assignment_text} octave={self.octave_offset:+d} "
            f"base_note={self.visualizer.midi_base_note}"
        )

    def close(self, *_args) -> None:
        if self.closed:
            return
        self.closed = True
        if hasattr(self, "timer"):
            self.timer.stop()
        self.visualizer.close_midi()
        if not self.args.no_preview:
            mocap.cv2.destroyWindow(combined.DEFAULT_COMBINED_WINDOW_NAME)
        for source in self.sources:
            source.close()


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    combined.apply_scaling_defaults(args)

    if mocap.cv2 is None:
        print("OpenCV is required for camera mocap. Install it with: python -m pip install opencv-python")
        return 1

    fl_debug.configure_vispy_backend()
    calibrations = combined.load_calibrations(args)
    sources = mocap.open_available_cameras(
        args.cameras,
        args.width,
        args.height,
        args.fps,
        args.exposure,
        args.auto_exposure,
        args.gain,
    )
    if not sources:
        print("[ball playing] no cameras opened")
        return 1

    connected_ids = {source.camera_id for source in sources}
    calibrated_connected_ids = connected_ids & set(calibrations)
    if len(calibrated_connected_ids) < 2:
        print(
            "[ball playing] fewer than two connected cameras have calibration. "
            "3D ball tracking needs at least two calibrated views."
        )

    print("[ball playing] controls: Space draw, 1-9/0 channel, Up/Down octave, C/R clear, Esc quit")
    app = ReflectiveBallPlayingApp(args, sources, calibrations)
    try:
        fl_debug.app.run()
    finally:
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

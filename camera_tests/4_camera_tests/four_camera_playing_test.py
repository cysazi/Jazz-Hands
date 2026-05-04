"""
Four-camera mocap playing test.

Cameras 1 and 2 are the front pair for Y/Z. Cameras 3 and 4 are the top pair
for X/Y. The shared Y coordinate fuses both stereo pairs, then the resulting
two tracked marker positions drive the same hand/plane/MIDI logic used by the
two-camera playing test.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

import numpy as np

CAMERA_TESTS_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = CAMERA_TESTS_DIR.parent
for path in (CAMERA_TESTS_DIR, ROOT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import fl_studio_debug_visualizer as fl_debug
import four_camera_shared as shared
import playing_test as two_camera_playing

mocap = shared.mocap


PREVIEW_WINDOW_NAME = "4 camera playing preview"
DEFAULT_PREVIEW_HZ = 30.0
DEFAULT_VISUALIZER_HZ = 120.0
DEFAULT_PLAYING_UPDATE_HZ = 120.0
HAND_LABELS = two_camera_playing.HAND_LABELS
HAND_SIDE_AXIS = two_camera_playing.HAND_SIDE_AXIS
RIGHT_HAND_HIGHER_ON_SIDE_AXIS = two_camera_playing.RIGHT_HAND_HIGHER_ON_SIDE_AXIS
CONTROLLED_HAND_LABEL = two_camera_playing.CONTROLLED_HAND_LABEL
USE_SERIAL_IMU_ROTATION = two_camera_playing.USE_SERIAL_IMU_ROTATION
IMU_PLANE_DRAW_HAND_LABEL = two_camera_playing.IMU_PLANE_DRAW_HAND_LABEL
IMU_CHANNEL_CYCLE_HAND_LABEL = two_camera_playing.IMU_CHANNEL_CYCLE_HAND_LABEL
IMU_CHANNEL_CYCLE_TARGET_HAND_LABEL = two_camera_playing.IMU_CHANNEL_CYCLE_TARGET_HAND_LABEL
IMU_SERIAL_PORT = two_camera_playing.IMU_SERIAL_PORT
IMU_SERIAL_BAUD = two_camera_playing.IMU_SERIAL_BAUD
IMU_PACKET_STALE_SECONDS = two_camera_playing.IMU_PACKET_STALE_SECONDS


def build_arg_parser() -> argparse.ArgumentParser:
    parser = mocap.build_arg_parser()
    parser.description = (
        "Track two mocap markers with four cameras, then feed LEFT/RIGHT hand "
        "positions into the FL Studio visualizer and MIDI plane logic."
    )
    parser.set_defaults(
        cameras=list(shared.CAMERA_IDS),
        calibration=str(shared.default_calibration_path()),
        update_hz=DEFAULT_PLAYING_UPDATE_HZ,
    )
    parser.add_argument("--front-cameras", type=shared.parse_camera_pair, default=tuple(shared.FRONT_CAMERA_IDS))
    parser.add_argument("--top-cameras", type=shared.parse_camera_pair, default=tuple(shared.TOP_CAMERA_IDS))
    parser.add_argument("--fusion-y-tolerance", type=float, default=shared.DEFAULT_FUSION_Y_TOLERANCE_M)
    parser.add_argument(
        "--max-fused-reprojection-error",
        type=float,
        default=shared.DEFAULT_MAX_FUSED_REPROJECTION_ERROR_PX,
    )
    parser.add_argument("--max-layout-measurements", type=int, default=shared.DEFAULT_TRACKED_POINT_COUNT)
    parser.add_argument("--min-measurement-separation", type=float, default=shared.DEFAULT_MIN_MEASUREMENT_SEPARATION_M)
    parser.add_argument("--pairing-track-bias-distance", type=float, default=shared.DEFAULT_PAIRING_TRACK_BIAS_DISTANCE_M)
    parser.add_argument("--tracked-point-count", type=int, default=shared.DEFAULT_TRACKED_POINT_COUNT)
    parser.add_argument("--scaling-factor", type=float, default=shared.DEFAULT_SCALING_FACTOR)
    parser.add_argument("--x-scaling-factor", type=float, default=None)
    parser.add_argument("--y-scaling-factor", type=float, default=None)
    parser.add_argument("--z-scaling-factor", type=float, default=None)
    parser.add_argument(
        "--position-scale",
        dest="scaling_factor",
        type=float,
        default=argparse.SUPPRESS,
        help="Alias for --scaling-factor.",
    )
    parser.add_argument("--visual-smoothing", type=float, default=shared.DEFAULT_VISUAL_SMOOTHING)
    parser.add_argument("--preview-hz", type=float, default=DEFAULT_PREVIEW_HZ)
    parser.add_argument("--visualizer-hz", type=float, default=DEFAULT_VISUALIZER_HZ)
    parser.add_argument("--panel-width", type=int, default=shared.DEFAULT_PANEL_WIDTH)
    parser.add_argument("--panel-height", type=int, default=shared.DEFAULT_PANEL_HEIGHT)
    parser.add_argument(
        "--hand-side-axis",
        choices=("x", "y", "z"),
        default=HAND_SIDE_AXIS,
        help="Mocap axis used to decide left hand vs right hand.",
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
        help="Only this hand responds to keyboard drawing/clear controls for now.",
    )
    parser.add_argument(
        "--use-serial-imu-rotation",
        action="store_true",
        default=USE_SERIAL_IMU_ROTATION,
        help="Read receiver ESP32 binary packets and use IMU quaternions/buttons.",
    )
    parser.add_argument(
        "--no-serial-imu-rotation",
        action="store_false",
        dest="use_serial_imu_rotation",
        help="Disable receiver ESP32 serial IMU input.",
    )
    parser.add_argument("--imu-serial-port", default=IMU_SERIAL_PORT)
    parser.add_argument("--imu-serial-baud", type=int, default=IMU_SERIAL_BAUD)
    parser.add_argument("--imu-packet-stale-seconds", type=float, default=IMU_PACKET_STALE_SECONDS)
    parser.add_argument(
        "--imu-plane-draw-hand",
        choices=HAND_LABELS,
        default=IMU_PLANE_DRAW_HAND_LABEL,
        help="Only this hand module's physical button draws the plane.",
    )
    parser.add_argument(
        "--imu-channel-cycle-hand",
        choices=HAND_LABELS,
        default=IMU_CHANNEL_CYCLE_HAND_LABEL,
        help="This hand module's physical button cycles a MIDI channel.",
    )
    parser.add_argument(
        "--imu-channel-cycle-target-hand",
        choices=(*HAND_LABELS, "CONTROLLED"),
        default=IMU_CHANNEL_CYCLE_TARGET_HAND_LABEL,
        help="Which hand channel the cycle button changes. CONTROLLED means --controlled-hand.",
    )
    parser.add_argument("--no-midi", action="store_true")
    parser.add_argument("--midi-output-hint", default=fl_debug.MIDI_OUTPUT_HINT)
    parser.add_argument("--midi-base-note", type=int, default=fl_debug.MIDI_BASE_NOTE)
    parser.add_argument("--midi-velocity", type=int, default=fl_debug.MIDI_VELOCITY)
    parser.add_argument(
        "--left-midi-channel",
        type=int,
        default=fl_debug.HAND_MIDI_CHANNELS_1_BASED["LEFT"],
    )
    parser.add_argument(
        "--right-midi-channel",
        type=int,
        default=fl_debug.HAND_MIDI_CHANNELS_1_BASED["RIGHT"],
    )
    parser.add_argument("--max-midi-channel", type=int, default=fl_debug.DEFAULT_MAX_MIDI_CHANNEL)
    parser.add_argument(
        "--require-inside-plane",
        action="store_true",
        help="Require the marker to stay inside the drawn plane bounds before notes play.",
    )
    return parser


class FourCameraPlayingTestApp:
    def __init__(
        self,
        args: argparse.Namespace,
        workers: list[shared.ThreadedMocapCamera],
        calibrations: dict[int, mocap.CameraCalibration],
        stop_event,
    ) -> None:
        self.args = args
        self.workers = workers
        self.calibrations = calibrations
        self.stop_event = stop_event
        self.camera_ids = [worker.camera_id for worker in workers]
        self.settings_by_camera = {worker.camera_id: worker.settings for worker in workers}
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
        self.display_track_ids: list[int] = []
        self.display_positions_by_track: dict[int, np.ndarray] = {}
        self.track_id_to_hand_label: dict[int, str] = {}
        self.last_hand_assignments: dict[str, int] = {}
        self.last_tracked_hand_labels: set[str] = set()
        self.last_processed_frame_numbers: dict[int, int] = {}
        self.last_measurement_count = 0
        self.last_live_track_count = 0
        self.last_fusion_diagnostics = shared.FusionDiagnostics(0, 0, 0, 0)
        self.last_print_time = 0.0
        self.last_camera_stats_time = 0.0
        self.last_preview_draw_time = 0.0
        self.last_visualizer_update_time = 0.0
        self.cv2_space_button = False
        self.last_imu_channel_cycle_button_pressed = False
        self.closed = False

        self.keyboard = two_camera_playing.KeyboardPoller()
        self.imu_reader: two_camera_playing.SerialImuReader | None = None
        self.visualizer = fl_debug.DualHandFLStudioVisualizer(
            keyboard_controlled=False,
            keyboard_buttons_enabled=True,
            enable_midi=not args.no_midi,
            midi_output_hint=args.midi_output_hint,
            midi_base_note=args.midi_base_note,
            midi_velocity=args.midi_velocity,
            midi_channels_1_based={"LEFT": args.left_midi_channel, "RIGHT": args.right_midi_channel},
            max_midi_channel=args.max_midi_channel,
            require_inside_plane_to_play=args.require_inside_plane,
            controlled_hand_label=args.controlled_hand,
            allow_hand_switching=False,
            update_hz=args.visualizer_hz,
            start_timer=False,
            show=True,
        )
        self.visualizer.canvas.events.close.connect(self.close)
        self._setup_preview_window()

        if args.use_serial_imu_rotation:
            self.imu_reader = two_camera_playing.SerialImuReader(
                args.imu_serial_port,
                args.imu_serial_baud,
                args.imu_packet_stale_seconds,
                self.stop_event,
            )
            self.imu_reader.start()
            print(f"[4cam playing] IMU plane draw button is mapped to {args.imu_plane_draw_hand}")
            print(
                "[4cam playing] IMU channel cycle button is mapped to "
                f"{args.imu_channel_cycle_hand} -> {args.imu_channel_cycle_target_hand}"
            )

        self.timer = fl_debug.app.Timer(
            interval=max(1.0 / max(float(args.update_hz), 1.0), 0.001),
            connect=self.update,
            start=True,
        )

    def _setup_preview_window(self) -> None:
        if self.args.no_preview:
            return
        mocap.cv2.namedWindow(PREVIEW_WINDOW_NAME, mocap.cv2.WINDOW_NORMAL)
        mocap.cv2.resizeWindow(
            PREVIEW_WINDOW_NAME,
            int(self.args.panel_width) * 4,
            int(self.args.panel_height) * 2,
        )

    def update(self, _event) -> None:
        if self.closed:
            return

        timestamp = time.time()
        self._handle_polled_keyboard()
        snapshots = shared.collect_snapshots(self.workers)
        if not self._has_new_camera_data(snapshots):
            self._pump_preview_keyboard()
            self._update_visualizer_if_due(timestamp)
            return
        self._mark_camera_data_processed(snapshots)

        frames = shared.frames_from_snapshots(snapshots)
        observations_by_camera = shared.observations_from_snapshots(snapshots)
        self.last_camera_stats_time = shared.print_threaded_camera_stats(
            "4cam playing",
            snapshots,
            max(float(self.args.print_interval), 0.25),
            self.last_camera_stats_time,
        )

        mocap.lock_observations_to_existing_tracks(
            frames,
            observations_by_camera,
            self.tracker.tracks,
            self.settings_by_camera,
            self.args.track_memory_pixels,
            timestamp,
        )
        measurements, diagnostics = shared.fuse_layout_measurements(
            observations_by_camera,
            self.calibrations,
            tuple(self.args.front_cameras),
            tuple(self.args.top_cameras),
            self.args.room_bounds,
            self.args.max_reprojection_error,
            self.args.fusion_y_tolerance,
            self.args.max_fused_reprojection_error,
            min(int(self.args.max_layout_measurements), int(self.args.tracked_point_count)),
            self.args.min_measurement_separation,
            self._pairing_reference_positions(timestamp),
            self.args.pairing_track_bias_distance,
        )
        self.last_fusion_diagnostics = diagnostics
        self.last_measurement_count = len(measurements)
        tracks = self.tracker.update(measurements, timestamp)
        live_tracks = [
            track for track in tracks if track.confirmed and track.missing_frames == 0
        ]
        self.last_live_track_count = len(live_tracks)
        selected_tracks = self._select_display_tracks(live_tracks)
        self._feed_tracks_to_visualizer(selected_tracks)
        self._update_visualizer_if_due(timestamp)
        self._print_status(timestamp, tracks, observations_by_camera, snapshots, selected_tracks)

        if not self.args.no_preview:
            preview_interval = 1.0 / max(float(self.args.preview_hz), 1.0)
            if timestamp - self.last_preview_draw_time >= preview_interval:
                self.last_preview_draw_time = timestamp
                preview = shared.build_four_camera_preview(
                    list(self.args.cameras),
                    snapshots,
                    tracks,
                    self.settings_by_camera,
                    self.args.track_memory_pixels,
                    int(self.args.panel_width),
                    int(self.args.panel_height),
                    tuple(self.args.front_cameras),
                    tuple(self.args.top_cameras),
                )
                mocap.cv2.imshow(PREVIEW_WINDOW_NAME, preview)
            key = mocap.cv2.waitKey(1) & 0xFF
            self._handle_cv2_key(key)

    def _select_display_tracks(self, live_tracks: list[mocap.MarkerTrack]) -> list[mocap.MarkerTrack]:
        max_tracks = max(min(int(self.args.tracked_point_count), len(HAND_LABELS)), 1)
        live_by_id = {track.track_id: track for track in live_tracks}
        selected_ids = [
            track_id for track_id in self.display_track_ids if track_id in live_by_id
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

    def _feed_tracks_to_visualizer(self, selected_tracks: list[mocap.MarkerTrack]) -> None:
        assigned_tracks = self._assign_tracks_to_hands(selected_tracks)
        self._prune_display_positions({track.track_id for track in selected_tracks})
        self.last_hand_assignments = {
            label: track.track_id for label, track in assigned_tracks.items()
        }
        self.last_tracked_hand_labels = set(assigned_tracks)

        for label in HAND_LABELS:
            track = assigned_tracks.get(label)
            if track is None:
                self.visualizer.set_hand_tracking_active(label, False)
                continue
            display_position = self._smoothed_display_position(track)
            self.visualizer.set_hand_pose(
                label,
                shared.scaled_position(display_position, self.args),
                rotation_quaternion=None,
                button_pressed=None,
            )
            self.visualizer.set_hand_tracking_active(label, True)
        self._feed_imu_to_visualizer(set(assigned_tracks))

    def _feed_imu_to_visualizer(self, tracked_hand_labels: set[str]) -> None:
        plane_draw_hand = str(self.args.imu_plane_draw_hand).upper()
        channel_cycle_hand = str(self.args.imu_channel_cycle_hand).upper()
        now = time.time()
        for label in HAND_LABELS:
            imu_snapshot = self._imu_snapshot_for_hand(label)
            button_pressed = False if label == plane_draw_hand else None
            if imu_snapshot is None:
                if label == channel_cycle_hand:
                    self.last_imu_channel_cycle_button_pressed = False
                self.visualizer.set_hand_imu_state(label, button_pressed=button_pressed)
                continue

            rotation_quaternion = (
                self._correct_imu_quaternion(label, imu_snapshot.quaternion)
                if imu_snapshot.has_fresh_quat(self.args.imu_packet_stale_seconds)
                else None
            )
            if label == channel_cycle_hand:
                channel_cycle_button_pressed = self._fresh_imu_button_pressed(
                    label,
                    imu_snapshot,
                    now,
                    tracked_hand_labels,
                    require_tracking=False,
                )
                if channel_cycle_button_pressed and not self.last_imu_channel_cycle_button_pressed:
                    self.visualizer.cycle_hand_midi_channel(self._channel_cycle_target_hand(), 1)
                self.last_imu_channel_cycle_button_pressed = channel_cycle_button_pressed
            if label == plane_draw_hand:
                button_pressed = self._fresh_imu_button_pressed(
                    label,
                    imu_snapshot,
                    now,
                    tracked_hand_labels,
                    require_tracking=True,
                )
            self.visualizer.set_hand_imu_state(
                label,
                rotation_quaternion=rotation_quaternion,
                button_pressed=button_pressed,
            )

    def _fresh_imu_button_pressed(
        self,
        label: str,
        imu_snapshot,
        now: float,
        tracked_hand_labels: set[str],
        require_tracking: bool,
    ) -> bool:
        button_is_fresh = (now - imu_snapshot.receive_time) <= self.args.imu_packet_stale_seconds
        if require_tracking and label not in tracked_hand_labels:
            return False
        return bool(
            button_is_fresh
            and (imu_snapshot.packet_type & two_camera_playing.IMU_PACKET_HAS_BUTTON)
            and imu_snapshot.button_pressed
        )

    def _channel_cycle_target_hand(self) -> str:
        target = str(self.args.imu_channel_cycle_target_hand).upper()
        if target == "CONTROLLED":
            return self.visualizer.controlled_hand_label
        return target

    def _correct_imu_quaternion(self, label: str, quaternion: np.ndarray) -> np.ndarray:
        signs = two_camera_playing.IMU_QUATERNION_COMPONENT_SIGNS.get(
            label,
            np.ones(4, dtype=np.float64),
        )
        corrected = fl_debug.normalize_quat(np.asarray(quaternion, dtype=np.float64) * signs)
        offset = two_camera_playing.IMU_MOUNTING_QUATERNION_OFFSETS.get(
            label,
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        )
        return fl_debug.normalize_quat(fl_debug.quaternion_multiply(corrected, offset))

    def _imu_snapshot_for_hand(self, label: str):
        if self.imu_reader is None:
            return None
        return self.imu_reader.snapshot_for_hand(label)

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

    def _prune_display_positions(self, valid_track_ids: set[int]) -> None:
        for track_id in list(self.display_positions_by_track):
            if track_id not in valid_track_ids:
                self.display_positions_by_track.pop(track_id, None)

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
                if (axis_value >= 0.0 and right_is_higher) or (axis_value < 0.0 and not right_is_higher):
                    assignments["RIGHT"] = track
                else:
                    assignments["LEFT"] = track

        self.track_id_to_hand_label = {
            track.track_id: label for label, track in assignments.items()
        }
        return assignments

    def _has_new_camera_data(self, snapshots: dict[int, shared.CameraSnapshot]) -> bool:
        for camera_id, snapshot in snapshots.items():
            if snapshot.frame_number <= 0:
                continue
            if self.last_processed_frame_numbers.get(camera_id) != snapshot.frame_number:
                return True
        return False

    def _mark_camera_data_processed(self, snapshots: dict[int, shared.CameraSnapshot]) -> None:
        for camera_id, snapshot in snapshots.items():
            self.last_processed_frame_numbers[camera_id] = snapshot.frame_number

    def _handle_polled_keyboard(self) -> None:
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

    def _pump_preview_keyboard(self) -> None:
        if self.args.no_preview:
            return
        key = mocap.cv2.waitKey(1) & 0xFF
        self._handle_cv2_key(key)

    def _handle_cv2_key(self, key: int) -> None:
        if key in (255, -1):
            return
        if key == 27:
            self.visualizer.canvas.close()
        elif key in (ord("c"), ord("r")):
            self.visualizer.clear_plane(self.visualizer.controlled_hand_label)
        elif key == ord(" ") and not self.keyboard.enabled:
            self.cv2_space_button = not self.cv2_space_button
            self.visualizer.set_hand_button(
                self.visualizer.controlled_hand_label,
                self.cv2_space_button,
            )

    def _update_visualizer_if_due(self, timestamp: float, force: bool = False) -> None:
        interval = 1.0 / max(float(self.args.visualizer_hz), 1.0)
        if force or timestamp - self.last_visualizer_update_time >= interval:
            self.last_visualizer_update_time = timestamp
            self._feed_imu_to_visualizer(self.last_tracked_hand_labels)
            self.visualizer.set_external_status(self._external_status_lines())
            self.visualizer.update(None)

    def _external_status_lines(self) -> list[str]:
        diagnostics = self.last_fusion_diagnostics
        mapping = ", ".join(
            f"{label}=track {track_id}" for label, track_id in sorted(self.last_hand_assignments.items())
        ) or "none"
        layout_line = (
            f"4cam front={tuple(self.args.front_cameras)} Y/Z "
            f"top={tuple(self.args.top_cameras)} X/Y "
            f"fused={diagnostics.fused_measurements} "
            f"front_candidates={diagnostics.front_candidates} "
            f"top_candidates={diagnostics.top_candidates} "
            f"hands={mapping} | {shared.scale_text(self.args)}"
        )
        return [layout_line, *self._imu_status_lines()]

    def _imu_status_lines(self) -> list[str]:
        if self.imu_reader is None:
            return [f"imu {label}: off" for label in HAND_LABELS]
        return self.imu_reader.status_lines()

    def _imu_status_text(self) -> str:
        return " | ".join(self._imu_status_lines())

    def _print_status(
        self,
        timestamp: float,
        tracks: list[mocap.MarkerTrack],
        observations_by_camera: dict[int, list[mocap.MarkerObservation]],
        snapshots: dict[int, shared.CameraSnapshot],
        selected_tracks: list[mocap.MarkerTrack],
    ) -> None:
        if timestamp - self.last_print_time < self.args.print_interval:
            return
        self.last_print_time = timestamp
        diagnostics = self.last_fusion_diagnostics
        blob_counts = ", ".join(
            f"cam {camera_id}: {len(observations)}"
            for camera_id, observations in sorted(observations_by_camera.items())
        )
        mapping = ", ".join(
            f"{label}=track {track_id}" for label, track_id in sorted(self.last_hand_assignments.items())
        ) or "none"
        camera_parts = []
        for camera_id, snapshot in sorted(snapshots.items()):
            if snapshot.error:
                camera_parts.append(f"cam {camera_id}: {snapshot.error}")
            elif snapshot.opened:
                camera_parts.append(
                    f"cam {camera_id}: {snapshot.fps:.1f}fps "
                    f"read={snapshot.read_ms:.1f}ms detect={snapshot.detect_ms:.1f}ms"
                )
        print(
            "[4cam playing] "
            f"measurements={self.last_measurement_count}, "
            f"live_tracks={self.last_live_track_count}, "
            f"selected={len(selected_tracks)}, "
            f"front_candidates={diagnostics.front_candidates}, "
            f"top_candidates={diagnostics.top_candidates}, "
            f"y_matches={diagnostics.y_matched_pairs}, "
            f"fused={diagnostics.fused_measurements}, "
            f"hands={mapping}, "
            f"blobs=({blob_counts}), "
            f"{self._imu_status_text()}, "
            + " | ".join(camera_parts)
        )

    def close(self, _event=None) -> None:
        if self.closed:
            return
        self.closed = True
        if hasattr(self, "timer"):
            self.timer.stop()
        self.stop_event.set()
        shared.stop_threaded_cameras(self.stop_event, self.workers)
        if self.imu_reader is not None:
            self.imu_reader.join(timeout=0.5)
        self.visualizer.close()
        if mocap.cv2 is not None:
            mocap.cv2.destroyAllWindows()


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    shared.apply_scaling_defaults(args)

    if mocap.cv2 is None:
        print("OpenCV is required for camera mocap. Install it with: python -m pip install opencv-python")
        return 1

    try:
        backend = fl_debug.configure_vispy_backend()
    except RuntimeError as error:
        print(f"[4cam playing] {error}")
        return 1
    print(f"[4cam playing] VisPy backend: {backend}")

    calibrations = shared.load_calibrations(args, "4cam playing")
    stop_event, workers = shared.start_threaded_cameras(
        args,
        list(args.cameras),
        build_masks=not args.no_preview,
        label="4cam playing",
    )
    shared.wait_for_open_attempts(workers)
    snapshots = shared.collect_snapshots(workers)
    if not shared.any_camera_open(snapshots) and shared.all_open_attempts_done(snapshots):
        print("[4cam playing] no cameras opened")
        shared.stop_threaded_cameras(stop_event, workers)
        return 1

    connected_ids = {camera_id for camera_id, snapshot in snapshots.items() if snapshot.opened}
    missing_front = set(args.front_cameras) - connected_ids
    missing_top = set(args.top_cameras) - connected_ids
    if missing_front or missing_top:
        print(
            "[4cam playing] warning: layout pair not fully connected yet "
            f"(missing front={sorted(missing_front)}, top={sorted(missing_top)})"
        )

    visualizer: FourCameraPlayingTestApp | None = None
    try:
        visualizer = FourCameraPlayingTestApp(args, workers, calibrations, stop_event)
        fl_debug.app.run()
    except KeyboardInterrupt:
        print("\n[4cam playing] stopped")
    finally:
        if visualizer is not None:
            visualizer.close()
        else:
            shared.stop_threaded_cameras(stop_event, workers)
            if mocap.cv2 is not None:
                mocap.cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import time
from collections import deque

import numpy as np

try:
    from vispy import app, scene  # type: ignore[import-untyped]
except ImportError:
    app = None
    scene = None

import four_camera_shared as shared

mocap = shared.mocap


PREVIEW_WINDOW_NAME = "4 camera mocap preview"
TRAIL_COLORS = [
    (0.15, 1.00, 0.25, 1.0),
    (1.00, 0.30, 0.20, 1.0),
    (0.20, 0.55, 1.00, 1.0),
    (1.00, 0.72, 0.15, 1.0),
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = mocap.build_arg_parser()
    parser.description = (
        "Track two mocap markers with four cameras: front pair supplies Y/Z, "
        "top pair supplies X/Y, and the shared Y coordinate fuses them."
    )
    parser.set_defaults(
        cameras=list(shared.CAMERA_IDS),
        calibration=str(shared.default_calibration_path()),
        update_hz=shared.DEFAULT_UPDATE_HZ,
    )
    parser.add_argument("--front-cameras", type=shared.parse_camera_pair, default=tuple(shared.FRONT_CAMERA_IDS))
    parser.add_argument("--top-cameras", type=shared.parse_camera_pair, default=tuple(shared.TOP_CAMERA_IDS))
    parser.add_argument("--fusion-y-tolerance", type=float, default=shared.DEFAULT_FUSION_Y_TOLERANCE_M)
    parser.add_argument(
        "--max-fused-reprojection-error",
        type=float,
        default=shared.DEFAULT_MAX_FUSED_REPROJECTION_ERROR_PX,
    )
    parser.add_argument("--max-layout-measurements", type=int, default=shared.DEFAULT_MAX_LAYOUT_MEASUREMENTS)
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
    parser.add_argument(
        "--min-measurement-separation",
        type=float,
        default=shared.DEFAULT_MIN_MEASUREMENT_SEPARATION_M,
        help="Minimum 3D distance between separately displayed fused measurements.",
    )
    parser.add_argument(
        "--pairing-track-bias-distance",
        type=float,
        default=shared.DEFAULT_PAIRING_TRACK_BIAS_DISTANCE_M,
        help="Prefer front/top blob pairings near existing predicted tracks.",
    )
    parser.add_argument(
        "--visual-smoothing",
        type=float,
        default=shared.DEFAULT_VISUAL_SMOOTHING,
        help="Display-only smoothing for VisPy points and trails. 0 is raw, higher is steadier.",
    )
    parser.add_argument("--trail-seconds", type=float, default=shared.DEFAULT_TRAIL_SECONDS)
    parser.add_argument("--update-hz", type=float, default=shared.DEFAULT_UPDATE_HZ)
    parser.add_argument("--panel-width", type=int, default=shared.DEFAULT_PANEL_WIDTH)
    parser.add_argument("--panel-height", type=int, default=shared.DEFAULT_PANEL_HEIGHT)
    parser.add_argument("--line-width", type=float, default=4.0)
    parser.add_argument("--point-size", type=float, default=12.0)
    return parser


class FourCameraTrailApp:
    def __init__(
        self,
        args: argparse.Namespace,
        workers: list[shared.ThreadedMocapCamera],
        calibrations: dict[int, mocap.CameraCalibration],
        stop_event,
    ) -> None:
        if app is None or scene is None:
            raise RuntimeError("VisPy is not available.")

        self.args = args
        self.workers = workers
        self.stop_event = stop_event
        self.calibrations = calibrations
        self.camera_ids = [worker.camera_id for worker in workers]
        self.settings_by_camera = {
            worker.camera_id: worker.settings for worker in workers
        }
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
        self.trail_points_by_track: dict[int, deque[tuple[float, np.ndarray]]] = {}
        self.track_lines: dict[int, object] = {}
        self.track_markers: dict[int, object] = {}
        self.track_labels: dict[int, object] = {}
        self.display_track_ids: list[int] = []
        self.display_positions_by_track: dict[int, np.ndarray] = {}
        self.last_print_time = 0.0
        self.last_camera_stats_time = 0.0
        self.closed = False

        self._setup_preview_window()
        self._setup_vispy_window()
        self.timer = app.Timer(
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

    def _setup_vispy_window(self) -> None:
        self.canvas = scene.SceneCanvas(
            keys="interactive",
            show=True,
            bgcolor="black",
            size=(1000, 750),
            title="Jazz Hands 4-camera mocap trail",
        )
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.cameras.TurntableCamera(
            fov=45,
            distance=4.0,
            elevation=25.0,
            azimuth=-35.0,
            center=(0, 0, 0),
        )
        scene.visuals.GridLines(
            scale=(0.25, 0.25),
            color=(0.25, 0.25, 0.25, 1.0),
            parent=self.view.scene,
        )
        scene.visuals.XYZAxis(width=2, parent=self.view.scene)
        self._add_axis_labels()
        self._add_camera_markers()
        self.status_text = scene.visuals.Text(
            "Waiting for fused 4-camera points...",
            color="white",
            font_size=9,
            pos=(10, 10),
            anchor_x="left",
            anchor_y="bottom",
            parent=self.canvas.scene,
        )
        self.canvas.events.close.connect(self.close)

    def _add_axis_labels(self) -> None:
        label_distance = 1.25 * float(self.args.scaling_factor)
        scene.visuals.Text("X", color="red", font_size=20, pos=(label_distance, 0, 0), parent=self.view.scene)
        scene.visuals.Text("Y", color="green", font_size=20, pos=(0, label_distance, 0), parent=self.view.scene)
        scene.visuals.Text("Z", color="blue", font_size=20, pos=(0, 0, label_distance), parent=self.view.scene)

    def _add_camera_markers(self) -> None:
        positions = []
        labels = []
        for camera_id, calibration in sorted(self.calibrations.items()):
            camera_position = -calibration.rotation.T @ calibration.translation.reshape(3)
            visual_position = self._visual_position(camera_position)
            positions.append(visual_position)
            labels.append((camera_id, visual_position))

        if not positions:
            return

        camera_points = scene.visuals.Markers(parent=self.view.scene)
        camera_points.set_data(
            pos=np.asarray(positions, dtype=np.float32),
            face_color=(0.2, 0.55, 1.0, 1.0),
            edge_color=(0.85, 0.95, 1.0, 1.0),
            size=8.0,
        )
        for camera_id, position in labels:
            scene.visuals.Text(
                f"cam {camera_id}",
                color=(0.75, 0.9, 1.0, 1.0),
                font_size=8,
                pos=tuple(position),
                parent=self.view.scene,
            )

    def _visual_position(self, position: np.ndarray) -> np.ndarray:
        return shared.scaled_position(position, self.args).astype(np.float32)

    def update(self, _event) -> None:
        if self.closed:
            return

        timestamp = time.time()
        snapshots = shared.collect_snapshots(self.workers)
        frames = shared.frames_from_snapshots(snapshots)
        observations_by_camera = shared.observations_from_snapshots(snapshots)
        self.last_camera_stats_time = shared.print_threaded_camera_stats(
            "4cam tracker",
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
            self.args.max_layout_measurements,
            self.args.min_measurement_separation,
            self._pairing_reference_positions(timestamp),
            self.args.pairing_track_bias_distance,
        )
        tracks = self.tracker.update(measurements, timestamp)
        live_tracks = [
            track for track in tracks if track.confirmed and track.missing_frames == 0
        ]
        selected_tracks = self._select_display_tracks(live_tracks)
        selected_track_ids = {track.track_id for track in selected_tracks}
        self._prune_display_positions({track.track_id for track in selected_tracks})
        for track in selected_tracks:
            display_position = self._smoothed_display_position(track)
            trail = self.trail_points_by_track.setdefault(track.track_id, deque())
            trail.append((timestamp, display_position.copy()))

        self._draw_trails(timestamp, selected_tracks, observations_by_camera, diagnostics)
        self._hide_unused_track_visuals(selected_track_ids)

        if not self.args.no_preview:
            combined_preview = shared.build_four_camera_preview(
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
            mocap.cv2.imshow(PREVIEW_WINDOW_NAME, combined_preview)
            key = mocap.cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                self.canvas.close()

        self._print_status(timestamp, tracks, observations_by_camera, diagnostics)

    def _select_display_tracks(self, live_tracks: list[mocap.MarkerTrack]) -> list[mocap.MarkerTrack]:
        max_tracks = max(int(self.args.tracked_point_count), 1)
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
        max_tracks = max(int(self.args.tracked_point_count), 1)
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

    def _track_color(self, track_id: int) -> tuple[float, float, float, float]:
        return TRAIL_COLORS[(track_id - 1) % len(TRAIL_COLORS)]

    def _ensure_track_visuals(self, track_id: int):
        if track_id not in self.track_lines:
            color = self._track_color(track_id)
            self.track_lines[track_id] = scene.visuals.Line(
                pos=np.zeros((1, 3), dtype=np.float32),
                color=(color[0], color[1], color[2], 0.0),
                width=float(self.args.line_width),
                parent=self.view.scene,
                method="gl",
            )
        if track_id not in self.track_markers:
            marker = scene.visuals.Markers(parent=self.view.scene)
            marker.set_data(pos=np.zeros((0, 3), dtype=np.float32))
            self.track_markers[track_id] = marker
        if track_id not in self.track_labels:
            self.track_labels[track_id] = scene.visuals.Text(
                "",
                color=self._track_color(track_id),
                font_size=9,
                pos=(0, 0, 0),
                parent=self.view.scene,
            )
        return self.track_lines[track_id]

    def _draw_trails(
        self,
        timestamp: float,
        selected_tracks: list[mocap.MarkerTrack],
        observations_by_camera: dict[int, list[mocap.MarkerObservation]],
        diagnostics: shared.FusionDiagnostics,
    ) -> None:
        self._draw_live_markers(selected_tracks)

        for track in selected_tracks:
            trail = self.trail_points_by_track.setdefault(track.track_id, deque())
            while trail and timestamp - trail[0][0] > self.args.trail_seconds:
                trail.popleft()

            line = self._ensure_track_visuals(track.track_id)
            color = self._track_color(track.track_id)
            if len(trail) < 2:
                line.set_data(
                    pos=np.zeros((1, 3), dtype=np.float32),
                    color=(color[0], color[1], color[2], 0.0),
                    width=float(self.args.line_width),
                )
                continue

            positions = np.asarray(
                [self._visual_position(position) for _point_time, position in trail],
                dtype=np.float32,
            )
            ages = np.asarray(
                [timestamp - point_time for point_time, _position in trail],
                dtype=np.float32,
            )
            alphas = np.clip(1.0 - ages / max(float(self.args.trail_seconds), 0.001), 0.05, 1.0)
            colors = np.tile(np.asarray(color, dtype=np.float32), (len(positions), 1))
            colors[:, 3] = alphas
            line.set_data(pos=positions, color=colors, width=float(self.args.line_width))

        blob_counts = ", ".join(
            f"cam {camera_id}: {len(observations)}"
            for camera_id, observations in sorted(observations_by_camera.items())
        )
        fusion_text = (
            f"front={diagnostics.front_candidates} top={diagnostics.top_candidates} "
            f"matched={diagnostics.y_matched_pairs} fused={diagnostics.fused_measurements}"
        )
        if not selected_tracks:
            self.status_text.text = (
                f"Waiting for fused 3D points | visible=0/{self.args.tracked_point_count} "
                f"| {fusion_text} | {blob_counts}"
            )
            return

        track_parts = []
        for track in selected_tracks:
            display_position = self.display_positions_by_track.get(track.track_id, track.position)
            x, y, z = display_position
            track_parts.append(
                f"id {track.track_id}: x={x:+.3f} y={y:+.3f} z={z:+.3f} "
                f"err={track.reprojection_error_px:.1f}px"
            )
        self.status_text.text = (
            " | ".join(track_parts)
            + f" | visible={len(selected_tracks)}/{self.args.tracked_point_count}"
            + f" | {shared.scale_text(self.args)} | {fusion_text} | {blob_counts}"
        )

    def _draw_live_markers(self, selected_tracks: list[mocap.MarkerTrack]) -> None:
        if not selected_tracks:
            for marker in self.track_markers.values():
                marker.set_data(pos=np.zeros((0, 3), dtype=np.float32))
            for label in self.track_labels.values():
                label.text = ""
            return

        label_offset = np.array([0.025, 0.025, 0.025], dtype=np.float32) * float(self.args.scaling_factor)
        for track in selected_tracks:
            self._ensure_track_visuals(track.track_id)
            display_position = self.display_positions_by_track.get(track.track_id, track.position)
            position = self._visual_position(display_position)
            color = self._track_color(track.track_id)
            marker = self.track_markers[track.track_id]
            marker.set_data(
                pos=np.asarray([position], dtype=np.float32),
                face_color=color,
                edge_color=(1.0, 1.0, 1.0, 1.0),
                size=float(self.args.point_size) * 1.6,
            )
            label = self.track_labels[track.track_id]
            label.text = f"id {track.track_id}"
            label.pos = tuple(position + label_offset)

    def _hide_unused_track_visuals(self, selected_track_ids: set[int]) -> None:
        for track_id, line in self.track_lines.items():
            if track_id in selected_track_ids:
                continue
            color = self._track_color(track_id)
            line.set_data(
                pos=np.zeros((1, 3), dtype=np.float32),
                color=(color[0], color[1], color[2], 0.0),
                width=float(self.args.line_width),
            )
            marker = self.track_markers.get(track_id)
            if marker is not None:
                marker.set_data(pos=np.zeros((0, 3), dtype=np.float32))
            label = self.track_labels.get(track_id)
            if label is not None:
                label.text = ""

    def _print_status(
        self,
        timestamp: float,
        tracks: list[mocap.MarkerTrack],
        observations_by_camera: dict[int, list[mocap.MarkerObservation]],
        diagnostics: shared.FusionDiagnostics,
    ) -> None:
        if timestamp - self.last_print_time < self.args.print_interval:
            return
        self.last_print_time = timestamp
        live = [
            f"id {track.track_id}: ({track.position[0]:+.3f},{track.position[1]:+.3f},{track.position[2]:+.3f})"
            for track in tracks
            if track.confirmed and track.missing_frames == 0
        ]
        blob_counts = ", ".join(
            f"cam {camera_id}: {len(observations)}"
            for camera_id, observations in sorted(observations_by_camera.items())
        )
        print(
            "[4cam tracker] "
            f"front candidates={diagnostics.front_candidates}, "
            f"top candidates={diagnostics.top_candidates}, "
            f"y matches={diagnostics.y_matched_pairs}, "
            f"fused={diagnostics.fused_measurements} | "
            f"{' | '.join(live) if live else 'no live tracks'} | {blob_counts}"
        )

    def close(self, _event=None) -> None:
        if self.closed:
            return
        self.closed = True
        if hasattr(self, "timer"):
            self.timer.stop()
        shared.stop_threaded_cameras(self.stop_event, self.workers)
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
        backend = shared.configure_vispy_backend(app)
    except RuntimeError as error:
        print(f"[4cam tracker] {error}")
        return 1
    print(f"[4cam tracker] VisPy backend: {backend}")

    calibrations = shared.load_calibrations(args, "4cam tracker")
    stop_event, workers = shared.start_threaded_cameras(
        args,
        list(args.cameras),
        build_masks=not args.no_preview,
        label="4cam tracker",
    )
    shared.wait_for_open_attempts(workers)
    snapshots = shared.collect_snapshots(workers)
    if not shared.any_camera_open(snapshots) and shared.all_open_attempts_done(snapshots):
        print("[4cam tracker] no cameras opened")
        shared.stop_threaded_cameras(stop_event, workers)
        return 1

    connected_ids = {
        camera_id for camera_id, snapshot in snapshots.items() if snapshot.opened
    }
    missing_front = set(args.front_cameras) - connected_ids
    missing_top = set(args.top_cameras) - connected_ids
    if missing_front or missing_top:
        print(
            "[4cam tracker] warning: layout pair not fully connected yet "
            f"(missing front={sorted(missing_front)}, top={sorted(missing_top)})"
        )

    visualizer: FourCameraTrailApp | None = None
    try:
        visualizer = FourCameraTrailApp(args, workers, calibrations, stop_event)
        app.run()
    except KeyboardInterrupt:
        print("\n[4cam tracker] stopped")
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

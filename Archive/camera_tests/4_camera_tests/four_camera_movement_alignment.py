from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

CAMERA_TESTS_DIR = Path(__file__).resolve().parents[1]
if str(CAMERA_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(CAMERA_TESTS_DIR))

import four_camera_shared as shared
import mocap_movement_alignment as movement

mocap = shared.mocap
app = movement.app


PREVIEW_WINDOW_NAME = "4 camera movement alignment preview"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = movement.build_arg_parser()
    parser.description = (
        "Four-camera threaded movement alignment. Front cameras provide Y/Z; "
        "top cameras provide X/Y."
    )
    parser.set_defaults(
        cameras=list(shared.CAMERA_IDS),
        calibration=str(shared.STANDARD_CALIBRATION_PATH),
        output=str(shared.ALIGNED_CALIBRATION_PATH),
        panel_width=shared.DEFAULT_PANEL_WIDTH,
        panel_height=shared.DEFAULT_PANEL_HEIGHT,
        update_hz=shared.DEFAULT_UPDATE_HZ,
        trail_seconds=shared.DEFAULT_TRAIL_SECONDS,
        scaling_factor=shared.DEFAULT_SCALING_FACTOR,
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
    return parser


class FourCameraMovementAlignmentApp(movement.MovementAlignmentApp):
    def __init__(
        self,
        args: argparse.Namespace,
        workers: list[shared.ThreadedMocapCamera],
        calibrations,
        raw_calibration: dict,
        input_path: Path,
        output_path: Path,
        stop_event,
    ) -> None:
        self.workers = workers
        self.stop_event = stop_event
        self.last_camera_stats_time = 0.0
        self.last_fusion_print_time = 0.0
        super().__init__(
            args,
            workers,
            calibrations,
            raw_calibration,
            input_path,
            output_path,
        )
        self.preview_camera_ids = list(args.cameras)
        self.settings_by_camera = {
            worker.camera_id: worker.settings for worker in workers
        }

    def _setup_combined_preview_window(self) -> None:
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
        snapshots = shared.collect_snapshots(self.workers)
        frames = shared.frames_from_snapshots(snapshots)
        observations_by_camera = shared.observations_from_snapshots(snapshots)
        self.last_camera_stats_time = shared.print_threaded_camera_stats(
            "4cam alignment",
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
        )
        tracks = self.tracker.update(measurements, timestamp)
        live_tracks = [
            track for track in tracks if track.confirmed and track.missing_frames == 0
        ]
        self.current_track = max(live_tracks, key=lambda track: track.confidence, default=None)

        if self.current_track is not None:
            self.trail_points.append((timestamp, self.current_track.position.copy()))
            self._collect_pending_sample(self.current_track)

        self._draw_vispy(timestamp, observations_by_camera)
        self._print_status(timestamp, tracks, observations_by_camera)
        if timestamp - self.last_fusion_print_time >= self.args.print_interval:
            self.last_fusion_print_time = timestamp
            print(
                "[4cam alignment] "
                f"front candidates={diagnostics.front_candidates}, "
                f"top candidates={diagnostics.top_candidates}, "
                f"y matches={diagnostics.y_matched_pairs}, "
                f"fused={diagnostics.fused_measurements}"
            )

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
            self._draw_preview_overlay(combined_preview)
            mocap.cv2.imshow(PREVIEW_WINDOW_NAME, combined_preview)
            key = mocap.cv2.waitKey(1) & 0xFF
            self._handle_key(key)

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
    args.position_scale = getattr(args, "scaling_factor", shared.DEFAULT_SCALING_FACTOR)

    if mocap.cv2 is None:
        print("OpenCV is required for camera mocap. Install it with: python -m pip install opencv-python")
        return 1

    input_path = Path(args.calibration).expanduser()
    output_path = Path(args.output).expanduser()
    if not input_path.exists():
        print(f"[4cam alignment] calibration file not found: {input_path}")
        return 1

    try:
        backend = shared.configure_vispy_backend(app)
    except RuntimeError as error:
        print(f"[4cam alignment] {error}")
        return 1
    print(f"[4cam alignment] VisPy backend: {backend}")

    raw_calibration = movement.load_raw_calibration(input_path)
    calibrations = mocap.load_calibration_file(
        str(input_path),
        args.width,
        args.height,
        args.focal_length_px,
    )
    stop_event, workers = shared.start_threaded_cameras(
        args,
        list(args.cameras),
        build_masks=not args.no_preview,
        label="4cam alignment",
    )
    shared.wait_for_open_attempts(workers)
    snapshots = shared.collect_snapshots(workers)
    if not shared.any_camera_open(snapshots) and shared.all_open_attempts_done(snapshots):
        print("[4cam alignment] no cameras opened")
        shared.stop_threaded_cameras(stop_event, workers)
        return 1

    print("[4cam alignment] controls:")
    print("[4cam alignment]   space/c = capture next point: origin, up, forward")
    print("[4cam alignment]   1/2/3 = recapture origin/up/forward")
    print("[4cam alignment]   s = save aligned calibration")
    print("[4cam alignment]   r = reset captures")
    print("[4cam alignment]   q/Esc = quit")
    print(f"[4cam alignment] input : {input_path}")
    print(f"[4cam alignment] output: {output_path}")

    visualizer: FourCameraMovementAlignmentApp | None = None
    try:
        visualizer = FourCameraMovementAlignmentApp(
            args,
            workers,
            calibrations,
            raw_calibration,
            input_path,
            output_path,
            stop_event,
        )
        app.run()
    except KeyboardInterrupt:
        print("\n[4cam alignment] stopped")
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

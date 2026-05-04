from __future__ import annotations

import argparse
from pathlib import Path
import sys

CAMERA_TESTS_DIR = Path(__file__).resolve().parents[1]
if str(CAMERA_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(CAMERA_TESTS_DIR))

import camera_uvc_settings
import four_camera_shared as shared


DEFAULT_SETTINGS_JSON_PATH = Path(__file__).resolve().with_name(
    "four_camera_uvc_settings_values.json"
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Snapshot the current AMCap/DirectShow UVC controls for the 4-camera rig.",
    )
    parser.add_argument(
        "--camera-ids",
        type=camera_uvc_settings.parse_camera_ids,
        default=list(shared.CAMERA_IDS),
        help="Comma-separated logical camera IDs. Default: 1,2,3,4.",
    )
    parser.add_argument(
        "--settings-json",
        type=Path,
        default=DEFAULT_SETTINGS_JSON_PATH,
        help=f"Where to save the profile. Default: {DEFAULT_SETTINGS_JSON_PATH}",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    saved_count = camera_uvc_settings.save_current_uvc_profile(
        camera_ids=list(args.camera_ids),
        path=args.settings_json,
    )
    return 0 if saved_count > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

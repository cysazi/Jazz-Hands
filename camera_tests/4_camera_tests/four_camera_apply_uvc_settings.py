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
        description="Apply saved AMCap/DirectShow UVC controls to the 4-camera rig.",
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
        help=f"Settings profile to apply. Default: {DEFAULT_SETTINGS_JSON_PATH}",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print writes without changing cameras.")
    parser.add_argument("--show-ranges", action="store_true", help="Print supported ranges/current values.")
    parser.add_argument("--list", action="store_true", help="List matched DirectShow devices before applying.")
    parser.add_argument("--list-only", action="store_true", help="List devices and do not apply settings.")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.list or args.list_only:
        camera_uvc_settings.list_devices(show_ranges=args.show_ranges)
        if args.list_only:
            return 0

    applied = camera_uvc_settings.apply_configured_camera_settings(
        camera_ids=list(args.camera_ids),
        dry_run=args.dry_run,
        show_ranges=args.show_ranges,
        settings_path=args.settings_json,
    )
    print(f"[4cam settings] controls {'checked' if args.dry_run else 'written'}: {applied}")
    return 0 if applied > 0 or args.dry_run else 1


if __name__ == "__main__":
    raise SystemExit(main())

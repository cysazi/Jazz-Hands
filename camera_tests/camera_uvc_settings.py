from __future__ import annotations

import argparse
import copy
import json
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ctypes import HRESULT, POINTER, c_long

from comtypes import COMMETHOD, GUID, IUnknown, client
from comtypes.persist import IPropertyBag
from pygrabber.dshow_core import ICreateDevEnum, qedit
from pygrabber.dshow_ids import DeviceCategories, clsids

# To apply run the following command
# .\.venv\Scripts\python.exe .\camera_tests\camera_uvc_settings.py --apply --camera-ids 1,2

AUTO_FLAG = 0x0001
MANUAL_FLAG = 0x0002
UVC_SETTINGS_JSON_PATH = Path(__file__).resolve().with_name("camera_uvc_settings_values.json")

# These are the stable identities written by the Arducam serial-number tool.
# Keep camera capture code using OpenCV indexes, but write UVC settings by these
# names/serials before the camera streams open.
CAMERA_UVC_IDENTITIES = {
    1: {"friendly_name": "Vertical_plane_camera_1", "serial": "UC001"},
    2: {"friendly_name": "Vertical_plane_camera_2", "serial": "UC002"},
    3: {"friendly_name": "Vertical_plane_camera_3", "serial": "UC003"},
    4: {"friendly_name": "Vertical_plane_camera_4", "serial": "UC004"},
}

# Values match the property-page style controls. Use None for controls you do
# not want Python to touch. auto=True means the Auto checkbox is enabled.
CAMERA_UVC_SETTINGS = {
    1: {
        "brightness": {"value": 45, "auto": False},
        "contrast": {"value": 64, "auto": False},
        "hue": {"value": 0, "auto": False},
        "saturation": {"value": 64, "auto": False},
        "sharpness": {"value": 3, "auto": False},
        "gamma": {"value": 72, "auto": False},
        "white_balance": {"value": 4600, "auto": True},
        "backlight_compensation": {"value": 2, "auto": False},
        "gain": {"value": 0, "auto": False},
        "powerline_frequency": {"value": 2, "auto": False},
        "exposure": {"value": -8, "auto": False},
    },
    2: {
        "brightness": {"value": 45, "auto": False},
        "contrast": {"value": 64, "auto": False},
        "hue": {"value": 0, "auto": False},
        "saturation": {"value": 64, "auto": False},
        "sharpness": {"value": 3, "auto": False},
        "gamma": {"value": 72, "auto": False},
        "white_balance": {"value": 4600, "auto": True},
        "backlight_compensation": {"value": 2, "auto": False},
        "gain": {"value": 0, "auto": False},
        "powerline_frequency": {"value": 2, "auto": False},
        "exposure": {"value": -8, "auto": False},
    },
    3: {
        "brightness": {"value": 45, "auto": False},
        "contrast": {"value": 64, "auto": False},
        "hue": {"value": 0, "auto": False},
        "saturation": {"value": 64, "auto": False},
        "sharpness": {"value": 3, "auto": False},
        "gamma": {"value": 72, "auto": False},
        "white_balance": {"value": 4600, "auto": True},
        "backlight_compensation": {"value": 2, "auto": False},
        "gain": {"value": 0, "auto": False},
        "powerline_frequency": {"value": 2, "auto": False},
        "exposure": {"value": -8, "auto": False},
    },
    4: {
        "brightness": {"value": 45, "auto": False},
        "contrast": {"value": 64, "auto": False},
        "hue": {"value": 0, "auto": False},
        "saturation": {"value": 64, "auto": False},
        "sharpness": {"value": 3, "auto": False},
        "gamma": {"value": 72, "auto": False},
        "white_balance": {"value": 4600, "auto": True},
        "backlight_compensation": {"value": 2, "auto": False},
        "gain": {"value": 0, "auto": False},
        "powerline_frequency": {"value": 2, "auto": False},
        "exposure": {"value": -8, "auto": False},
    },
}

LIVE_UVC_CONTROL_NAMES = [
    "exposure",
    "gain",
    "brightness",
    "contrast",
    "gamma",
    "backlight_compensation",
    "sharpness",
    "saturation",
    "white_balance",
]


VIDEO_PROC_AMP_CONTROLS = {
    "brightness": 0,
    "contrast": 1,
    "hue": 2,
    "saturation": 3,
    "sharpness": 4,
    "gamma": 5,
    "color_enable": 6,
    "white_balance": 7,
    "backlight_compensation": 8,
    "gain": 9,
    "powerline_frequency": 10,
}

CAMERA_CONTROL_CONTROLS = {
    "pan": 0,
    "tilt": 1,
    "roll": 2,
    "zoom": 3,
    "exposure": 4,
    "iris": 5,
    "focus": 6,
}


class IAMVideoProcAmp(IUnknown):
    _case_insensitive_ = True
    _iid_ = GUID("{C6E13360-30AC-11D0-A18C-00A0C9118956}")
    _idlflags_ = []


IAMVideoProcAmp._methods_ = [
    COMMETHOD(
        [],
        HRESULT,
        "GetRange",
        (["in"], c_long, "Property"),
        (["out"], POINTER(c_long), "pMin"),
        (["out"], POINTER(c_long), "pMax"),
        (["out"], POINTER(c_long), "pSteppingDelta"),
        (["out"], POINTER(c_long), "pDefault"),
        (["out"], POINTER(c_long), "pCapsFlags"),
    ),
    COMMETHOD(
        [],
        HRESULT,
        "Set",
        (["in"], c_long, "Property"),
        (["in"], c_long, "lValue"),
        (["in"], c_long, "Flags"),
    ),
    COMMETHOD(
        [],
        HRESULT,
        "Get",
        (["in"], c_long, "Property"),
        (["out"], POINTER(c_long), "lValue"),
        (["out"], POINTER(c_long), "Flags"),
    ),
]


class IAMCameraControl(IUnknown):
    _case_insensitive_ = True
    _iid_ = GUID("{C6E13370-30AC-11D0-A18C-00A0C9118956}")
    _idlflags_ = []


IAMCameraControl._methods_ = IAMVideoProcAmp._methods_


@dataclass(slots=True)
class DirectShowCameraDevice:
    index: int
    friendly_name: str
    device_path: str | None
    pnp_device_id: str | None
    parent_device_id: str | None
    filter_object: Any

    @property
    def serial(self) -> str | None:
        if not self.parent_device_id:
            return None
        return self.parent_device_id.rsplit("\\", 1)[-1]


def parse_camera_ids(text: str) -> list[int]:
    values = [value.strip() for value in text.split(",") if value.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one camera id.")
    return [int(value) for value in values]


def normalize_key(value: str | None) -> str:
    return (value or "").casefold()


def directshow_path_to_pnp_id(device_path: str | None) -> str | None:
    if not device_path:
        return None
    path = device_path
    prefix = "\\\\?\\"
    if path.startswith(prefix):
        path = path[len(prefix) :]
    path = path.split("#{", 1)[0]
    return path.replace("#", "\\").upper()


def read_property_bag_value(property_bag: IPropertyBag, name: str) -> str | None:
    try:
        value = property_bag.Read(name, pErrorLog=None)
    except Exception:
        return None
    return str(value) if value is not None else None


def current_timestamp_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def camera_settings_from_saved_payload(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    if isinstance(payload.get("settings"), dict):
        return payload["settings"]
    return payload


def load_saved_uvc_settings(path: Path = UVC_SETTINGS_JSON_PATH) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        print(f"[uvc] could not load saved settings {path}: {error}")
        return {}

    settings: dict[int, dict[str, Any]] = {}
    if not isinstance(data, dict):
        return settings

    camera_entries = data.get("cameras") if isinstance(data.get("cameras"), dict) else data
    for camera_id_text, camera_payload in camera_entries.items():
        try:
            camera_id = int(camera_id_text)
        except (TypeError, ValueError):
            continue
        camera_settings = camera_settings_from_saved_payload(camera_payload)
        if isinstance(camera_settings, dict):
            settings[camera_id] = camera_settings
    return settings


def effective_uvc_settings(path: Path = UVC_SETTINGS_JSON_PATH) -> dict[int, dict[str, Any]]:
    settings = copy.deepcopy(CAMERA_UVC_SETTINGS)
    saved_settings = load_saved_uvc_settings(path)
    for camera_id, camera_settings in saved_settings.items():
        settings.setdefault(camera_id, {})
        settings[camera_id].update(camera_settings)
    return settings


def save_uvc_settings_snapshot(
    camera_settings: dict[int, dict[str, Any]],
    path: Path = UVC_SETTINGS_JSON_PATH,
) -> None:
    serializable = {
        str(camera_id): settings
        for camera_id, settings in sorted(camera_settings.items())
    }
    with path.open("w", encoding="utf-8") as file:
        json.dump(serializable, file, indent=2, sort_keys=True)
        file.write("\n")
    print(f"[uvc] saved live settings to {path}")


def load_pnp_parent_by_instance_id() -> dict[str, str]:
    command = r"""
$items = Get-PnpDevice -PresentOnly -Class Camera | ForEach-Object {
    $parent = $null
    try {
        $parent = (Get-PnpDeviceProperty -InstanceId $_.InstanceId -KeyName 'DEVPKEY_Device_Parent' -ErrorAction Stop).Data
    } catch {}
    [PSCustomObject]@{
        InstanceId = $_.InstanceId
        FriendlyName = $_.FriendlyName
        Parent = $parent
    }
}
$items | ConvertTo-Json -Compress
"""
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            check=False,
            capture_output=True,
            text=True,
            timeout=25,
        )
    except Exception:
        return {}

    if result.returncode != 0 or not result.stdout.strip():
        return {}

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {}

    if isinstance(data, dict):
        rows = [data]
    else:
        rows = data if isinstance(data, list) else []

    parents: dict[str, str] = {}
    for row in rows:
        instance_id = str(row.get("InstanceId", "")).upper()
        parent = row.get("Parent")
        if instance_id and parent:
            parents[instance_id] = str(parent).upper()
    return parents


def enumerate_video_devices() -> list[DirectShowCameraDevice]:
    parent_by_instance_id = load_pnp_parent_by_instance_id()
    system_device_enum = client.CreateObject(
        clsids.CLSID_SystemDeviceEnum,
        interface=ICreateDevEnum,
    )
    filter_enumerator = system_device_enum.CreateClassEnumerator(
        GUID(DeviceCategories.VideoInputDevice),
        dwFlags=0,
    )

    devices: list[DirectShowCameraDevice] = []
    try:
        moniker, count = filter_enumerator.Next(1)
    except Exception:
        return devices

    index = 0
    while count:
        property_bag = moniker.BindToStorage(0, 0, IPropertyBag._iid_).QueryInterface(IPropertyBag)
        friendly_name = read_property_bag_value(property_bag, "FriendlyName") or f"camera_{index}"
        device_path = read_property_bag_value(property_bag, "DevicePath")
        pnp_device_id = directshow_path_to_pnp_id(device_path)
        parent_device_id = parent_by_instance_id.get((pnp_device_id or "").upper())
        filter_object = moniker.BindToObject(0, 0, qedit.IBaseFilter._iid_).QueryInterface(
            qedit.IBaseFilter
        )
        devices.append(
            DirectShowCameraDevice(
                index=index,
                friendly_name=friendly_name,
                device_path=device_path,
                pnp_device_id=pnp_device_id,
                parent_device_id=parent_device_id,
                filter_object=filter_object,
            )
        )
        try:
            moniker, count = filter_enumerator.Next(1)
        except Exception:
            break
        index += 1

    return devices


def find_device_for_camera_id(
    camera_id: int,
    devices: list[DirectShowCameraDevice],
) -> DirectShowCameraDevice | None:
    identity = CAMERA_UVC_IDENTITIES.get(camera_id, {})
    directshow_index = identity.get("directshow_index")
    if directshow_index is not None:
        for device in devices:
            if device.index == int(directshow_index):
                return device

    friendly_name = normalize_key(identity.get("friendly_name"))
    serial = normalize_key(identity.get("serial"))
    device_path_part = normalize_key(identity.get("device_path_contains"))

    matches = []
    for device in devices:
        if friendly_name and normalize_key(device.friendly_name) != friendly_name:
            continue
        if serial:
            serial_targets = [
                normalize_key(device.serial),
                normalize_key(device.parent_device_id),
                normalize_key(device.pnp_device_id),
                normalize_key(device.device_path),
            ]
            if not any(serial in target for target in serial_targets):
                continue
        if device_path_part and device_path_part not in normalize_key(device.device_path):
            continue
        matches.append(device)

    if len(matches) == 1:
        return matches[0]
    return None


def normalize_setting(setting: Any) -> tuple[int | None, bool | None, bool]:
    if setting is None:
        return None, None, False
    if isinstance(setting, dict):
        enabled = bool(setting.get("enabled", True))
        value = setting.get("value")
        auto = setting.get("auto")
        return (int(value) if value is not None else None), auto, enabled
    return int(setting), False, True


def adjusted_to_range(value: int, min_value: int, max_value: int, step: int) -> int:
    value = max(min_value, min(max_value, int(value)))
    if step > 1:
        value = min_value + round((value - min_value) / step) * step
        value = max(min_value, min(max_value, int(value)))
    return int(value)


def desired_flags(auto: bool | None, current_flags: int, caps_flags: int, value_is_set: bool) -> int:
    if auto is True:
        return AUTO_FLAG if caps_flags & AUTO_FLAG else current_flags
    if auto is False:
        return MANUAL_FLAG if caps_flags & MANUAL_FLAG else current_flags
    if value_is_set and caps_flags & MANUAL_FLAG:
        return MANUAL_FLAG
    return current_flags


def flag_label(flags: int) -> str:
    if flags & AUTO_FLAG:
        return "auto"
    if flags & MANUAL_FLAG:
        return "manual"
    return str(flags)


def get_interface(filter_object: Any, interface_type: Any) -> Any | None:
    try:
        return filter_object.QueryInterface(interface_type)
    except Exception:
        return None


def device_metadata(device: DirectShowCameraDevice) -> dict[str, Any]:
    return {
        "directshow_index": int(device.index),
        "friendly_name": device.friendly_name,
        "serial": device.serial or "unknown",
        "pnp_device_id": device.pnp_device_id,
        "parent_device_id": device.parent_device_id,
        "device_path": device.device_path,
    }


def read_current_control_group(
    interface: Any,
    controls: dict[str, int],
    group_name: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    settings: dict[str, dict[str, Any]] = {}
    snapshots: dict[str, dict[str, Any]] = {}
    if interface is None:
        return settings, snapshots

    for control_name, property_id in controls.items():
        try:
            min_value, max_value, step, default_value, caps_flags = interface.GetRange(property_id)
            current_value, current_flags = interface.Get(property_id)
        except Exception:
            continue

        current_value = int(current_value)
        current_flags = int(current_flags)
        caps_flags = int(caps_flags)
        settings[control_name] = {
            "value": current_value,
            "auto": bool(current_flags & AUTO_FLAG),
        }
        snapshots[control_name] = {
            "group": group_name,
            "property_id": int(property_id),
            "value": current_value,
            "auto": bool(current_flags & AUTO_FLAG),
            "flags": current_flags,
            "flags_label": flag_label(current_flags),
            "range": {
                "min": int(min_value),
                "max": int(max_value),
                "step": int(step),
                "default": int(default_value),
                "caps_flags": caps_flags,
                "supports_auto": bool(caps_flags & AUTO_FLAG),
                "supports_manual": bool(caps_flags & MANUAL_FLAG),
            },
        }
    return settings, snapshots


def read_current_device_controls(
    device: DirectShowCameraDevice,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    settings: dict[str, dict[str, Any]] = {}
    snapshots: dict[str, dict[str, Any]] = {}

    video_settings, video_snapshots = read_current_control_group(
        get_interface(device.filter_object, IAMVideoProcAmp),
        VIDEO_PROC_AMP_CONTROLS,
        "video_proc_amp",
    )
    camera_settings, camera_snapshots = read_current_control_group(
        get_interface(device.filter_object, IAMCameraControl),
        CAMERA_CONTROL_CONTROLS,
        "camera_control",
    )
    settings.update(video_settings)
    settings.update(camera_settings)
    snapshots.update(video_snapshots)
    snapshots.update(camera_snapshots)
    return settings, snapshots


def save_current_uvc_profile(
    camera_ids: list[int] | None = None,
    path: Path = UVC_SETTINGS_JSON_PATH,
) -> int:
    selected_camera_ids = camera_ids or sorted(CAMERA_UVC_IDENTITIES)
    devices = enumerate_video_devices()
    cameras: dict[str, dict[str, Any]] = {}

    for camera_id in selected_camera_ids:
        device = find_device_for_camera_id(camera_id, devices)
        if device is None:
            identity = CAMERA_UVC_IDENTITIES.get(camera_id, {})
            print(f"[uvc snapshot] camera {camera_id}: no matching DirectShow device for {identity}")
            continue

        settings, controls = read_current_device_controls(device)
        cameras[str(camera_id)] = {
            "identity": dict(CAMERA_UVC_IDENTITIES.get(camera_id, {})),
            "matched_device": device_metadata(device),
            "settings": settings,
            "controls": controls,
        }
        readable_settings = ", ".join(
            f"{name}={setting['value']}/{('auto' if setting['auto'] else 'manual')}"
            for name, setting in sorted(settings.items())
        )
        print(
            f"[uvc snapshot] camera {camera_id}: {device.friendly_name} "
            f"serial={device.serial or 'unknown'} directshow_index={device.index}"
        )
        print(f"    {readable_settings or 'no readable controls'}")

    payload = {
        "schema_version": 2,
        "created_at": current_timestamp_utc(),
        "description": (
            "Snapshot of current DirectShow UVC controls. The settings blocks are "
            "apply-compatible; controls include AMCap-style range/device metadata."
        ),
        "cameras": cameras,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")
    print(f"[uvc snapshot] saved {len(cameras)} camera profile(s) to {path}")
    return len(cameras)


def apply_control_group(
    device: DirectShowCameraDevice,
    interface: Any,
    controls: dict[str, int],
    settings: dict[str, Any],
    dry_run: bool,
    show_ranges: bool,
) -> int:
    if interface is None:
        return 0

    applied = 0
    for control_name, property_id in controls.items():
        try:
            min_value, max_value, step, default_value, caps_flags = interface.GetRange(property_id)
            current_value, current_flags = interface.Get(property_id)
        except Exception:
            continue

        if show_ranges:
            print(
                f"[uvc] {device.friendly_name}: {control_name} "
                f"range={min_value}..{max_value} step={step} default={default_value} "
                f"current={current_value} flags={flag_label(current_flags)} caps={caps_flags}"
            )

        desired_value, auto, enabled = normalize_setting(settings.get(control_name))
        if not enabled:
            continue
        if desired_value is None and auto is None:
            continue

        value_to_set = current_value if desired_value is None else desired_value
        value_to_set = adjusted_to_range(value_to_set, min_value, max_value, step)
        flags_to_set = desired_flags(auto, current_flags, caps_flags, desired_value is not None)

        if dry_run:
            print(
                f"[uvc dry-run] {device.friendly_name}: {control_name} "
                f"{current_value}/{flag_label(current_flags)} -> "
                f"{value_to_set}/{flag_label(flags_to_set)}"
            )
            applied += 1
            continue

        try:
            interface.Set(property_id, value_to_set, flags_to_set)
            new_value, new_flags = interface.Get(property_id)
            print(
                f"[uvc] {device.friendly_name}: {control_name} "
                f"{current_value}/{flag_label(current_flags)} -> "
                f"{new_value}/{flag_label(new_flags)}"
            )
            applied += 1
        except Exception as error:
            print(
                f"[uvc] {device.friendly_name}: could not set {control_name}: "
                f"{str(error).splitlines()[0]}"
            )

    return applied


def apply_settings_to_device(
    camera_id: int,
    device: DirectShowCameraDevice,
    settings: dict[str, Any],
    dry_run: bool = False,
    show_ranges: bool = False,
) -> int:
    if not settings:
        print(f"[uvc] camera {camera_id}: no settings configured")
        return 0

    video_proc_amp = get_interface(device.filter_object, IAMVideoProcAmp)
    camera_control = get_interface(device.filter_object, IAMCameraControl)
    print(
        f"[uvc] camera {camera_id}: matched {device.friendly_name} "
        f"serial={device.serial or 'unknown'} directshow_index={device.index}"
    )
    applied = apply_control_group(
        device,
        video_proc_amp,
        VIDEO_PROC_AMP_CONTROLS,
        settings,
        dry_run,
        show_ranges,
    )
    applied += apply_control_group(
        device,
        camera_control,
        CAMERA_CONTROL_CONTROLS,
        settings,
        dry_run,
        show_ranges,
    )
    return applied


def apply_configured_camera_settings(
    camera_ids: list[int] | None = None,
    dry_run: bool = False,
    show_ranges: bool = False,
    settings_path: Path = UVC_SETTINGS_JSON_PATH,
) -> int:
    selected_camera_ids = camera_ids or sorted(CAMERA_UVC_SETTINGS)
    devices = enumerate_video_devices()
    settings_by_camera = effective_uvc_settings(settings_path)
    total = 0
    for camera_id in selected_camera_ids:
        device = find_device_for_camera_id(camera_id, devices)
        if device is None:
            identity = CAMERA_UVC_IDENTITIES.get(camera_id, {})
            print(f"[uvc] camera {camera_id}: no matching DirectShow device for {identity}")
            continue
        total += apply_settings_to_device(
            camera_id,
            device,
            settings_by_camera.get(camera_id, {}),
            dry_run=dry_run,
            show_ranges=show_ranges,
        )
    return total


def list_devices(show_ranges: bool = False) -> None:
    devices = enumerate_video_devices()
    for device in devices:
        configured_ids = [
            camera_id
            for camera_id in sorted(CAMERA_UVC_IDENTITIES)
            if find_device_for_camera_id(camera_id, [device]) is not None
        ]
        print(
            f"[{device.index}] {device.friendly_name} "
            f"serial={device.serial or 'unknown'} configured_ids={configured_ids or '-'}"
        )
        print(f"    pnp:    {device.pnp_device_id or '-'}")
        print(f"    parent: {device.parent_device_id or '-'}")
        print(f"    dshow:  {device.device_path or '-'}")
        if show_ranges:
            apply_control_group(
                device,
                get_interface(device.filter_object, IAMVideoProcAmp),
                VIDEO_PROC_AMP_CONTROLS,
                {},
                dry_run=True,
                show_ranges=True,
            )
            apply_control_group(
                device,
                get_interface(device.filter_object, IAMCameraControl),
                CAMERA_CONTROL_CONTROLS,
                {},
                dry_run=True,
                show_ranges=True,
            )


@dataclass(slots=True)
class UvcControlRef:
    camera_id: int
    device: DirectShowCameraDevice
    name: str
    interface: Any
    property_id: int
    min_value: int
    max_value: int
    step: int
    default_value: int
    caps_flags: int

    def get(self) -> tuple[int, int]:
        value, flags = self.interface.Get(self.property_id)
        return int(value), int(flags)

    def slider_max(self) -> int:
        return max(1, int(round((self.max_value - self.min_value) / max(self.step, 1))))

    def value_to_slider(self, value: int) -> int:
        return int(round((int(value) - self.min_value) / max(self.step, 1)))

    def slider_to_value(self, slider_value: int) -> int:
        return adjusted_to_range(
            self.min_value + int(slider_value) * max(self.step, 1),
            self.min_value,
            self.max_value,
            self.step,
        )

    def set(self, value: int | None = None, auto: bool | None = None) -> tuple[int, int]:
        current_value, current_flags = self.get()
        value_to_set = current_value if value is None else int(value)
        value_to_set = adjusted_to_range(
            value_to_set,
            self.min_value,
            self.max_value,
            self.step,
        )
        flags_to_set = desired_flags(
            auto,
            current_flags,
            self.caps_flags,
            value is not None,
        )
        self.interface.Set(self.property_id, value_to_set, flags_to_set)
        return self.get()


class LiveFpsCounter:
    def __init__(self) -> None:
        self.window_start = time.perf_counter()
        self.frames_this_window = 0
        self.fps = 0.0

    def mark_frame(self) -> float:
        self.frames_this_window += 1
        now = time.perf_counter()
        elapsed = now - self.window_start
        if elapsed >= 1.0:
            self.fps = self.frames_this_window / elapsed
            self.frames_this_window = 0
            self.window_start = now
        return self.fps


class LiveTuningState:
    def __init__(self, camera_ids: list[int], camera_settings_module: Any) -> None:
        self.threshold = int(camera_settings_module.THRESHOLD)
        self.blur_by_camera = {
            camera_id: camera_settings_module.blur_kernel_for_camera(camera_id)
            for camera_id in camera_ids
        }

    def set_threshold(self, value: int) -> None:
        self.threshold = int(max(0, min(255, value)))

    def set_blur(self, camera_id: int, value: int) -> None:
        self.blur_by_camera[camera_id] = int(value)


SHORT_CONTROL_NAMES = {
    "brightness": "bright",
    "contrast": "contrast",
    "hue": "hue",
    "saturation": "sat",
    "sharpness": "sharp",
    "gamma": "gamma",
    "white_balance": "whitebal",
    "backlight_compensation": "backlight",
    "gain": "gain",
    "exposure": "exposure",
}


def build_control_ref(
    camera_id: int,
    device: DirectShowCameraDevice,
    control_name: str,
) -> UvcControlRef | None:
    if control_name in VIDEO_PROC_AMP_CONTROLS:
        interface = get_interface(device.filter_object, IAMVideoProcAmp)
        property_id = VIDEO_PROC_AMP_CONTROLS[control_name]
    elif control_name in CAMERA_CONTROL_CONTROLS:
        interface = get_interface(device.filter_object, IAMCameraControl)
        property_id = CAMERA_CONTROL_CONTROLS[control_name]
    else:
        return None

    if interface is None:
        return None
    try:
        min_value, max_value, step, default_value, caps_flags = interface.GetRange(property_id)
        interface.Get(property_id)
    except Exception:
        return None

    return UvcControlRef(
        camera_id=camera_id,
        device=device,
        name=control_name,
        interface=interface,
        property_id=int(property_id),
        min_value=int(min_value),
        max_value=int(max_value),
        step=max(int(step), 1),
        default_value=int(default_value),
        caps_flags=int(caps_flags),
    )


def build_live_control_refs(
    camera_ids: list[int],
    devices_by_camera: dict[int, DirectShowCameraDevice],
) -> dict[int, dict[str, UvcControlRef]]:
    refs_by_camera: dict[int, dict[str, UvcControlRef]] = {}
    for camera_id in camera_ids:
        device = devices_by_camera.get(camera_id)
        if device is None:
            continue
        refs_by_camera[camera_id] = {}
        for control_name in LIVE_UVC_CONTROL_NAMES:
            ref = build_control_ref(camera_id, device, control_name)
            if ref is not None:
                refs_by_camera[camera_id][control_name] = ref
    return refs_by_camera


def current_settings_from_refs(
    camera_ids: list[int],
    refs_by_camera: dict[int, dict[str, UvcControlRef]],
) -> dict[int, dict[str, Any]]:
    settings: dict[int, dict[str, Any]] = {}
    for camera_id in camera_ids:
        camera_settings: dict[str, Any] = {}
        for control_name, ref in refs_by_camera.get(camera_id, {}).items():
            try:
                value, flags = ref.get()
            except Exception:
                continue
            camera_settings[control_name] = {
                "value": int(value),
                "auto": bool(flags & AUTO_FLAG),
            }
        settings[camera_id] = camera_settings
    return settings


def save_current_live_settings(
    camera_ids: list[int],
    refs_by_camera: dict[int, dict[str, UvcControlRef]],
    path: Path = UVC_SETTINGS_JSON_PATH,
) -> None:
    saved_settings = load_saved_uvc_settings(path)
    saved_settings.update(current_settings_from_refs(camera_ids, refs_by_camera))
    save_uvc_settings_snapshot(saved_settings, path)


def draw_live_panel_title(cv2: Any, panel: Any, title: str) -> None:
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 32), (0, 0, 0), -1)
    cv2.putText(
        panel,
        title,
        (12, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def resize_live_panel(cv2: Any, frame: Any, width: int, height: int) -> Any:
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def blank_live_panel(cv2: Any, np: Any, title: str, width: int, height: int) -> Any:
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    draw_live_panel_title(cv2, panel, title)
    cv2.putText(
        panel,
        "no frame",
        (24, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return panel


def draw_threshold_contours(cv2: Any, panel: Any, mask: Any) -> None:
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area <= 0.0:
            continue
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(round(x)), int(round(y)))
        cv2.circle(panel, center, max(2, int(round(radius))), (0, 255, 0), 1)


def build_live_preview(
    cv2: Any,
    np: Any,
    camera_settings_module: Any,
    camera_ids: list[int],
    captures: dict[int, Any],
    devices_by_camera: dict[int, DirectShowCameraDevice],
    fps_by_camera: dict[int, LiveFpsCounter],
    state: LiveTuningState,
    panel_width: int,
    panel_height: int,
) -> Any:
    sections = []
    for camera_id in camera_ids:
        capture = captures.get(camera_id)
        device = devices_by_camera.get(camera_id)
        ok, frame = (False, None)
        if capture is not None:
            ok, frame = capture.read()

        if not ok or frame is None:
            raw_panel = blank_live_panel(cv2, np, f"camera {camera_id} raw", panel_width, panel_height)
            threshold_panel = blank_live_panel(
                cv2,
                np,
                f"camera {camera_id} threshold",
                panel_width,
                panel_height,
            )
        else:
            fps = fps_by_camera[camera_id].mark_frame()
            blur_kernel = camera_settings_module.blur_kernel_for_camera(
                camera_id,
                state.blur_by_camera.get(camera_id, 1),
            )
            mask = camera_settings_module.build_threshold_mask(
                frame,
                camera_id,
                threshold=state.threshold,
                blur_kernel=blur_kernel,
            )
            raw_panel = frame.copy()
            threshold_panel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            draw_threshold_contours(cv2, threshold_panel, mask)
            name = device.friendly_name if device is not None else "unknown"
            draw_live_panel_title(
                cv2,
                raw_panel,
                f"cam {camera_id} {name} | {fps:.1f} fps",
            )
            draw_live_panel_title(
                cv2,
                threshold_panel,
                f"cam {camera_id} threshold | t {state.threshold} blur {blur_kernel}",
            )

        sections.append(
            cv2.hconcat(
                [
                    resize_live_panel(cv2, raw_panel, panel_width, panel_height),
                    resize_live_panel(cv2, threshold_panel, panel_width, panel_height),
                ]
            )
        )

    if not sections:
        return blank_live_panel(cv2, np, "no cameras", panel_width * 2, panel_height)
    if len(sections) == 1:
        return sections[0]
    return cv2.vconcat(sections[:2])


def build_live_controls_preview(
    cv2: Any,
    np: Any,
    camera_ids: list[int],
    refs_by_camera: dict[int, dict[str, UvcControlRef]],
    state: LiveTuningState,
) -> Any:
    lines = [
        "q/Esc quit | s save settings | p print settings",
        f"threshold={state.threshold}",
    ]
    for camera_id in camera_ids:
        parts = []
        for control_name in LIVE_UVC_CONTROL_NAMES:
            ref = refs_by_camera.get(camera_id, {}).get(control_name)
            if ref is None:
                continue
            try:
                value, flags = ref.get()
            except Exception:
                continue
            parts.append(f"{SHORT_CONTROL_NAMES.get(control_name, control_name)}={value}/{flag_label(flags)}")
        lines.append(f"cam {camera_id}: " + " | ".join(parts))

    height = max(180, 32 + 28 * len(lines))
    width = 1080
    image = np.zeros((height, width, 3), dtype=np.uint8)
    y = 26
    for line in lines:
        cv2.putText(
            image,
            line[:160],
            (14, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 28
    return image


def create_live_tuning_windows(
    cv2: Any,
    camera_settings_module: Any,
    camera_ids: list[int],
    refs_by_camera: dict[int, dict[str, UvcControlRef]],
    state: LiveTuningState,
    control_window_name: str,
    preview_window_name: str,
    panel_width: int,
    panel_height: int,
) -> None:
    cv2.namedWindow(preview_window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(preview_window_name, panel_width * 2, panel_height * max(len(camera_ids), 1))
    cv2.namedWindow(control_window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(control_window_name, 1100, 520)
    cv2.createTrackbar("threshold", control_window_name, state.threshold, 255, state.set_threshold)

    for camera_id in camera_ids:
        cv2.createTrackbar(
            f"cam {camera_id} blur",
            control_window_name,
            camera_settings_module.blur_kernel_for_camera(camera_id),
            camera_settings_module.MAX_BLUR_TRACKBAR_VALUE,
            lambda value, selected_camera_id=camera_id: state.set_blur(selected_camera_id, value),
        )
        for control_name, ref in refs_by_camera.get(camera_id, {}).items():
            short_name = SHORT_CONTROL_NAMES.get(control_name, control_name)
            try:
                current_value, current_flags = ref.get()
            except Exception:
                current_value, current_flags = ref.default_value, MANUAL_FLAG

            def set_value_callback(position: int, selected_ref: UvcControlRef = ref) -> None:
                try:
                    selected_ref.set(selected_ref.slider_to_value(position), auto=False)
                except Exception as error:
                    print(f"[uvc live] could not set {selected_ref.name}: {str(error).splitlines()[0]}")

            cv2.createTrackbar(
                f"cam {camera_id} {short_name}",
                control_window_name,
                ref.value_to_slider(current_value),
                ref.slider_max(),
                set_value_callback,
            )
            if ref.caps_flags & AUTO_FLAG:
                def set_auto_callback(position: int, selected_ref: UvcControlRef = ref) -> None:
                    try:
                        selected_ref.set(value=None, auto=bool(position))
                    except Exception as error:
                        print(f"[uvc live] could not set auto {selected_ref.name}: {str(error).splitlines()[0]}")

                cv2.createTrackbar(
                    f"cam {camera_id} {short_name} auto",
                    control_window_name,
                    1 if current_flags & AUTO_FLAG else 0,
                    1,
                    set_auto_callback,
                )


def open_live_capture(cv2: Any, device: DirectShowCameraDevice, width: int, height: int, fps: int) -> Any:
    capture = cv2.VideoCapture(device.index, cv2.CAP_DSHOW)
    if not capture.isOpened():
        capture.release()
        capture = cv2.VideoCapture(device.index)
    if capture.isOpened():
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        capture.set(cv2.CAP_PROP_FPS, fps)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def run_live_tuner(args: argparse.Namespace) -> int:
    import cv2
    import numpy as np
    import multithreaded_camera_testing as camera_settings_module

    camera_ids = args.camera_ids or list(camera_settings_module.CAMERA_IDS)
    devices = enumerate_video_devices()
    devices_by_camera: dict[int, DirectShowCameraDevice] = {}
    for camera_id in camera_ids:
        device = find_device_for_camera_id(camera_id, devices)
        if device is None:
            print(f"[uvc live] camera {camera_id}: no matching DirectShow device")
            continue
        devices_by_camera[camera_id] = device
        print(
            f"[uvc live] camera {camera_id}: {device.friendly_name} "
            f"serial={device.serial or 'unknown'} directshow_index={device.index}"
        )

    if not devices_by_camera:
        print("[uvc live] no matched cameras to tune")
        return 1

    if not args.no_initial_apply:
        apply_configured_camera_settings(
            camera_ids=list(devices_by_camera),
            settings_path=args.settings_json,
        )

    refs_by_camera = build_live_control_refs(list(devices_by_camera), devices_by_camera)
    state = LiveTuningState(list(devices_by_camera), camera_settings_module)
    captures = {
        camera_id: open_live_capture(cv2, device, args.width, args.height, args.fps)
        for camera_id, device in devices_by_camera.items()
    }
    captures = {
        camera_id: capture
        for camera_id, capture in captures.items()
        if capture is not None and capture.isOpened()
    }
    if not captures:
        print("[uvc live] no cameras opened for preview")
        return 1

    preview_window_name = "UVC Live Camera Preview"
    control_window_name = "UVC Live Camera Controls"
    create_live_tuning_windows(
        cv2,
        camera_settings_module,
        list(captures),
        refs_by_camera,
        state,
        control_window_name,
        preview_window_name,
        args.panel_width,
        args.panel_height,
    )
    fps_by_camera = {camera_id: LiveFpsCounter() for camera_id in captures}

    try:
        while True:
            preview = build_live_preview(
                cv2,
                np,
                camera_settings_module,
                list(captures),
                captures,
                devices_by_camera,
                fps_by_camera,
                state,
                args.panel_width,
                args.panel_height,
            )
            controls_preview = build_live_controls_preview(
                cv2,
                np,
                list(captures),
                refs_by_camera,
                state,
            )
            cv2.imshow(preview_window_name, preview)
            cv2.imshow(control_window_name, controls_preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                save_current_live_settings(list(captures), refs_by_camera, args.settings_json)
            if key == ord("p"):
                print(json.dumps(current_settings_from_refs(list(captures), refs_by_camera), indent=2))
    finally:
        for capture in captures.values():
            capture.release()
        cv2.destroyAllWindows()
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="List and apply Windows DirectShow UVC camera settings by camera name/serial.",
    )
    parser.add_argument("--list", action="store_true", help="List DirectShow camera devices.")
    parser.add_argument("--snapshot", action="store_true", help="Read current UVC controls and save them to JSON.")
    parser.add_argument("--apply", action="store_true", help="Apply saved JSON settings plus CAMERA_UVC_SETTINGS defaults.")
    parser.add_argument("--live", action="store_true", help="Open a live raw/threshold preview with UVC sliders.")
    parser.add_argument("--dry-run", action="store_true", help="Print changes without writing them.")
    parser.add_argument("--show-ranges", action="store_true", help="Print supported ranges/current values.")
    parser.add_argument("--no-initial-apply", action="store_true", help="Do not apply saved/default settings before live tuning.")
    parser.add_argument(
        "--settings-json",
        type=Path,
        default=UVC_SETTINGS_JSON_PATH,
        help=f"Settings JSON to snapshot/apply. Default: {UVC_SETTINGS_JSON_PATH}",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--fps", type=int, default=120)
    parser.add_argument("--panel-width", type=int, default=640)
    parser.add_argument("--panel-height", type=int, default=400)
    parser.add_argument(
        "--camera-ids",
        type=parse_camera_ids,
        default=None,
        help="Comma-separated logical camera IDs to apply. Default applies every configured camera.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.live:
        return run_live_tuner(args)
    if args.snapshot:
        saved_count = save_current_uvc_profile(
            camera_ids=args.camera_ids,
            path=args.settings_json,
        )
        return 0 if saved_count > 0 else 1
    if args.list or not args.apply:
        list_devices(show_ranges=args.show_ranges)
        if not args.apply:
            return 0
    applied = apply_configured_camera_settings(
        camera_ids=args.camera_ids,
        dry_run=args.dry_run,
        show_ranges=args.show_ranges,
        settings_path=args.settings_json,
    )
    print(f"[uvc] controls {'checked' if args.dry_run else 'written'}: {applied}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

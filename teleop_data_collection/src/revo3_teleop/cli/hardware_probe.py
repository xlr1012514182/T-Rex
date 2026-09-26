"""Read-only capability probe for the real Revo/EMG/camera assembly."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

from revo3_teleop.hardware import (
    BrainCoEduArmbandConfig,
    BrainCoEduGloveConfig,
    BrainCoEduSdkEMGClient,
    BrainCoEduSdkGloveClient,
    BrainCoRevo3SdkAssembly,
    CameraProbeConfig,
    Revo3ProbeConfig,
    VisionTouchForce6DConfig,
    VisionTouchForce6DSource,
    probe_opencv_camera,
)


SCHEMA_VERSION = "revo3-hardware-probe-config-v1"


def load_probe_config(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"hardware probe config must use {SCHEMA_VERSION}")
    if data.get("allow_hardware_probe") is not True:
        raise PermissionError(
            "config keeps hardware probe disabled; make an operator-reviewed copy and opt in"
        )
    if data.get("allow_hardware_write") is not False:
        raise ValueError("probe config must set allow_hardware_write=false")
    return data


def _component(data: dict[str, Any], name: str) -> dict[str, Any] | None:
    raw = data.get(name)
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"{name} config must be an object")
    if raw.get("enabled") is not True:
        return None
    return raw


async def run_probe(data: dict[str, Any]) -> dict[str, object]:
    result: dict[str, object] = {
        "schema_version": "revo3-hardware-capability-manifest-v1",
        "probe_only": True,
        "hardware_write_performed": False,
        "real_hardware_function_verified": False,
        "components": {},
    }
    components = result["components"]
    assert isinstance(components, dict)

    revo = _component(data, "revo")
    if revo is not None:
        assembly = BrainCoRevo3SdkAssembly(
            Revo3ProbeConfig(
                port=revo.get("port"),
                slave_id=revo.get("slave_id"),
                expected_serial=revo.get("expected_serial"),
                expected_sdk_version=revo.get("expected_sdk_version", "1.5.1"),
                allow_hardware_probe=True,
            )
        )
        connection = await assembly.probe()
        try:
            components["revo3_u21vt"] = connection.report.to_json()
        finally:
            await connection.close()

    emg = _component(data, "emg")
    if emg is not None:
        client = BrainCoEduSdkEMGClient(
            BrainCoEduArmbandConfig(
                port_name=emg.get("port_name"),
                baudrate=int(emg.get("baudrate", 115_200)),
                emg_buffer_length=int(emg.get("emg_buffer_length", 1_250)),
                expected_sdk_version=emg.get("expected_sdk_version", "0.5.0"),
                allow_hardware_discovery=True,
                allow_hardware_stream=False,
            )
        )
        components["brainco_edu_emg"] = client.discover().to_json()

    glove = _component(data, "glove")
    if glove is not None:
        glove_client = BrainCoEduSdkGloveClient(
            BrainCoEduGloveConfig(
                port_name=glove.get("port_name"),
                baudrate=int(glove.get("baudrate", 115_200)),
                expected_serial=glove.get("expected_serial"),
                expected_sdk_version=glove.get("expected_sdk_version", "0.5.0"),
                allow_hardware_discovery=True,
                allow_hardware_stream=False,
            )
        )
        components["brainco_edu_glove"] = glove_client.discover().to_json()

    visiontouch = _component(data, "visiontouch")
    if visiontouch is not None:
        source = VisionTouchForce6DSource(
            VisionTouchForce6DConfig(
                force_model_dir=Path(visiontouch["force_model_dir"]),
                finger_serials=visiontouch["finger_serials"],
                expected_model_sha256=visiontouch["expected_model_sha256"],
                expected_sdk_version=visiontouch.get("expected_sdk_version", "1.0.10"),
                allow_hardware_probe=True,
                allow_hardware_stream=False,
            )
        )
        components["revo3_u21vt_visiontouch_force6d"] = source.probe().to_json()

    camera = _component(data, "camera")
    if camera is not None:
        device: int | str = camera.get("device", 0)
        capability = probe_opencv_camera(
            CameraProbeConfig(
                device=device,
                requested_width=camera.get("width"),
                requested_height=camera.get("height"),
                requested_fps=camera.get("fps"),
                backend=camera.get("backend"),
                require_exact_resolution=bool(camera.get("require_exact_resolution", True)),
                fps_tolerance=float(camera.get("fps_tolerance", 1.0)),
            ),
            allow_hardware_probe=True,
        )
        components["rgb_fisheye_camera"] = capability.to_json()
    if not components:
        raise ValueError("probe config enables no components")
    return result


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only Revo3/pressure/U21VT VisionTouch, BrainCo EDU EMG/glove, "
            "and camera capability probe"
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--allow-hardware-probe",
        action="store_true",
        help="second operator opt-in; the config must also explicitly allow probing",
    )
    args = parser.parse_args(argv)
    if not args.allow_hardware_probe:
        parser.error("--allow-hardware-probe is required")
    try:
        config = load_probe_config(args.config)
    except (OSError, ValueError, PermissionError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    manifest = asyncio.run(run_probe(config))
    _atomic_json(args.output, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Save and load a single D435 RGB/depth frame for offline vision work."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import numpy as np
from PIL import Image


FIXTURE_SCHEMA_VERSION = 1
RGB_FILENAME = "rgb.png"
DEPTH_FILENAME = "depth.npy"
METADATA_FILENAME = "metadata.json"


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    model: str = "unknown"
    coeffs: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "width", _positive_int(self.width, "width"))
        object.__setattr__(self, "height", _positive_int(self.height, "height"))
        for name in ("fx", "fy"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        for name in ("cx", "cy"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if not isinstance(self.model, str) or not self.model:
            raise ValueError("model must be a non-empty string")
        coeffs = tuple(float(value) for value in self.coeffs)
        if not all(np.isfinite(value) for value in coeffs):
            raise ValueError("coeffs must contain only finite values")
        object.__setattr__(self, "coeffs", coeffs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "model": self.model,
            "coeffs": list(self.coeffs),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CameraIntrinsics":
        if not isinstance(data, Mapping):
            raise ValueError("intrinsics must be a mapping")
        required = ("width", "height", "fx", "fy", "cx", "cy")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"intrinsics missing fields: {', '.join(missing)}")
        return cls(
            width=data["width"],
            height=data["height"],
            fx=data["fx"],
            fy=data["fy"],
            cx=data["cx"],
            cy=data["cy"],
            model=data.get("model", "unknown"),
            coeffs=tuple(data.get("coeffs", ())),
        )


@dataclass(frozen=True)
class D435Frame:
    rgb: np.ndarray
    depth: np.ndarray
    color_intrinsics: CameraIntrinsics
    depth_scale: float
    timestamp_ns: int
    depth_aligned_to_color: bool
    depth_intrinsics: Optional[CameraIntrinsics] = None
    frame_number: Optional[int] = None
    camera_model: str = "D435"
    device_timestamp_ms: Optional[float] = None

    def __post_init__(self) -> None:
        rgb = np.asarray(self.rgb)
        depth = np.asarray(self.depth)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError("rgb must have shape (height, width, 3) and dtype uint8")
        if depth.ndim != 2 or depth.dtype.kind != "u":
            raise ValueError("depth must be a 2-D unsigned integer array")
        if tuple(rgb.shape[:2]) != (
            self.color_intrinsics.height,
            self.color_intrinsics.width,
        ):
            raise ValueError("rgb shape does not match color intrinsics resolution")
        if self.depth_aligned_to_color and depth.shape != rgb.shape[:2]:
            raise ValueError("aligned depth must have the same resolution as rgb")
        if self.depth_intrinsics is not None and depth.shape != (
            self.depth_intrinsics.height,
            self.depth_intrinsics.width,
        ) and not self.depth_aligned_to_color:
            raise ValueError("depth shape does not match depth intrinsics resolution")
        depth_scale = float(self.depth_scale)
        if not np.isfinite(depth_scale) or depth_scale <= 0:
            raise ValueError("depth_scale must be finite and positive")
        timestamp_ns = _positive_int(self.timestamp_ns, "timestamp_ns")
        if self.frame_number is not None:
            object.__setattr__(self, "frame_number", _positive_int(self.frame_number, "frame_number"))
        if not isinstance(self.camera_model, str) or not self.camera_model:
            raise ValueError("camera_model must be a non-empty string")
        if self.device_timestamp_ms is not None:
            device_timestamp_ms = float(self.device_timestamp_ms)
            if not np.isfinite(device_timestamp_ms) or device_timestamp_ms < 0:
                raise ValueError("device_timestamp_ms must be finite and non-negative")
            object.__setattr__(self, "device_timestamp_ms", device_timestamp_ms)
        object.__setattr__(self, "rgb", rgb.copy())
        object.__setattr__(self, "depth", depth.copy())
        object.__setattr__(self, "depth_scale", depth_scale)
        object.__setattr__(self, "timestamp_ns", timestamp_ns)

    @property
    def color_resolution(self) -> tuple[int, int]:
        return self.color_intrinsics.width, self.color_intrinsics.height

    @property
    def depth_resolution(self) -> tuple[int, int]:
        return int(self.depth.shape[1]), int(self.depth.shape[0])

    @property
    def timestamp_utc(self) -> str:
        return datetime.fromtimestamp(
            self.timestamp_ns / 1_000_000_000, tz=timezone.utc
        ).isoformat()

    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": FIXTURE_SCHEMA_VERSION,
            "camera_model": self.camera_model,
            "timestamp_ns": self.timestamp_ns,
            "timestamp_utc": self.timestamp_utc,
            "frame_number": self.frame_number,
            "device_timestamp_ms": self.device_timestamp_ms,
            "color_intrinsics": self.color_intrinsics.to_dict(),
            "depth_intrinsics": (
                self.depth_intrinsics.to_dict()
                if self.depth_intrinsics is not None
                else None
            ),
            "depth_scale": self.depth_scale,
            "depth_scale_units": "metres_per_raw_unit",
            "resolution": {
                "color": list(self.color_resolution),
                "depth": list(self.depth_resolution),
            },
            "rgb": {"path": RGB_FILENAME, "format": "RGB8", "dtype": str(self.rgb.dtype)},
            "depth": {
                "path": DEPTH_FILENAME,
                "format": "numpy",
                "dtype": str(self.depth.dtype),
                "units": "raw_depth_units",
                "aligned_to": "color" if self.depth_aligned_to_color else None,
            },
        }

    def save(self, directory: Union[str, Path]) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        Image.fromarray(self.rgb, mode="RGB").save(directory / RGB_FILENAME)
        np.save(directory / DEPTH_FILENAME, self.depth, allow_pickle=False)
        with (directory / METADATA_FILENAME).open("w", encoding="utf-8") as handle:
            json.dump(self.metadata(), handle, indent=2, sort_keys=True)
            handle.write("\n")
        return directory

    @classmethod
    def load(cls, directory: Union[str, Path]) -> "D435Frame":
        directory = Path(directory)
        metadata_path = directory / METADATA_FILENAME
        if not metadata_path.is_file():
            raise FileNotFoundError(f"fixture metadata not found: {metadata_path}")
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if metadata.get("schema_version") != FIXTURE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported fixture schema: {metadata.get('schema_version')}"
            )
        rgb_path = directory / metadata.get("rgb", {}).get("path", RGB_FILENAME)
        depth_path = directory / metadata.get("depth", {}).get("path", DEPTH_FILENAME)
        if not rgb_path.is_file():
            raise FileNotFoundError(f"fixture RGB image not found: {rgb_path}")
        if not depth_path.is_file():
            raise FileNotFoundError(f"fixture depth array not found: {depth_path}")
        with Image.open(rgb_path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        depth = np.load(depth_path, allow_pickle=False)
        resolution = metadata.get("resolution", {})
        actual_color = (rgb.shape[1], rgb.shape[0])
        actual_depth = (depth.shape[1], depth.shape[0]) if depth.ndim == 2 else ()
        if resolution.get("color") and actual_color != tuple(resolution["color"]):
            raise ValueError(f"RGB resolution mismatch: metadata={resolution['color']}, actual={actual_color}")
        if resolution.get("depth") and actual_depth != tuple(resolution["depth"]):
            raise ValueError(f"depth resolution mismatch: metadata={resolution['depth']}, actual={actual_depth}")
        return cls(
            rgb=rgb,
            depth=depth,
            color_intrinsics=CameraIntrinsics.from_dict(metadata["color_intrinsics"]),
            depth_scale=metadata["depth_scale"],
            timestamp_ns=metadata["timestamp_ns"],
            depth_aligned_to_color=metadata.get("depth", {}).get("aligned_to") == "color",
            depth_intrinsics=(
                CameraIntrinsics.from_dict(metadata["depth_intrinsics"])
                if metadata.get("depth_intrinsics") is not None
                else None
            ),
            frame_number=metadata.get("frame_number"),
            camera_model=metadata.get("camera_model", "D435"),
            device_timestamp_ms=metadata.get("device_timestamp_ms"),
        )


def save_d435_fixture(frame: D435Frame, directory: Union[str, Path]) -> Path:
    if not isinstance(frame, D435Frame):
        raise TypeError("frame must be a D435Frame")
    return frame.save(directory)


def load_d435_fixture(directory: Union[str, Path]) -> D435Frame:
    return D435Frame.load(directory)

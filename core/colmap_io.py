"""flight2world.core.colmap_io

Pure COLMAP text-format readers, extracted faithfully from `fusion_v10.py`
(the immutable V10 baseline) plus minimal, pure parsing for cameras.txt.

Coordinate convention preserved from V10 (COLMAP convention):

    X_cam = R @ X_world + t
    camera_center = C = -R^T @ t

No georeferencing, GPS handling, or coordinate-system conversion is
performed here. Metric scale is unknown.

Reader functions `load_colmap_images`, `load_colmap_points`, `qvec2rotmat`
and `camera_center` reproduce the exact V10 behavior (byte-for-byte logic).
New, purely-additive helpers (`parse_cameras`, accessors, projection) are
provided for reuse and testing; they do not alter V10.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from core.config import CameraIntrinsics


# ============================================================
# Quaternion -> rotation matrix (V10 verbatim, lines 133-139)
# ============================================================
def qvec2rotmat(q):
    """Convert COLMAP quaternion [qw, qx, qy, qz] to a 3x3 rotation matrix.

    Identical to fusion_v10.py qvec2rotmat.
    """
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy]
    ], dtype=np.float64)


# ============================================================
# Registered images (V10 verbatim, lines 142-195)
# ============================================================
def load_colmap_images(path):
    """Parse a COLMAP images.txt into {image_id: image dict}.

    Identical to fusion_v10.py load_colmap_images. Each image dict has
    keys: id, name, R (3x3), t (3,), observations (list of
    (x, y, point3d_id) where point3d_id >= 0).
    """
    images = {}
    with open(path, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue

        parts = line.split()
        if len(parts) >= 10:
            try:
                image_id = int(parts[0])
                qw, qx, qy, qz = map(float, parts[1:5])
                tx, ty, tz = map(float, parts[5:8])
                camera_id = int(parts[8])
                name = parts[9]
            except ValueError:
                i += 1
                continue

            R = qvec2rotmat(np.array([qw, qx, qy, qz]))
            t = np.array([tx, ty, tz], dtype=np.float64)

            observations = []
            if i + 1 < len(lines):
                obs_line = lines[i + 1].strip()
                if obs_line and not obs_line.startswith("#"):
                    obs_parts = obs_line.split()
                    for j in range(0, len(obs_parts) - 2, 3):
                        try:
                            x = float(obs_parts[j])
                            y = float(obs_parts[j + 1])
                            point_id = int(obs_parts[j + 2])
                            if point_id >= 0:
                                observations.append((x, y, point_id))
                        except ValueError:
                            pass
                    i += 2
                    images[image_id] = {
                        "id": image_id,
                        "name": name,
                        "R": R,
                        "t": t,
                        "observations": observations,
                    }
                    continue

        i += 1

    return images


# ============================================================
# 3D points (V10 verbatim, lines 198-215)
# ============================================================
def load_colmap_points(path):
    """Parse a points3D.txt into {point3d_id: np.ndarray XYZ (3,)}.

    Identical to fusion_v10.py load_colmap_points. Only lines with >= 8
    whitespace-separated fields are accepted, matching V10.
    """
    points = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            try:
                point_id = int(parts[0])
                points[point_id] = np.array(
                    list(map(float, parts[1:4])), dtype=np.float64
                )
            except ValueError:
                continue
    return points


# ============================================================
# Camera center (V10 verbatim, lines 218-220)
# ============================================================
def camera_center(R, t):
    """C = -R^T t (COLMAP convention: X_cam = R X_world + t)."""
    return -R.T @ t


# ============================================================
# Cameras.txt parsing (pure; ADDITIVE, not part of V10)
# ============================================================

# COLMAP intrinsics models and their parameter layouts (camera model names
# and param order per COLMAP text format):
#   SIMPLE_PINHOLE : [f, cx, cy]
#   PINHOLE        : [fx, fy, cx, cy]
#   SIMPLE_RADIAL  : [f, cx, cy, k]
#   RADIAL         : [f, cx, cy, k1, k2]
#   OPENCV         : [fx, fy, cx, cy, k1, k2, p1, p2]
#   ...
# Pinhole-readable models (fx, fy directly derivable) are handled here.
_CAMERA_PARAM_LAYOUTS = {
    "SIMPLE_PINHOLE": ("f", "cx", "cy"),
    "PINHOLE": ("fx", "fy", "cx", "cy"),
    "SIMPLE_RADIAL": ("f", "cx", "cy", "k"),
    "RADIAL": ("f", "cx", "cy", "k1", "k2"),
    "OPENCV": ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2"),
    "OPENCV_FISHEYE": ("fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4"),
}


@dataclass(frozen=True)
class Camera:
    """A parsed COLMAP camera.

    ``params`` holds the float parameters in COLMAP's order for the model.
    ``fx``/``fy`` are provided for all pinhole-family models (f used for
    both axes in single-focal models).
    """

    camera_id: int
    model: str
    width: int
    height: int
    params: Tuple[float, ...]

    @property
    def fx(self) -> float:
        layout = _CAMERA_PARAM_LAYOUTS[self.model]
        if "fx" in layout:
            return self.params[layout.index("fx")]
        return self.params[layout.index("f")]

    @property
    def fy(self) -> float:
        layout = _CAMERA_PARAM_LAYOUTS[self.model]
        if "fy" in layout:
            return self.params[layout.index("fy")]
        return self.params[layout.index("f")]

    @property
    def cx(self) -> float:
        return self.params[_CAMERA_PARAM_LAYOUTS[self.model].index("cx")]

    @property
    def cy(self) -> float:
        return self.params[_CAMERA_PARAM_LAYOUTS[self.model].index("cy")]

    def intrinsics(self) -> CameraIntrinsics:
        """Return a CameraIntrinsics in the pinhole convention V10 uses.

        V10 ignores the radial distortion term (SIMPLE_RADIAL k) and uses
        a pure pinhole with fx == fy == f.
        """
        return CameraIntrinsics(
            fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy,
            width=self.width, height=self.height,
        )


def parse_cameras(path) -> Dict[int, Camera]:
    """Parse a cameras.txt into {camera_id: Camera}.

    Pure COLMAP parsing (additive helper; V10 hardcodes the intrinsics
    rather than reading this file).
    """
    cameras = {}
    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            # skip blank lines, comments, and the "Number of cameras:" trailer
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            try:
                camera_id = int(parts[0])
                model = parts[1]
                width = int(parts[2])
                height = int(parts[3])
                params = tuple(float(p) for p in parts[4:])
            except (ValueError, IndexError):
                continue
            if model not in _CAMERA_PARAM_LAYOUTS:
                # Unknown model: keep raw params; fx/fy unavailable.
                cameras[camera_id] = Camera(camera_id, model, width,
                                            height, params)
                continue
            cameras[camera_id] = Camera(camera_id, model, width, height,
                                        params)
    return cameras


# ============================================================
# Accessors / pure projection helpers (ADDITIVE, for reuse + tests)
# ============================================================

def image_pose(image_data) -> Tuple[np.ndarray, np.ndarray]:
    """Return (R, t) for a loaded image dict, in V10's convention."""
    return image_data["R"], image_data["t"]


def image_observations(image_data) -> List[Tuple[float, float, int]]:
    """Return the list of (x, y, point3d_id) observations for an image."""
    return image_data["observations"]


def world_to_cam(X_world, R, t):
    """X_cam = R @ X_world + t (COLMAP convention, as V10)."""
    return R @ X_world + t


def project_to_image(X_cam, intrinsics) -> Tuple[np.ndarray, np.ndarray]:
    """Pinhole-project camera coordinates to (u, v) pixel coordinates.

    Uses the exact pinhole convention V10 employs (lines 699-700, 761-762):
        x = (u - cx) / fx * z   =>  u = fx * x / z + cx
        y = (v - cy) / fy * z   =>  v = fy * y / z + cy
    """
    fx, fy = intrinsics.fx, intrinsics.fy
    cx, cy = intrinsics.cx, intrinsics.cy
    x, y, z = X_cam
    if z <= 0:
        return np.nan, np.nan
    return fx * x / z + cx, fy * y / z + cy
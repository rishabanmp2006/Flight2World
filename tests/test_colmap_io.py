"""tests.test_colmap_io

Deterministic tests for core.colmap_io using the real COLMAP artifacts
in test/sparse_txt/. Covers requirement A (camera parsing), B (55 images),
C (20265 points), and D (pose convention).
"""

import math
import pathlib
import sys

# Make the project root importable regardless of how this file is invoked
# (`python tests/test_x.py` puts tests/ on sys.path, not the project root).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from tests import SPARSE_TXT, run_module_tests

import core.colmap_io as cio
import core.config as cfg

CAMERAS = SPARSE_TXT / "cameras.txt"
IMAGES = SPARSE_TXT / "images.txt"
POINTS = SPARSE_TXT / "points3D.txt"

# Reference camera parameters from cameras.txt
#   1 SIMPLE_RADIAL 1280 720 1167.4277386087481 640 360 -0.05865183452009938
REF_FX = 1167.4277386087481
REF_CX = 640.0
REF_CY = 360.0
REF_W = 1280
REF_H = 720


# ------------------------------------------------------------
# A. Camera parsing
# ------------------------------------------------------------
def test_camera_parsing_width_height():
    cams = cio.parse_cameras(CAMERAS)
    assert len(cams) == 1, "expected exactly one camera, got %d" % len(cams)
    cam = cams[1]
    assert cam.width == REF_W
    assert cam.height == REF_H


def test_camera_parsing_intrinsics():
    cam = cio.parse_cameras(CAMERAS)[1]
    assert cam.model == "SIMPLE_RADIAL"
    assert math.isclose(cam.fx, REF_FX, rel_tol=1e-12)
    assert math.isclose(cam.fy, REF_FX, rel_tol=1e-12)
    assert math.isclose(cam.cx, REF_CX, rel_tol=1e-12)
    assert math.isclose(cam.cy, REF_CY, abs_tol=1e-12)


def test_camera_distortion_param_preserved():
    cam = cio.parse_cameras(CAMERAS)[1]
    # SIMPLE_RADIAL params = [f, cx, cy, k]; k is preserved raw.
    assert math.isclose(cam.params[3], -0.05865183452009938, rel_tol=1e-12)


def test_camera_intrinsics_object_matches_config():
    cam = cio.parse_cameras(CAMERAS)[1]
    k = cam.intrinsics()
    assert (k.fx, k.fy, k.cx, k.cy, k.width, k.height) == (
        cfg.FX, cfg.FY, cfg.CX, cfg.CY, cfg.IMAGE_W, cfg.IMAGE_H)


def test_config_constants_match_reference():
    assert cfg.FX == REF_FX
    assert cfg.FY == REF_FX
    assert cfg.CX == REF_CX
    assert cfg.CY == REF_CY
    assert cfg.IMAGE_W == REF_W
    assert cfg.IMAGE_H == REF_H
    assert cfg.METRIC_SCALE_UNKNOWN is True


# ------------------------------------------------------------
# B. Image parsing (55 registered images)
# ------------------------------------------------------------
def test_image_count_is_55():
    imgs = cio.load_colmap_images(IMAGES)
    assert len(imgs) == 55, "expected 55 registered images, got %d" % len(imgs)


def test_image_record_structure():
    imgs = cio.load_colmap_images(IMAGES)
    for ident, rec in imgs.items():
        assert isinstance(ident, int)
        for key in ("id", "name", "R", "t", "observations"):
            assert key in rec, "missing key %r in %s" % (key, ident)
        assert rec["R"].shape == (3, 3)
        assert rec["t"].shape == (3,)
    assert imgs[27]["name"] == "frame_0027.jpg"


def test_image_pose_content():
    imgs = cio.load_colmap_images(IMAGES)
    R, t = cio.image_pose(imgs[27])
    # Pose recorded in images.txt for image 27 (frame_0027.jpg)
    expected_t = np.array([0.14151307778977879, -0.65908826377446417,
                           1.237283557129194])
    assert np.allclose(t, expected_t, atol=1e-12)
    # R must be a proper rotation.
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert math.isclose(np.linalg.det(R), 1.0, abs_tol=1e-9)


def test_every_image_has_observations():
    imgs = cio.load_colmap_images(IMAGES)
    for ident, rec in imgs.items():
        obs = cio.image_observations(rec)
        assert len(obs) > 0, "image %d has no observations" % ident
        for (x, y, pid) in obs:
            assert pid >= 0          # only triangulated observations kept
            assert np.isfinite(x) and np.isfinite(y)


def test_images_sorted_by_name_are_consecutive_frames():
    imgs = cio.load_colmap_images(IMAGES)
    names = sorted(rec["name"] for rec in imgs.values())
    assert names[0] == "frame_0027.jpg"
    # frame numbers are consecutive from 0027 .. 0081 (55 frames)
    nums = [int(n.split("_")[1].split(".")[0]) for n in names]
    assert nums == list(range(27, 27 + 55))


# ------------------------------------------------------------
# C. 3D point parsing (20265 points)
# ------------------------------------------------------------
def test_point_count_is_20262():
    # points3D.txt has 20265 physical lines: 3 comment ('#') header lines
    # and 20262 data lines. Every data line has >= 8 whitespace fields
    # (verified: zero fragmented lines), so V10's loader -- which skips
    # '#' lines and requires len(parts) >= 8 -- returns exactly 20262
    # points. The file's own header agrees: "Number of points: 20262".
    with open(POINTS) as f:
        data_lines = [ln for ln in f if ln.strip() and not ln.startswith("#")]
    assert len(data_lines) == 20262, "data lines != 20262: %d" % len(data_lines)
    pts = cio.load_colmap_points(POINTS)
    assert len(pts) == 20262, "parsed %d points, expected 20262" % len(pts)


def test_point_members_are_finite_xyz():
    pts = cio.load_colmap_points(POINTS)
    # point id 1 (first data line in points3D.txt)
    assert np.allclose(pts[1],
                       [2.1544089327054774, 1.1554149734400936,
                        3.3712715334084771], atol=1e-9)
    allpts = np.array(list(pts.values()), dtype=np.float64)
    assert np.all(np.isfinite(allpts)), "non-finite coordinate in points"

    # A point observed by image 27 (frame_0027) exists; cross-check that
    # some parsed point is referenced by an observation.
    imgs = cio.load_colmap_images(IMAGES)
    obs_ids = {pid for _x, _y, pid in cio.image_observations(imgs[27])}
    assert obs_ids & set(pts.keys()), "no shared point between image and cloud"


# ------------------------------------------------------------
# D. Pose convention
# ------------------------------------------------------------
def test_pose_convention_synthetic_exact():
    # Airtight convention test independent of real data.
    theta = 0.7
    R = np.array([[math.cos(theta), -math.sin(theta), 0.0],
                  [math.sin(theta), math.cos(theta), 0.0],
                  [0.0, 0.0, 1.0]])
    t = np.array([1.0, 2.0, 3.0])
    X_world = np.array([4.0, 5.0, 6.0])

    X_cam = cio.world_to_cam(X_world, R, t)
    expected = R @ X_world + t
    assert np.allclose(X_cam, expected, atol=1e-12)

    C = cio.camera_center(R, t)
    assert np.allclose(C, -R.T @ t, atol=1e-12)

    # Reprojection round trips with the pinhole convention V10 uses.
    k = cio.CameraIntrinsics(fx=800.0, fy=800.0, cx=320.0, cy=240.0,
                             width=640, height=480)
    u, v = cio.project_to_image(X_cam, k)
    u_expect = k.fx * X_cam[0] / X_cam[2] + k.cx
    v_expect = k.fy * X_cam[1] / X_cam[2] + k.cy
    assert math.isclose(u, u_expect, rel_tol=1e-9)
    assert math.isclose(v, v_expect, rel_tol=1e-9)


def test_pose_convention_real_observation_reprojects():
    # Take a real triangulated point observed by image 27, place it in
    # world space, transform to camera space with the V10 convention and
    # pinhole-project; the result must land on the recorded observation.
    imgs = cio.load_colmap_images(IMAGES)
    pts = cio.load_colmap_points(POINTS)
    cam = cio.parse_cameras(CAMERAS)[1]
    k = cam.intrinsics()

    im27 = imgs[27]
    R, t = cio.image_pose(im27)
    found = None
    for (x, y, pid) in cio.image_observations(im27):
        if pid in pts:
            found = (x, y, pid, pts[pid])
            break
    assert found is not None, "frame_0027 has no triangulated observation"
    x_obs, y_obs, pid, X_world = found

    X_cam = cio.world_to_cam(X_world, R, t)
    assert X_cam[2] > 0, "observed point must lie in front of the camera"
    u, v = cio.project_to_image(X_cam, k)

    # tolerance generous to the reconstruction's small reprojection error;
    # the CONVENTION (not BA quality) is what we assert.
    assert abs(u - x_obs) < 2.0, (u, x_obs)
    assert abs(v - y_obs) < 2.0, (v, y_obs)


if __name__ == "__main__":
    run_module_tests(globals())
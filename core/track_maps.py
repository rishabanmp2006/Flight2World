"""flight2world.core.track_maps

V10 track-map construction, extracted faithfully from `fusion_v10.py`
(`build_track_maps`, lines 330-382). The algorithm is NOT redesigned.

For every pixel of a frame, produce the camera-space depth Z of the
nearest TRIANGULATED COLMAP track. Tracks are sparse (thousands of
pixels per image), so V10 rasterizes them and fills the gaps by
iterative 8-neighbour dilation, propagating depth as the mean of the
valid neighbours. Computed with only cv2/numpy (no scipy), as in V10.

Outputs (all of shape (image_h, image_w); defaults are the V10 camera
1280x720):

    z_grid     float32; NaN where no track depth could be propagated
    dist_grid  int32;   0 on real track pixels, dilation step > 0 elsewhere
    n_tracks   int;     number of rasterized track pixels (pre-dilation)

Coordinate convention is preserved from V10: X_cam = R @ X_world + t,
and rasterization indexes z_grid[row=yi, col=xi] -- no axis inversion.
"""

import cv2
import numpy as np

from core.config import (
    IMAGE_H,
    IMAGE_W,
    TRACK_MAP_KERNEL,
    TRACK_MAP_MAX_STEPS,
)


def build_track_maps(image_data, sparse_points,
                     image_w=IMAGE_W, image_h=IMAGE_H,
                     max_steps=TRACK_MAP_MAX_STEPS,
                     kernel_size=TRACK_MAP_KERNEL):
    """Rasterize a frame's triangulated tracks and fill gaps by dilation.

    Byte-for-byte faithful to fusion_v10.py `build_track_maps`
    (lines 330-382); the image dimensions are threaded as parameters
    (defaulting to the V10 camera values) instead of module globals.
    With defaults the behavior is identical to V10.

    Returns (z_grid, dist_grid, n_tracks).
    """
    R = image_data["R"]
    t = image_data["t"]

    z_grid = np.full((image_h, image_w), np.nan, dtype=np.float32)
    valid = np.zeros((image_h, image_w), dtype=np.uint8)

    for x, y, point_id in image_data["observations"]:
        if point_id not in sparse_points:
            continue

        X_cam = R @ sparse_points[point_id] + t
        z = X_cam[2]
        if z <= 0:
            continue

        xi = int(round(x))
        yi = int(round(y))
        if 0 <= xi < image_w and 0 <= yi < image_h:
            z_grid[yi, xi] = z
            valid[yi, xi] = 1

    mask = valid.astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dist_grid = np.full((image_h, image_w), 0, dtype=np.int32)

    step = 0
    while not mask.all() and step < max_steps:
        step += 1
        prev = mask
        mask = cv2.dilate(mask, kernel)
        added = (mask.astype(bool)) & (~prev.astype(bool))
        if not added.any():
            break

        zsum = cv2.filter2D(
            np.where(prev.astype(bool), z_grid, 0.0).astype(np.float64),
            -1, kernel.astype(np.float64),
            borderType=cv2.BORDER_CONSTANT,
        )
        cnt = cv2.filter2D(
            prev.astype(np.float64),
            -1, kernel.astype(np.float64),
            borderType=cv2.BORDER_CONSTANT,
        )
        fill = added & (cnt > 0)
        if not fill.any():
            break
        z_grid[fill] = (zsum[fill] / np.maximum(cnt[fill], 1.0)).astype(
            np.float32
        )
        dist_grid[added] = step

    # dist_grid = 0 on real track pixels; >0 elsewhere.
    return z_grid, dist_grid, int(np.sum(valid.astype(bool)))

import os
import math
import numpy as np
import cv2
import open3d as o3d
from PIL import Image
from transformers import pipeline


# ============================================================
# FLIGHT2WORLD V8
# 55-frame hybrid COLMAP + Depth Anything reconstruction
# ============================================================

BASE = os.path.expanduser("~/flight2world")
FRAME_DIR = os.path.join(BASE, "test", "frames")
SPARSE_TXT = os.path.join(BASE, "test", "sparse_txt")

OUTPUT = os.path.join(BASE, "test", "fused_55_v8.ply")

# ------------------------------------------------------------
# Reconstruction settings
# ------------------------------------------------------------

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

# Generate one depth point every N pixels.
PIXEL_STRIDE = 8

# Multi-view consistency.
RELATIVE_TOLERANCE = 0.20
ABSOLUTE_TOLERANCE = 0.08

# Number of neighbouring registered frames to inspect.
NEIGHBOR_RADIUS = 3

# Minimum number of OTHER views that must agree.
MIN_NEIGHBOR_SUPPORT = 2

# Final voxel size.
VOXEL_SIZE = 0.020

# Statistical cleanup.
STAT_NB_NEIGHBORS = 20
STAT_STD_RATIO = 2.0


# ============================================================
# Camera intrinsics
# ============================================================

FX = 1167.4277386087481
FY = 1167.4277386087481
CX = 640.0
CY = 360.0


# ============================================================
# Quaternion -> rotation matrix
# COLMAP convention:
# X_cam = R @ X_world + t
# ============================================================

def qvec2rotmat(q):
    qw, qx, qy, qz = q

    return np.array([
        [
            1 - 2*qy*qy - 2*qz*qz,
            2*qx*qy - 2*qz*qw,
            2*qx*qz + 2*qy*qw
        ],
        [
            2*qx*qy + 2*qz*qw,
            1 - 2*qx*qx - 2*qz*qz,
            2*qy*qz - 2*qx*qw
        ],
        [
            2*qx*qz - 2*qy*qw,
            2*qy*qz + 2*qx*qw,
            1 - 2*qx*qx - 2*qy*qy
        ]
    ], dtype=np.float64)


# ============================================================
# Read COLMAP images.txt
# ============================================================

def load_colmap_images(path):

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

        # Image pose line has:
        # IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME

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

            R = qvec2rotmat(
                np.array([qw, qx, qy, qz])
            )

            t = np.array(
                [tx, ty, tz],
                dtype=np.float64
            )

            observations = []

            # Next line contains:
            # X Y POINT3D_ID X Y POINT3D_ID ...

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
                                observations.append(
                                    (x, y, point_id)
                                )

                        except ValueError:
                            pass

                    i += 2
                    images[image_id] = {
                        "id": image_id,
                        "name": name,
                        "R": R,
                        "t": t,
                        "observations": observations
                    }

                    continue

        i += 1

    return images


# ============================================================
# Read COLMAP points3D.txt
# ============================================================

def load_colmap_points(path):

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

                xyz = np.array(
                    [
                        float(parts[1]),
                        float(parts[2]),
                        float(parts[3])
                    ],
                    dtype=np.float64
                )

                points[point_id] = xyz

            except ValueError:
                continue

    return points


# ============================================================
# Calibration
# AI depth -> COLMAP camera-space Z
#
# Z_colmap = scale * AI_depth + offset
# ============================================================

def calibrate_depth(image_data, sparse_points):

    ai_values = []
    z_values = []

    R = image_data["R"]
    t = image_data["t"]

    for x, y, point_id in image_data["observations"]:

        if point_id not in sparse_points:
            continue

        X_world = sparse_points[point_id]

        X_cam = R @ X_world + t

        z = X_cam[2]

        if z <= 0:
            continue

        xi = int(round(x))
        yi = int(round(y))

        if xi < 0 or xi >= IMAGE_WIDTH:
            continue

        if yi < 0 or yi >= IMAGE_HEIGHT:
            continue

        ai_values.append(
            (xi, yi)
        )

        z_values.append(z)

    return ai_values, np.array(z_values, dtype=np.float64)


def fit_linear_calibration(depth_map, pixels, z_values):

    if len(pixels) < 50:
        return None

    ai = np.array(
        [depth_map[y, x] for x, y in pixels],
        dtype=np.float64
    )

    valid = np.isfinite(ai) & np.isfinite(z_values)

    ai = ai[valid]
    z = z_values[valid]

    if len(ai) < 50:
        return None

    # Remove extreme observations.
    lo_ai, hi_ai = np.percentile(ai, [2, 98])
    lo_z, hi_z = np.percentile(z, [2, 98])

    mask = (
        (ai >= lo_ai) &
        (ai <= hi_ai) &
        (z >= lo_z) &
        (z <= hi_z)
    )

    ai = ai[mask]
    z = z[mask]

    if len(ai) < 50:
        return None

    A = np.vstack(
        [ai, np.ones_like(ai)]
    ).T

    scale, offset = np.linalg.lstsq(
        A,
        z,
        rcond=None
    )[0]

    predicted = scale * ai + offset

    rmse = float(
        np.sqrt(
            np.mean(
                (predicted - z) ** 2
            )
        )
    )

    corr = float(
        np.corrcoef(ai, z)[0, 1]
    )

    return {
        "scale": float(scale),
        "offset": float(offset),
        "rmse": rmse,
        "corr": corr,
        "samples": len(ai)
    }


# ============================================================
# Project world point into camera
# ============================================================

def project_world_to_image(X_world, image_data):

    R = image_data["R"]
    t = image_data["t"]

    X_cam = R @ X_world + t

    z = X_cam[2]

    if z <= 0:
        return None

    u = FX * X_cam[0] / z + CX
    v = FY * X_cam[1] / z + CY

    if (
        u < 0 or
        u >= IMAGE_WIDTH or
        v < 0 or
        v >= IMAGE_HEIGHT
    ):
        return None

    return u, v, z


# ============================================================
# Main
# ============================================================

print()
print("=" * 70)
print("FLIGHT2WORLD V8")
print("55-FRAME HYBRID RECONSTRUCTION")
print("=" * 70)
print()


# ------------------------------------------------------------
# Load COLMAP
# ------------------------------------------------------------

images_txt = os.path.join(
    SPARSE_TXT,
    "images.txt"
)

points_txt = os.path.join(
    SPARSE_TXT,
    "points3D.txt"
)

print("Loading COLMAP cameras...")

images = load_colmap_images(images_txt)

print(
    "Registered COLMAP images:",
    len(images)
)

print("Loading COLMAP sparse points...")

sparse_points = load_colmap_points(points_txt)

print(
    "COLMAP sparse points:",
    len(sparse_points)
)

if len(images) < 20:
    raise RuntimeError(
        "Too few registered COLMAP images."
    )


# ------------------------------------------------------------
# Sort frames chronologically
# ------------------------------------------------------------

registered = []

for image_id, data in images.items():

    path = os.path.join(
        FRAME_DIR,
        data["name"]
    )

    if os.path.exists(path):

        registered.append(
            (image_id, data)
        )

registered.sort(
    key=lambda x: x[1]["name"]
)

print(
    "Usable registered frames:",
    len(registered)
)

if len(registered) != 55:

    print(
        "WARNING: Expected 55 registered frames, "
        "but found",
        len(registered)
    )


# ------------------------------------------------------------
# Build sparse Open3D cloud
# ------------------------------------------------------------

print()
print("Building sparse COLMAP spatial gate...")

sparse_xyz = np.array(
    list(sparse_points.values()),
    dtype=np.float64
)

sparse_pcd = o3d.geometry.PointCloud()

sparse_pcd.points = o3d.utility.Vector3dVector(
    sparse_xyz
)

sparse_tree = o3d.geometry.KDTreeFlann(
    sparse_pcd
)

# Estimate characteristic sparse spacing.
nn_distances = []

sample_count = min(
    len(sparse_xyz),
    5000
)

sample_indices = np.linspace(
    0,
    len(sparse_xyz) - 1,
    sample_count,
    dtype=int
)

for idx in sample_indices:

    p = sparse_xyz[idx]

    count, indices, distances = sparse_tree.search_knn_vector_3d(
        p,
        2
    )

    if count >= 2:

        d = math.sqrt(
            distances[1]
        )

        if np.isfinite(d) and d > 0:
            nn_distances.append(d)

if nn_distances:

    median_spacing = float(
        np.median(nn_distances)
    )

else:

    median_spacing = 0.03

# Adaptive gate.
colmap_radius = np.clip(
    median_spacing * 8.0,
    0.08,
    0.20
)

print(
    "Median sparse spacing:",
    median_spacing
)

print(
    "COLMAP spatial gate radius:",
    colmap_radius
)


# ------------------------------------------------------------
# Load Depth Anything V2
# ------------------------------------------------------------

print()
print("Loading Depth Anything V2 Small...")

depth_pipe = pipeline(
    "depth-estimation",
    model="depth-anything/Depth-Anything-V2-Small-hf",
    device="mps"
)

print("Depth model loaded.")
print()


# ------------------------------------------------------------
# Storage for calibrated depths
# ------------------------------------------------------------

frame_records = []

successful = 0


# ------------------------------------------------------------
# First pass:
# Run AI depth and calibrate every frame.
# ------------------------------------------------------------

for index, (image_id, image_data) in enumerate(
    registered
):

    name = image_data["name"]

    path = os.path.join(
        FRAME_DIR,
        name
    )

    print(
        f"[{index + 1:02d}/{len(registered):02d}] "
        f"{name}"
    )

    image = Image.open(path).convert("RGB")

    result = depth_pipe(image)

    # IMPORTANT:
    # predicted_depth = actual model geometry.
    # result["depth"] is only the visualized depth image.
    predicted = result["predicted_depth"]

    if hasattr(predicted, "detach"):
        predicted = predicted.detach()

    predicted = predicted.squeeze().cpu().numpy()

    predicted = predicted.astype(
        np.float32
    )

    if predicted.shape != (
        IMAGE_HEIGHT,
        IMAGE_WIDTH
    ):

        predicted = cv2.resize(
            predicted,
            (
                IMAGE_WIDTH,
                IMAGE_HEIGHT
            ),
            interpolation=cv2.INTER_LINEAR
        )

    pixels, z_values = calibrate_depth(
        image_data,
        sparse_points
    )

    calibration = fit_linear_calibration(
        predicted,
        pixels,
        z_values
    )

    if calibration is None:

        print(
            "   calibration FAILED - skipped"
        )

        continue

    print(
        "   samples:",
        calibration["samples"],
        "scale:",
        f'{calibration["scale"]:.5f}',
        "offset:",
        f'{calibration["offset"]:.5f}',
        "RMSE:",
        f'{calibration["rmse"]:.4f}',
        "corr:",
        f'{calibration["corr"]:.4f}'
    )

    # Keep image as numpy RGB.
    rgb = np.asarray(
        image,
        dtype=np.uint8
    )

    frame_records.append(
        {
            "index": len(frame_records),
            "image_id": image_id,
            "name": name,
            "R": image_data["R"],
            "t": image_data["t"],
            "depth": predicted,
            "rgb": rgb,
            "scale": calibration["scale"],
            "offset": calibration["offset"],
            "rmse": calibration["rmse"],
            "corr": calibration["corr"]
        }
    )

    successful += 1


print()
print(
    "Successfully calibrated frames:",
    successful
)

if successful < 10:
    raise RuntimeError(
        "Too few calibrated frames."
    )


# ============================================================
# Multi-view dense reconstruction
# ============================================================

print()
print("=" * 70)
print("GENERATING DENSE CANDIDATES")
print("=" * 70)
print()

all_points = []
all_colors = []

total_candidates = 0
total_gate_pass = 0
total_consistent = 0


for frame_idx, frame in enumerate(
    frame_records
):

    print(
        f"[DENSE {frame_idx + 1:02d}/{len(frame_records):02d}] "
        f"{frame['name']}"
    )

    depth = frame["depth"]
    rgb = frame["rgb"]

    scale = frame["scale"]
    offset = frame["offset"]

    R = frame["R"]
    t = frame["t"]

    # --------------------------------------------------------
    # Pixel grid
    # --------------------------------------------------------

    ys = np.arange(
        0,
        IMAGE_HEIGHT,
        PIXEL_STRIDE
    )

    xs = np.arange(
        0,
        IMAGE_WIDTH,
        PIXEL_STRIDE
    )

    u, v = np.meshgrid(
        xs,
        ys
    )

    u = u.reshape(-1)
    v = v.reshape(-1)

    ai_depth = depth[
        v,
        u
    ].astype(np.float64)

    valid = (
        np.isfinite(ai_depth) &
        (ai_depth > 0)
    )

    u = u[valid]
    v = v[valid]
    ai_depth = ai_depth[valid]

    # --------------------------------------------------------
    # AI depth -> calibrated COLMAP Z
    # --------------------------------------------------------

    z = (
        scale * ai_depth +
        offset
    )

    valid = (
        np.isfinite(z) &
        (z > 0.15) &
        (z < 50.0)
    )

    u = u[valid]
    v = v[valid]
    z = z[valid]

    total_candidates += len(z)

    # --------------------------------------------------------
    # Backproject into camera coordinates
    # --------------------------------------------------------

    x = (
        (u - CX) /
        FX *
        z
    )

    y = (
        (v - CY) /
        FY *
        z
    )

    cam_points = np.column_stack(
        [x, y, z]
    )

    # --------------------------------------------------------
    # Camera -> world
    #
    # X_cam = R X_world + t
    #
    # X_world = R.T (X_cam - t)
    # --------------------------------------------------------

    world_points = (
        (R.T @
         (cam_points - t).T)
        .T
    )

    # --------------------------------------------------------
    # COLMAP spatial gate
    # --------------------------------------------------------

    gate_mask = np.zeros(
        len(world_points),
        dtype=bool
    )

    for i in range(
        len(world_points)
    ):

        p = world_points[i]

        count, _, distances = (
            sparse_tree.search_radius_vector_3d(
                p,
                colmap_radius
            )
        )

        if count > 0:
            gate_mask[i] = True

    world_points = world_points[
        gate_mask
    ]

    u_gate = u[
        gate_mask
    ]

    v_gate = v[
        gate_mask
    ]

    z_gate = z[
        gate_mask
    ]

    total_gate_pass += len(
        world_points
    )

    if len(world_points) == 0:
        continue

    # --------------------------------------------------------
    # Multi-view consistency
    # --------------------------------------------------------

    neighbors = []

    start = max(
        0,
        frame_idx - NEIGHBOR_RADIUS
    )

    end = min(
        len(frame_records),
        frame_idx + NEIGHBOR_RADIUS + 1
    )

    for j in range(start, end):

        if j == frame_idx:
            continue

        neighbors.append(j)

    # Keep candidate if enough neighbouring cameras agree.
    keep = np.zeros(
        len(world_points),
        dtype=bool
    )

    support = np.zeros(
        len(world_points),
        dtype=np.uint8
    )

    # --------------------------------------------------------
    # Check every neighbouring camera.
    # --------------------------------------------------------

    for neighbor_idx in neighbors:

        neighbor = frame_records[
            neighbor_idx
        ]

        Rn = neighbor["R"]
        tn = neighbor["t"]

        scale_n = neighbor["scale"]
        offset_n = neighbor["offset"]
        depth_n = neighbor["depth"]

        # World -> neighbor camera
        cam_n = (
            (Rn @ world_points.T)
            .T + tn
        )

        zn = cam_n[:, 2]

        valid_z = (
            np.isfinite(zn) &
            (zn > 0.15)
        )

        if not np.any(valid_z):
            continue

        un = (
            FX *
            cam_n[:, 0] /
            np.maximum(zn, 1e-8)
            + CX
        )

        vn = (
            FY *
            cam_n[:, 1] /
            np.maximum(zn, 1e-8)
            + CY
        )

        ui = np.rint(un).astype(int)
        vi = np.rint(vn).astype(int)

        inside = (
            valid_z &
            (ui >= 0) &
            (ui < IMAGE_WIDTH) &
            (vi >= 0) &
            (vi < IMAGE_HEIGHT)
        )

        if not np.any(inside):
            continue

        indices = np.where(
            inside
        )[0]

        observed_ai = depth_n[
            vi[indices],
            ui[indices]
        ].astype(np.float64)

        observed_z = (
            scale_n *
            observed_ai +
            offset_n
        )

        predicted_z = zn[
            indices
        ]

        tolerance = np.maximum(
            np.abs(observed_z) *
            RELATIVE_TOLERANCE,
            ABSOLUTE_TOLERANCE
        )

        agreement = (
            np.isfinite(observed_z) &
            (observed_z > 0.15) &
            (
                np.abs(
                    predicted_z -
                    observed_z
                ) <= tolerance
            )
        )

        support[
            indices[agreement]
        ] += 1

    keep = (
        support >=
        MIN_NEIGHBOR_SUPPORT
    )

    world_points = world_points[
        keep
    ]

    u_final = u_gate[
        keep
    ]

    v_final = v_gate[
        keep
    ]

    total_consistent += len(
        world_points
    )

    if len(world_points) == 0:
        continue

    # --------------------------------------------------------
    # Colors
    # --------------------------------------------------------

    colors = rgb[
        v_final,
        u_final
    ].astype(
        np.float64
    ) / 255.0

    all_points.append(
        world_points
    )

    all_colors.append(
        colors
    )

    print(
        "   candidates:",
        len(z),
        "gate:",
        len(z_gate),
        "consistent:",
        len(world_points)
    )


# ============================================================
# Merge
# ============================================================

print()
print("=" * 70)
print("MERGING")
print("=" * 70)
print()

if not all_points:
    raise RuntimeError(
        "No points survived reconstruction."
    )

points = np.vstack(
    all_points
)

colors = np.vstack(
    all_colors
)

print(
    "Raw reconstructed points:",
    len(points)
)


# ============================================================
# Open3D point cloud
# ============================================================

pcd = o3d.geometry.PointCloud()

pcd.points = o3d.utility.Vector3dVector(
    points
)

pcd.colors = o3d.utility.Vector3dVector(
    colors
)


# ============================================================
# Voxel downsample
# ============================================================

print()
print(
    "Voxel downsampling:",
    VOXEL_SIZE
)

pcd = pcd.voxel_down_sample(
    VOXEL_SIZE
)

print(
    "After voxel downsample:",
    len(pcd.points)
)


# ============================================================
# Statistical cleanup
# ============================================================

print()
print("Statistical outlier removal...")

pcd_clean, inliers = (
    pcd.remove_statistical_outlier(
        nb_neighbors=STAT_NB_NEIGHBORS,
        std_ratio=STAT_STD_RATIO
    )
)

print(
    "Final retained points:",
    len(pcd_clean.points)
)

print(
    "Removed:",
    len(pcd.points) -
    len(pcd_clean.points)
)


# ============================================================
# Save
# ============================================================

print()
print("Saving:")

o3d.io.write_point_cloud(
    OUTPUT,
    pcd_clean
)

print(
    OUTPUT
)

print()
print("=" * 70)
print("V8 COMPLETE")
print("=" * 70)
print()
print(
    "Calibration frames:",
    successful
)

print(
    "Raw candidates:",
    total_candidates
)

print(
    "COLMAP gate survivors:",
    total_gate_pass
)

print(
    "Multi-view survivors:",
    total_consistent
)

print(
    "Final points:",
    len(pcd_clean.points)
)

print()

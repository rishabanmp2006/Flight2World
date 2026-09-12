import os
import re
import numpy as np
import open3d as o3d

from PIL import Image
from transformers import pipeline


# ============================================================
# FLIGHT2WORLD
# 15-FRAME V6
#
# HYBRID COLMAP + AI DEPTH RECONSTRUCTION
#
# COLMAP = geometric authority
# Depth Anything = dense surface proposal
#
# Pipeline:
#
#   COLMAP poses
#       +
#   COLMAP sparse geometry
#       +
#   Depth Anything
#       ↓
#   calibrated dense candidates
#       ↓
#   multi-view consistency
#       ↓
#   COLMAP geometric gate
#       ↓
#   clean dense point cloud
# ============================================================


# ============================================================
# PATHS
# ============================================================

ROOT = os.path.expanduser("~/flight2world/test")

TXT = os.path.join(ROOT, "sparse_txt")
FRAMES = os.path.join(ROOT, "frames")

OUTPUT = os.path.join(
    ROOT,
    "fused_15_v6.ply"
)


# ============================================================
# CAMERA INTRINSICS
# ============================================================

fx = 1167.4277386087481
fy = 1167.4277386087481

cx = 640.0
cy = 360.0


# ============================================================
# SETTINGS
# ============================================================

NUM_FRAMES = 15

DEPTH_STEP = 6

# Multi-view relative depth tolerance.
RELATIVE_TOLERANCE = 0.20

# Minimum number of supporting views.
MIN_SUPPORT = 2

# ------------------------------------------------------------
# COLMAP GEOMETRIC GATE
# ------------------------------------------------------------
#
# Dense AI points must remain reasonably close to the
# reconstructed COLMAP sparse surface.
#
# The threshold is calculated automatically from the
# density of the sparse cloud.
#
# MULTIPLIER controls how permissive the gate is.
#
# Smaller = cleaner but sparser
# Larger  = denser but more artifacts
#
COLMAP_RADIUS_MULTIPLIER = 6.0

# Absolute safety minimum in COLMAP coordinate units.
MIN_COLMAP_RADIUS = 0.15

# Maximum safety radius.
MAX_COLMAP_RADIUS = 3.0

# Final voxel size.
VOXEL_SIZE = 0.04


# ============================================================
# COLMAP IMAGE READER
# ============================================================

def read_images(path):

    images = {}

    with open(path) as f:

        lines = [
            x.strip()
            for x in f
            if x.strip()
            and not x.startswith("#")
        ]

    for i in range(0, len(lines), 2):

        p = lines[i].split()

        q = np.array(
            list(map(float, p[1:5])),
            dtype=np.float64
        )

        t = np.array(
            list(map(float, p[5:8])),
            dtype=np.float64
        )

        name = p[9]

        pts = lines[i + 1].split()

        xy = []
        ids = []

        for j in range(0, len(pts), 3):

            xy.append([
                float(pts[j]),
                float(pts[j + 1])
            ])

            ids.append(
                int(pts[j + 2])
            )

        images[name] = {

            "q": q,
            "t": t,

            "xy": np.array(
                xy,
                dtype=np.float64
            ),

            "ids": np.array(
                ids,
                dtype=np.int64
            )

        }

    return images


# ============================================================
# COLMAP POINT READER
# ============================================================

def read_points(path):

    points = {}

    with open(path) as f:

        for line in f:

            if (
                line.startswith("#")
                or not line.strip()
            ):
                continue

            p = line.split()

            pid = int(p[0])

            points[pid] = np.array(
                list(map(float, p[1:4])),
                dtype=np.float64
            )

    return points


# ============================================================
# QUATERNION -> ROTATION
# ============================================================

def qvec_to_rotmat(q):

    w, x, y, z = q

    return np.array([

        [
            1 - 2*y*y - 2*z*z,
            2*x*y - 2*z*w,
            2*x*z + 2*y*w
        ],

        [
            2*x*y + 2*z*w,
            1 - 2*x*x - 2*z*z,
            2*y*z - 2*x*w
        ],

        [
            2*x*z - 2*y*w,
            2*y*z + 2*x*w,
            1 - 2*x*x - 2*y*y
        ]

    ], dtype=np.float64)


# ============================================================
# CAMERA -> WORLD
# ============================================================

def camera_to_world(points, R, t):

    return (
        R.T
        @ (
            points.T
            - t.reshape(3, 1)
        )
    ).T


# ============================================================
# WORLD -> CAMERA
# ============================================================

def world_to_camera(points, R, t):

    return (
        R
        @ points.T
        + t.reshape(3, 1)
    ).T


# ============================================================
# FRAME NUMBER
# ============================================================

def frame_number(name):

    match = re.search(
        r"(\d+)",
        name
    )

    if match:
        return int(match.group(1))

    return 0


# ============================================================
# HEADER
# ============================================================

print()
print("================================================")
print("FLIGHT2WORLD - 15 FRAME V6")
print("HYBRID COLMAP + AI DEPTH")
print("================================================")


# ============================================================
# LOAD COLMAP
# ============================================================

images = read_images(
    os.path.join(
        TXT,
        "images.txt"
    )
)

points3d = read_points(
    os.path.join(
        TXT,
        "points3D.txt"
    )
)


print(
    "Registered COLMAP images:",
    len(images)
)

print(
    "COLMAP 3D points:",
    len(points3d)
)


# ============================================================
# BUILD SPARSE COLMAP ARRAY
# ============================================================

sparse_points = np.asarray(
    list(points3d.values()),
    dtype=np.float64
)


if len(sparse_points) < 100:

    raise RuntimeError(
        "Too few COLMAP sparse points."
    )


print(
    "Sparse geometry loaded:",
    len(sparse_points),
    "points"
)


# ============================================================
# BUILD COLMAP KD TREE
# ============================================================

print()
print("Building COLMAP geometric index...")


sparse_pcd = o3d.geometry.PointCloud()

sparse_pcd.points = (
    o3d.utility.Vector3dVector(
        sparse_points
    )
)


sparse_tree = o3d.geometry.KDTreeFlann(
    sparse_pcd
)


# ============================================================
# ESTIMATE SPARSE CLOUD DENSITY
# ============================================================

print(
    "Estimating sparse-cloud spacing..."
)


sample_count = min(
    len(sparse_points),
    5000
)


rng = np.random.default_rng(42)

sample_indices = rng.choice(
    len(sparse_points),
    size=sample_count,
    replace=False
)


nearest_distances = []


for idx in sample_indices:

    point = sparse_points[idx]

    count, indices, distances = (
        sparse_tree.search_knn_vector_3d(
            point,
            2
        )
    )

    if count >= 2:

        d = np.sqrt(
            distances[1]
        )

        if np.isfinite(d) and d > 0:

            nearest_distances.append(d)


nearest_distances = np.asarray(
    nearest_distances
)


if len(nearest_distances) == 0:

    raise RuntimeError(
        "Could not estimate COLMAP point spacing."
    )


median_spacing = float(
    np.median(
        nearest_distances
    )
)


p90_spacing = float(
    np.percentile(
        nearest_distances,
        90
    )
)


# Use a robust spacing estimate.
base_spacing = max(
    median_spacing,
    p90_spacing * 0.35
)


colmap_radius = (
    base_spacing
    * COLMAP_RADIUS_MULTIPLIER
)


colmap_radius = max(
    MIN_COLMAP_RADIUS,
    colmap_radius
)


colmap_radius = min(
    MAX_COLMAP_RADIUS,
    colmap_radius
)


print(
    "Median sparse spacing:",
    round(median_spacing, 5)
)

print(
    "P90 sparse spacing:",
    round(p90_spacing, 5)
)

print(
    "COLMAP gate radius:",
    round(colmap_radius, 5)
)


# ============================================================
# SELECT REGISTERED FRAMES
# ============================================================

registered_names = sorted(
    images.keys(),
    key=frame_number
)


indices = np.linspace(
    0,
    len(registered_names) - 1,
    NUM_FRAMES
)


indices = np.round(
    indices
).astype(int)


selected = [
    registered_names[i]
    for i in indices
]


selected = list(
    dict.fromkeys(selected)
)


print()
print("================================================")
print("SELECTED FRAMES")
print("================================================")


for i, name in enumerate(selected):

    print(
        f"{i + 1:02d}. {name}"
    )


print(
    "Total:",
    len(selected)
)


# ============================================================
# PREPARE POSES
# ============================================================

poses = {}


for name in selected:

    poses[name] = {

        "R": qvec_to_rotmat(
            images[name]["q"]
        ),

        "t": images[name]["t"]

    }


# ============================================================
# LOAD DEPTH ANYTHING
# ============================================================

print()
print("================================================")
print("LOADING DEPTH ANYTHING V2")
print("================================================")


pipe = pipeline(
    "depth-estimation",
    model="depth-anything/Depth-Anything-V2-Small-hf",
    device="mps"
)


print(
    "Depth model loaded."
)


# ============================================================
# GENERATE DEPTH MAPS
# ============================================================

depth_maps = {}
rgb_maps = {}


print()
print("================================================")
print("GENERATING DEPTH MAPS")
print("================================================")


for i, name in enumerate(selected):

    print(
        f"[{i + 1}/{len(selected)}] {name}"
    )


    path = os.path.join(
        FRAMES,
        name
    )


    image = Image.open(
        path
    ).convert("RGB")


    rgb_maps[name] = np.asarray(
        image
    )


    result = pipe(
        image
    )


    depth = (
        result["predicted_depth"]
        .squeeze()
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )


    depth_maps[name] = depth


    print(
        "    Shape:",
        depth.shape
    )


    print(
        "    Range:",
        round(
            float(np.min(depth)),
            5
        ),
        "->",
        round(
            float(np.max(depth)),
            5
        )
    )


# ============================================================
# CALIBRATE AI DEPTH TO COLMAP Z
# ============================================================

calibration = {}


print()
print("================================================")
print("CALIBRATING AI DEPTH")
print("================================================")


for name in selected:

    print()
    print("----------------------------------------------")
    print(name)
    print("----------------------------------------------")


    info = images[name]

    R = poses[name]["R"]

    t = poses[name]["t"]

    depth = depth_maps[name]

    H, W = depth.shape


    z_values = []
    d_values = []


    for xy, pid in zip(
        info["xy"],
        info["ids"]
    ):

        if pid < 0:
            continue

        if pid not in points3d:
            continue


        x, y = xy


        ix = int(
            round(x)
        )

        iy = int(
            round(y)
        )


        if (
            ix < 0
            or ix >= W
            or iy < 0
            or iy >= H
        ):
            continue


        Xw = points3d[pid]


        Xc = (
            R @ Xw
            + t
        )


        z = float(
            Xc[2]
        )


        d = float(
            depth[iy, ix]
        )


        if (

            z > 0
            and d > 0
            and np.isfinite(z)
            and np.isfinite(d)

        ):

            z_values.append(z)

            d_values.append(d)


    z = np.asarray(
        z_values,
        dtype=np.float64
    )

    d = np.asarray(
        d_values,
        dtype=np.float64
    )


    print(
        "Raw samples:",
        len(z)
    )


    if len(z) < 50:

        raise RuntimeError(
            f"Too few calibration points: {name}"
        )


    z_low, z_high = np.percentile(
        z,
        [2, 98]
    )


    d_low, d_high = np.percentile(
        d,
        [2, 98]
    )


    good = (

        (z >= z_low)
        &
        (z <= z_high)
        &
        (d >= d_low)
        &
        (d <= d_high)

    )


    z = z[good]

    d = d[good]


    A = np.column_stack([
        d,
        np.ones_like(d)
    ])


    coef, *_ = np.linalg.lstsq(
        A,
        z,
        rcond=None
    )


    scale = float(
        coef[0]
    )

    offset = float(
        coef[1]
    )


    prediction = (
        scale * d
        + offset
    )


    rmse = float(
        np.sqrt(
            np.mean(
                (
                    z
                    - prediction
                ) ** 2
            )
        )
    )


    corr = float(
        np.corrcoef(
            z,
            d
        )[0, 1]
    )


    calibration[name] = {

        "scale": scale,
        "offset": offset,
        "rmse": rmse,
        "corr": corr

    }


    print(
        "Filtered:",
        len(z)
    )


    print(
        "Scale:",
        round(scale, 6)
    )


    print(
        "Offset:",
        round(offset, 6)
    )


    print(
        "RMSE:",
        round(rmse, 5)
    )


    print(
        "Correlation:",
        round(corr, 5)
    )


# ============================================================
# CREATE Z MAPS
# ============================================================

z_maps = {}


print()
print("================================================")
print("CREATING CALIBRATED Z MAPS")
print("================================================")


for name in selected:

    depth = depth_maps[name]


    scale = calibration[
        name
    ]["scale"]


    offset = calibration[
        name
    ]["offset"]


    z = (
        scale * depth
        + offset
    )


    z = z.astype(
        np.float32
    )


    invalid = (

        ~np.isfinite(z)
        |
        (z <= 0.5)
        |
        (z > 50.0)

    )


    z[invalid] = -1


    z_maps[name] = z


    print(
        name,
        "valid:",
        int(
            np.sum(z > 0)
        )
    )


# ============================================================
# MULTI-VIEW + COLMAP HYBRID FUSION
# ============================================================

print()
print("================================================")
print("HYBRID MULTI-VIEW FUSION")
print("COLMAP = GEOMETRIC AUTHORITY")
print("================================================")


all_points = []
all_colors = []


# ============================================================
# SOURCE FRAME LOOP
# ============================================================

for source_index, source_name in enumerate(
    selected
):

    print()
    print("----------------------------------------------")

    print(
        f"SOURCE {source_index + 1}/{len(selected)}:",
        source_name
    )

    print("----------------------------------------------")


    z_source = z_maps[
        source_name
    ]


    rgb_source = rgb_maps[
        source_name
    ]


    Hs, Ws = z_source.shape


    # --------------------------------------------------------
    # SAMPLE DEPTH
    # --------------------------------------------------------

    yy, xx = np.mgrid[
        0:Hs:DEPTH_STEP,
        0:Ws:DEPTH_STEP
    ]


    source_z = z_source[
        yy,
        xx
    ]


    valid = (
        source_z > 0
    )


    yy = yy[
        valid
    ]


    xx = xx[
        valid
    ]


    source_z = source_z[
        valid
    ]


    print(
        "Initial samples:",
        len(source_z)
    )


    if len(source_z) == 0:
        continue


    # --------------------------------------------------------
    # BACKPROJECT
    # --------------------------------------------------------

    X = (
        (xx - cx)
        / fx
        * source_z
    )


    Y = (
        (yy - cy)
        / fy
        * source_z
    )


    Z = source_z


    camera_points = np.column_stack([
        X,
        Y,
        Z
    ])


    # --------------------------------------------------------
    # CAMERA -> WORLD
    # --------------------------------------------------------

    world_points = camera_to_world(
        camera_points,
        poses[source_name]["R"],
        poses[source_name]["t"]
    )


    # --------------------------------------------------------
    # MULTI-VIEW SUPPORT
    # --------------------------------------------------------

    support = np.zeros(
        len(world_points),
        dtype=np.int16
    )


    # --------------------------------------------------------
    # OTHER VIEWS
    # --------------------------------------------------------

    for target_name in selected:

        if target_name == source_name:
            continue


        z_target = z_maps[
            target_name
        ]


        Ht, Wt = z_target.shape


        R_target = poses[
            target_name
        ]["R"]


        t_target = poses[
            target_name
        ]["t"]


        target_camera = world_to_camera(
            world_points,
            R_target,
            t_target
        )


        tx = target_camera[:, 0]

        ty = target_camera[:, 1]

        tz = target_camera[:, 2]


        valid_z = (

            np.isfinite(tz)
            &
            (tz > 0.5)

        )


        u = np.full(
            len(tz),
            -1.0
        )


        v = np.full(
            len(tz),
            -1.0
        )


        u[valid_z] = (
            fx
            * tx[valid_z]
            / tz[valid_z]
            + cx
        )


        v[valid_z] = (
            fy
            * ty[valid_z]
            / tz[valid_z]
            + cy
        )


        inside = (

            valid_z
            &
            np.isfinite(u)
            &
            np.isfinite(v)
            &
            (u >= 0)
            &
            (u < Wt - 1)
            &
            (v >= 0)
            &
            (v < Ht - 1)

        )


        if not np.any(inside):
            continue


        ids = np.where(
            inside
        )[0]


        ui = np.rint(
            u[ids]
        ).astype(
            np.int32
        )


        vi = np.rint(
            v[ids]
        ).astype(
            np.int32
        )


        safe = (

            (ui >= 0)
            &
            (ui < Wt)
            &
            (vi >= 0)
            &
            (vi < Ht)

        )


        if not np.any(safe):
            continue


        ids = ids[safe]

        ui = ui[safe]

        vi = vi[safe]


        observed_z = z_target[
            vi,
            ui
        ]


        predicted_z = tz[
            ids
        ]


        valid_depth = (

            (observed_z > 0)
            &
            np.isfinite(observed_z)
            &
            np.isfinite(predicted_z)

        )


        if not np.any(valid_depth):
            continue


        ids = ids[
            valid_depth
        ]


        observed = observed_z[
            valid_depth
        ]


        predicted = predicted_z[
            valid_depth
        ]


        error = (

            np.abs(
                observed
                - predicted
            )
            /
            np.maximum(
                np.abs(observed),
                1e-6
            )

        )


        consistent = (
            error
            <= RELATIVE_TOLERANCE
        )


        if np.any(consistent):

            good_ids = ids[
                consistent
            ]


            support[
                good_ids
            ] += 1


    # --------------------------------------------------------
    # REQUIRE MULTI-VIEW SUPPORT
    # --------------------------------------------------------

    supported = (
        support >= MIN_SUPPORT
    )


    print(
        "Multi-view supported:",
        int(
            np.sum(supported)
        ),
        "/",
        len(world_points)
    )


    if not np.any(supported):

        print(
            "No multi-view supported points."
        )

        continue


    candidate_points = (
        world_points[
            supported
        ]
    )


    candidate_x = (
        xx[
            supported
        ]
    )


    candidate_y = (
        yy[
            supported
        ]
    )


    # ========================================================
    # COLMAP GEOMETRIC GATE
    # ========================================================

    print(
        "Applying COLMAP geometric gate..."
    )


    keep_indices = []


    for i, point in enumerate(
        candidate_points
    ):

        count, indices, distances = (
            sparse_tree.search_knn_vector_3d(
                point,
                1
            )
        )


        if count < 1:
            continue


        nearest_distance = np.sqrt(
            distances[0]
        )


        if (
            np.isfinite(nearest_distance)
            and nearest_distance
            <= colmap_radius
        ):

            keep_indices.append(i)


    keep_indices = np.asarray(
        keep_indices,
        dtype=np.int64
    )


    print(
        "Passed COLMAP gate:",
        len(keep_indices),
        "/",
        len(candidate_points)
    )


    if len(keep_indices) == 0:

        print(
            "No points passed COLMAP gate."
        )

        continue


    accepted_points = (
        candidate_points[
            keep_indices
        ]
    )


    accepted_x = (
        candidate_x[
            keep_indices
        ]
    )


    accepted_y = (
        candidate_y[
            keep_indices
        ]
    )


    # --------------------------------------------------------
    # COLOR
    # --------------------------------------------------------

    colors = (

        rgb_source[
            accepted_y,
            accepted_x
        ]

        / 255.0

    )


    all_points.append(
        accepted_points
    )


    all_colors.append(
        colors
    )


    print(
        "Accepted:",
        len(accepted_points)
    )


# ============================================================
# MERGE
# ============================================================

print()
print("================================================")
print("MERGING HYBRID GEOMETRY")
print("================================================")


if len(all_points) == 0:

    raise RuntimeError(
        "No points survived hybrid reconstruction."
    )


points_np = np.vstack(
    all_points
)


colors_np = np.vstack(
    all_colors
)


print(
    "Raw hybrid points:",
    len(points_np)
)


# ============================================================
# OPEN3D CLOUD
# ============================================================

pcd = o3d.geometry.PointCloud()


pcd.points = (
    o3d.utility.Vector3dVector(
        points_np
    )
)


pcd.colors = (
    o3d.utility.Vector3dVector(
        colors_np
    )
)


# ============================================================
# VOXEL DOWNSAMPLE
# ============================================================

print()
print("Voxel downsampling...")


pcd = pcd.voxel_down_sample(
    voxel_size=VOXEL_SIZE
)


print(
    "After voxel:",
    len(pcd.points)
)


# ============================================================
# STATISTICAL CLEANUP
# ============================================================

print()
print("Statistical cleanup...")


if len(pcd.points) > 100:

    pcd, ind = (
        pcd.remove_statistical_outlier(
            nb_neighbors=30,
            std_ratio=1.5
        )
    )


print(
    "After cleanup:",
    len(pcd.points)
)


# ============================================================
# SAVE
# ============================================================

o3d.io.write_point_cloud(
    OUTPUT,
    pcd
)


print()
print("================================================")
print("🔥 FLIGHT2WORLD V6 COMPLETE")
print("================================================")


print(
    "Saved:",
    OUTPUT
)


print(
    "Final points:",
    len(pcd.points)
)


# ============================================================
# VISUALIZE
# ============================================================

print()
print(
    "Opening Open3D viewer..."
)


o3d.visualization.draw_geometries(
    [
        pcd
    ],
    window_name=(
        "FLIGHT2WORLD - "
        "15 Frame Hybrid V6"
    )
)
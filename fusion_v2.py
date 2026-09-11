import os
import numpy as np
import cv2
import open3d as o3d
from PIL import Image
from transformers import pipeline


# ============================================================
# PATHS
# ============================================================

ROOT = os.path.expanduser("~/flight2world/test")

TXT = os.path.join(ROOT, "sparse_txt")
FRAMES = os.path.join(ROOT, "frames")

OUTPUT = os.path.join(ROOT, "fused_5_v2.ply")


# ============================================================
# COLMAP CAMERA INTRINSICS
# ============================================================

fx = 1167.4277386087481
fy = 1167.4277386087481

cx = 640.0
cy = 360.0


# ============================================================
# READ COLMAP IMAGES.TXT
# ============================================================

def read_images(path):

    images = {}

    with open(path) as f:

        lines = [
            x.strip()
            for x in f
            if x.strip() and not x.startswith("#")
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
            "xy": np.array(xy),
            "ids": np.array(ids)
        }

    return images


# ============================================================
# READ COLMAP POINTS3D.TXT
# ============================================================

def read_points(path):

    points = {}

    with open(path) as f:

        for line in f:

            if line.startswith("#") or not line.strip():
                continue

            p = line.split()

            pid = int(p[0])

            points[pid] = np.array(
                list(map(float, p[1:4])),
                dtype=np.float64
            )

    return points


# ============================================================
# QUATERNION -> ROTATION MATRIX
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
# LOAD COLMAP
# ============================================================

images = read_images(
    os.path.join(TXT, "images.txt")
)

points = read_points(
    os.path.join(TXT, "points3D.txt")
)

print()
print("==============================================")
print("FLIGHT2WORLD FUSION V2")
print("==============================================")

print(
    "Registered images:",
    len(images)
)

print(
    "COLMAP points:",
    len(points)
)


# ============================================================
# LOAD DEPTH ANYTHING
# ============================================================

print()
print("Loading Depth Anything V2...")

pipe = pipeline(
    "depth-estimation",
    model="depth-anything/Depth-Anything-V2-Small-hf",
    device="mps"
)

print("Depth model ready.")


# ============================================================
# FIVE FRAMES
# ============================================================

selected = [

    "frame_0027.jpg",
    "frame_0040.jpg",
    "frame_0054.jpg",
    "frame_0067.jpg",
    "frame_0081.jpg"

]


# ============================================================
# PRE-COMPUTED CALIBRATION QUALITY
#
# Lower RMSE = higher confidence.
# ============================================================

calibration = {

    "frame_0027.jpg": {
        "scale": -0.2222956819,
        "offset": 4.6871058242,
        "rmse": 0.10209,
        "corr": -0.8435
    },

    "frame_0040.jpg": {
        "scale": -0.5794626,
        "offset": 6.7545033,
        "rmse": 0.14639,
        "corr": -0.9637
    },

    "frame_0054.jpg": {
        "scale": -0.2441202,
        "offset": 2.3597157,
        "rmse": 0.09211,
        "corr": -0.9806
    },

    "frame_0067.jpg": {
        "scale": -0.1612020,
        "offset": 1.6823068,
        "rmse": 0.04775,
        "corr": -0.9657
    },

    "frame_0081.jpg": {
        "scale": -0.6552737,
        "offset": 8.7565520,
        "rmse": 0.18132,
        "corr": -0.7423
    }

}


# ============================================================
# BUILD DENSE CLOUD
# ============================================================

all_points = []

all_colors = []


for frame_index, name in enumerate(selected):

    print()
    print("==============================================")
    print(
        f"Processing {frame_index + 1}/{len(selected)}:",
        name
    )
    print("==============================================")


    info = images[name]

    R = qvec_to_rotmat(
        info["q"]
    )

    t = info["t"]


    # --------------------------------------------------------
    # CALIBRATION
    # --------------------------------------------------------

    cal = calibration[name]

    scale = cal["scale"]
    offset = cal["offset"]

    rmse = cal["rmse"]
    corr = abs(cal["corr"])


    # Frame confidence.
    #
    # Better RMSE + stronger correlation = better confidence.
    #

    confidence = (
        np.exp(-rmse * 4.0)
        * min(corr, 1.0)
    )


    print(
        "Calibration RMSE:",
        round(rmse, 4)
    )

    print(
        "Calibration correlation:",
        round(cal["corr"], 4)
    )

    print(
        "Frame confidence:",
        round(confidence, 4)
    )


    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    path = os.path.join(
        FRAMES,
        name
    )

    pil_img = Image.open(
        path
    ).convert("RGB")


    rgb = np.asarray(
        pil_img
    )


    # --------------------------------------------------------
    # DEPTH
    # --------------------------------------------------------

    result = pipe(
        pil_img
    )

    depth = np.asarray(
        result["depth"],
        dtype=np.float32
    )


    H, W = depth.shape


    print(
        "Depth resolution:",
        W,
        "x",
        H
    )


    # ========================================================
    # DEPTH SMOOTHNESS / EDGE FILTER
    # ========================================================

    depth_float = depth.astype(
        np.float32
    )

    grad_x = cv2.Sobel(
        depth_float,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    grad_y = cv2.Sobel(
        depth_float,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    gradient = np.sqrt(
        grad_x * grad_x
        + grad_y * grad_y
    )


    # Robust threshold.
    edge_threshold = np.percentile(
        gradient,
        92
    )


    # ========================================================
    # DEPTH -> CAMERA Z
    # ========================================================

    z_cam = (
        scale * depth_float
        + offset
    )


    # Keep only sensible positive depths.
    valid = (

        np.isfinite(z_cam)
        &
        (z_cam > 0.5)
        &
        (z_cam < 50.0)

    )


    # Remove extreme depth discontinuities.
    valid &= (
        gradient < edge_threshold
    )


    # ========================================================
    # SAMPLE EVERY 6 PIXELS
    # ========================================================

    step = 6

    yy, xx = np.mgrid[
        0:H:step,
        0:W:step
    ]


    sample_valid = valid[
        yy,
        xx
    ]


    yy = yy[
        sample_valid
    ]

    xx = xx[
        sample_valid
    ]


    z = z_cam[
        yy,
        xx
    ]


    # ========================================================
    # CAMERA BACKPROJECTION
    # ========================================================

    X = (
        (xx - cx)
        / fx
        * z
    )

    Y = (
        (yy - cy)
        / fy
        * z
    )

    Z = z


    camera_points = np.column_stack([
        X,
        Y,
        Z
    ])


    # ========================================================
    # CAMERA -> WORLD
    #
    # Xworld = R.T @ (Xcamera - t)
    # ========================================================

    world_points = (
        R.T
        @ (
            camera_points.T
            - t.reshape(3, 1)
        )
    ).T


    # ========================================================
    # BASIC GEOMETRIC SANITY FILTER
    # ========================================================

    finite = np.all(
        np.isfinite(world_points),
        axis=1
    )

    world_points = world_points[
        finite
    ]

    yy = yy[
        finite
    ]

    xx = xx[
        finite
    ]


    # ========================================================
    # COLOR
    # ========================================================

    colors = (
        rgb[
            yy,
            xx
        ]
        / 255.0
    )


    # ========================================================
    # CONFIDENCE WEIGHTING
    #
    # Keep stronger frames more heavily represented.
    # ========================================================

    keep_probability = (
        0.55
        + 0.45 * confidence
    )

    rng = np.random.default_rng(
        frame_index + 1234
    )

    keep = (
        rng.random(
            len(world_points)
        )
        < keep_probability
    )


    world_points = world_points[
        keep
    ]

    colors = colors[
        keep
    ]


    print(
        "Generated points:",
        len(world_points)
    )


    all_points.append(
        world_points
    )

    all_colors.append(
        colors
    )


# ============================================================
# MERGE
# ============================================================

print()
print("==============================================")
print("MERGING")
print("==============================================")


points_np = np.vstack(
    all_points
)

colors_np = np.vstack(
    all_colors
)


print(
    "Raw merged points:",
    len(points_np)
)


# ============================================================
# OPEN3D
# ============================================================

pcd = o3d.geometry.PointCloud()

pcd.points = o3d.utility.Vector3dVector(
    points_np
)

pcd.colors = o3d.utility.Vector3dVector(
    colors_np
)


# ============================================================
# VOXEL DOWNSAMPLE
# ============================================================

print()
print("Voxel downsampling...")

pcd = pcd.voxel_down_sample(
    voxel_size=0.04
)


print(
    "After voxel downsampling:",
    len(pcd.points)
)


# ============================================================
# STATISTICAL OUTLIER REMOVAL
# ============================================================

print()
print("Removing statistical outliers...")

pcd, ind = (
    pcd.remove_statistical_outlier(
        nb_neighbors=30,
        std_ratio=1.5
    )
)


print(
    "After outlier removal:",
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
print("==============================================")
print("🔥 FUSION V2 COMPLETE")
print("==============================================")

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
print("Opening visualization...")

o3d.visualization.draw_geometries(
    [pcd],
    window_name="FLIGHT2WORLD - Fusion V2"
)
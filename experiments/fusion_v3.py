import os
import numpy as np
import open3d as o3d
from PIL import Image
from transformers import pipeline


# ============================================================
# PATHS
# ============================================================

ROOT = os.path.expanduser("~/flight2world/test")

TXT = os.path.join(ROOT, "sparse_txt")
FRAMES = os.path.join(ROOT, "frames")

OUTPUT = os.path.join(ROOT, "fused_5_v3.ply")


# ============================================================
# CAMERA INTRINSICS FROM COLMAP
# ============================================================

fx = 1167.4277386087481
fy = 1167.4277386087481

cx = 640.0
cy = 360.0


# ============================================================
# FIVE TEST FRAMES
# ============================================================

selected = [
    "frame_0027.jpg",
    "frame_0040.jpg",
    "frame_0054.jpg",
    "frame_0067.jpg",
    "frame_0081.jpg"
]


# ============================================================
# READ COLMAP images.txt
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
            "xy": np.array(xy),
            "ids": np.array(ids)
        }

    return images


# ============================================================
# READ COLMAP points3D.txt
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


print()
print("==============================================")
print("FLIGHT2WORLD FUSION V3")
print("==============================================")

print(
    "Registered COLMAP images:",
    len(images)
)

print(
    "COLMAP 3D points:",
    len(points3d)
)


# ============================================================
# PREPARE CAMERA POSES
# ============================================================

poses = {}

for name in selected:

    R = qvec_to_rotmat(
        images[name]["q"]
    )

    t = images[name]["t"]

    poses[name] = {
        "R": R,
        "t": t
    }


# ============================================================
# LOAD DEPTH ANYTHING V2
# ============================================================

print()
print("Loading Depth Anything V2...")

pipe = pipeline(
    "depth-estimation",
    model="depth-anything/Depth-Anything-V2-Small-hf",
    device="mps"
)

print("Depth model loaded.")


# ============================================================
# GENERATE RAW DEPTH MAPS
# ============================================================

depth_maps = {}
rgb_maps = {}


print()
print("==============================================")
print("GENERATING RAW DEPTH MAPS")
print("==============================================")


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


    # IMPORTANT:
    #
    # Use the actual model prediction.
    # result["depth"] is a processed visualization.
    #

    predicted = result["predicted_depth"]


    depth = (
        predicted
        .squeeze()
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )


    depth_maps[name] = depth


    print(
        "    Raw depth shape:",
        depth.shape
    )

    print(
        "    Raw depth range:",
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
# CALIBRATE EACH FRAME AGAINST COLMAP
# ============================================================

calibration = {}


print()
print("==============================================")
print("CALIBRATING RAW DEPTH AGAINST COLMAP")
print("==============================================")


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


    zs = []
    ds = []


    for xy, pid in zip(
        info["xy"],
        info["ids"]
    ):

        if pid < 0:
            continue

        if pid not in points3d:
            continue


        x, y = xy

        ix = int(round(x))
        iy = int(round(y))


        if (
            ix < 0
            or ix >= W
            or iy < 0
            or iy >= H
        ):
            continue


        Xw = points3d[pid]


        # COLMAP:
        #
        # Xcamera = R @ Xworld + t
        #

        Xc = R @ Xw + t


        z = Xc[2]

        d = float(
            depth[iy, ix]
        )


        if (
            z > 0
            and np.isfinite(z)
            and np.isfinite(d)
            and d > 0
        ):

            zs.append(z)
            ds.append(d)


    z = np.asarray(
        zs,
        dtype=np.float64
    )

    d = np.asarray(
        ds,
        dtype=np.float64
    )


    print(
        "Raw matched samples:",
        len(z)
    )


    if len(z) < 50:

        raise RuntimeError(
            f"Too few calibration samples for {name}"
        )


    # --------------------------------------------------------
    # REMOVE EXTREME OUTLIERS
    # --------------------------------------------------------

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
        & (z <= z_high)
        & (d >= d_low)
        & (d <= d_high)

    )


    z = z[good]
    d = d[good]


    print(
        "Filtered samples:",
        len(z)
    )


    # --------------------------------------------------------
    # LINEAR CALIBRATION
    #
    # Z = scale * depth + offset
    # --------------------------------------------------------

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


    predicted_z = (
        scale * d
        + offset
    )


    residuals = (
        z
        - predicted_z
    )


    rmse = float(
        np.sqrt(
            np.mean(
                residuals ** 2
            )
        )
    )


    correlation = float(
        np.corrcoef(
            z,
            d
        )[0, 1]
    )


    calibration[name] = {

        "scale": scale,
        "offset": offset,
        "rmse": rmse,
        "correlation": correlation

    }


    print(
        "Scale:",
        scale
    )

    print(
        "Offset:",
        offset
    )

    print(
        "RMSE:",
        round(rmse, 5)
    )

    print(
        "Correlation:",
        round(correlation, 5)
    )


# ============================================================
# CREATE CALIBRATED Z MAPS
# ============================================================

z_maps = {}


print()
print("==============================================")
print("CREATING CALIBRATED Z MAPS")
print("==============================================")


for name in selected:

    depth = depth_maps[name]

    scale = calibration[name]["scale"]
    offset = calibration[name]["offset"]


    z = (
        scale * depth
        + offset
    )


    z = z.astype(
        np.float32
    )


    # Basic physical validity only.
    #

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
        "valid pixels:",
        int(np.sum(z > 0))
    )


# ============================================================
# MULTI-VIEW CONSISTENCY
# ============================================================

print()
print("==============================================")
print("MULTI-VIEW CONSISTENCY FUSION")
print("==============================================")


# Allow moderate disagreement because Depth Anything
# is relative monocular depth, not perfect metric depth.
#

RELATIVE_TOLERANCE = 0.30


# Sample every 6 pixels.
#

STEP = 6


all_points = []
all_colors = []


# ============================================================
# PROCESS EACH SOURCE FRAME
# ============================================================

for source_index, source_name in enumerate(selected):

    print()
    print("----------------------------------------------")
    print(
        f"SOURCE {source_index + 1}/{len(selected)}:",
        source_name
    )
    print("----------------------------------------------")


    z_source = z_maps[source_name]

    rgb_source = rgb_maps[source_name]

    Hs, Ws = z_source.shape


    # --------------------------------------------------------
    # SOURCE GRID
    # --------------------------------------------------------

    yy, xx = np.mgrid[
        0:Hs:STEP,
        0:Ws:STEP
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
    # BACKPROJECT SOURCE IMAGE
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
    # SOURCE CAMERA -> WORLD
    # --------------------------------------------------------

    world_points = camera_to_world(
        camera_points,
        poses[source_name]["R"],
        poses[source_name]["t"]
    )


    # --------------------------------------------------------
    # SUPPORT COUNTER
    # --------------------------------------------------------

    support = np.zeros(
        len(world_points),
        dtype=np.int16
    )


    # ========================================================
    # TEST AGAINST EVERY OTHER FRAME
    # ========================================================

    for target_name in selected:

        if target_name == source_name:
            continue


        target_z_map = z_maps[target_name]

        Ht, Wt = target_z_map.shape


        R_target = poses[target_name]["R"]
        t_target = poses[target_name]["t"]


        # ----------------------------------------------------
        # WORLD -> TARGET CAMERA
        # ----------------------------------------------------

        target_camera = world_to_camera(
            world_points,
            R_target,
            t_target
        )


        tx = target_camera[:, 0]
        ty = target_camera[:, 1]
        tz = target_camera[:, 2]


        # ----------------------------------------------------
        # VALID POSITIVE DEPTH
        # ----------------------------------------------------

        valid_z = (

            np.isfinite(tz)
            &
            (tz > 0.5)

        )


        # ----------------------------------------------------
        # PROJECT TO TARGET IMAGE
        # ----------------------------------------------------

        u = np.full(
            len(tz),
            -1.0,
            dtype=np.float64
        )

        v = np.full(
            len(tz),
            -1.0,
            dtype=np.float64
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


        # ----------------------------------------------------
        # FLOATING-POINT IMAGE BOUNDS
        # ----------------------------------------------------

        inside = (

            valid_z
            &
            np.isfinite(u)
            &
            np.isfinite(v)
            &
            (u >= 0)
            &
            (u < Wt)
            &
            (v >= 0)
            &
            (v < Ht)

        )


        if not np.any(inside):

            print(
                "    ",
                target_name,
                ": no projected points"
            )

            continue


        indices = np.where(
            inside
        )[0]


        # ----------------------------------------------------
        # ROUND TO INTEGER PIXELS
        # ----------------------------------------------------

        ui = np.rint(
            u[indices]
        ).astype(
            np.int32
        )

        vi = np.rint(
            v[indices]
        ).astype(
            np.int32
        )


        # ====================================================
        # IMPORTANT FIX:
        #
        # Rounding can turn 719.8 -> 720
        # or 1279.7 -> 1280.
        #
        # Re-check bounds AFTER rounding.
        # ====================================================

        inside_rounded = (

            (ui >= 0)
            &
            (ui < Wt)
            &
            (vi >= 0)
            &
            (vi < Ht)

        )


        if not np.any(inside_rounded):

            print(
                "    ",
                target_name,
                ": no points after integer bounds check"
            )

            continue


        indices = indices[
            inside_rounded
        ]

        ui = ui[
            inside_rounded
        ]

        vi = vi[
            inside_rounded
        ]


        # ----------------------------------------------------
        # TARGET DEPTH
        # ----------------------------------------------------

        observed_z = target_z_map[
            vi,
            ui
        ]


        predicted_z = tz[
            indices
        ]


        # ----------------------------------------------------
        # VALID TARGET DEPTH
        # ----------------------------------------------------

        depth_valid = (

            (observed_z > 0)
            &
            np.isfinite(observed_z)

        )


        if not np.any(depth_valid):

            print(
                "    ",
                target_name,
                ": no valid target depth"
            )

            continue


        good_indices = indices[
            depth_valid
        ]


        observed = observed_z[
            depth_valid
        ]

        predicted = predicted_z[
            depth_valid
        ]


        # ----------------------------------------------------
        # RELATIVE DEPTH ERROR
        # ----------------------------------------------------

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
            < RELATIVE_TOLERANCE
        )


        support[
            good_indices[consistent]
        ] += 1


        print(
            "    ",
            target_name,
            ":",
            int(np.sum(consistent)),
            "consistent"
        )


    # ========================================================
    # ACCEPT POINTS WITH AT LEAST ONE OTHER VIEW
    # ========================================================

    keep = (
        support >= 1
    )


    supported_count = int(
        np.sum(keep)
    )


    print()
    print(
        "Multi-view supported:",
        supported_count,
        "/",
        len(world_points)
    )


    if supported_count == 0:

        print(
            "No supported geometry in this frame."
        )

        continue


    world_points = world_points[
        keep
    ]


    yy_keep = yy[
        keep
    ]

    xx_keep = xx[
        keep
    ]


    # --------------------------------------------------------
    # COLOR
    # --------------------------------------------------------

    colors = (
        rgb_source[
            yy_keep,
            xx_keep
        ]
        / 255.0
    )


    all_points.append(
        world_points
    )

    all_colors.append(
        colors
    )


    print(
        "Accepted:",
        len(world_points)
    )


# ============================================================
# MERGE
# ============================================================

print()
print("==============================================")
print("MERGING CONSISTENT POINTS")
print("==============================================")


if len(all_points) == 0:

    raise RuntimeError(
        "No points survived multi-view consistency."
    )


points_np = np.vstack(
    all_points
)

colors_np = np.vstack(
    all_colors
)


print(
    "Raw consistent points:",
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
# VOXEL FUSION
# ============================================================

print()
print("Voxel downsampling...")

pcd = pcd.voxel_down_sample(
    voxel_size=0.04
)


print(
    "After voxel fusion:",
    len(pcd.points)
)


# ============================================================
# LIGHT OUTLIER CLEANUP
# ============================================================

print()
print("Light statistical cleanup...")

pcd, ind = (
    pcd.remove_statistical_outlier(
        nb_neighbors=30,
        std_ratio=2.0
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
print("==============================================")
print("🔥 FUSION V3 COMPLETE")
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
print("Opening Open3D viewer...")

o3d.visualization.draw_geometries(
    [pcd],
    window_name="FLIGHT2WORLD - Multi-View Fusion V3"
)
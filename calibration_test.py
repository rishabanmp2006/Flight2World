import os
import numpy as np
from PIL import Image
from transformers import pipeline


# ============================================================
# PATHS
# ============================================================

ROOT = os.path.expanduser("~/flight2world/test")

TXT = os.path.join(ROOT, "sparse_txt")
FRAMES = os.path.join(ROOT, "frames")


# ============================================================
# READ COLMAP images.txt
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

        image_id = int(p[0])

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

            xyz = np.array(
                list(map(float, p[1:4])),
                dtype=np.float64
            )

            points[pid] = xyz

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
# LOAD COLMAP DATA
# ============================================================

images = read_images(
    os.path.join(TXT, "images.txt")
)

points = read_points(
    os.path.join(TXT, "points3D.txt")
)

print()
print("======================================")
print("COLMAP DATA")
print("======================================")

print("Registered images:", len(images))
print("3D points:", len(points))


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
# TEST FRAMES
# ============================================================

selected = [

    "frame_0027.jpg",
    "frame_0040.jpg",
    "frame_0054.jpg",
    "frame_0067.jpg",
    "frame_0081.jpg"

]


# ============================================================
# CALIBRATION TEST
# ============================================================

print()
print("======================================")
print("DEPTH CALIBRATION COMPARISON")
print("======================================")


for name in selected:

    print()
    print("--------------------------------------")
    print(name)
    print("--------------------------------------")


    # --------------------------------------------------------
    # COLMAP CAMERA
    # --------------------------------------------------------

    info = images[name]

    R = qvec_to_rotmat(
        info["q"]
    )

    t = info["t"]


    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    image_path = os.path.join(
        FRAMES,
        name
    )

    img = Image.open(
        image_path
    ).convert("RGB")


    # --------------------------------------------------------
    # RUN DEPTH MODEL
    # --------------------------------------------------------

    result = pipe(img)

    depth = np.asarray(
        result["depth"],
        dtype=np.float32
    )


    H, W = depth.shape


    # --------------------------------------------------------
    # MATCH COLMAP POINTS TO DEPTH
    # --------------------------------------------------------

    zs = []
    ds = []


    for xy, pid in zip(
        info["xy"],
        info["ids"]
    ):

        if pid < 0:
            continue

        if pid not in points:
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


        # World-space COLMAP point
        Xw = points[pid]


        # COLMAP camera coordinates
        #
        # Xcam = R @ Xworld + t
        #

        Xc = R @ Xw + t


        z = Xc[2]

        d = depth[iy, ix]


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


    # --------------------------------------------------------
    # OUTLIER FILTER
    # --------------------------------------------------------

    if len(z) < 20:

        print(
            "Not enough samples."
        )

        continue


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


    # ========================================================
    # MODEL 1: LINEAR
    #
    # Z = a * Depth + b
    # ========================================================

    A1 = np.column_stack([

        d,
        np.ones_like(d)

    ])


    coef1, *_ = np.linalg.lstsq(
        A1,
        z,
        rcond=None
    )


    pred1 = A1 @ coef1


    rmse1 = np.sqrt(
        np.mean(
            (z - pred1) ** 2
        )
    )


    # ========================================================
    # MODEL 2: INVERSE DEPTH
    #
    # Z = a / Depth + b
    # ========================================================

    invd = 1.0 / d


    A2 = np.column_stack([

        invd,
        np.ones_like(invd)

    ])


    coef2, *_ = np.linalg.lstsq(
        A2,
        z,
        rcond=None
    )


    pred2 = A2 @ coef2


    rmse2 = np.sqrt(
        np.mean(
            (z - pred2) ** 2
        )
    )


    # ========================================================
    # CORRELATION
    # ========================================================

    corr = np.corrcoef(
        z,
        d
    )[0, 1]


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()
    print("Samples       :", len(z))
    print("Correlation   :", round(corr, 4))
    print("Linear RMSE   :", round(rmse1, 5))
    print("Inverse RMSE  :", round(rmse2, 5))


    # ========================================================
    # WINNER
    # ========================================================

    if rmse2 < rmse1:

        improvement = (
            (rmse1 - rmse2)
            / rmse1
            * 100
        )

        print()
        print(">>> INVERSE MODEL WINS")
        print(
            ">>> Improvement:",
            round(improvement, 2),
            "%"
        )

    else:

        improvement = (
            (rmse2 - rmse1)
            / rmse2
            * 100
        )

        print()
        print(">>> LINEAR MODEL WINS")
        print(
            ">>> Difference:",
            round(improvement, 2),
            "%"
        )


# ============================================================
# COMPLETE
# ============================================================

print()
print("======================================")
print("CALIBRATION TEST COMPLETE")
print("======================================")
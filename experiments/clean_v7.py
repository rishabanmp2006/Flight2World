import open3d as o3d
import numpy as np
from pathlib import Path


# ============================================================
# FLIGHT2WORLD - V7 GEOMETRY CLEANER
# ============================================================

INPUT = Path("test/fused_15_v6.ply")
OUTPUT = Path("test/fused_15_v7_clean.ply")


print()
print("==============================================")
print("FLIGHT2WORLD - V7 GEOMETRY CLEANER")
print("==============================================")
print()


# ============================================================
# LOAD
# ============================================================

print("Loading V6 point cloud...")

pcd = o3d.io.read_point_cloud(str(INPUT))

if pcd.is_empty():
    raise RuntimeError("V6 point cloud is empty.")


print("Loaded points:", len(pcd.points))


# ============================================================
# BASIC CLEANUP
# ============================================================

points = np.asarray(pcd.points)

valid = np.isfinite(points).all(axis=1)

if not np.all(valid):
    print("Removing invalid points:", np.sum(~valid))
    pcd = pcd.select_by_index(np.where(valid)[0])


print("Valid points:", len(pcd.points))


# ============================================================
# VOXEL DOWNSAMPLE
#
# This makes clustering much more stable and prevents tiny
# point-to-point noise from dominating the algorithm.
# ============================================================

print()
print("==============================================")
print("VOXEL PREPROCESSING")
print("==============================================")

bbox = pcd.get_axis_aligned_bounding_box()
extent = np.asarray(bbox.get_extent())

print("Cloud extent:")
print("X:", extent[0])
print("Y:", extent[1])
print("Z:", extent[2])


# Adaptive voxel size.
# The cloud is in COLMAP-relative coordinates, so we don't
# assume metres here.

largest_dimension = float(np.max(extent))

voxel_size = largest_dimension * 0.003

# Prevent extreme values.
voxel_size = max(voxel_size, 0.005)
voxel_size = min(voxel_size, 0.10)

print("Voxel size:", voxel_size)

down = pcd.voxel_down_sample(voxel_size)

print("After voxel downsample:", len(down.points))


# ============================================================
# DBSCAN CLUSTERING
#
# Main reconstruction should form the dominant spatial cluster.
# Small detached pieces are treated as floating artifacts.
# ============================================================

print()
print("==============================================")
print("SPATIAL CLUSTERING")
print("==============================================")

cluster_eps = voxel_size * 3.5

print("DBSCAN epsilon:", cluster_eps)

labels = np.array(
    down.cluster_dbscan(
        eps=cluster_eps,
        min_points=8,
        print_progress=True
    )
)


# ============================================================
# CLUSTER STATISTICS
# ============================================================

valid_labels = labels[labels >= 0]

if len(valid_labels) == 0:
    raise RuntimeError(
        "DBSCAN found no valid clusters. "
        "We should not continue with cleanup."
    )


unique_labels, counts = np.unique(
    valid_labels,
    return_counts=True
)

order = np.argsort(counts)[::-1]

print()
print("Detected clusters:", len(unique_labels))
print()

print("Largest clusters:")

for rank, idx in enumerate(order[:10], start=1):
    label = unique_labels[idx]
    count = counts[idx]

    print(
        f"  #{rank}: "
        f"cluster {label} "
        f"-> {count} points"
    )


# ============================================================
# KEEP MAIN CLUSTER(S)
#
# We keep the largest cluster as the main reconstruction.
#
# We also keep reasonably large clusters close to the main
# cluster, because the real scene can contain legitimate
# disconnected surfaces.
# ============================================================

largest_label = unique_labels[order[0]]

main_indices = np.where(labels == largest_label)[0]

main_points = np.asarray(
    down.points
)[main_indices]


main_center = np.mean(main_points, axis=0)

print()
print("Main cluster:", largest_label)
print("Main cluster points:", len(main_points))


# ------------------------------------------------------------
# Determine which secondary clusters are legitimate.
#
# A secondary cluster must:
#   1. Have enough points
#   2. Be spatially close to the main reconstruction
# ------------------------------------------------------------

keep_labels = [largest_label]

main_bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
    o3d.utility.Vector3dVector(main_points)
)

main_min = np.asarray(main_bbox.get_min_bound())
main_max = np.asarray(main_bbox.get_max_bound())


for label, count in zip(unique_labels, counts):

    if label == largest_label:
        continue

    # Ignore tiny fragments.
    if count < max(30, int(len(main_points) * 0.001)):
        continue

    cluster_indices = np.where(labels == label)[0]

    cluster_points = np.asarray(
        down.points
    )[cluster_indices]

    center = np.mean(cluster_points, axis=0)

    distance = np.linalg.norm(center - main_center)

    # Distance relative to overall scene size.
    relative_distance = distance / largest_dimension

    print(
        f"Secondary cluster {label}: "
        f"{count} points, "
        f"distance={relative_distance:.3f} scene units"
    )

    # Keep substantial clusters that aren't extremely far away.
    if relative_distance < 0.35:
        keep_labels.append(label)


print()
print("Clusters retained:", keep_labels)


# ============================================================
# BUILD CLEAN CLOUD
# ============================================================

keep_mask = np.isin(labels, keep_labels)

clean_points = np.asarray(down.points)[keep_mask]


clean_colors = None

if down.has_colors():
    clean_colors = np.asarray(down.colors)[keep_mask]


clean = o3d.geometry.PointCloud()

clean.points = o3d.utility.Vector3dVector(clean_points)

if clean_colors is not None:
    clean.colors = o3d.utility.Vector3dVector(clean_colors)


print()
print("Points after spatial cleanup:", len(clean.points))


# ============================================================
# STATISTICAL OUTLIER REMOVAL
# ============================================================

print()
print("==============================================")
print("STATISTICAL OUTLIER REMOVAL")
print("==============================================")

if len(clean.points) > 100:

    clean_filtered, removed = clean.remove_statistical_outlier(
        nb_neighbors=20,
        std_ratio=2.5
    )

    print("Removed statistical outliers:", len(removed))
    print(
        "Final points after outlier removal:",
        len(clean_filtered.points)
    )

    clean = clean_filtered


# ============================================================
# SAVE
# ============================================================

print()
print("==============================================")
print("SAVING V7")
print("==============================================")

success = o3d.io.write_point_cloud(
    str(OUTPUT),
    clean
)

if not success:
    raise RuntimeError("Failed to save V7 point cloud.")


print()
print("🔥 V7 CLEAN RECONSTRUCTION CREATED")
print()
print("Input :", INPUT)
print("Output:", OUTPUT)
print("Points:", len(clean.points))
print()
print("==============================================")
print("V7 COMPLETE")
print("==============================================")
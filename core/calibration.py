"""flight2world.core.calibration

V10 per-frame depth calibration logic, extracted faithfully from
`fusion_v10.py` for reuse. Many of these functions are byte-for-byte
transcriptions of the V10 baseline; the algorithm is NOT redesigned.

Supported V10 behaviors (unchanged):
    * sparse observation collection (depth vs COLMAP camera-space Z)
    * 2-98 percentile tail trimming
    * iteratively re-weighted least squares with 3*1.4826*MAD trimming
    * linear model          Z = a * depth + b
    * inverse-depth model   Z = a / depth + b   (predicted depth is
      treated as inverse depth, guard-epsilon 1e-6)
    * RMSE and correlation computation
    * linear-vs-inverse model selection (lower RMSE wins)
    * frame quality gating  (min samples |corr| >= CALIB_MIN_CORR)

The metric scale is unknown (COLMAP arbitrary units); no conversion is
performed.
"""

import numpy as np

from core.config import (
    CALIB_MIN_CORR,
    CALIB_MIN_SAMPLES,
    INVERSE_DEPTH_EPS,
    IMAGE_H,
    IMAGE_W,
    ROBUST_FIT_ITERATIONS,
)


def robust_fit(x, z, iterations=ROBUST_FIT_ITERATIONS):
    """Robust linear fit z = a*x + b on paired arrays.

    Byte-for-byte faithful to fusion_v10.py robust_fit (lines 234-283).
    Returns None when fewer than ROBUST_FIT_MIN_SAMPLES valid pairs
    survive (a degenerate/invalid input fails cleanly, never NaN).

    Returns a dict {a, b, rmse, corr, samples} or None.
    """
    min_samples = 50  # fusion_v10.py line 236 (hardcoded lower bound)
    if len(x) < min_samples:
        return None

    x = np.asarray(x, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    valid = np.isfinite(x) & np.isfinite(z)
    x, z = x[valid], z[valid]
    if len(x) < min_samples:
        return None

    # Trim extreme tails once (fixed 2-98 percentile).
    lo_x, hi_x = np.percentile(x, [2, 98])
    lo_z, hi_z = np.percentile(z, [2, 98])
    mask = (
        (x >= lo_x) & (x <= hi_x) &
        (z >= lo_z) & (z <= hi_z)
    )
    x, z = x[mask], z[mask]
    if len(x) < min_samples:
        return None

    A = np.vstack([x, np.ones_like(x)]).T

    coef = np.linalg.lstsq(A, z, rcond=None)[0]

    for _ in range(iterations):
        pred = A @ coef
        resid = z - pred
        mad = np.median(np.abs(resid - np.median(resid)))
        if mad < 1e-12:
            break
        good = np.abs(resid) <= 3.0 * 1.4826 * mad
        if good.sum() < min_samples:
            break
        coef = np.linalg.lstsq(A[good], z[good], rcond=None)[0]

    pred = A @ coef
    rmse = float(np.sqrt(np.mean((z - pred) ** 2)))
    corr = float(np.corrcoef(x, z)[0, 1])

    return {
        "a": float(coef[0]),
        "b": float(coef[1]),
        "rmse": rmse,
        "corr": corr,
        "samples": len(x),
    }


def calibrate_frame(image_data, sparse_points, depth_map,
                    image_w=IMAGE_W, image_h=IMAGE_H):
    """Collect (depth_at_obs, COLMAP camera Z at obs) paired arrays.

    Faithful to fusion_v10.py calibrate_frame (lines 286-316). The image
    dimensions are threaded as parameters (defaulting to the V10 camera
    values) instead of module globals; with defaults the behavior is
    identical to V10.

    Returns (d_vals, z_vals) numpy float64 arrays, or empty arrays when a
    frame has no usable observations.
    """
    R = image_data["R"]
    t = image_data["t"]

    d_vals = []
    z_vals = []

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
        if xi < 0 or xi >= image_w or yi < 0 or yi >= image_h:
            continue

        d = depth_map[yi, xi]
        if not (np.isfinite(d) and d > 0):
            continue

        d_vals.append(d)
        z_vals.append(z)

    return np.asarray(d_vals), np.asarray(z_vals)


def fit_depth_calibration(d_vals, z_vals,
                          min_samples=CALIB_MIN_SAMPLES,
                          min_corr=CALIB_MIN_CORR,
                          inverse_eps=INVERSE_DEPTH_EPS,
                          iterations=ROBUST_FIT_ITERATIONS):
    """Select the depth-to-Z calibration model with gating, as V10 does.

    Faithful transcription of the V10 main-loop model-selection block
    (fusion_v10.py lines 542-576):

      1. reject if fewer than min_samples paired observations,
      2. fit BOTH linear (Z = a*d + b) and inverse (Z = a/d + b),
      3. pick the lower-RMSE candidate,
      4. reject if its |corr| < min_corr.

    Predicted depth is treated as inverse depth (higher = closer), hence
    the inverse model; the sign of corr(d, Z) encodes only this convention
    and is gated by magnitude.

    Returns a dict {a, b, model, rmse, corr, n_pairs, n_samples} on
    success, or None (a degenerate/invalid frame fails cleanly, never
    producing NaN or a nonsense fit).
    """
    n_pairs = int(len(d_vals))

    def _none():
        return None

    if n_pairs < min_samples:
        return _none()

    d_vals = np.asarray(d_vals, dtype=np.float64)
    z_vals = np.asarray(z_vals, dtype=np.float64)

    linear = robust_fit(d_vals, z_vals, iterations=iterations)
    inv_d = 1.0 / np.maximum(d_vals, inverse_eps)
    inverse = robust_fit(inv_d, z_vals, iterations=iterations)

    candidate = None
    model = None
    if linear and inverse:
        if inverse["rmse"] <= linear["rmse"]:
            candidate, model = inverse, "inverse"
        else:
            candidate, model = linear, "linear"
    elif linear:
        candidate, model = linear, "linear"
    elif inverse:
        candidate, model = inverse, "inverse"

    if candidate is None:
        return _none()

    # Degenerate-frame safety guard: if the selected fit is non-finite
    # (e.g. constant depth makes corrcoef NaN), reject the frame instead
    # of silently accepting a NaN-corr calibration. This does not change
    # V10 behavior on any real frame (all V10 |corr| values are finite,
    # >= 0.685); it only closes a whole-frame NaN acceptance hole that
    # V10's inline `abs(corr) < CALIB_MIN_CORR` gate would not catch.
    if not np.isfinite(candidate["corr"]):
        return _none()

    if abs(candidate["corr"]) < min_corr:
        return _none()

    return {
        "a": candidate["a"],
        "b": candidate["b"],
        "model": model,
        "rmse": candidate["rmse"],
        "corr": candidate["corr"],
        "n_pairs": n_pairs,
        "n_samples": candidate["samples"],
    }


def calibrated_depth(frame, d):
    """Apply the selected calibration model to a raw depth value.

    Byte-for-byte faithful to fusion_v10.py calibrated_depth
    (lines 656-659). ``frame`` is a frame record with a "model" ("linear"
    or "inverse"), "a" and "b".
    """
    if frame["model"] == "inverse":
        return frame["a"] / np.maximum(d, INVERSE_DEPTH_EPS) + frame["b"]
    return frame["a"] * d + frame["b"]


# ---------------------------------------------------------------------------
# RMSE outlier filtering (V10 lines 623-632) — Phase 4 extraction
# ---------------------------------------------------------------------------

def filter_rmse_outliers(
    frame_records,
    factor: float | None = None,
):
    """Drop calibration outliers whose RMSE exceeds factor * median RMSE.

    Faithful to the V10 post-calibration gate (fusion_v10.py lines 623-632):

        rmse_all = np.array([f["rmse"] for f in frame_records])
        rmse_median = float(np.median(rmse_all))
        rmse_keep = rmse_all <= CALIB_RMSE_OUTLIER_FACTOR * rmse_median

    ``factor`` defaults to ``CALIB_RMSE_OUTLIER_FACTOR`` (2.5) from
    ``core.config``. Returns ``(kept, dropped, diagnostics)`` where
    ``diagnostics`` is a dict with ``median``, ``threshold``, ``kept``,
    ``dropped`` and ``n_total``. An empty input yields empty outputs and
    ``median == threshold == 0.0`` (deterministic, no crash).

    The function is pure and testable; it does not depend on model loading
    or filesystem state.
    """
    from core.config import CALIB_RMSE_OUTLIER_FACTOR as _DEFAULT

    if factor is None:
        factor = _DEFAULT

    if not frame_records:
        return [], [], {
            "median": 0.0,
            "threshold": 0.0,
            "kept": 0,
            "dropped": 0,
            "n_total": 0,
            "factor": float(factor),
        }

    rmse_all = np.array([float(f["rmse"]) for f in frame_records], dtype=np.float64)
    median = float(np.median(rmse_all))
    threshold = float(factor * median)
    keep_mask = rmse_all <= threshold

    kept = [f for f, ok in zip(frame_records, keep_mask) if ok]
    dropped = [f for f, ok in zip(frame_records, keep_mask) if not ok]

    diagnostics = {
        "median": median,
        "threshold": threshold,
        "kept": len(kept),
        "dropped": len(dropped),
        "n_total": len(frame_records),
        "factor": float(factor),
    }
    return kept, dropped, diagnostics
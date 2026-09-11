"""tests.smoke_depth_integration  (OPT-IN -- never part of the unit suite)

End-to-end smoke for core.depth with the REAL Depth Anything V2 model.

This test downloads the model weights and runs real GPU/MPS inference, so
it is deliberately EXCLUDED from the deterministic unit suite. Run it
only when a live model download is acceptable:

    .venv/bin/python tests/smoke_depth_integration.py

It exits 0 on success, 1 on any failure (model download, pipeline
construction, or inference), and never runs as part of `tests/test_*.py`.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from core import config as cfg
from core import depth as depth_mod
from tests import FRAMES


def main():
    # A real registered frame image (frame_0027.jpg is part of the
    # artifacts, already present -- no download needed for the image).
    img_path = FRAMES / "frame_0027.jpg"
    if not img_path.exists():
        sys.stderr.write("missing image: %s\n" % img_path)
        return 1

    from PIL import Image

    pipe = depth_mod.load_depth_pipe(
        model=cfg.DEFAULT_DEPTH_MODEL, device=cfg.DEFAULT_DEPTH_DEVICE)
    img = Image.open(img_path).convert("RGB")

    depth = depth_mod.predict_depth(pipe, img)
    assert depth.dtype == np.float32, depth.dtype
    assert depth.shape == (cfg.IMAGE_H, cfg.IMAGE_W), depth.shape
    assert np.all(np.isfinite(depth)), "non-finite depth values"
    assert float(depth.min()) >= 0.0, "negative depth values"

    # Depth Anything V2 returns inverse depth: small -> far, large -> near.
    print("model:      %s" % cfg.DEFAULT_DEPTH_MODEL)
    print("device:     %s" % cfg.DEFAULT_DEPTH_DEVICE)
    print("shape:      %s" % (depth.shape,))
    print("inverse_depth min/max: %f / %f" % (depth.min(), depth.max()))
    print("OK -- real-model depth smoke passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

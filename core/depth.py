"""flight2world.core.depth

Thin Depth Anything V2 wrapper, extracted faithfully from `fusion_v10.py`
(model loading, lines 499-503; inference, lines 525-539). The algorithm
is NOT redesigned; this module only factorizes the model/device
configuration and the predicted-depth conversion so the rest of the
pipeline does not hardcode them.

    load_depth_pipe(...)     build a Depth Anything V2 pipeline
    predict_depth(pipe, ...) V10-faithful inference -> float32 (H, W)
    DepthAnythingV2          small stateful wrapper (pipe + config)

IMPORTANT: importing this module does NOT load or download any model.
The pipeline is only constructed when `load_depth_pipe` (or the class
constructor) is called, and the factory is injectable so tests can use a
fake inference object without touching transformers or the network.
"""

import cv2
import numpy as np

from core.config import (
    DEFAULT_DEPTH_DEVICE,
    DEFAULT_DEPTH_MODEL,
    IMAGE_H,
    IMAGE_W,
)


def predict_depth(pipe, image, image_w=IMAGE_W, image_h=IMAGE_H):
    """Run one depth-estimation inference, V10-faithfully.

    ``pipe`` is any callable accepting a PIL image and returning a dict
    with a ``predicted_depth`` key (e.g. a transformers pipeline). The
    conversion sequence -- detach, squeeze, cpu, numpy, float32, resize
    when needed -- is byte-for-byte that of fusion_v10.py lines 528-539.

    Returns a float32 numpy array of shape (image_h, image_w).
    """
    result = pipe(image)
    predicted = result["predicted_depth"]
    if hasattr(predicted, "detach"):
        predicted = predicted.detach()
    predicted = predicted.squeeze().cpu().numpy().astype(np.float32)

    if predicted.shape != (image_h, image_w):
        # cv2.resize takes (width, height); V10 compares against (H, W).
        predicted = cv2.resize(
            predicted, (image_w, image_h),
            interpolation=cv2.INTER_LINEAR,
        )
    return predicted


def load_depth_pipe(model=DEFAULT_DEPTH_MODEL,
                    device=DEFAULT_DEPTH_DEVICE,
                    pipeline_factory=None):
    """Build a Depth Anything V2 depth-estimation pipeline.

    ``pipeline_factory`` is injectable for tests and must behave like
    ``transformers.pipeline``. Defaults reproduce fusion_v10.py lines
    499-503 exactly (model + device come from core.config).
    """
    if pipeline_factory is None:
        from transformers import pipeline as pipeline_factory
    return pipeline_factory(
        "depth-estimation",
        model=model,
        device=device,
    )


class DepthAnythingV2:
    """Stateful thin wrapper: holds the loaded pipeline and its config.

    The model/device are exposed as attributes so callers can report or
    override them without hardcoding the values throughout the pipeline.
    """

    def __init__(self, model=DEFAULT_DEPTH_MODEL,
                 device=DEFAULT_DEPTH_DEVICE,
                 pipeline_factory=None):
        self.model = model
        self.device = device
        self.pipe = load_depth_pipe(
            model=model, device=device, pipeline_factory=pipeline_factory,
        )

    def predict_depth(self, image, image_w=IMAGE_W, image_h=IMAGE_H):
        """Depth Anything inference on a PIL image -> float32 (H, W)."""
        return predict_depth(self.pipe, image, image_w=image_w, image_h=image_h)

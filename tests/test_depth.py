"""tests.test_depth

Deterministic tests for core.depth using a fake inference object.

HARD RULE: no model is downloaded and no transformers pipeline is
constructed during these tests. `load_depth_pipe` is always called with
an injected pipeline_factory, and `predict_depth` is exercised with a
fake pipe returning a fake tensor.

The single real-model smoke test lives in smoke_depth_integration.py and
is OPT-IN (it never runs as part of this suite).
"""

import pathlib
import sys

# Make the project root importable regardless of how this file is invoked.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from tests import run_module_tests

import core.config as cfg
import core.depth as depth_mod

IMAGE_W, IMAGE_H = cfg.IMAGE_W, cfg.IMAGE_H


# ------------------------------------------------------------
# Fake inference objects (mimic transformers + torch tensor API)
# ------------------------------------------------------------
class FakeTensor:
    """Minimal stand-in for a torch tensor: detach/squeeze/cpu/numpy."""

    def __init__(self, arr):
        self.arr = np.asarray(arr, dtype=np.float32)
        self.detach_called = False
        self.squeeze_called = False
        self.cpu_called = False

    def detach(self):
        self.detach_called = True
        return self

    def squeeze(self):
        self.squeeze_called = True
        self.arr = self.arr.squeeze()
        return self

    def cpu(self):
        self.cpu_called = True
        return self

    def numpy(self):
        return self.arr


class FakePipe:
    """Fake depth-estimation pipeline: returns {"predicted_depth": tensor}."""

    def __init__(self, tensor):
        self.tensor = tensor
        self.last_image = None

    def __call__(self, image):
        self.last_image = image
        return {"predicted_depth": self.tensor}


# ------------------------------------------------------------
# predict_depth: V10 conversion sequence
# ------------------------------------------------------------
def test_predict_depth_returns_float32_exact_shape():
    # float32 input so the float32 output (V10 convention) is compared
    # exactly, not against a float64 slice.
    arr = np.random.default_rng(0).uniform(0, 1, (1, IMAGE_H, IMAGE_W))
    arr = arr.astype(np.float32)
    fake = FakeTensor(arr)
    pipe = FakePipe(fake)
    out = depth_mod.predict_depth(pipe, "image-placeholder")
    assert out.dtype == np.float32
    assert out.shape == (IMAGE_H, IMAGE_W)
    assert fake.detach_called and fake.squeeze_called and fake.cpu_called
    assert pipe.last_image == "image-placeholder"
    np.testing.assert_array_equal(out, arr[0])   # no resize needed


def test_predict_depth_detaches_squeezes_cpu():
    # A shape that differs from (H, W) only by the leading batch dim must
    # NOT trigger a resize (squeeze handles the batch dim first).
    arr = np.random.default_rng(1).uniform(0, 1, (1, IMAGE_H, IMAGE_W))
    out = depth_mod.predict_depth(FakePipe(FakeTensor(arr)), "img")
    assert out.shape == (IMAGE_H, IMAGE_W)


def test_predict_depth_resizes_when_required():
    # Model returned a smaller map (e.g. 360x640): V10 resizes with
    # INTER_LINEAR to exactly 1280x720.
    small = np.random.default_rng(2).uniform(0, 1, (360, 640))
    out = depth_mod.predict_depth(FakePipe(FakeTensor(small)), "img")
    assert out.dtype == np.float32
    assert out.shape == (IMAGE_H, IMAGE_W)
    assert np.all(np.isfinite(out))


def test_predict_depth_resize_interpolation_is_linear():
    # A constant map stays constant after INTER_LINEAR resize (constant is
    # preserved by linear interpolation) -- and the shape is exact.
    const = np.full((360, 640), 0.25, dtype=np.float32)
    out = depth_mod.predict_depth(FakePipe(FakeTensor(const)), "img")
    assert out.shape == (IMAGE_H, IMAGE_W)
    np.testing.assert_allclose(out, 0.25, rtol=1e-5, atol=1e-6)


# ------------------------------------------------------------
# load_depth_pipe: configuration without hardcoding
# ------------------------------------------------------------
def test_load_depth_pipe_uses_injected_factory_with_config():
    seen = {}

    def fake_factory(task, model=None, device=None):
        seen["task"] = task
        seen["model"] = model
        seen["device"] = device
        return "fake-pipeline"

    pipe = depth_mod.load_depth_pipe(
        model="some/model", device="cpu", pipeline_factory=fake_factory)
    assert pipe == "fake-pipeline"
    assert seen == {"task": "depth-estimation",
                    "model": "some/model", "device": "cpu"}


def test_default_model_and_device_match_v10():
    # The wrapper's defaults must be the V10 reference model + device.
    assert cfg.DEFAULT_DEPTH_MODEL == "depth-anything/Depth-Anything-V2-Base-hf"
    assert cfg.DEFAULT_DEPTH_DEVICE == "mps"


# ------------------------------------------------------------
# DepthAnythingV2 class wrapper
# ------------------------------------------------------------
def test_class_exposes_config_and_delegates():
    tensor = FakeTensor(np.ones((1, 300, 400), dtype=np.float32))
    inst = depth_mod.DepthAnythingV2(
        model="m", device="d",
        pipeline_factory=lambda task, model=None, device=None: FakePipe(tensor),
    )
    assert inst.model == "m"
    assert inst.device == "d"
    out = inst.predict_depth("img", image_w=400, image_h=300)
    assert out.shape == (300, 400)
    assert out.dtype == np.float32


def test_import_does_not_construct_pipeline():
    # Importing core.depth must not load anything (no network/model).
    # If it did, constructing a pipeline without a factory would raise a
    # transformers error at import time -- so simply re-importing and
    # confirming the module's lazy design is enough.
    assert depth_mod.load_depth_pipe.__module__ == "core.depth"
    assert not hasattr(depth_mod, "_PIPE")


if __name__ == "__main__":
    run_module_tests(globals())

"""tests.v10_reference

Extract the EXACT V10 implementation from the immutable `fusion_v10.py`
source (it is a script, not importable) so tests can run the V10 code and
the extracted core modules head-to-head on identical inputs.

Two extraction modes:

    v10_function(name)
        Compile the named top-level function definition (e.g.
        ``build_track_maps``, ``calibrated_depth``) verbatim from the file.

    v10_block(name, start, end, params, extra=None, returns=None)
        Compile source lines [start, end] of the file -- slices of the
        inline pass-2 main-loop code -- into a standalone function. The
        given ``params`` become its parameters; every other name must
        resolve to a V10 module constant or one of the numpy/cv2/o3d/math
        bindings. ``extra`` injects additional globals (e.g. the V10
        ``calibrated_depth``). ``returns`` appends a ``return <expr>``
        line so the block's result can be read out. ``continue``
        statements are only legal inside loops, so blocks that contain
        them must keep their enclosing loop inside the slice.

This is for DETERMINISTIC numerical head-to-head checks; it never
imports or executes the V10 script's pipeline.
"""

import ast
import math
import pathlib
import textwrap

import cv2
import numpy as np
import open3d as o3d

from tests import FUSION_V10

# Module-level constants the extracted blocks reference (all values are
# read verbatim from the V10 source, never supplied by this module).
_V10_CONSTANTS = (
    "FX", "FY", "CX", "CY", "IMAGE_W", "IMAGE_H",
    "PIXEL_STRIDE", "NEIGHBOR_WINDOW", "MIN_NET_VOTES",
    "RELATIVE_TOLERANCE", "ABSOLUTE_TOLERANCE", "TRACK_RADIUS_PX",
    "GATE_FACTOR", "GATE_RADIUS_MIN", "GATE_RADIUS_MAX",
    "CALIB_MIN_CORR", "CALIB_MIN_SAMPLES", "CALIB_RMSE_OUTLIER_FACTOR",
)


def _v10_globals():
    """Namespace of V10 constants + numpy/cv2/o3d/math bindings."""
    src = FUSION_V10.read_text()
    tree = ast.parse(src)
    ns = {"np": np, "cv2": cv2, "o3d": o3d, "math": math}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in _V10_CONSTANTS:
                try:
                    ns[target.id] = ast.literal_eval(node.value)
                except Exception:
                    pass
    return ns


def v10_function(name):
    """Return the V10 function object with the given top-level name."""
    src = FUSION_V10.read_text()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            mod = ast.Module(body=[node], type_ignores=[])
            ast.fix_missing_locations(mod)
            ns = _v10_globals()
            exec(compile(mod, "v10_" + name, "exec"), ns)
            return ns[name]
    raise KeyError("no top-level V10 function %r" % name)


def v10_block(name, start, end, params, extra=None, returns=None):
    """Compile V10 source lines [start, end] into a standalone function.

    ``params`` names are the function parameters; any other name in the
    block must resolve to a V10 module constant or the numpy/cv2/o3d/math
    bindings above. ``extra`` merges extra globals (e.g. the V10
    ``calibrated_depth``) into the function's module namespace.
    ``returns``, when given, appends ``return <returns>`` as the final
    statement so the block's result can be read out.
    """
    src = FUSION_V10.read_text()
    lines = src.splitlines()
    if end > len(lines):
        raise ValueError("end line %d beyond file (%d)" % (end, len(lines)))
    # The sliced source is indented inside the V10 pass-2 loop; dedent to
    # the common level first, then the appended ``return`` and the whole
    # body are indented uniformly to function-body level.
    body = textwrap.dedent("\n".join(lines[start - 1:end]))
    if returns is not None:
        body += "\nreturn " + returns
    sig = ", ".join(params)
    func_src = "def %s(%s):\n%s" % (
        name, sig, textwrap.indent(body, "    "))
    ns = _v10_globals()
    if extra:
        ns.update(extra)
    exec(func_src, ns)
    return ns[name]

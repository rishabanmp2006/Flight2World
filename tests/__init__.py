"""flight2world tests package (Phase 2).

Tests are self-running (each file has a ``main`` runner) so they execute
deterministically without requiring any package install (rule: zero new
packages). They are also pytest-compatible: every ``test_*`` function is a
pytest test.
"""

import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
# Ensure the project root is importable whether tests run via
#   python -m tests.test_x        (cwd already on path)
#   python tests/test_x.py        (sys.path[0] would be tests/ only)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SPARSE_TXT = PROJECT_ROOT / "test" / "sparse_txt"
FRAMES = PROJECT_ROOT / "test" / "frames"
FUSION_V10 = PROJECT_ROOT / "fusion_v10.py"


def run_module_tests(module_ns):
    """Run every test_* callable in the given namespace; print results.

    Accepts either a module object or its ``globals()`` dict. (Note:
    ``dir(<module>)`` reveals module globals, but ``dir(<globals dict>)``
    only reveals the dict's methods -- so we branch on type.)
    """
    import traceback

    if isinstance(module_ns, dict):
        names = list(module_ns)

        def get(name):
            return module_ns[name]
    else:
        names = dir(module_ns)

        def get(name):
            return getattr(module_ns, name)

    tests = sorted(
        name for name in names
        if name.startswith("test_") and callable(get(name))
    )
    failures = 0
    for name in tests:
        try:
            get(name)()
            print("PASS  %s" % name)
        except Exception:
            failures += 1
            print("FAIL  %s" % name)
            traceback.print_exc()
    print("\n%d of %d tests passed." % (len(tests) - failures, len(tests)))
    return failures
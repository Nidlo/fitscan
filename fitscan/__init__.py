"""FitScan — AI-Powered Body Measurement Engine."""

# Python 3.11+ removed `inspect.getargspec` (deprecated since 3.0). chumpy —
# the SMPL `.pkl` deserialiser pulled in by smplx — still references it on
# import. Without the shim, loading any SMPL model file crashes with
# `AttributeError: module 'inspect' has no attribute 'getargspec'`. We install
# the shim at package-init time so every chumpy-using path (body_fitter,
# any future SMPLX/SMPLH user) inherits it without per-call repetition.
# chumpy is unmaintained; migrating to a chumpy-free SMPL loader is tracked
# as a separate sprint.
import inspect as _inspect

if not hasattr(_inspect, "getargspec"):
    _inspect.getargspec = _inspect.getfullargspec  # type: ignore[attr-defined]

# chumpy also imports `numpy.bool` / `numpy.int` / `numpy.float` etc. — those
# aliases were deprecated in numpy 1.20 and removed in 1.25 (we're on 1.26).
# Restore them as aliases for the Python built-ins so chumpy's top-level
# `from numpy import bool, int, ...` doesn't raise ImportError at deserialise
# time. Affects only this venv; numpy itself isn't mutated for callers that
# don't touch chumpy.
import numpy as _np

for _alias, _builtin in (
    ("bool", bool),
    ("int", int),
    ("float", float),
    ("complex", complex),
    ("object", object),
    ("str", str),
    ("unicode", str),
):
    if not hasattr(_np, _alias):
        setattr(_np, _alias, _builtin)

__version__ = "0.1.0"

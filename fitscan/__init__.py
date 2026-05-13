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

# PyTorch 2.6 flipped `torch.load`'s `weights_only` default to True, which
# is the right call for production servers loading attacker-controlled
# weights. Our model files come from trusted academic sources (cs.utexas.edu
# for HMR2.0, smpl.is.tue.mpg.de for SMPL), bundled by the official hmr2
# `download_models()` and the smpl-x package — same trust model as any
# pip-installed dependency. Restore the pre-2.6 default here so the HMR2
# checkpoint loads without enumerating every omegaconf internal type for
# `add_safe_globals`. Doing it at package import means the patch is in
# place before pytorch_lightning binds `from torch import load`.
try:
    import torch as _torch

    _original_torch_load = _torch.load

    def _trusted_torch_load(*args, **kwargs):  # type: ignore[no-untyped-def]
        # Force-override rather than setdefault — lightning_fabric.cloud_io
        # explicitly passes `weights_only=None` (which 2.6+ treats as True),
        # so a plain setdefault wouldn't bite. Worker-wide trust: every
        # torch.load consumer in fitscan reads files we control or pulled
        # from trusted academic mirrors. Untrusted user-uploaded weights
        # would need a separate audited code path anyway.
        kwargs["weights_only"] = False
        return _original_torch_load(*args, **kwargs)

    _torch.load = _trusted_torch_load  # type: ignore[assignment]
except ImportError:
    # torch isn't a hard import-time dep for the API surface; if it isn't
    # installed yet (CI lint, fresh checkout) skip silently.
    pass

__version__ = "0.1.0"

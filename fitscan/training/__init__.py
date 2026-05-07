"""Continuous-learning calibrator pipeline (Phase 5, scaffold-only).

See [README.md](README.md) for the data shape, phase plan, and
privacy contract.

Step 1 (data plumbing) lives in the Laravel backend at
`backend/app/Console/Commands/FitscanExportCorrections.php`. Step 2
(model trainer) will land in this package once accumulated correction
data justifies the build — gated on Snad's strategic review at the
~3-month mark of beta usage.
"""

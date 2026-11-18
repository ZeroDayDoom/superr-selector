"""Feature engineering package.

Every module documents its features in a FEATURE_DOCS dict consumed by
feature_registry.Registry. All features are computed from strictly
pre-delivery information; the leakage audit in scripts/run_features.py
verifies this dynamically by perturbing the current delivery's outcomes and
checking that no feature value changes.
"""

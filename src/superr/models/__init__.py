"""Models package: win probability, leverage, pressure.

All models train on match-level chronological splits (never random ball
splits). The Phase-5 protocol: train seasons <= 2024, validation 2025,
holdout 2026; the selected configuration is then refit on all seasons to
score every delivery (documented in the reports).
"""

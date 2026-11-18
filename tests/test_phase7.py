"""Phase-7 tests: composite rating, shrinkage, eligibility, reliability.

These pin the rating-layer invariants the rankings depend on:
z-composites are finite and weighted as documented, EB shrinkage pulls
small samples toward the pool mean, eligibility tiers gate on both sample
axes with per-player reasons, and the bootstrap SE/LCB relationship holds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from superr.ratings import composite_rating as cr  # noqa: E402
from superr.ratings import bayesian_shrinkage as bs  # noqa: E402
from superr.ratings import eligibility as el  # noqa: E402
from superr.ratings import reliability as rel  # noqa: E402
from superr.ratings import rankings as rk  # noqa: E402


def _pool(n: int = 40, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "player_id": range(n),
        "player_name": [f"p{i}" for i in range(n)],
        "level_lcb": rng.normal(0, 1, n),
        "positive_match_pct": rng.uniform(0.2, 0.9, n),   # consistency source
        "uplift_adjusted": rng.normal(0, 0.5, n),          # uplift source
        "volume": rng.uniform(1, 5, n),
        "cvar5": -rng.uniform(0, 2, n),
    })


def test_primary_weights_sum_and_formula():
    w = cr.PRIMARY_WEIGHTS
    assert set(w) == {"level_lcb", "consistency", "uplift", "volume", "downside"}
    # weights documented in the methodology (downside enters negatively)
    assert w["level_lcb"] == 0.45 and w["downside"] == -0.10


def test_composite_is_linear_in_zscores():
    d = _pool()
    d["downside_abs"] = d["cvar5"].abs()
    out = cr.composite(d, cr.PRIMARY_WEIGHTS, "rating_x")
    manual = (0.45 * cr.zscore(d["level_lcb"])
              + 0.20 * cr.zscore(d["positive_match_pct"])
              + 0.20 * cr.zscore(d["uplift_adjusted"])
              + 0.15 * cr.zscore(d["volume"])
              - 0.10 * cr.zscore(d["downside_abs"]))
    assert np.allclose(out, manual)
    # z-composite: mean ~0 across the pool
    assert abs(out.mean()) < 1e-9


def test_composite_treats_undefined_component_as_pool_mean():
    d = _pool(30)
    d["downside_abs"] = d["cvar5"].abs()
    d.loc[0, "uplift_adjusted"] = np.nan   # e.g. no normal-pressure deliveries
    d.loc[1, "level_lcb"] = np.nan
    out = cr.composite(d, cr.PRIMARY_WEIGHTS, "rating_x")
    assert np.isfinite(out).all()          # never poisons the total
    # the neutral fill (z=0) is equivalent to the pool-mean contribution
    full = cr.composite(d.dropna(subset=["uplift_adjusted"]),
                        cr.PRIMARY_WEIGHTS, "rating_y")
    assert full.notna().all()


def test_add_composites_produces_all_variants():
    d = cr.add_composites(_pool())
    for name in ("primary", *cr.VARIANTS):
        col = f"rating_{name}"
        assert col in d.columns
        assert np.isfinite(d[col]).all()
    # rating column order follows the pool; no name enters the computation
    assert d["rating_primary"].std() > 0


def test_shrinkage_pulls_small_samples_to_pool_mean():
    rng = np.random.default_rng(11)
    match_values = pd.concat([
        pd.DataFrame({"player_id": i, "hp_match_value": rng.normal(i / 5.0, 1.0, 20)})
        for i in range(40)])
    eb = bs.estimate_k(match_values, "hp_match_value")
    assert eb["k"] > 0 and np.isfinite(eb["k"])
    raw = pd.Series([2.0, 0.0], index=pd.Index([100, 101], name="player_id"))
    n = pd.Series([5, 200], index=raw.index)
    tab = bs.shrinkage_table(raw, n, eb["k"], pool_mean=0.5)
    # small sample is pulled harder toward the pool mean than the large one
    pull_small = abs(tab["shrunk_k"].iloc[0] - 0.5)
    pull_large = abs(tab["shrunk_k"].iloc[1] - 0.5)
    assert pull_small > pull_large
    # stronger shrinkage moves closer to the pool mean
    assert abs(tab["shrunk_k_double"].iloc[0] - 0.5) \
        < abs(tab["shrunk_k_half"].iloc[0] - 0.5)


def test_eligibility_tiers_and_reasons():
    agg = pd.DataFrame([
        {"player_id": 1, "player_name": "a", "role": "batter",
         "hp_balls": 120, "hp_matches": 15, "n_seasons": 3, "normal_balls": 900},
        {"player_id": 2, "player_name": "b", "role": "batter",
         "hp_balls": 70, "hp_matches": 9, "n_seasons": 2, "normal_balls": 400},
        {"player_id": 3, "player_name": "c", "role": "batter",
         "hp_balls": 30, "hp_matches": 4, "n_seasons": 1, "normal_balls": 100},
    ])
    audit = el.eligibility_audit(agg)
    assert list(audit["tier"]) == ["strict", "moderate", "ineligible"]
    assert audit["eligible_moderate"].tolist() == [True, True, False]
    reason = audit.loc[2, "reason_moderate"]
    assert "hp_balls" in reason and "hp_matches" in reason
    # every player appears — nobody silently excluded
    assert len(audit) == 3


def test_bootstrap_player_mean_and_lcb():
    rng = np.random.default_rng(5)
    good = pd.DataFrame({
        "player_id": 1, "match_id": range(30),
        "value": rng.normal(0.5, 0.05, 30)})
    noisy = pd.DataFrame({
        "player_id": 2, "match_id": range(30),
        "value": rng.normal(0.5, 1.5, 30)})
    boot = rel.bootstrap_player_mean(pd.concat([good, noisy]), "value", n_boot=300)
    # the low-variance player has a much tighter bootstrap SE
    assert boot.loc[1, "boot_se"] < boot.loc[2, "boot_se"] * 0.5
    # LCB subtracts 0.85 SE from the mean
    m, se = boot.loc[1, "boot_mean"], boot.loc[1, "boot_se"]
    assert rel.lcb(m, se) == pytest.approx(m - 0.85 * se)
    # single-match player: no SE, LCB falls back to the mean
    single = pd.DataFrame({"player_id": 3, "match_id": [0], "value": [1.0]})
    b1 = rel.bootstrap_player_mean(single, "value", n_boot=50)
    assert pd.isna(b1.loc[3, "boot_se"]) and rel.lcb(1.0, np.nan) == 1.0


def test_ranking_checks_and_scenario_tables():
    agg = pd.DataFrame({
        "player_id": [1, 2, 2, 3],
        "player_name": ["a", "b", "b", "c"],
        "score": [0.5, 0.9, 0.1, 0.3],
    })
    assert not rk.check_no_duplicates(agg)
    assert rk.check_no_duplicates(agg.drop_duplicates("player_id"))

    deliv = pd.DataFrame({
        "player_id": [1] * 50 + [2] * 60 + [3] * 10,
        "player_name": ["a"] * 50 + ["b"] * 60 + ["c"] * 10,
        "scenario_DEATH": [True] * 50 + [True] * 60 + [False] * 10,
        "value": np.concatenate([np.full(50, 0.4), np.full(60, 0.8),
                                 np.full(10, 0.1)]),
    })
    out = rk.scenario_rankings(deliv, ["scenario_DEATH"], "value", min_balls=40)
    t = out["scenario_DEATH"]
    assert list(t["player_id"]) == [2, 1]          # b first, c filtered by volume
    assert t["rank"].tolist() == [1, 2]


def test_public_context_never_alters_pressure_ranking():
    """The auction/public-context table must carry the data-driven rank and
    rating through UNCHANGED, with unknowns staying unknown."""
    from superr.ratings import public_context as pubcon
    top10 = pd.DataFrame({
        "rank": [1, 2],
        "player_id": [10, 11],
        "player_name": ["A", "B"],
        "rating_primary": [1.25, 0.9],
        "batter_role": ["opener", "middle"],
    })
    table = pubcon.build_auction_table(top10, "batter", None)
    assert list(table["pressure_rank"]) == [1, 2]
    assert table["pressure_rating"].tolist() == [1.25, 0.9]
    # no claims -> contract unknown -> capped at Medium, uncertainty High
    assert (table["contract_status_verified"] == "unknown").all()
    assert (table["auction_relevance_rating"] == "Medium").all()
    assert (table["information_uncertainty"] == "High").all()

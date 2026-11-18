"""Match-context and match-pressure features (per delivery).

Includes the WP-independent pressure proxy used until the win-probability
model (Phase 5) supplies the calibrated index: balance/severity components
computed from state only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MODULE = "match_pressure_features"

FEATURE_DOCS: Dict[str, dict] = {
    "innings_phase": {
        "definition": "Coarse innings phase of the current over",
        "formula": "powerplay if over<=6; middle if over<=15; death otherwise",
        "source": MODULE + " (from legal_balls_before)",
        "available_before_delivery": True,
        "leakage_risk": "None — over number is known before the ball",
        "missing_handling": "n/a (always computable)",
        "relevance": "Scoring and wicket rates differ sharply by phase; baselines must be phase-aware",
    },
    "over_number_before": {
        "definition": "1-based over index containing this delivery",
        "formula": "floor(legal_balls_before / 6) + 1",
        "source": "match_state.add_pre_delivery_state",
        "available_before_delivery": True,
        "leakage_risk": "None",
        "missing_handling": "n/a",
        "relevance": "Fine-grained phase control for expected-outcome models",
    },
    "ball_in_over_before": {
        "definition": "Legal balls already bowled in the current over",
        "formula": "legal_balls_before mod 6",
        "source": "match_state.add_pre_delivery_state",
        "available_before_delivery": True,
        "leakage_risk": "None",
        "missing_handling": "n/a",
        "relevance": "Captures over-end effects (batter targeting last ball, bowler pressure)",
    },
    "is_knockout": {
        "definition": "Match is a knockout (eliminator/qualifier/semi/final)",
        "formula": "matches.knock_out == 1",
        "source": "matches.csv",
        "available_before_delivery": True,
        "leakage_risk": "None — pre-match metadata",
        "missing_handling": "False when null",
        "relevance": "Elimination matches are a distinct pressure regime (stage metadata)",
    },
    "is_home_batting_team": {
        "definition": "Batting team is the designated home side",
        "formula": "batting_team_id == home_team_id",
        "source": "matches.csv",
        "available_before_delivery": True,
        "leakage_risk": "None (venue designations are pre-match)",
        "missing_handling": "NaN when neither team is home (neutral venues)",
        "relevance": "Home advantage is a contextual baseline adjustment",
    },
    "toss_winner_bats_first": {
        "definition": "Toss winner chose to bat (proxy for pitch conditions)",
        "formula": "toss_win_team_id == batting_team_id and toss_opted=='Batting'",
        "source": "matches.csv",
        "available_before_delivery": True,
        "leakage_risk": "None",
        "missing_handling": "NaN when toss fields missing (15 matches)",
        "relevance": "Toss decisions encode expected surface behaviour",
    },
    "season": {
        "definition": "Competition season label",
        "formula": "competitions.season",
        "source": "competitions.csv",
        "available_before_delivery": True,
        "leakage_risk": "None",
        "missing_handling": "n/a",
        "relevance": "Era effects in scoring environment; walk-forward split key",
    },
    "competition_class": {
        "definition": "IPL vs senior T20I",
        "formula": "competition_class from competition_filter",
        "source": "competition_filter.classify_competitions",
        "available_before_delivery": True,
        "leakage_risk": "None",
        "missing_handling": "n/a",
        "relevance": "Tournament-level scoring environment differences",
    },
    "match_balance_proxy": {
        "definition": "How close the match is, without a WP model (Phase-4 proxy)",
        "formula": "chase: 1-|2*target_progress-1|; innings 1: 1-|2*(expected_final_score/venue_season_par)-1|, clipped [0,1]",
        "source": MODULE,
        "available_before_delivery": True,
        "leakage_risk": "Proxy only; replaced by calibrated WP balance in Phase 5",
        "missing_handling": "NaN when target/par unavailable",
        "relevance": "Pressure requires the outcome to still be in balance",
    },
    "severity_proxy": {
        "definition": "Batting-side resource stress, 0-1 (Phase-4 proxy)",
        "formula": "0.5*clip(RRR/14 or 0,0,1) + 0.3*(1 - wickets_in_hand/10) + 0.2*(over>=16)",
        "source": MODULE,
        "available_before_delivery": True,
        "leakage_risk": "Proxy; Phase 5 replaces with model-based severity",
        "missing_handling": "NaN when chase components unavailable",
        "relevance": "Resource scarcity is the core of situational pressure",
    },
}


def add_match_pressure_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    over_idx = (df["legal_balls_before"] // 6).astype("int64") + 1
    df["innings_phase"] = np.select(
        [over_idx <= 6, over_idx <= 15], ["powerplay", "middle"], default="death"
    )

    df["is_knockout"] = df["knock_out"].astype("boolean").fillna(False)
    df["is_home_batting_team"] = np.where(
        df["batting_team_id"].eq(df["home_team_id"]), 1.0,
        np.where(df["batting_team_id"].eq(df["away_team_id"]), 0.0, np.nan),
    )
    df["toss_winner_bats_first"] = np.where(
        df["toss_win_team_id"].eq(df["batting_team_id"])
        & df["toss_opted"].astype("string").str.lower().eq("batting"),
        1.0,
        np.where(
            df["toss_win_team_id"].eq(df["bowling_team_id"]), 0.0, np.nan
        ),
    )

    # match balance proxy
    chase_balance = 1.0 - (2.0 * df["target_progress"] - 1.0).abs()
    par_ratio = (df["expected_final_score"] / df["venue_season_par"]).replace(
        [np.inf, -np.inf], np.nan
    )
    inn1_balance = 1.0 - (2.0 * par_ratio - 1.0).abs()
    df["match_balance_proxy"] = np.where(
        df["innings"].eq(2), chase_balance.clip(0, 1), inn1_balance.clip(0, 1)
    )

    # severity proxy
    rrr_component = (
        df["required_run_rate"] / 14.0 if "required_run_rate" in df else pd.Series(np.nan, index=df.index)
    )
    rrr_component = pd.Series(rrr_component, index=df.index).clip(0, 1).fillna(0.0)
    wicket_component = 1.0 - df["wickets_in_hand"] / 10.0
    death_component = (over_idx >= 16).astype(float)
    df["severity_proxy"] = (
        0.5 * rrr_component + 0.3 * wicket_component + 0.2 * death_component
    )
    return df

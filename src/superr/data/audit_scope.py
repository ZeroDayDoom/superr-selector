"""Competition-scope and match-exclusion audits.

Produces the two CSV artefacts that document, match by match and competition
by competition, what is included in the analysis and why. No competition or
match is silently included or dropped: every decision is recorded here.
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from superr.data.audit_data import parse_result_string

# Classification by competition name. The two A-team / development tournaments
# present in the pack are flagged explicitly.
A_TEAM_PATTERNS = ["Emerging", "Rising Stars"]


def classify_competitions(competitions: pd.DataFrame) -> pd.DataFrame:
    c = competitions.copy()
    name = c["comp_name"].astype(str)

    def cls(n: str) -> str:
        if n.startswith("IPL"):
            return "IPL"
        if any(p.lower() in n.lower() for p in A_TEAM_PATTERNS):
            return "A-team/development"
        if "T20I" in n or True:  # everything else in this pack is senior T20I
            return "senior T20I"

    c["competition_class"] = name.map(cls)
    c["is_womens"] = name.str.contains("Women", case=False)
    return c


def build_competition_audit(
    competitions: pd.DataFrame,
    matches: pd.DataFrame,
    balls: pd.DataFrame,
    mapping: pd.DataFrame,
) -> pd.DataFrame:
    c = classify_competitions(competitions)
    m_counts = matches.groupby("comp_id").agg(
        n_matches=("match_id", "size"),
        n_knockouts=("knock_out", "sum"),
    )
    bb = balls.merge(matches[["match_id", "comp_id"]], on="match_id", how="left")
    b_counts = bb.groupby("comp_id").agg(n_deliveries=("match_id", "size"))
    p_counts = mapping.groupby("comp_id")["player_id"].nunique().rename("n_players")

    out = c.set_index("comp_id").join(m_counts).join(b_counts).join(p_counts)
    out[["n_matches", "n_deliveries", "n_players"]] = out[
        ["n_matches", "n_deliveries", "n_players"]
    ].fillna(0).astype(int)

    def policy(row: pd.Series) -> str:
        if row["competition_class"] == "A-team/development":
            return "excluded (development/A-team tournament; not senior international or IPL)"
        if row["n_matches"] == 0:
            return "excluded (no matches in ball-by-ball data)"
        return "included"

    out["inclusion_policy"] = out.apply(policy, axis=1)
    out = out.reset_index()[
        [
            "comp_id",
            "comp_name",
            "match_format",
            "season",
            "competition_class",
            "is_womens",
            "n_matches",
            "n_knockouts",
            "n_deliveries",
            "n_players",
            "inclusion_policy",
        ]
    ]
    return out.sort_values(["competition_class", "season", "comp_name"]).reset_index(drop=True)


def build_match_exclusion_audit(
    matches: pd.DataFrame,
    balls: pd.DataFrame,
    comp_audit: pd.DataFrame,
) -> pd.DataFrame:
    m = matches.copy()
    parsed = pd.DataFrame([parse_result_string(r) for r in m["result"]])
    m["_rt"] = parsed["result_type"].values

    comp_scope = dict(zip(comp_audit["comp_id"], comp_audit["inclusion_policy"]))
    comp_class = dict(zip(comp_audit["comp_id"], comp_audit["competition_class"]))
    m["_comp_policy"] = m["comp_id"].map(comp_scope).fillna("unknown competition")
    m["_comp_class"] = m["comp_id"].map(comp_class).fillna("unknown")

    has_balls = m["match_id"].isin(set(balls["match_id"]))
    n_inn = balls.groupby("match_id")["innings"].nunique()
    super_over = balls[balls["innings"] >= 3].groupby("match_id").size()
    dls = (
        m["first_inn_revised_overs"].notna() | m["second_inn_revised_overs"].notna()
    )

    decisions = []
    reasons = []
    for i, row in m.iterrows():
        rs = []
        if not has_balls.iloc[i]:
            rs.append("no ball-by-ball data")
        if row["_rt"] in ("no_result", "abandoned"):
            rs.append("no result / abandoned (no outcome to contextualise pressure)")
        if row["_comp_policy"].startswith("excluded"):
            rs.append(f"competition excluded: {row['_comp_policy']}")
        if rs:
            decisions.append("excluded")
        else:
            decisions.append("included")
        extra = []
        if decisions[-1] == "included":
            if row["_rt"] == "tied":
                extra.append("tied match; super-over innings (3/4) dropped from delivery analysis")
            if dls.iloc[i]:
                extra.append("DLS-revised overs/target; chase state uses revised values")
        reasons.append("; ".join(rs + extra) if (rs or extra) else "included in full")

    out = pd.DataFrame(
        {
            "match_id": m["match_id"],
            "comp_id": m["comp_id"],
            "comp_name": m["comp_id"].map(
                dict(zip(comp_audit["comp_id"], comp_audit["comp_name"]))
            ),
            "competition_class": m["_comp_class"],
            "match_date": m["match_date"],
            "stage": m["stage"],
            "knock_out": m["knock_out"],
            "home_team": m["home_team"],
            "away_team": m["away_team"],
            "result": m["result"],
            "result_type": m["_rt"],
            "has_ball_data": has_balls,
            "n_innings_with_balls": m["match_id"].map(n_inn).fillna(0).astype(int),
            "has_super_over_innings": m["match_id"].isin(set(super_over.index)),
            "dls_revised": dls.values,
            "revised_target": m["revised_target"],
            "exclusion_decision": decisions,
            "exclusion_or_special_handling_reason": reasons,
        }
    )
    return out.sort_values("match_id").reset_index(drop=True)


def summarize_match_audit(match_audit: pd.DataFrame) -> Dict[str, Any]:
    inc = match_audit["exclusion_decision"].eq("included")
    exc = ~inc
    reasons = match_audit["exclusion_or_special_handling_reason"]
    return {
        "total_matches": int(len(match_audit)),
        "included": int(inc.sum()),
        "excluded": int(exc.sum()),
        "excluded_no_ball_data": int((exc & reasons.str.contains("no ball-by-ball")).sum()),
        "excluded_no_result": int((exc & reasons.str.contains("no result / abandoned")).sum()),
        "excluded_competition_scope": int((exc & reasons.str.contains("competition excluded")).sum()),
        "tied_matches_kept": int((inc & match_audit["result_type"].eq("tied")).sum()),
        "dls_revised_kept": int((inc & match_audit["dls_revised"]).sum()),
        "super_over_matches_flagged": int(match_audit["has_super_over_innings"].sum()),
    }

"""Competition classification and configurable scope filtering.

Scope policy (approved in the Phase-2 methodology and Phase-1 audit):
IPL 2022-2026 and senior T20I competitions are included; A-team/development
tournaments are excluded by default (configurable); No Result / Abandoned
matches are excluded (no outcome to contextualise pressure); super-over
innings (3/4 of tied matches) are dropped from delivery-level analysis.
"""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

A_TEAM_PATTERNS = ("Emerging", "Rising Stars")

# Label saved alongside the data for every competition.
COMPETITION_CLASS_IPL = "IPL"
COMPETITION_CLASS_T20I = "senior T20I"
COMPETITION_CLASS_ATEAM = "A-team/development"


def classify_competitions(competitions: pd.DataFrame) -> pd.DataFrame:
    """Add competition_class / is_ipl / is_t20i / is_a_team labels."""
    c = competitions.copy()
    name = c["comp_name"].astype(str)

    def cls(n: str) -> str:
        if n.startswith("IPL"):
            return COMPETITION_CLASS_IPL
        if any(p.lower() in n.lower() for p in A_TEAM_PATTERNS):
            return COMPETITION_CLASS_ATEAM
        return COMPETITION_CLASS_T20I

    c["competition_class"] = name.map(cls)
    c["is_ipl"] = c["competition_class"].eq(COMPETITION_CLASS_IPL)
    c["is_t20i"] = c["competition_class"].eq(COMPETITION_CLASS_T20I)
    c["is_a_team"] = c["competition_class"].eq(COMPETITION_CLASS_ATEAM)
    c["is_womens"] = name.str.contains("Women", case=False)
    return c


def build_scope_filter(
    comp_labels: pd.DataFrame,
    matches: pd.DataFrame,
    scope: str = "approved",
    include_a_team: bool = False,
    include_no_result: bool = False,
    keep_super_over_innings: bool = False,
) -> pd.DataFrame:
    """Return a match-level table with inclusion flags and reasons.

    Parameters
    ----------
    scope : one of
        "approved"   default analysis scope per the methodology document
        "ipl"        IPL competitions only
        "t20i"       senior T20I competitions only
        "all_t20"    every competition present (including A-team)
    """
    labels = comp_labels[["comp_id", "competition_class", "is_ipl", "is_t20i",
                          "is_a_team"]]
    m = matches.merge(labels, on="comp_id", how="left", validate="many_to_one")

    if m["competition_class"].isna().any():
        raise ValueError("matches reference competitions missing from the registry")

    if scope == "approved":
        comp_ok = (pd.Series(True, index=m.index) if include_a_team
                   else ~m["is_a_team"])
    elif scope == "ipl":
        comp_ok = m["is_ipl"]
    elif scope == "t20i":
        comp_ok = m["is_t20i"]
    elif scope == "all_t20":
        comp_ok = pd.Series(True, index=m.index)
    else:
        raise ValueError(f"unknown scope: {scope}")

    has_result = m["win_team_id"].notna() & (m["result"].notna())
    is_nr = m["result"].astype("string").str.strip().str.lower().isin(
        ["no result", "match abandoned"]
    )
    result_ok = (~is_nr) if not include_no_result else pd.Series(True, index=m.index)

    m["scope_include"] = (comp_ok & result_ok).fillna(False)
    reasons = []
    for i, row in m.iterrows():
        r = []
        if not comp_ok.iloc[i]:
            r.append("competition out of scope")
        if is_nr.iloc[i] and not include_no_result:
            r.append("no result / abandoned")
        reasons.append("; ".join(r) if r else "included")
    m["scope_reason"] = reasons

    # result_type for downstream handling
    res = m["result"].astype("string").str.strip().str.lower()
    m["result_type"] = "decisive"
    m.loc[res.eq("match tied"), "result_type"] = "tied"
    m.loc[is_nr.fillna(False), "result_type"] = "no_result"
    return m


def competition_scope_summary(match_scope: pd.DataFrame) -> Dict[str, object]:
    inc = match_scope["scope_include"]
    return {
        "matches_total": int(len(match_scope)),
        "matches_in_scope": int(inc.sum()),
        "by_class": match_scope[inc]
        .groupby("competition_class")
        .size()
        .to_dict(),
        "excluded_competition": int((~inc & match_scope["scope_reason"].str.contains("competition")).sum()),
        "excluded_no_result": int((~inc & match_scope["scope_reason"].str.contains("no result")).sum()),
    }

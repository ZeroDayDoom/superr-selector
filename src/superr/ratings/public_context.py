"""Public player context: claims registry + auction-relevance framework.

STRICT SEPARATION: public information NEVER feeds the pressure ranking.
It is used only to interpret players, verify roles, and assess auction
relevance and availability uncertainty. The Auction Relevance rating is
computed from public fields alone and is reported SEPARATELY from the
data-driven Pressure Reliability rating.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

CLAIMS_COLUMNS = [
    "player_name", "player_id", "claim", "category", "value",
    "source_name", "url", "publication_date", "confidence",
    "official",
    # confidence: high | medium | low
    # official: yes | no
]

AUCTION_COLUMNS = [
    "player_name", "player_id", "pressure_rank", "pressure_rating",
    "data_role", "public_role", "role_consistent",
    "age_estimate", "ipl_experience_seasons",
    "contract_status_verified", "auction_eligibility",
    "availability_notes", "potential_rr_squad_role",
    "auction_relevance_rating",  # High / Medium / Low (public-info only)
    "information_uncertainty",   # High / Medium / Low
    "key_sources",
]


def load_claims(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    return pd.read_csv(path)


def validate_claims(df: pd.DataFrame) -> list:
    """Every claim must carry source, url, date, confidence, official flag."""
    problems = []
    required = ["player_name", "claim", "source_name", "url",
                "publication_date", "confidence", "official"]
    for i, row in df.iterrows():
        for c in required:
            if pd.isna(row.get(c)) or str(row.get(c)).strip() == "":
                problems.append(f"row {i}: missing '{c}'")
        if row.get("confidence") not in ("high", "medium", "low"):
            problems.append(f"row {i}: bad confidence")
        if row.get("official") not in ("yes", "no"):
            problems.append(f"row {i}: bad official flag")
    return problems


def auction_relevance_rating(row: pd.Series) -> str:
    """Public-information-only rating (never mixed with pressure rating).

    Heuristic, transparent: High = verified recent IPL role + plausible
    availability; Low = contracted long-term without known release, or
    unavailable; Medium otherwise. Availability unknown caps at Medium.
    """
    avail = str(row.get("contract_status_verified", "unknown")).lower()
    if "unavailable" in avail or "injury" in avail:
        return "Low"
    if avail.startswith("unknown"):
        return "Medium"
    ipl_exp = row.get("ipl_experience_seasons")
    if pd.notna(ipl_exp) and float(ipl_exp) >= 2:
        return "High"
    return "Medium"


def uncertainty_rating(df_player_claims: pd.DataFrame) -> str:
    if len(df_player_claims) == 0 or "official" not in df_player_claims.columns:
        return "High"
    n_official = int((df_player_claims["official"] == "yes").sum())
    share_official = n_official / len(df_player_claims)
    if share_official >= 0.6:
        return "Low"
    if share_official >= 0.3:
        return "Medium"
    return "High"


def build_auction_table(top10: pd.DataFrame, role: str, claims: pd.DataFrame,
                        pressure_col: str = "rating_primary") -> pd.DataFrame:
    """Merge the data-driven ranking with public context (read-only join)."""
    rows = []
    for _, r in top10.iterrows():
        pc = claims[claims["player_name"] == r["player_name"]] \
            if claims is not None else pd.DataFrame()
        def val(cat: str, default: str = "") -> str:
            sub = pc[pc["category"] == cat] if len(pc) else pd.DataFrame()
            return str(sub["value"].iloc[0]) if len(sub) else default
        rec = {
            "player_name": r["player_name"],
            "player_id": r["player_id"],
            "pressure_rank": int(r["rank"]),
            "pressure_rating": r[pressure_col],
            "data_role": (r.get("batter_role") if role == "batter"
                          else r.get("bowler_role")),
            "public_role": val("role"),
            "age_estimate": val("age", "unknown"),
            "ipl_experience_seasons": val("ipl_experience", None),
            "contract_status_verified": val("contract_status", "unknown"),
            "availability_notes": val("availability"),
            "auction_relevance_rating": None,
            "information_uncertainty": uncertainty_rating(pc),
            "key_sources": "; ".join(pc["source_name"].dropna().unique())
            if len(pc) else "",
        }
        rec["auction_relevance_rating"] = auction_relevance_rating(rec)
        rows.append(rec)
    return pd.DataFrame(rows, columns=AUCTION_COLUMNS)


def context_report_md(role: str, table: pd.DataFrame, claims: pd.DataFrame) -> str:
    lines = [f"# Public Player Context — {role.title()}",
             "",
             "_Public information is used ONLY to interpret players, verify "
             "roles, and assess auction relevance. The pressure ranking is "
             "untouched by this section. Claims without a verifiable source "
             "are not included; unknowns are stated as unknown._",
             ""]
    for _, r in table.iterrows():
        lines.append(f"## {int(r['pressure_rank'])}. {r['player_name']} "
                     f"({role})")
        lines.append(f"- Pressure rank: **{int(r['pressure_rank'])}** "
                     f"(data-driven; unchanged by public information)")
        lines.append(f"- Data-driven role: {r['data_role']} · "
                     f"public role: {r['public_role']}")
        lines.append(f"- Age (estimate): {r['age_estimate']}")
        lines.append(f"- IPL experience: {r['ipl_experience_seasons']}")
        lines.append(f"- Contract status (verified only): "
                     f"{r['contract_status_verified']}")
        lines.append(f"- Availability notes: {r['availability_notes'] or 'unknown'}")
        lines.append(f"- Potential RR squad role: {r.get('potential_rr_squad_role','—')}")
        lines.append(f"- Auction relevance: **{r['auction_relevance_rating']}** "
                     f"(public-info rating, separate from pressure rating)")
        lines.append(f"- Information uncertainty: "
                     f"{r['information_uncertainty']}")
        lines.append("")
        pc = claims[claims["player_name"] == r["player_name"]] \
            if claims is not None else pd.DataFrame()
        if len(pc):
            lines.append("| Claim | Value | Source | URL | Date | Confidence | Official |")
            lines.append("|---|---|---|---|---|---|---|")
            for _, c in pc.iterrows():
                lines.append(f"| {c['claim']} | {c['value']} | {c['source_name']} "
                             f"| {c['url']} | {c['publication_date']} "
                             f"| {c['confidence']} | {c['official']} |")
        else:
            lines.append("_No verifiable public claims recorded._")
        lines.append("")
    return "\n".join(lines)

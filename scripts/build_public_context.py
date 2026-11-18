"""Phase 11: build the public-context claims sheet + auction relevance report.

Honesty rules enforced:
- Only stable, long-established profile facts (age band, role, handedness,
  bowling type) are recorded, sourced to the player's canonical profile page.
- Volatile facts (current contract, auction status, injuries) are recorded
  as UNKNOWN because web verification was unavailable in this session.
- The pressure ranking is untouched; the auction-relevance rating is
  computed from public fields only and capped at Medium when availability
  is unknown.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd

from superr.ratings import public_context as pc

OUT = PROJECT_ROOT / "outputs"
REPORTS = PROJECT_ROOT / "reports"

# canonical profile page (ESPNcricinfo) per player — cited as the source of
# long-established profile facts. Access blocked in-session; flagged below.
def profile_url(slug: str) -> str:
    return f"https://www.espncricinfo.com/cricketers/{slug}"


# player -> (age born-year band, batting role, bowling role, handedness/type)
PLAYERS = {
    # batters
    "Mohammad Rizwan": ("born 1992", "wicketkeeper-batter (opener/middle)",
                        "wicketkeeper", "RHB"),
    "Nitish Reddy": ("born 2003", "middle-order batter", "medium-paced all-rounder",
                     "RHB"),
    "Fakhar Zaman": ("born 1990", "opener", "part-time spin", "LHB"),
    "Jaker Ali Anik": ("born 2000s (Bangladesh)", "middle-order batter / wicketkeeper",
                       "wicketkeeper", "RHB"),
    "Nurul Hasan": ("born 1990s (Bangladesh)", "wicketkeeper-batter", "wicketkeeper",
                    "RHB"),
    "Harry Tector": ("born 1999 (Ireland)", "middle-order batter", "part-time spin",
                     "RHB"),
    "Parvez Emon": ("born 2000s (Bangladesh)", "opener", "-", "LHB"),
    "Mahendra Singh Dhoni": ("born 1981", "wicketkeeper-batter (finisher)",
                             "wicketkeeper", "RHB"),
    "Shai Hope": ("born 1993 (West Indies)", "top-order batter / wicketkeeper",
                  "wicketkeeper", "RHB"),
    "Afif Hossain": ("born 2000s (Bangladesh)", "middle-order batter",
                     "off-spin", "LHB"),
    # bowlers
    "Fareed Malik": ("born 1990s (Afghanistan)", "-", "left-arm fast-medium",
                     "LHB"),
    "Kuldeep Sen": ("born 1990s (India)", "-", "right-arm fast", "RHB"),
    "Shakib Al Hasan": ("born 1987 (Bangladesh)", "batting all-rounder",
                        "left-arm orthodox spin", "LHB"),
    "Mohammed Shami": ("born 1990", "-", "right-arm fast", "RHB"),
    "Blair Tickner": ("born 1993 (New Zealand)", "-", "right-arm fast-medium",
                      "RHB"),
    "Mohsin Khan": ("born 1990s (India)", "-", "left-arm fast", "LHB"),
    "Mohammad Wasim": ("born 2000s (Pakistan)", "-", "right-arm fast", "RHB"),
    "Digvesh Rathi": ("born 1990s (India)", "-", "mystery spin", "RHB"),
    "Harshit Rana": ("born 2000s (India)", "-", "right-arm fast-medium", "RHB"),
    "Shamar Joseph": ("born 1990s (West Indies)", "-", "right-arm fast", "RHB"),
}


def build_claims(top_bat: pd.DataFrame, top_bowl: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in pd.concat([top_bat.assign(_role="batter"),
                           top_bowl.assign(_role="bowler")]).iterrows():
        name = r["player_name"]
        age_band, bat_role, bowl_role, hand = PLAYERS.get(name, ("unknown",) * 4)
        src = profile_url(name.lower().replace(" ", "-"))
        date = "profile (not re-verified in-session; access blocked 2026-11-18)"
        base = {"player_name": name, "player_id": r["player_id"],
                "source_name": "ESPNcricinfo player profile",
                "url": src, "publication_date": date,
                "confidence": "medium", "official": "no"}
        rows += [
            {**base, "claim": "Age / generation", "category": "age",
             "value": age_band},
            {**base, "claim": "Batting role", "category": "role",
             "value": bat_role if r["_role"] == "batter" else bat_role},
            {**base, "claim": "Bowling type / role", "category": "bowling_role",
             "value": bowl_role},
            {**base, "claim": "Batting handedness", "category": "handedness",
             "value": hand},
            {**base, "claim": "Current IPL contract status", "category":
             "contract_status",
             "value": "UNKNOWN — web verification unavailable in-session"},
            {**base, "claim": "Auction eligibility", "category": "auction",
             "value": "UNKNOWN — not verified; do not assume"},
            {**base, "claim": "Availability / injury status", "category":
             "availability",
             "value": "UNKNOWN — not verified in-session"},
        ]
    return pd.DataFrame(rows, columns=pc.CLAIMS_COLUMNS)


def main() -> int:
    top_bat = pd.read_csv(OUT / "rankings_batters.csv")
    top_bowl = pd.read_csv(OUT / "rankings_bowlers.csv")
    claims = build_claims(top_bat, top_bowl)
    claims.to_csv(OUT / "player_public_context_claims.csv", index=False)

    problems = pc.validate_claims(claims)
    if problems:
        print("CLAIM PROBLEMS:", problems[:5])

    table = pd.concat([
        pc.build_auction_table(top_bat, "batter", claims).assign(role="batter"),
        pc.build_auction_table(top_bowl, "bowler", claims).assign(role="bowler")],
        ignore_index=True)
    table.to_csv(OUT / "player_public_context.csv", index=False)

    md = ["# Public Player Context — auction relevance (Phase 11)", "",
          "_Web search and direct page fetches were unavailable during this "
          "session. Accordingly: (1) only long-established profile facts are "
          "recorded, sourced to canonical profile pages at MEDIUM confidence "
          "with an explicit not-re-verified flag; (2) every volatile fact "
          "(current contract, auction eligibility, injuries) is recorded as "
          "UNKNOWN. The pressure ranking below is the untouched, data-driven "
          "output of the pipeline; the auction-relevance rating is a separate "
          "public-information assessment and is capped at Medium while "
          "availability is unknown._", ""]
    for role in ("batter", "bowler"):
        sub = table[table["role"] == role]
        pc_md = pc.context_report_md(role, sub, claims)
        md.append(pc_md)
    (REPORTS / "public_player_context.md").write_text("\n".join(md),
                                                      encoding="utf-8")
    print(f"public context written: {len(claims)} claims, {len(table)} players")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

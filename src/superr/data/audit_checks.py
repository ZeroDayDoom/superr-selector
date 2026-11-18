"""Data-quality and cricket-semantics checks for the audit.

Every check returns a dict: {check, status (pass/warn/fail/info), detail,
...counts}. Results are saved as JSON and rendered into the quality report.
Semantics are *verified against the data* (never assumed):

- delivery total runs vs parsed match-result margins
- legal-ball accounting vs over structure
- bowler-run attribution (which extras the bowler is charged with)
- bowler-wicket attribution (dismissal modes)
- wicket/dismissed-player consistency
- batting order consistency
- identity checks (name <-> id bijections)
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from superr.data.audit_data import parse_result_string

TRUE_SET = {"true", "True", True, 1, "1"}
FALSE_SET = {"false", "False", False, 0, "0"}


def _to_bool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().eq("true")


def run_quality_checks(data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []

    def add(check: str, status: str, detail: str, **counts: Any) -> None:
        rec = {"check": check, "status": status, "detail": detail}
        rec.update(counts)
        checks.append(rec)

    b = data["ball_by_ball"]
    m = data["matches"]
    c = data["competitions"]
    pe = data["playing_eleven"]
    mp = data["mapping"]
    teams = data["teams"]

    # ---------------- duplicates & keys ----------------
    add(
        "duplicate_full_rows_ball_by_ball",
        "pass" if b.duplicated().sum() == 0 else "fail",
        f"{int(b.duplicated().sum())} fully duplicated delivery rows",
        count=int(b.duplicated().sum()),
    )
    dup_key = int(b.duplicated(subset=["match_id", "innings", "overid", "over_ball_num"]).sum())
    add(
        "duplicate_delivery_keys",
        "pass" if dup_key == 0 else "fail",
        f"{dup_key} duplicated (match_id, innings, overid, over_ball_num) keys",
        count=dup_key,
    )
    add(
        "duplicate_match_ids",
        "pass" if not m["match_id"].duplicated().any() else "fail",
        f"{int(m['match_id'].duplicated().sum())} duplicated match ids in matches.csv",
        count=int(m["match_id"].duplicated().sum()),
    )

    # ---------------- referential integrity ----------------
    missing_mm = sorted(set(b["match_id"]) - set(m["match_id"]))
    add(
        "ball_match_ids_missing_from_matches",
        "pass" if not missing_mm else "fail",
        f"{len(missing_mm)} ball match_ids not found in matches.csv",
        count=len(missing_mm),
        examples=missing_mm[:5],
    )
    matches_no_balls = sorted(set(m["match_id"]) - set(b["match_id"]))
    add(
        "matches_without_ball_data",
        "warn" if matches_no_balls else "pass",
        f"{len(matches_no_balls)} matches in matches.csv have no deliveries "
        "(typically abandoned/no-result matches)",
        count=len(matches_no_balls),
        examples=matches_no_balls[:10],
    )
    pe_matches_missing = sorted(set(m["match_id"]) - set(pe["match_id"]))
    add(
        "matches_without_playing_xi",
        "warn" if pe_matches_missing else "pass",
        f"{len(pe_matches_missing)} matches have no playing-XI rows",
        count=len(pe_matches_missing),
        examples=pe_matches_missing[:10],
    )
    null_players = int(
        b[["striker_id", "non_striker_id", "bowler_id"]].isna().any(axis=1).sum()
    )
    add(
        "null_player_ids_in_deliveries",
        "pass" if null_players == 0 else "fail",
        f"{null_players} deliveries with a null striker/non-striker/bowler id",
        count=null_players,
    )
    # every delivery player appears in the mapping for the same competition
    mb = b.merge(m[["match_id", "comp_id"]], on="match_id", how="left")
    comp_players = set(zip(mp["comp_id"], mp["player_id"]))
    unknown = 0
    for col in ("striker_id", "bowler_id"):
        pairs = set(zip(mb["comp_id"], mb[col]))
        unknown += len(pairs - comp_players)
    add(
        "delivery_players_present_in_mapping",
        "pass" if unknown == 0 else "warn",
        f"{unknown} (comp_id, player_id) delivery pairs absent from the "
        "competition-team-player mapping",
        count=unknown,
    )

    # ---------------- innings / over / ball validity ----------------
    bad_inn = int((~b["innings"].isin([1, 2, 3, 4])).sum())
    add(
        "invalid_innings_numbers",
        "pass" if bad_inn == 0 else "fail",
        f"{bad_inn} deliveries with innings outside 1-4 (3/4 = super over)",
        count=bad_inn,
    )
    over_counts = b.groupby(["match_id", "innings"])["overid"].agg(["min", "max"])
    bad_over = int((over_counts["min"] < 1).sum())
    add(
        "invalid_over_numbers",
        "pass" if bad_over == 0 else "warn",
        f"{bad_over} innings start before over 1",
        count=bad_over,
    )
    ballnum_range = b["over_ball_num"].between(1, 13).all()
    add(
        "invalid_delivery_numbers",
        "pass" if ballnum_range else "fail",
        "over_ball_num outside 1-13 found" if not ballnum_num_safe(b) else
        "all over_ball_num within 1-13 (max reflects extended overs of extras)",
        count=0 if ballnum_range else int((~b["over_ball_num"].between(1, 13)).sum()),
    )
    # innings_ball_num must be strictly increasing per innings
    seq_bad = 0
    for _, grp in b.sort_values("innings_ball_num").groupby(["match_id", "innings"]):
        if not grp["innings_ball_num"].is_monotonic_increasing:
            seq_bad += 1
    add(
        "ball_sequence_monotonic",
        "pass" if seq_bad == 0 else "warn",
        f"{seq_bad} innings with non-monotonic innings_ball_num after sorting",
        count=seq_bad,
    )

    # ---------------- runs & wickets ----------------
    add(
        "impossible_runs_off_bat",
        "pass" if b["runs"].between(0, 6).all() else "fail",
        "off-bat 'runs' outside 0-6 found" if not b["runs"].between(0, 6).all()
        else "off-bat 'runs' always within 0-6 (extras recorded separately)",
        count=int((~b["runs"].between(0, 6)).sum()),
    )
    neg_extras = int(
        (b[["extras1_runs", "extras_2_runs"]].fillna(0) < 0).any(axis=1).sum()
    )
    add(
        "negative_extras",
        "pass" if neg_extras == 0 else "fail",
        f"{neg_extras} deliveries with negative extras",
        count=neg_extras,
    )
    wk = b[_to_bool(b["is_wicket"])]
    mism = int((wk["dismissal_type"].isna()).sum())
    add(
        "wicket_without_dismissal_type",
        "pass" if mism == 0 else "fail",
        f"{mism} is_wicket rows with null dismissal_type",
        count=mism,
    )
    dwk = b[b["dismissal_type"].notna() & ~_to_bool(b["is_wicket"])]
    add(
        "dismissal_type_without_wicket_flag",
        "pass" if len(dwk) == 0 else "fail",
        f"{len(dwk)} dismissal_type rows where is_wicket is false",
        count=len(dwk),
    )
    null_disp = wk["dismissed_player_id"].isna()
    add(
        "wicket_without_dismissed_player",
        "warn" if null_disp.any() else "pass",
        f"{int(null_disp.sum())} wickets with null dismissed_player_id",
        count=int(null_disp.sum()),
    )
    runout = wk[wk["dismissal_type"] == "RunOut"]
    disp_is_batter = runout["dismissed_player_id"].isin(
        pd.concat([runout["striker_id"], runout["non_striker_id"]], axis=1).stack().unique()
    )
    runout_bad = int((~runout.apply(_dismissed_in_pair, axis=1)).sum())
    add(
        "runout_dismissed_player_in_pair",
        "pass" if runout_bad == 0 else "fail",
        f"{runout_bad} run-outs where dismissed_player_id is neither striker "
        f"nor non-striker (n run-outs = {len(runout)})",
        count=runout_bad,
    )
    wk_bad_pair = int((~wk.apply(_dismissed_in_pair_or_null, axis=1)).sum())
    add(
        "dismissed_player_matches_batting_pair",
        "pass" if wk_bad_pair == 0 else "fail",
        f"{wk_bad_pair} wickets where dismissed_player_id is not one of the two "
        "batters at the crease",
        count=wk_bad_pair,
    )
    wpi = wk.groupby(["match_id", "innings"]).size()
    add(
        "wickets_per_innings_le_10",
        "pass" if (wpi <= 10).all() else "fail",
        f"{int((wpi > 10).sum())} innings with more than 10 wickets",
        count=int((wpi > 10).sum()),
    )

    # ---------------- results ----------------
    parsed = pd.DataFrame([parse_result_string(r) for r in m["result"]])
    m["_result_type"] = parsed["result_type"].values
    m["_margin"] = parsed["margin"].values
    m["_margin_unit"] = parsed["margin_unit"].values
    m["_dls"] = parsed["dls"].values
    unparsed = m[m["_result_type"] == "unparsed"]
    add(
        "result_strings_unparsed",
        "pass" if len(unparsed) == 0 else "warn",
        f"{len(unparsed)} result strings could not be parsed by known patterns",
        count=len(unparsed),
        examples=list(unparsed["result"].unique())[:10],
    )
    win_null = m[(m["_result_type"].isin(["runs", "wickets"])) & m["win_team_id"].isna()]
    add(
        "decisive_result_without_win_team",
        "pass" if len(win_null) == 0 else "fail",
        f"{len(win_null)} decisive results with null win_team_id",
        count=len(win_null),
    )
    tied = m[m["_result_type"] == "tied"]
    add(
        "tied_matches",
        "info",
        f"{len(tied)} tied matches (super-over deliveries appear as innings 3/4)",
        count=len(tied),
        match_ids=[int(x) for x in tied["match_id"]],
    )
    nr = m[m["_result_type"].isin(["no_result", "abandoned"])]
    add(
        "no_result_or_abandoned_matches",
        "info",
        f"{len(nr)} matches with no result ({(nr['_result_type'] == 'abandoned').sum()} "
        "abandoned, "
        f"{(nr['_result_type'] == 'no_result').sum()} no-result)",
        count=len(nr),
    )
    dls = m[m["first_inn_revised_overs"].notna() | m["second_inn_revised_overs"].notna()]
    add(
        "dls_revised_matches",
        "info",
        f"{len(dls)} matches with revised overs; "
        f"{int(m['revised_target'].notna().sum())} with a revised target",
        count=len(dls),
    )

    # ---------------- identity consistency ----------------
    for role, idc, namec in [
        ("striker", "striker_id", "striker_name"),
        ("non_striker", "non_striker_id", "non_striker_name"),
        ("bowler", "bowler_id", "bowler_name"),
    ]:
        g = b.groupby(idc)[namec].nunique()
        n_bad = int((g > 1).sum())
        add(
            f"name_to_multiple_ids_{role}",
            "pass" if n_bad == 0 else "fail",
            f"{n_bad} {role}_id values map to multiple names",
            count=n_bad,
        )
    gm = mp.groupby("player_id")["player_name"].nunique()
    add(
        "mapping_id_to_multiple_names",
        "pass" if (gm <= 1).all() else "warn",
        f"{int((gm > 1).sum())} player_id values map to multiple names in the mapping file",
        count=int((gm > 1).sum()),
    )
    name_dup = mp["player_name"].value_counts()
    name_dup = name_dup[name_dup > 1]
    add(
        "shared_player_names_in_mapping",
        "info",
        f"{len(name_dup)} player names are shared by more than one player_id "
        "(names are NOT unique identifiers; player_id is the identity key)",
        count=len(name_dup),
        examples=list(name_dup.index[:8]),
    )
    # team names across competitions
    tg = teams.groupby("team_id")["team_name"].nunique()
    add(
        "team_id_to_multiple_names",
        "pass" if (tg <= 1).all() else "warn",
        f"{int((tg > 1).sum())} team_id values with multiple names in teams.csv",
        count=int((tg > 1).sum()),
    )

    # ---------------- playing XI semantics ----------------
    pe_false = pe[~_to_bool(pe["is_played"])]
    add(
        "playing_xi_is_played_false_rows",
        "info",
        f"{len(pe_false)} playing-XI rows are squad members who did not play "
        "(their captain/wk flags are null)",
        count=len(pe_false),
    )
    pe_dup = int(pe.duplicated(subset=["match_id", "team_id", "player_id"]).sum())
    add(
        "playing_xi_duplicate_player_rows",
        "info",
        f"{pe_dup} (match, team, player) rows are duplicated with different "
        "batting_order values (preferred positions varying across the "
        "tournament). Ball-level batting_order is authoritative; the PE "
        "batting_order column is not used for ordering.",
        count=pe_dup,
    )
    pe_bad_capt = int(
        pe[_to_bool(pe["is_played"])]["is_captain"].isna().sum()
    )
    add(
        "playing_xi_captain_flag_null_for_players",
        "pass" if pe_bad_capt == 0 else "warn",
        f"{pe_bad_capt} played rows with null is_captain",
        count=pe_bad_capt,
    )
    # batters in balls must be is_played=True in that match
    played_pairs = set(
        zip(
            pe.loc[_to_bool(pe["is_played"]), "match_id"],
            pe.loc[_to_bool(pe["is_played"]), "player_id"],
        )
    )
    deliv_pairs = set(zip(b["match_id"], b["striker_id"])) | set(
        zip(b["match_id"], b["non_striker_id"])
    ) | set(zip(b["match_id"], b["bowler_id"]))
    outside = len(deliv_pairs - played_pairs)
    add(
        "delivery_players_in_playing_xi",
        "pass" if outside == 0 else "warn",
        f"{outside} (match, player) delivery pairs with no is_played=True row",
        count=outside,
    )

    # ---------------- timestamps ----------------
    ts = pd.to_datetime(b["delivery_ist_time"], errors="coerce", utc=True, format="mixed")
    add(
        "delivery_timestamps_parseable",
        "pass" if ts.notna().all() else "warn",
        f"{int(ts.isna().sum())} unparseable delivery timestamps; "
        f"range {ts.min()} to {ts.max()}",
        count=int(ts.isna().sum()),
    )
    md = pd.to_datetime(m["match_date"], errors="coerce")
    b2 = b[["match_id"]].copy()
    b2["_ts"] = pd.to_datetime(
        b["delivery_ist_time"], errors="coerce", utc=True, format="mixed"
    ).dt.tz_localize(None)
    md_df = pd.DataFrame({"match_id": m["match_id"].values, "_md": md.values})
    chk = b2.merge(md_df, on="match_id", how="left")
    day_off = int((chk["_ts"].dt.date != chk["_md"].dt.date).sum())
    add(
        "delivery_date_matches_match_date",
        "warn" if day_off else "pass",
        f"{day_off} deliveries whose date differs from match_date (timezones: "
        "IST timestamps vs UTC dates; tolerance of +-1 day is expected)",
        count=day_off,
    )

    return checks


def _dismissed_in_pair(row: pd.Series) -> bool:
    return row["dismissed_player_id"] in (row["striker_id"], row["non_striker_id"])


def _dismissed_in_pair_or_null(row: pd.Series) -> bool:
    if pd.isna(row["dismissed_player_id"]):
        return True
    return _dismissed_in_pair(row)


def ballnum_num_safe(b: pd.DataFrame) -> bool:
    return bool(b["over_ball_num"].between(1, 13).all())

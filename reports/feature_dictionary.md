# Feature Dictionary — SupeRR Selector Season 2

_Generated 2026-11-18T10:03:51+00:00. 128 documented features._

All features are computed from strictly pre-delivery information
unless explicitly marked otherwise. Leakage risk and its control are
documented per feature; the dynamic leakage audit is described in
`scripts/run_features.py` and reported alongside the outputs.

## `ball_in_over_before`
- **Definition:** Legal balls already bowled in the current over
- **Formula:** legal_balls_before mod 6
- **Source:** match_state.add_pre_delivery_state
- **Available before delivery:** YES
- **Leakage risk:** None
- **Missing-data handling:** n/a
- **Relevance:** Captures over-end effects (batter targeting last ball, bowler pressure)

## `balls_per_wicket_remaining`
- **Definition:** Legal balls available per wicket in hand
- **Formula:** legal_balls_remaining / max(1, wickets_in_hand)
- **Source:** batting_pressure_features
- **Available before delivery:** YES
- **Leakage risk:** None
- **Missing-data handling:** n/a
- **Relevance:** Low values = double jeopardy (must score fast with few wickets)

## `balls_since_last_boundary`
- **Definition:** Deliveries since team boundary
- **Formula:** position - last boundary position
- **Source:** match_state.add_momentum_features
- **Available before delivery:** YES
- **Leakage risk:** Pre-delivery only
- **Missing-data handling:** NaN before first boundary
- **Relevance:** Boundary drought pressure

## `batter_balls_faced_before`
- **Definition:** Striker's legal balls faced this innings
- **Formula:** per-striker exclusive cumsum
- **Source:** player_state.add_striker_state
- **Available before delivery:** YES
- **Leakage risk:** Per-player grouping; exclusive of current ball
- **Missing-data handling:** 0 at entry
- **Relevance:** Setness / responsibility

## `batter_entry_ball`
- **Definition:** Legal ball at which batter entered
- **Formula:** legal_balls_before at entry
- **Source:** player_state.add_striker_state
- **Available before delivery:** YES
- **Leakage risk:** None
- **Missing-data handling:** n/a
- **Relevance:** Entry difficulty

## `batter_entry_wickets`
- **Definition:** Wickets lost when batter came in
- **Formula:** wickets_before at entry (ffill/bfill)
- **Source:** player_state.add_striker_state
- **Available before delivery:** YES
- **Leakage risk:** None
- **Missing-data handling:** n/a
- **Relevance:** Entry difficulty

## `batter_is_new`
- **Definition:** Striker has faced < 6 balls
- **Formula:** balls_faced_before < 6 (config)
- **Source:** player_state.add_striker_state
- **Available before delivery:** YES
- **Leakage risk:** None
- **Missing-data handling:** n/a
- **Relevance:** New-batter vulnerability

## `batter_is_set`
- **Definition:** Striker has faced >= 12 balls
- **Formula:** balls_faced_before >= 12 (config)
- **Source:** player_state.add_striker_state
- **Available before delivery:** YES
- **Leakage risk:** None
- **Missing-data handling:** n/a
- **Relevance:** Set batters are prized wickets

## `batter_role`
- **Definition:** Batter role from actual positions + death exposure
- **Formula:** role bands + finisher refinement
- **Source:** player_state.add_batter_roles
- **Available before delivery:** YES
- **Leakage risk:** None
- **Missing-data handling:** 'unknown' when order missing
- **Relevance:** Role-conditioned baselines

## `batter_runs_before`
- **Definition:** Striker's runs this innings
- **Formula:** per-striker exclusive cumsum
- **Source:** player_state.add_striker_state
- **Available before delivery:** YES
- **Leakage risk:** Per-player grouping
- **Missing-data handling:** 0 at entry
- **Relevance:** Setness / responsibility

## `batter_vs_pace_spin`
- **Definition:** Matchup cell: batter hand x bowler class
- **Formula:** batting_type x pace_or_spin (e.g. RHB_vs_Pace)
- **Source:** role_features
- **Available before delivery:** YES
- **Leakage risk:** None
- **Missing-data handling:** 'unknown_vs_unknown'
- **Relevance:** League-rate matchup baselines

## `batting_rate_vs_par`
- **Definition:** Batting run rate minus par run rate
- **Formula:** current_run_rate - par_run_rate
- **Source:** bowling_pressure_features
- **Available before delivery:** YES
- **Leakage risk:** Venue par leave-one-match-out
- **Missing-data handling:** NaN when par unavailable
- **Relevance:** Whether the bowler has contained below the environment

## `batting_resources_remaining`
- **Definition:** Batting-side resource proxy (balls x wickets)
- **Formula:** (balls_left/scheduled)*(wickets_left/10)
- **Source:** bowling_pressure_features
- **Available before delivery:** YES
- **Leakage risk:** None
- **Missing-data handling:** n/a
- **Relevance:** Mirror of resources_remaining for the fielding side

## `batting_type`
- **Definition:** Batter handedness
- **Formula:** raw batting_type (RHB/LHB)
- **Source:** ball_by_ball.csv
- **Available before delivery:** YES
- **Leakage risk:** None
- **Missing-data handling:** 'unknown' when blank
- **Relevance:** Matchup physics

## `boundary_streak_before`
- **Definition:** Consecutive boundaries immediately before
- **Formula:** streak length
- **Source:** match_state.add_momentum_features
- **Available before delivery:** YES
- **Leakage risk:** Pre-delivery only
- **Missing-data handling:** 0 when not on a streak
- **Relevance:** Momentum in batter's favour

## `bowler_boundary_streak_before`
- **Definition:** Consecutive boundaries conceded immediately before
- **Formula:** streak length
- **Source:** player_state.add_bowler_state (re-exposed)
- **Available before delivery:** YES
- **Leakage risk:** Pre-delivery streak
- **Missing-data handling:** 0 when none
- **Relevance:** Bowler under boundary siege

## `bowler_last12_boundaries`
- **Definition:** Boundaries conceded in previous 12 own deliveries
- **Formula:** shifted rolling count
- **Source:** player_state.add_bowler_state (re-exposed)
- **Available before delivery:** YES
- **Leakage risk:** Shifted
- **Missing-data handling:** NaN early
- **Relevance:** Leakage of boundaries

## `bowler_last12_dots`
- **Definition:** Dots in previous 12 own deliveries
- **Formula:** shifted rolling count
- **Source:** player_state.add_bowler_state (re-exposed)
- **Available before delivery:** YES
- **Leakage risk:** Shifted
- **Missing-data handling:** NaN early
- **Relevance:** Containment

## `bowler_last12_legal`
- **Definition:** Legal balls in previous 12 own deliveries
- **Formula:** shifted rolling legal count
- **Source:** bowling_pressure_features
- **Available before delivery:** YES
- **Leakage risk:** Shifted
- **Missing-data handling:** NaN early
- **Relevance:** Denominator for recent economy/dot rate

## `bowler_last12_legalruns`
- **Definition:** Runs conceded in previous 12 own deliveries
- **Formula:** shifted rolling runs
- **Source:** bowling_pressure_features
- **Available before delivery:** YES
- **Leakage risk:** Shifted
- **Missing-data handling:** NaN early
- **Relevance:** Recent economy numerator

## `bowler_last12_runs`
- **Definition:** Runs conceded in previous 12 own deliveries
- **Formula:** shifted rolling sum
- **Source:** player_state.add_bowler_state (re-exposed)
- **Available before delivery:** YES
- **Leakage risk:** Shifted
- **Missing-data handling:** NaN early
- **Relevance:** Recent economy

## `bowler_last12_wickets`
- **Definition:** Wickets in previous 12 own deliveries
- **Formula:** shifted rolling count
- **Source:** player_state.add_bowler_state (re-exposed)
- **Available before delivery:** YES
- **Leakage risk:** Shifted
- **Missing-data handling:** NaN early
- **Relevance:** Striking

## `bowler_last18_dots`
- **Definition:** Dots in previous 18 own deliveries
- **Formula:** shifted rolling count
- **Source:** bowling_pressure_features
- **Available before delivery:** YES
- **Leakage risk:** Shifted
- **Missing-data handling:** NaN early
- **Relevance:** Containment

## `bowler_last18_legal`
- **Definition:** Legal balls in previous 18 own deliveries
- **Formula:** shifted rolling legal count
- **Source:** bowling_pressure_features
- **Available before delivery:** YES
- **Leakage risk:** Shifted
- **Missing-data handling:** NaN early
- **Relevance:** Denominator

## `bowler_last18_legalruns`
- **Definition:** Runs conceded in previous 18 own deliveries
- **Formula:** shifted rolling runs
- **Source:** bowling_pressure_features
- **Available before delivery:** YES
- **Leakage risk:** Shifted
- **Missing-data handling:** NaN early
- **Relevance:** Recent economy numerator

## `bowler_last18_runs`
- **Definition:** Runs conceded in previous 18 own deliveries
- **Formula:** shifted rolling sum
- **Source:** bowling_pressure_features
- **Available before delivery:** YES
- **Leakage risk:** Shifted
- **Missing-data handling:** NaN early
- **Relevance:** Recent economy

## `bowler_last18_wickets`
- **Definition:** Wickets in previous 18 own deliveries
- **Formula:** shifted rolling count
- **Source:** bowling_pressure_features
- **Available before delivery:** YES
- **Leakage risk:** Shifted
- **Missing-data handling:** NaN early
- **Relevance:** Striking

## `bowler_last24_legal`
- **Definition:** Legal balls in previous 24 own deliveries
- **Formula:** shifted rolling legal count
- **Source:** bowling_pressure_features
- **Available before delivery:** YES
- **Leakage risk:** Shifted
- **Missing-data handling:** NaN early
- **Relevance:** Denominator for recent wicket rate

## `bowler_last24_legalruns`
- **Definition:** Runs conceded in previous 24 own deliveries
- **Formula:** shifted rolling runs
- **Source:** bowling_pressure_features
- **Available before delivery:** YES
- **Leakage risk:** Shifted
- **Missing-data handling:** NaN early
- **Relevance:** Recent economy numerator

## `bowler_last6_legal`
- **Definition:** Legal balls in previous 6 own deliveries
- **Formula:** shifted rolling legal count
- **Source:** bowling_pressure_features
- **Available before delivery:** YES
- **Leakage risk:** Shifted
- **Missing-data handling:** NaN early
- **Relevance:** Denominator for recent economy

## `bowler_last6_legalruns`
- **Definition:** Runs conceded in previous 6 own legal-denominated deliveries
- **Formula:** shifted rolling runs
- **Source:** bowling_pressure_features
- **Available before delivery:** YES
- **Leakage risk:** Shifted
- **Missing-data handling:** NaN early
- **Relevance:** Recent economy numerator

## `bowler_last6_runs`
- **Definition:** Runs conceded in previous 6 own deliveries
- **Formula:** shifted rolling sum
- **Source:** player_state.add_bowler_state (re-exposed)
- **Available before delivery:** YES
- **Leakage risk:** Shifted
- **Missing-data handling:** NaN early
- **Relevance:** Short-form pressure

## `bowler_overs_completed`
- **Definition:** Completed overs by this bowler in the innings
- **Formula:** floor(bowler_balls_before/6)
- **Source:** player_state.add_bowler_state (re-exposed)
- **Available before delivery:** YES
- **Leakage risk:** None
- **Missing-data handling:** 0 at start
- **Relevance:** Spell management

## `bowler_recent_dot_rate`
- **Definition:** Dot share over previous 12 own deliveries
- **Formula:** shifted dots / shifted legal
- **Source:** bowling_pressure_features
- **Available before delivery:** YES
- **Leakage risk:** Shifted
- **Missing-data handling:** NaN early
- **Relevance:** Containment form

## `bowler_recent_economy`
- **Definition:** Bowler's economy over previous 12 own deliveries
- **Formula:** shifted rolling runs / (shifted rolling legal/6)
- **Source:** bowling_pressure_features
- **Available before delivery:** YES
- **Leakage risk:** Shifted; excludes current ball
- **Missing-data handling:** NaN in first 12 own deliveries
- **Relevance:** Recent form of the bowler

## `bowler_recent_wicket_rate`
- **Definition:** Wickets per legal ball over previous 24 own deliveries
- **Formula:** shifted wkts / shifted legal
- **Source:** bowling_pressure_features
- **Available before delivery:** YES
- **Leakage risk:** Shifted
- **Missing-data handling:** NaN early
- **Relevance:** Striking form

## `bowler_role`
- **Definition:** Bowler phase role: powerplay/middle/death or hybrid-*
- **Formula:** share of legal balls by phase within innings; main phase >=70% => role, else hybrid-main
- **Source:** player_state.add_bowler_roles (re-exposed)
- **Available before delivery:** YES
- **Leakage risk:** Same-innings composition (pre-delivery part dominates; used categorically)
- **Missing-data handling:** n/a
- **Relevance:** Bowler phase specialisation

## `bowler_spell_overs_before`
- **Definition:** Overs in the current spell before this over
- **Formula:** consecutive same-bowler overs
- **Source:** player_state.add_bowler_state (re-exposed)
- **Available before delivery:** YES
- **Leakage risk:** None
- **Missing-data handling:** 0 at spell start
- **Relevance:** Fatigue proxy

## `bowling_style_group`
- **Definition:** Grouped bowling style
- **Formula:** fast (fast/fast-medium/medium), medium-fast, spin (break/orthodox/wrist/googly), other
- **Source:** role_features
- **Available before delivery:** YES
- **Leakage risk:** None
- **Missing-data handling:** 'unknown'
- **Relevance:** Finer matchup axis than pace/spin

## `bowling_style_group_bat_hand`
- **Definition:** Combined hand x style-group cell key
- **Formula:** batting_type + '_vs_' + bowling_style_group
- **Source:** matchup_features
- **Available before delivery:** YES
- **Leakage risk:** None
- **Missing-data handling:** 'unknown_vs_unknown'
- **Relevance:** Grouping key for style-level matchup rates

## `bowling_team_wicket_drought`
- **Definition:** Deliveries since the bowling team last took a wicket
- **Formula:** position - last team-wicket position
- **Source:** player_state.add_bowler_state
- **Available before delivery:** YES
- **Leakage risk:** Pre-delivery only
- **Missing-data handling:** NaN before first wicket
- **Relevance:** Extended droughts raise defensive pressure

## `collapse_severity`
- **Definition:** Continuous collapse intensity, 0-1
- **Formula:** 0.5*min(1, wk12/3) + 0.3*1[wickets_in_hand<=4] + 0.2*1[over>=7]
- **Source:** batting_pressure_features
- **Available before delivery:** YES
- **Leakage risk:** Derived from pre-delivery state
- **Missing-data handling:** 0 when no recent wickets
- **Relevance:** Single collapse axis for scenario labels and ratings

## `competition_class`
- **Definition:** IPL vs senior T20I
- **Formula:** competition_class from competition_filter
- **Source:** competition_filter.classify_competitions
- **Available before delivery:** YES
- **Leakage risk:** None
- **Missing-data handling:** n/a
- **Relevance:** Tournament-level scoring environment differences

## `consecutive_dots_before`
- **Definition:** Dot streak immediately before
- **Formula:** streak of consecutive dot deliveries
- **Source:** match_state.add_momentum_features
- **Available before delivery:** YES
- **Leakage risk:** Pre-delivery streak only
- **Missing-data handling:** 0 at innings start
- **Relevance:** Scoreboard pressure

## `consecutive_wickets`
- **Definition:** Wickets in the current collapse cluster
- **Formula:** wickets in last 18 legal balls when the most recent wicket is within 12 legal balls, else 1 if wickets_in_last_12>0
- **Source:** batting_pressure_features
- **Available before delivery:** YES
- **Leakage risk:** Derived from shifted windows only
- **Missing-data handling:** 0 when no recent wicket
- **Relevance:** Clustered wickets define a collapse, not isolated losses

## `current_run_rate`
- **Definition:** Batting run rate so far
- **Formula:** score_before / (legal_balls_before/6)
- **Source:** match_state.add_pre_delivery_state
- **Available before delivery:** YES
- **Leakage risk:** None
- **Missing-data handling:** 0 when no balls faced
- **Relevance:** Context for required-rate gap

## `delivery_line`
- **Definition:** Line side of the delivery
- **Formula:** raw on_or_off
- **Source:** ball_by_ball.csv
- **Available before delivery:** YES
- **Leakage risk:** None
- **Missing-data handling:** 'unknown'
- **Relevance:** Line context

## `entering_after_wicket`
- **Definition:** Batter's first delivery comes directly after a wicket
- **Formula:** partnership_balls_before == 0
- **Source:** batting_pressure_features
- **Available before delivery:** YES
- **Leakage risk:** None
- **Missing-data handling:** False otherwise
- **Relevance:** Incoming-batter vulnerability window

## `entry_balls_remaining`
- **Definition:** Balls remaining when batter entered
- **Formula:** legal_balls_remaining at entry
- **Source:** batting_pressure_features
- **Available before delivery:** YES
- **Leakage risk:** None
- **Missing-data handling:** n/a
- **Relevance:** Time pressure inherited at entry

## `entry_required_rate`
- **Definition:** Required run rate when batter entered (chases)
- **Formula:** required_run_rate at entry
- **Source:** batting_pressure_features
- **Available before delivery:** YES
- **Leakage risk:** None
- **Missing-data handling:** NaN in innings 1
- **Relevance:** Chase difficulty inherited at entry

## `entry_score`
- **Definition:** Team score when batter entered
- **Formula:** score_before at entry (ffill/bfill within batter-innings)
- **Source:** batting_pressure_features
- **Available before delivery:** YES
- **Leakage risk:** None
- **Missing-data handling:** n/a
- **Relevance:** Context of the situation inherited by the batter

## `expected_final_score`
- **Definition:** Projected final total from state
- **Formula:** expected_score_at_state + expected_runs_remaining
- **Source:** match_state.add_par_state
- **Available before delivery:** YES
- **Leakage risk:** LOO curves; no own-innings information
- **Missing-data handling:** Fallback hierarchy
- **Relevance:** Batting-first difficulty / target-set strength

## `expected_remaining_runs`
- **Definition:** Expected runs the batting side will still score
- **Formula:** expected_runs_remaining (LOO curve)
- **Source:** match_state.add_par_state (re-exposed)
- **Available before delivery:** YES
- **Leakage risk:** LOO curves
- **Missing-data handling:** Fallback hierarchy
- **Relevance:** Damage control baseline for the bowler

## `expected_runs_remaining`
- **Definition:** Expected runs still to be scored
- **Formula:** LOO mean of (final - score) at (balls-left, wickets)
- **Source:** match_state.add_par_state
- **Available before delivery:** YES
- **Leakage risk:** LOO curves
- **Missing-data handling:** Fallback hierarchy
- **Relevance:** Bowling-first pressure and chase context

## `innings_phase`
- **Definition:** Coarse innings phase of the current over
- **Formula:** powerplay if over<=6; middle if over<=15; death otherwise
- **Source:** match_pressure_features (from legal_balls_before)
- **Available before delivery:** YES
- **Leakage risk:** None — over number is known before the ball
- **Missing-data handling:** n/a (always computable)
- **Relevance:** Scoring and wicket rates differ sharply by phase; baselines must be phase-aware

## `is_home_batting_team`
- **Definition:** Batting team is the designated home side
- **Formula:** batting_team_id == home_team_id
- **Source:** matches.csv
- **Available before delivery:** YES
- **Leakage risk:** None (venue designations are pre-match)
- **Missing-data handling:** NaN when neither team is home (neutral venues)
- **Relevance:** Home advantage is a contextual baseline adjustment

## `is_impact_player_rule`
- **Definition:** Match played under the IPL impact-player rule (2023+ IPL)
- **Formula:** competition_class=='IPL' AND season>=2023
- **Source:** role_features
- **Available before delivery:** YES
- **Leakage risk:** None
- **Missing-data handling:** False
- **Relevance:** Rule changes alter batting depth and scoring

## `is_knockout`
- **Definition:** Match is a knockout (eliminator/qualifier/semi/final)
- **Formula:** matches.knock_out == 1
- **Source:** matches.csv
- **Available before delivery:** YES
- **Leakage risk:** None — pre-match metadata
- **Missing-data handling:** False when null
- **Relevance:** Elimination matches are a distinct pressure regime (stage metadata)

## `last12_boundaries`
- **Definition:** Boundaries in previous 12 deliveries
- **Formula:** shifted rolling count
- **Source:** match_state.add_momentum_features
- **Available before delivery:** YES
- **Leakage risk:** Shifted
- **Missing-data handling:** NaN early
- **Relevance:** Recent boundary flow

## `last12_dot_balls`
- **Definition:** Dots in previous 12 deliveries
- **Formula:** shifted rolling count
- **Source:** match_state.add_momentum_features
- **Available before delivery:** YES
- **Leakage risk:** Shifted
- **Missing-data handling:** NaN early
- **Relevance:** Recent dot pressure

## `last12_run_rate`
- **Definition:** Recent scoring rate (12 deliveries)
- **Formula:** last12_runs / (last12_legal/6)
- **Source:** match_state.add_momentum_features
- **Available before delivery:** YES
- **Leakage risk:** Shifted
- **Missing-data handling:** NaN early in innings
- **Relevance:** Recent momentum

## `last12_wickets`
- **Definition:** Team wickets in previous 12 deliveries
- **Formula:** shifted rolling count
- **Source:** match_state.add_momentum_features
- **Available before delivery:** YES
- **Leakage risk:** Shifted; excludes current ball
- **Missing-data handling:** NaN in first 12 deliveries
- **Relevance:** Collapse detection

## `last18_wickets`
- **Definition:** Team wickets in previous 18 deliveries
- **Formula:** shifted rolling count
- **Source:** match_state.add_momentum_features
- **Available before delivery:** YES
- **Leakage risk:** Shifted; excludes current ball
- **Missing-data handling:** NaN in first 18 deliveries
- **Relevance:** Collapse detection

## `last24_wickets`
- **Definition:** Team wickets in previous 24 deliveries
- **Formula:** shifted rolling count
- **Source:** match_state.add_momentum_features
- **Available before delivery:** YES
- **Leakage risk:** Shifted; excludes current ball
- **Missing-data handling:** NaN in first 24 deliveries
- **Relevance:** Collapse detection

## `last6_run_rate`
- **Definition:** Recent scoring rate (6 deliveries)
- **Formula:** last6_runs / (last6_legal/6)
- **Source:** match_state.add_momentum_features
- **Available before delivery:** YES
- **Leakage risk:** Shifted; excludes current ball
- **Missing-data handling:** NaN in first 6 deliveries
- **Relevance:** Recent momentum

## `last6_wickets`
- **Definition:** Team wickets in previous 6 deliveries
- **Formula:** shifted rolling count
- **Source:** match_state.add_momentum_features
- **Available before delivery:** YES
- **Leakage risk:** Shifted by one row; excludes current ball
- **Missing-data handling:** NaN in first 6 deliveries of innings
- **Relevance:** Collapse detection

## `legal_balls_remaining`
- **Definition:** Legal balls remaining
- **Formula:** scheduled_legal_balls - legal_balls_before
- **Source:** match_state.add_pre_delivery_state
- **Available before delivery:** YES
- **Leakage risk:** None (DLS-aware)
- **Missing-data handling:** n/a
- **Relevance:** Time pressure

## `match_balance_proxy`
- **Definition:** How close the match is, without a WP model (Phase-4 proxy)
- **Formula:** chase: 1-|2*target_progress-1|; innings 1: 1-|2*(expected_final_score/venue_season_par)-1|, clipped [0,1]
- **Source:** match_pressure_features
- **Available before delivery:** YES
- **Leakage risk:** Proxy only; replaced by calibrated WP balance in Phase 5
- **Missing-data handling:** NaN when target/par unavailable
- **Relevance:** Pressure requires the outcome to still be in balance

## `matchup_difficulty_index`
- **Definition:** Cell difficulty relative to league average, 0-centred
- **Formula:** matchup_lboro_runs_per_ball - global runs per legal ball
- **Source:** matchup_features
- **Available before delivery:** YES
- **Leakage risk:** Leave-one-season-out
- **Missing-data handling:** 0 when unavailable
- **Relevance:** Single-axis matchup difficulty

## `matchup_lboro_runs_per_ball`
- **Definition:** League runs-per-legal-ball for this hand x bowler-class cell
- **Formula:** EB-shrunk mean: (S_cell - S_own_season + k*global) / (N_cell - N_own_season + k), leave-one-season-out
- **Source:** matchup_features
- **Available before delivery:** YES
- **Leakage risk:** Leave-one-season-out: the delivery's own season never contributes to its baseline; no player-specific stat involved
- **Missing-data handling:** Falls back to hand-level, then global mean
- **Relevance:** Pre-delivery difficulty of the handedness matchup

## `matchup_lboro_wicket_rate`
- **Definition:** League wicket probability for this hand x bowler-class cell
- **Formula:** as above on bowler-wicket frequency per legal ball
- **Source:** matchup_features
- **Available before delivery:** YES
- **Leakage risk:** Leave-one-season-out
- **Missing-data handling:** Fallback hierarchy
- **Relevance:** Dismissal likelihood of the matchup

## `matchup_style_runs_per_ball`
- **Definition:** League runs-per-legal-ball for hand x bowling-style group
- **Formula:** as matchup_lboro_runs_per_ball on style groups
- **Source:** matchup_features
- **Available before delivery:** YES
- **Leakage risk:** Leave-one-season-out
- **Missing-data handling:** Fallback to class-level
- **Relevance:** Finer matchup axis (fast vs medium vs spin variants)

## `matchup_style_wicket_rate`
- **Definition:** League wicket probability for hand x bowling-style group
- **Formula:** as matchup_lboro_wicket_rate on style groups
- **Source:** matchup_features
- **Available before delivery:** YES
- **Leakage risk:** Leave-one-season-out
- **Missing-data handling:** Fallback to class-level
- **Relevance:** Finer matchup dismissal axis

## `over_number_before`
- **Definition:** 1-based over index containing this delivery
- **Formula:** floor(legal_balls_before / 6) + 1
- **Source:** match_state.add_pre_delivery_state
- **Available before delivery:** YES
- **Leakage risk:** None
- **Missing-data handling:** n/a
- **Relevance:** Fine-grained phase control for expected-outcome models

## `over_or_around_wicket`
- **Definition:** Bowling action position
- **Formula:** raw over_or_around
- **Source:** ball_by_ball.csv
- **Available before delivery:** YES
- **Leakage risk:** None
- **Missing-data handling:** 'unknown'
- **Relevance:** Angle information

## `overs_remaining`
- **Definition:** Overs left for the bowling side
- **Formula:** legal_balls_remaining/6
- **Source:** match_state.add_pre_delivery_state (re-exposed)
- **Available before delivery:** YES
- **Leakage risk:** None
- **Missing-data handling:** n/a
- **Relevance:** Time to take wickets

## `pace_or_spin`
- **Definition:** Bowler class
- **Formula:** raw pace_or_spin (Pace/Spin)
- **Source:** ball_by_ball.csv
- **Available before delivery:** YES
- **Leakage risk:** None
- **Missing-data handling:** 'unknown' when blank
- **Relevance:** Matchup physics

## `par_run_rate`
- **Definition:** Venue-season expected run rate
- **Formula:** venue_season_par / scheduled overs
- **Source:** match_state.add_par_state
- **Available before delivery:** YES
- **Leakage risk:** Venue par shrinkage k=8, leave-one-match-out
- **Missing-data handling:** Filled with global par when venue unseen
- **Relevance:** Baseline scoring environment

## `partnership_balls_before`
- **Definition:** Legal balls in current partnership
- **Formula:** exclusive cumsum within block
- **Source:** player_state.add_striker_state
- **Available before delivery:** YES
- **Leakage risk:** As partnership_runs_before
- **Missing-data handling:** 0 after wicket
- **Relevance:** Partnership stability / rebuild detection

## `partnership_runs_before`
- **Definition:** Runs in current partnership before this ball
- **Formula:** exclusive cumsum within partnership block
- **Source:** player_state.add_striker_state
- **Available before delivery:** YES
- **Leakage risk:** Block = pre-delivery wicket count
- **Missing-data handling:** 0 after wicket
- **Relevance:** Partnership stability

## `required_boundaries_estimate`
- **Definition:** Sixes needed to close the chase
- **Formula:** ceil(runs_required/6)
- **Source:** match_state.add_chase_state
- **Available before delivery:** YES
- **Leakage risk:** None
- **Missing-data handling:** NaN in innings 1
- **Relevance:** Upper bound on boundary burden

## `required_run_rate`
- **Definition:** Chase: required run rate
- **Formula:** runs_required / (balls_remaining/6)
- **Source:** match_state.add_chase_state
- **Available before delivery:** YES
- **Leakage risk:** None
- **Missing-data handling:** NaN in innings 1
- **Relevance:** Primary chase steepness measure

## `resources_remaining`
- **Definition:** Simple resource proxy: balls left x wickets left
- **Formula:** (legal_balls_remaining/scheduled) * (wickets_in_hand/10), clipped [0,1]
- **Source:** batting_pressure_features
- **Available before delivery:** YES
- **Leakage risk:** Proxy — official DLS resource tables are not supplied in the pack
- **Missing-data handling:** n/a
- **Relevance:** Unified scarcity measure across chases and first innings

## `rrr_crr_gap`
- **Definition:** Chase: required minus current run rate
- **Formula:** required_run_rate - current_run_rate
- **Source:** match_state.add_chase_state
- **Available before delivery:** YES
- **Leakage risk:** None
- **Missing-data handling:** NaN in innings 1
- **Relevance:** Positive gap = escalating chase pressure

## `runs_required_before`
- **Definition:** Chase: runs still needed
- **Formula:** target - score_before (clipped at 0)
- **Source:** match_state.add_chase_state
- **Available before delivery:** YES
- **Leakage risk:** None — target known at innings break
- **Missing-data handling:** NaN in innings 1
- **Relevance:** Core chase difficulty quantity

## `scenario_ACCELERATION_REQUIRED`
- **Definition:** Scenario label ACCELERATION_REQUIRED (multi-label boolean)
- **Formula:** rrr_crr_gap>2 (chase) OR (innings1 AND over>=10 AND current_run_rate<0.85*par_run_rate)
- **Source:** scenario_features
- **Available before delivery:** YES
- **Leakage risk:** All inputs are pre-delivery state; HIGH_LEVERAGE/MATCH_ON_LINE use pre-delivery proxies pending the Phase-5 WP model
- **Missing-data handling:** False when inputs unavailable
- **Relevance:** Explicit pressure regimes for ratings, dossiers and scenario diversity

## `scenario_ANCHOR_UNDER_PAR`
- **Definition:** Scenario label ANCHOR_UNDER_PAR (multi-label boolean)
- **Formula:** score_vs_par<=-15 AND batter_is_set
- **Source:** scenario_features
- **Available before delivery:** YES
- **Leakage risk:** All inputs are pre-delivery state; HIGH_LEVERAGE/MATCH_ON_LINE use pre-delivery proxies pending the Phase-5 WP model
- **Missing-data handling:** False when inputs unavailable
- **Relevance:** Explicit pressure regimes for ratings, dossiers and scenario diversity

## `scenario_BOUNDARY_DROUGHT`
- **Definition:** Scenario label BOUNDARY_DROUGHT (multi-label boolean)
- **Formula:** balls_since_last_boundary>=12
- **Source:** scenario_features
- **Available before delivery:** YES
- **Leakage risk:** All inputs are pre-delivery state; HIGH_LEVERAGE/MATCH_ON_LINE use pre-delivery proxies pending the Phase-5 WP model
- **Missing-data handling:** False when inputs unavailable
- **Relevance:** Explicit pressure regimes for ratings, dossiers and scenario diversity

## `scenario_BOUNDARY_SUPPRESSION`
- **Definition:** Scenario label BOUNDARY_SUPPRESSION (multi-label boolean)
- **Formula:** bowler_boundary_streak_before==0 AND balls_since_last_boundary>=8
- **Source:** scenario_features
- **Available before delivery:** YES
- **Leakage risk:** All inputs are pre-delivery state; HIGH_LEVERAGE/MATCH_ON_LINE use pre-delivery proxies pending the Phase-5 WP model
- **Missing-data handling:** False when inputs unavailable
- **Relevance:** Explicit pressure regimes for ratings, dossiers and scenario diversity

## `scenario_DEATH_DEFENCE`
- **Definition:** Scenario label DEATH_DEFENCE (multi-label boolean)
- **Formula:** over>=16
- **Source:** scenario_features
- **Available before delivery:** YES
- **Leakage risk:** All inputs are pre-delivery state; HIGH_LEVERAGE/MATCH_ON_LINE use pre-delivery proxies pending the Phase-5 WP model
- **Missing-data handling:** False when inputs unavailable
- **Relevance:** Explicit pressure regimes for ratings, dossiers and scenario diversity

## `scenario_DEATH_OVERS_PRESSURE`
- **Definition:** Scenario label DEATH_OVERS_PRESSURE (multi-label boolean)
- **Formula:** over>=16
- **Source:** scenario_features
- **Available before delivery:** YES
- **Leakage risk:** All inputs are pre-delivery state; HIGH_LEVERAGE/MATCH_ON_LINE use pre-delivery proxies pending the Phase-5 WP model
- **Missing-data handling:** False when inputs unavailable
- **Relevance:** Explicit pressure regimes for ratings, dossiers and scenario diversity

## `scenario_EARLY_COLLAPSE_RESCUE`
- **Definition:** Scenario label EARLY_COLLAPSE_RESCUE (multi-label boolean)
- **Formula:** over<=6 AND (wickets_before>=2 OR last6_wickets>=1)
- **Source:** scenario_features
- **Available before delivery:** YES
- **Leakage risk:** All inputs are pre-delivery state; HIGH_LEVERAGE/MATCH_ON_LINE use pre-delivery proxies pending the Phase-5 WP model
- **Missing-data handling:** False when inputs unavailable
- **Relevance:** Explicit pressure regimes for ratings, dossiers and scenario diversity

## `scenario_FINAL_OVER_DEFENCE`
- **Definition:** Scenario label FINAL_OVER_DEFENCE (multi-label boolean)
- **Formula:** legal_balls_remaining<=6
- **Source:** scenario_features
- **Available before delivery:** YES
- **Leakage risk:** All inputs are pre-delivery state; HIGH_LEVERAGE/MATCH_ON_LINE use pre-delivery proxies pending the Phase-5 WP model
- **Missing-data handling:** False when inputs unavailable
- **Relevance:** Explicit pressure regimes for ratings, dossiers and scenario diversity

## `scenario_FINAL_OVER_PRESSURE`
- **Definition:** Scenario label FINAL_OVER_PRESSURE (multi-label boolean)
- **Formula:** legal_balls_remaining<=6
- **Source:** scenario_features
- **Available before delivery:** YES
- **Leakage risk:** All inputs are pre-delivery state; HIGH_LEVERAGE/MATCH_ON_LINE use pre-delivery proxies pending the Phase-5 WP model
- **Missing-data handling:** False when inputs unavailable
- **Relevance:** Explicit pressure regimes for ratings, dossiers and scenario diversity

## `scenario_FINISHER_PRESSURE`
- **Definition:** Scenario label FINISHER_PRESSURE (multi-label boolean)
- **Formula:** over>=16 AND batter_role in {finisher, middle}
- **Source:** scenario_features
- **Available before delivery:** YES
- **Leakage risk:** All inputs are pre-delivery state; HIGH_LEVERAGE/MATCH_ON_LINE use pre-delivery proxies pending the Phase-5 WP model
- **Missing-data handling:** False when inputs unavailable
- **Relevance:** Explicit pressure regimes for ratings, dossiers and scenario diversity

## `scenario_HIGH_LEVERAGE`
- **Definition:** Scenario label HIGH_LEVERAGE (multi-label boolean)
- **Formula:** match_balance_proxy>=0.6 (WP-based in Phase 5)
- **Source:** scenario_features
- **Available before delivery:** YES
- **Leakage risk:** All inputs are pre-delivery state; HIGH_LEVERAGE/MATCH_ON_LINE use pre-delivery proxies pending the Phase-5 WP model
- **Missing-data handling:** False when inputs unavailable
- **Relevance:** Explicit pressure regimes for ratings, dossiers and scenario diversity

## `scenario_HIGH_LEVERAGE_DELIVERY`
- **Definition:** Scenario label HIGH_LEVERAGE_DELIVERY (multi-label boolean)
- **Formula:** match_balance_proxy>=0.6
- **Source:** scenario_features
- **Available before delivery:** YES
- **Leakage risk:** All inputs are pre-delivery state; HIGH_LEVERAGE/MATCH_ON_LINE use pre-delivery proxies pending the Phase-5 WP model
- **Missing-data handling:** False when inputs unavailable
- **Relevance:** Explicit pressure regimes for ratings, dossiers and scenario diversity

## `scenario_HIGH_RRR_CHASE`
- **Definition:** Scenario label HIGH_RRR_CHASE (multi-label boolean)
- **Formula:** chase AND required_run_rate>=10
- **Source:** scenario_features
- **Available before delivery:** YES
- **Leakage risk:** All inputs are pre-delivery state; HIGH_LEVERAGE/MATCH_ON_LINE use pre-delivery proxies pending the Phase-5 WP model
- **Missing-data handling:** False when inputs unavailable
- **Relevance:** Explicit pressure regimes for ratings, dossiers and scenario diversity

## `scenario_HIGH_RRR_DEFENCE`
- **Definition:** Scenario label HIGH_RRR_DEFENCE (multi-label boolean)
- **Formula:** chase AND required_run_rate>=10
- **Source:** scenario_features
- **Available before delivery:** YES
- **Leakage risk:** All inputs are pre-delivery state; HIGH_LEVERAGE/MATCH_ON_LINE use pre-delivery proxies pending the Phase-5 WP model
- **Missing-data handling:** False when inputs unavailable
- **Relevance:** Explicit pressure regimes for ratings, dossiers and scenario diversity

## `scenario_LOW_WICKET_ATTACK`
- **Definition:** Scenario label LOW_WICKET_ATTACK (multi-label boolean)
- **Formula:** wickets_in_hand<=3
- **Source:** scenario_features
- **Available before delivery:** YES
- **Leakage risk:** All inputs are pre-delivery state; HIGH_LEVERAGE/MATCH_ON_LINE use pre-delivery proxies pending the Phase-5 WP model
- **Missing-data handling:** False when inputs unavailable
- **Relevance:** Explicit pressure regimes for ratings, dossiers and scenario diversity

## `scenario_LOW_WICKET_RESOURCE_CHASE`
- **Definition:** Scenario label LOW_WICKET_RESOURCE_CHASE (multi-label boolean)
- **Formula:** chase AND wickets_in_hand<=3
- **Source:** scenario_features
- **Available before delivery:** YES
- **Leakage risk:** All inputs are pre-delivery state; HIGH_LEVERAGE/MATCH_ON_LINE use pre-delivery proxies pending the Phase-5 WP model
- **Missing-data handling:** False when inputs unavailable
- **Relevance:** Explicit pressure regimes for ratings, dossiers and scenario diversity

## `scenario_MATCH_ON_LINE`
- **Definition:** Scenario label MATCH_ON_LINE (multi-label boolean)
- **Formula:** match_balance_proxy>=0.75 AND (chase OR over>=15) (WP-based in Phase 5)
- **Source:** scenario_features
- **Available before delivery:** YES
- **Leakage risk:** All inputs are pre-delivery state; HIGH_LEVERAGE/MATCH_ON_LINE use pre-delivery proxies pending the Phase-5 WP model
- **Missing-data handling:** False when inputs unavailable
- **Relevance:** Explicit pressure regimes for ratings, dossiers and scenario diversity

## `scenario_MIDDLE_ORDER_RESCUE`
- **Definition:** Scenario label MIDDLE_ORDER_RESCUE (multi-label boolean)
- **Formula:** 7<=over<=15 AND wickets_before>=4
- **Source:** scenario_features
- **Available before delivery:** YES
- **Leakage risk:** All inputs are pre-delivery state; HIGH_LEVERAGE/MATCH_ON_LINE use pre-delivery proxies pending the Phase-5 WP model
- **Missing-data handling:** False when inputs unavailable
- **Relevance:** Explicit pressure regimes for ratings, dossiers and scenario diversity

## `scenario_MIDDLE_OVERS_CONTROL`
- **Definition:** Scenario label MIDDLE_OVERS_CONTROL (multi-label boolean)
- **Formula:** 7<=over<=15
- **Source:** scenario_features
- **Available before delivery:** YES
- **Leakage risk:** All inputs are pre-delivery state; HIGH_LEVERAGE/MATCH_ON_LINE use pre-delivery proxies pending the Phase-5 WP model
- **Missing-data handling:** False when inputs unavailable
- **Relevance:** Explicit pressure regimes for ratings, dossiers and scenario diversity

## `scenario_NEW_BATTER_AFTER_WICKET`
- **Definition:** Scenario label NEW_BATTER_AFTER_WICKET (multi-label boolean)
- **Formula:** batter_is_new AND entering_after_wicket
- **Source:** scenario_features
- **Available before delivery:** YES
- **Leakage risk:** All inputs are pre-delivery state; HIGH_LEVERAGE/MATCH_ON_LINE use pre-delivery proxies pending the Phase-5 WP model
- **Missing-data handling:** False when inputs unavailable
- **Relevance:** Explicit pressure regimes for ratings, dossiers and scenario diversity

## `scenario_NEW_BATTER_PRESSURE`
- **Definition:** Scenario label NEW_BATTER_PRESSURE (multi-label boolean)
- **Formula:** batter_is_new
- **Source:** scenario_features
- **Available before delivery:** YES
- **Leakage risk:** All inputs are pre-delivery state; HIGH_LEVERAGE/MATCH_ON_LINE use pre-delivery proxies pending the Phase-5 WP model
- **Missing-data handling:** False when inputs unavailable
- **Relevance:** Explicit pressure regimes for ratings, dossiers and scenario diversity

## `scenario_PARTNERSHIP_REBUILD`
- **Definition:** Scenario label PARTNERSHIP_REBUILD (multi-label boolean)
- **Formula:** partnership_balls_before<=6 AND wickets_before>=2
- **Source:** scenario_features
- **Available before delivery:** YES
- **Leakage risk:** All inputs are pre-delivery state; HIGH_LEVERAGE/MATCH_ON_LINE use pre-delivery proxies pending the Phase-5 WP model
- **Missing-data handling:** False when inputs unavailable
- **Relevance:** Explicit pressure regimes for ratings, dossiers and scenario diversity

## `scenario_POWERPLAY_PRESSURE`
- **Definition:** Scenario label POWERPLAY_PRESSURE (multi-label boolean)
- **Formula:** over<=6
- **Source:** scenario_features
- **Available before delivery:** YES
- **Leakage risk:** All inputs are pre-delivery state; HIGH_LEVERAGE/MATCH_ON_LINE use pre-delivery proxies pending the Phase-5 WP model
- **Missing-data handling:** False when inputs unavailable
- **Relevance:** Explicit pressure regimes for ratings, dossiers and scenario diversity

## `scenario_RAPID_COLLAPSE`
- **Definition:** Scenario label RAPID_COLLAPSE (multi-label boolean)
- **Formula:** last6_wickets>=2 OR last12_wickets>=3
- **Source:** scenario_features
- **Available before delivery:** YES
- **Leakage risk:** All inputs are pre-delivery state; HIGH_LEVERAGE/MATCH_ON_LINE use pre-delivery proxies pending the Phase-5 WP model
- **Missing-data handling:** False when inputs unavailable
- **Relevance:** Explicit pressure regimes for ratings, dossiers and scenario diversity

## `scenario_SET_BATTER_DISMISSAL`
- **Definition:** Scenario label SET_BATTER_DISMISSAL (multi-label boolean)
- **Formula:** batter_is_set AND batter_runs_before>=25
- **Source:** scenario_features
- **Available before delivery:** YES
- **Leakage risk:** All inputs are pre-delivery state; HIGH_LEVERAGE/MATCH_ON_LINE use pre-delivery proxies pending the Phase-5 WP model
- **Missing-data handling:** False when inputs unavailable
- **Relevance:** Explicit pressure regimes for ratings, dossiers and scenario diversity

## `scenario_WICKET_DROUGHT_BREAKTHROUGH`
- **Definition:** Scenario label WICKET_DROUGHT_BREAKTHROUGH (multi-label boolean)
- **Formula:** wicket_drought_severity>=0.5
- **Source:** scenario_features
- **Available before delivery:** YES
- **Leakage risk:** All inputs are pre-delivery state; HIGH_LEVERAGE/MATCH_ON_LINE use pre-delivery proxies pending the Phase-5 WP model
- **Missing-data handling:** False when inputs unavailable
- **Relevance:** Explicit pressure regimes for ratings, dossiers and scenario diversity

## `scenario_count`
- **Definition:** Number of scenario labels active on the delivery
- **Formula:** sum of all scenario labels
- **Source:** scenario_features
- **Available before delivery:** YES
- **Leakage risk:** None beyond the labels
- **Missing-data handling:** 0
- **Relevance:** Scenario stacking intensity

## `scenario_diversity_index`
- **Definition:** Per-match entropy of active scenario labels (delivery-level)
- **Formula:** -sum p_i log p_i over scenario labels within the match so far (pre-delivery)
- **Source:** scenario_features
- **Available before delivery:** YES
- **Leakage risk:** Computed over PRIOR deliveries only
- **Missing-data handling:** 0 at match start
- **Relevance:** Whether the player has faced varied pressure situations

## `score_deficit_vs_par`
- **Definition:** Alias of score_vs_par
- **Formula:** score_before - expected_score_at_state
- **Source:** match_state.add_par_state
- **Available before delivery:** YES
- **Leakage risk:** As score_vs_par
- **Missing-data handling:** As score_vs_par
- **Relevance:** Batting-first difficulty

## `score_vs_par`
- **Definition:** Runs above/below state-expected score
- **Formula:** score_before - expected_score_at_state (LOO curve)
- **Source:** match_state.add_par_state
- **Available before delivery:** YES
- **Leakage risk:** LOO curve excludes own innings; venue par leave-one-match-out
- **Missing-data handling:** Fallback hierarchy documented in match_state
- **Relevance:** Batting-first difficulty vs historical state baselines

## `season`
- **Definition:** Competition season label
- **Formula:** competitions.season
- **Source:** competitions.csv
- **Available before delivery:** YES
- **Leakage risk:** None
- **Missing-data handling:** n/a
- **Relevance:** Era effects in scoring environment; walk-forward split key

## `severity_proxy`
- **Definition:** Batting-side resource stress, 0-1 (Phase-4 proxy)
- **Formula:** 0.5*clip(RRR/14 or 0,0,1) + 0.3*(1 - wickets_in_hand/10) + 0.2*(over>=16)
- **Source:** match_pressure_features
- **Available before delivery:** YES
- **Leakage risk:** Proxy; Phase 5 replaces with model-based severity
- **Missing-data handling:** NaN when chase components unavailable
- **Relevance:** Resource scarcity is the core of situational pressure

## `striker_dismissal_risk_career`
- **Definition:** Career-to-date dismissal rate
- **Formula:** dismissals/legal balls over PRIOR matches (>=30 balls)
- **Source:** player_state.add_career_dismissal_risk
- **Available before delivery:** YES
- **Leakage risk:** Shifted by one match; prior seasons only in walk-forward
- **Missing-data handling:** NaN until 30 career legal balls
- **Relevance:** Prior vulnerability

## `striker_runs`
- **Definition:** Striker runs so far
- **Formula:** batter_runs_before
- **Source:** player_state.add_striker_state (re-exposed)
- **Available before delivery:** YES
- **Leakage risk:** None
- **Missing-data handling:** 0 at entry
- **Relevance:** Threat level of the set batter

## `striker_setness`
- **Definition:** Striker balls faced (setness)
- **Formula:** batter_balls_faced_before
- **Source:** player_state.add_striker_state (re-exposed)
- **Available before delivery:** YES
- **Leakage risk:** None
- **Missing-data handling:** 0 at entry
- **Relevance:** Set batters are the drought's cause

## `target_progress`
- **Definition:** Fraction of target scored
- **Formula:** score_before / target
- **Source:** match_state.add_chase_state
- **Available before delivery:** YES
- **Leakage risk:** None
- **Missing-data handling:** NaN in innings 1
- **Relevance:** Chase position on the win-probability curve

## `toss_winner_bats_first`
- **Definition:** Toss winner chose to bat (proxy for pitch conditions)
- **Formula:** toss_win_team_id == batting_team_id and toss_opted=='Batting'
- **Source:** matches.csv
- **Available before delivery:** YES
- **Leakage risk:** None
- **Missing-data handling:** NaN when toss fields missing (15 matches)
- **Relevance:** Toss decisions encode expected surface behaviour

## `venue_id`
- **Definition:** Venue identifier
- **Formula:** matches.venue_id
- **Source:** matches.csv
- **Available before delivery:** YES
- **Leakage risk:** None
- **Missing-data handling:** n/a
- **Relevance:** Venue par / environment

## `wicket_drought_severity`
- **Definition:** Drought relative to expectation, 0-1
- **Formula:** clip(drought / (2/league_wicket_rate), 0, 1) with league_wicket_rate a FIXED pre-computed constant
- **Source:** bowling_pressure_features
- **Available before delivery:** YES
- **Leakage risk:** None — normalizer is a fixed constant (10392/178000), never frame-derived
- **Missing-data handling:** 0 before first wicket
- **Relevance:** Scales drought by how unusual it is

## `wicket_ends_drought`
- **Definition:** Would a wicket here end a long drought (context flag)
- **Formula:** wicket_drought_severity >= 0.5
- **Source:** bowling_pressure_features
- **Available before delivery:** YES
- **Leakage risk:** None
- **Missing-data handling:** False
- **Relevance:** Marks drought-ending breakthrough opportunities

## `wickets_in_hand`
- **Definition:** Wickets remaining
- **Formula:** 10 - wickets_before
- **Source:** match_state.add_pre_delivery_state
- **Available before delivery:** YES
- **Leakage risk:** None
- **Missing-data handling:** n/a
- **Relevance:** Resource availability

## `wickets_lost_vs_expected`
- **Definition:** Wickets lost relative to state-expected wickets
- **Formula:** wickets_before - LOO mean wickets at (balls_elapsed_over, innings type)
- **Source:** batting_pressure_features
- **Available before delivery:** YES
- **Leakage risk:** LOO curve excludes the current innings
- **Missing-data handling:** Fallback to global wicket curve
- **Relevance:** Positive = batting side behind the typical wicket-loss schedule

## `wickets_remaining`
- **Definition:** Wickets the batting side has left
- **Formula:** 10 - wickets_before
- **Source:** match_state.add_pre_delivery_state (re-exposed)
- **Available before delivery:** YES
- **Leakage risk:** None
- **Missing-data handling:** n/a
- **Relevance:** Fewer wickets = attacking opportunity

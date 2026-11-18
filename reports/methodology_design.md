# Methodology Design — SupeRR Selector Season 2

_Phase 2 research & design document. Written before any ranking implementation.
Every design decision below cites either (A) established analytics literature,
(B) an inspected independent submission, (C) this project's own reasoning, or
(D) an explicitly rejected approach. Companion artefacts: the Phase-1 audit
(`data_audit_decisions.md`) and the implementation that follows this spec._

---

## 0. Research synthesis

### 0.1 Inspected independent submissions (Category B)

| # | Submission | Core pressure definition | Performance engine | Notable strengths | Notable weaknesses |
|---|---|---|---|---|---|
| 1 | **Rishi Saraf** (`RR-Round1-Submission-Code`) | "How far the next ball could swing the match" — leverage vs balls at same stage; top 20% = HP (chases hold 60%) | Ridge-regression context-adjusted ball impact; EB-shrunk consistency; match Elo form; `level_lcb85` lower-bound scoring | Backwards win-chance recalibration; split-halves reliability check; 25-setting sensitivity sweep; LCB (lower-confidence-bound) scoring bakes uncertainty into the rank metric | Innings-specific backwards DP per match is expensive; recency "form" term risks conflating schedule strength with clutch |
| 2 | **Saket** (`saket0x07/SupeRR`) | PI = 0.50·leverage + 0.30·closeness + 0.20·stress; HP = top 25% of deliveries | GBDT xRuns/xOut; dynamic wicket value λ(balls_left, wickets_lost); BVOE = (R−xR) − λ(O−xO); residual clutch c2 = y_HP − (α+β·y_norm); EB shrinkage k≈119/111 balls | Dual WP models (chase GBDT + 1st-innings structural par); REML-estimated shrinkage constants; 13-scenario multiverse; out-of-sample persistence test | 1st-innings WP model is weak (AUC 0.642) and acknowledged; composite weights (40/25/20/15) are asserted, not tuned |
| 3 | **Halla Bol Pressure Index** (Lohithaswin V) | OR-of-triggers: LI≥1.5 (top 15% swing), RRR≥10 chases, collapse windows (2 wkts in PP / 4 by over 12), death (<36 runs) | XGBoost xR on 18 features; WPA-based factors (counter-attack, crisis rescue, death finishing); runs-saved by phase | Rich scenario taxonomy; opposition tiering awareness; phase-level bowling audit | Hard-coded named-country opposition tiers (1.25x/0.40x); non-compensatory "flaw penalties" with magic multipliers (7.5x, 3.0x); ≥750-ball cutoff; many asserted weights |
| 4 | **Aniket Kumar** (`aniketku23.github.io/SuperrSelector`) | CPI = situational formula (50+(RRR−8.5)·7.5)·wicket-factor + death stress + closeness; hybrid 70% CPI + 30% leverage | Logistic WP (AUC 0.890); WPA totals; pressure-vs-normal ΔSR/ΔEcon; EB shrinkage K=45 | Explicit hybrid weighting; transparent simulator; WPA reported per player | 47% of all deliveries labelled high-pressure (94,841/200,999) — pressure loses meaning; K=45 shrinkage asserted; knockout/IPL multipliers (1.1–1.2x) hard-coded |
| 5 | **theic-alt** (`superr-selector`) | Rule-based: chase — ≤60 legal balls left AND (RRR≥9.5 OR wkts≤4 OR RRR−CRR≥2.5); 1st inns — ≤30 balls & ≥5 wkts (bat) / CRR≥8 (bowl) | Z-score composite (SR, boundary rate, dismissal rate, uplifts); reliability = n/(n+60); min-max scale to 0–100 | Fully transparent rules; clean separation of batting/bowling pressure; on-demand calculator | Reliability **multiplies** the raw score (shrinks toward 0, not toward the pool mean — systematically biases small-sample players downward); min-max scaling is outlier-driven; thresholds untested |

Cross-submission consensus (B):
- Every serious submission uses a **continuous pressure index**, then a quantile threshold.
- Every serious submission separates **win-probability leverage** from **situation stress**.
- The strongest submissions (1, 2) combine **expected-outcome baselines** with **dynamic wicket valuation** and **empirical shrinkage**.
- The weakest methodological elements across submissions are **asserted constants** (weights, multipliers, cutoffs) that are never sensitivity-tested, and **reliability handling that distorts calibration** (submission 5) or **thresholds so loose that "pressure" covers half of all cricket** (submission 4).

### 0.2 Established analytical concepts (Category A)

- **Win probability (WP) models** — logistic regression or gradient boosting on match state; standard in soccer/baseball/cricket analytics. Calibration (reliability of predicted probabilities) matters more than raw AUC for downstream use.
- **Win Probability Added (WPA)** — the change in win probability attributable to a play; the natural outcome-level value currency when an outcome exists.
- **Leverage index** (Tango, *The Book*, baseball) — LI = expected absolute WP swing of a situation ÷ average swing; normalizes "how much can this moment matter" across game state.
- **Run expectancy / expected runs (xR)** — average runs from a state (Reyes/leverage-aware variants); here, learned per-delivery.
- **Expected wickets (xW)** — probability of dismissal from state, conditional on bowling quality; the bowling analogue.
- **Contextual performance** (above/below expectation) rather than raw averages — prevents talent-free context (flat pitches, weak opponents) from inflating ratings.
- **Dynamic wicket valuation** — the run-value of a wicket varies with balls/wickets remaining (a 9th wicket with 2 balls left is nearly worthless; a 3rd wicket in over 8 is expensive).
- **Clutch / pressure performance** — persistence of performance deltas in high-leverage states; empirically small in most sports, so **uncertainty quantification is not optional**.
- **Empirical-Bayes / hierarchical shrinkage** — rate estimates shrink toward a population prior with strength learned from the data (method of moments, REML, or full hierarchical fitting).
- **Match-level (cluster) bootstrap** — deliveries within a match are not independent; resampling *matches*, not balls, preserves within-match correlation.
- **Consistency metrics** — positive-match share, split-half reliability; distinguish stable skill from variance.
- **Role adjustment** — comparing finishers to openers on the same axis is biased; role stratification or role-conditioned baselines are standard practice.
- **Opposition adjustment** — plus-minus / iterative ratings tradition; conditioning expectation models on opponent quality.

### 0.3 Methods rejected or restricted (Category D)

| Rejected method | Why |
|---|---|
| Hard-coded named opposition tiers ("minnows get 0.40x") | Subjective country lists invite bias; we estimate opponent strength from data instead |
| Non-compensatory "flaw penalties" with magic multipliers | A ranking forced by penalties encodes analyst taste, not evidence; penalties are folded into the value framework instead |
| Multiplicative reliability (score × n/(n+k)) | Shrinks toward zero instead of the pool mean; miscalibrates rankings (submission 5 pattern) |
| Min-max scaling to 0–100 | Rank order becomes hostage to single outliers |
| Treating every ball as independent | Within-match correlation inflates certainty; use cluster bootstrap |
| Random ball-level train/test splits | Serial correlation + near-duplicate states leak between train and test; use season walk-forward |
| Labelling ~half of deliveries "high pressure" (submission 4 pattern) | If 47% of cricket is pressure, the label carries no information; threshold chosen by quantile grid + validation, not assumption |
| Reputation, team, or name-based features anywhere in scoring | Explicit project constraint; also bad science |
| Using post-delivery information in pre-delivery pressure | Leakage; the feature taxonomy (§3) enforces this |
| Forcing deep learning | Neural nets are *evaluated* (§9) and included only if they win |

---

## 1. Research questions — position statements

**Q1. How should pressure be defined?**
Pressure is a property of the *match state before a delivery*: the joint condition that (a) the match outcome is still substantially in balance, (b) the situation can swing materially on this delivery (leverage), and (c) the batting side faces situational stress (resource scarcity: wickets/runs/time). This mirrors the consensus of submissions 1–2 (B) and the leverage concept (A). We formalize pressure as a continuous index on [0,1]; "high pressure" is a quantile threshold on that index, never an assumed constant.

**Q2. Continuous measurement?**
`PressureIndex ∈ [0,1]` built from three normalised components: balance `B = 1 − |2·WP − 1|` (closeness of the match), leverage `L = E|ΔWP| / mean(E|ΔWP|)` scaled via logistic squashing, and severity `S` (resource stress, 0–1). Primary definition (Model C) is `P = w_B·B + w_L·logistic(L−1) + w_S·S` with weights from a small validated grid. Categories are deciles of `P`.

**Q3. Batting-first pressure?**
Innings 1 has no chase calculus. WP for innings 1 is defined as *probability the batting side reaches a competitive total*, operationalized via the structural-par approach of submission 2 (B): project the final total from state + venue-era par; WP₁ = P(final total > par score) via a calibrated model. Severity comes from wickets lost vs phase expectation and collapse state. This avoids the weak first-innings WP models that hurt submission 2 (their AUC 0.642) by not pretending to model "which team wins" from mid-innings-1 state alone.

**Q4. Chase pressure?**
WP₂ from a calibrated gradient-boosting model on pre-delivery state (runs needed, balls left, wickets left, RRR, par context, batter/bowler quality terms). Chase states are where WP is most reliable (A, B).

**Q5. Bowling pressure?**
The same index applies — bowling under pressure means the bowling side is under threat (WP of batting side near 0.5+ with leverage), or defending a knife-edge total. Bowler performance in HP states is scored against xR/xW conditioned on state, so "pressure bowling" = suppressing expected outcomes precisely when they are hardest to suppress.

**Q6. Win probability calculation?**
Two calibrated models:
- **WP₂ (chase)**: LightGBM binary classifier, target = batting side wins; features = pre-delivery state; trained season walk-forward; Platt/isotonic calibrated; validated by calibration curves + Brier + AUC (targets: AUC ≥ 0.85 chase; calibration slope ≈ 1).
- **WP₁ (innings 1)**: probabilistic total-projection — model distribution of final total given state; WP₁ = P(total > venue-era par). Par = expected 1st-innings score for venue×season from historical distribution, shrinkage-adjusted.
Both models include player-quality terms (see Q10) so they adjust for who is actually at the crease.

**Q7. Delivery leverage?**
`ΔWP` per outcome branch estimated empirically: for each state-bin, the distribution of `WP(after) − WP(before)` across realized outcomes. `LI(state) = E|ΔWP| / E_allsates|ΔWP|` (A: Tango leverage). Leverage enters the pressure index and is also reported per player as average leverage faced (scenario diversity evidence).

**Q8. Dynamic wicket valuation?**
λ(state) = expected run-cost of losing a wicket in that state, estimated from data: fit total-projection model; λ = ∂E[final total]/∂(wickets lost) locally, binned on (balls remaining × wickets lost) with monotonic smoothing (A concept; B submits same idea). Used to convert a dismissal into runs in batter/bowler value: `value = Δruns − λ·(wicket)`.

**Q9. Batter vs expected?**
Ball-level: `BV = (runs_off_bat + runs_from_extras_while_on_strike_conceded_by_bowler_responsibility) − xR(state) − λ(state)·(wicket − xW(state))`, where xR/xW come from models conditioned on phase, venue-era, bowler class, batter/bowler quality terms. Batter rating = shrunk mean `BV` over HP balls (primary), with consistency and uplift terms.

**Q10. Bowler vs expected?**
Ball-level: `BoV = xR_conceded(state, bowler) − bowler_conceded_runs(state) + λ(state)·(bowler_wicket − xW(state))`. Bowler-conceded runs follow the audit definition (off-bat + wides/no-balls; byes/leg-byes excluded). xR for the bowler is the model expectation *for this bowler given the batters faced*, so it implicitly prices opposition batting quality; an iterative two-pass fit adds opponent-quality adjustment (Q15).

**Q11. Pressure vs normal (clutch uplift)?**
Uplift = HP mean value − Normal mean value for the same player, computed within a hierarchical framework: the raw difference is noisy, so we regress `y_HP` on `y_Norm` across players and keep the residual (the *conditional* elevation given the player's normal level) — the same identifying idea as submission 2's residual clutch (B), implemented here as part of a composite rating rather than the whole rating. Uplift is reported with CI; it informs but does not dominate the composite.

**Q12. Sample size?**
Three mechanisms: (i) eligibility floors (grid-tested, §7); (ii) Empirical-Bayes shrinkage toward the position-role pool mean with strength `k` estimated from the between/within variance structure (method of moments on match-level aggregates); (iii) all reported metrics carry match-cluster-bootstrap CIs. Ranking score uses shrunk quantities; raw quantities shown as evidence.

**Q13. Role differences?**
Batter roles from *actual* batting positions in ball data (data-driven, not labels): opener (batting_order 1–2), top (3–4), middle (5–6), finisher (7 + median overs-batting in last 5), lower (8–11). Bowler roles from *actual* overs bowled distribution: powerplay-heavy, middle-heavy, death-heavy, all-phase. Ratings are compared within role pools for the "vs peers" evidence, while the main composite uses role-conditioned baselines (xR/xW include role terms) so cross-role ranking is fair (A).

**Q14. IPL vs T20I differences?**
(a) Expectation models include competition fixed effects (IPL vs T20I) capturing tournament scoring environment; (b) the WP/pressure layer is competition-agnostic (state-based); (c) sensitivity: full ranking recomputed IPL-only and T20I-only; a player's rank stability is reported. No fixed 1.1–1.2x multipliers (rejected as asserted constants; D).

**Q15. Opposition strength?**
Two-pass expectation models: pass 1 fits xR/xW with batter/bowler quality terms initialised at pool means; pass 2 refits using pass-1 career quality estimates as features. This yields opponent-adjusted expectations without country lists (C; plus-minus tradition, A). For batting, the relevant opponent is the bowler (and bowling team quality); for bowling, the batter (and batting team quality).

**Q16. Consistency?**
Per player: share of HP matches with positive total value (`positive-match %`), variance of per-match mean value, and split-half reliability (200 random match splits, correlation of ratings across halves — the submission-1 check, reimplemented). Consistency enters the composite EB-shrunk.

**Q17. Downside risk?**
Per player: 5th percentile of per-match mean value; share of HP balls with value below −λ (i.e., costly dismissals); "blow-up rate" = share of HP matches below −1 SD of pool. Reported in dossiers; a modest negative weight in the composite, sensitivity-tested.

**Q18. Ranking uncertainty?**
Match-level cluster bootstrap (B draws of matches with replacement): per draw, recompute eligibility, ratings, ranks. Outputs: 90% rank CI, P(top-10). The headline composite deliberately uses an LCB-flavoured component (shrunk level minus 0.85·SE, submission-1 idea, A/B) so that rankings require evidence, not variance — with the exact quantile sensitivity-tested.

**Q19. Validate the pressure model?**
A pressure definition is accepted only if it passes all:
1. **Variance test**: HP balls have materially higher outcome variance (runs/wickets) than non-HP at matched phase.
2. **WPA swing test**: mean |ΔWP| on HP balls ≥ 2× non-HP balls.
3. **Face validity**: HP share concentrated in chases' final overs, collapses, knife-edge defenses; HP rate for dead rubbers ≪ knockouts.
4. **Predictive validity**: WP calibration (slope, Brier) on held-out seasons.
5. **Rating persistence**: player HP ratings from 2022–24 predict 2025–26 HP ratings (Spearman r and its CI; B).
6. **Model-comparison**: Model B (WP-based) vs Model A (rules) vs Model C (hybrid) compared on 1–5; the winner becomes primary; others retained for the multiverse.

**Q20. Robustness?**
Multiverse analysis (B): ranking recomputed across a pre-registered grid — pressure threshold quantiles (10/15/20/25/30%), eligibility floors (3 values), metric weight perturbations (±0.1 on composite weights), IPL-only/T20I-only, with/without A-team comps. Outputs: per-player top-10 appearance counts and rank ranges. No setting is chosen post hoc to move any named player; the grid and its results are published wholesale.

---

## 2. Feature taxonomy (leakage control)

**T1. Available strictly before the delivery (allowed in pressure + expectation models):**
match/innings context (innings no., legal balls bowled, legal balls left); state (runs, wickets lost, target/DLS-adjusted target, runs required, RRR); phase (over bucket); venue-era par (from *prior* matches only); recent form-of-innings (wickets in last N legal balls, dot streak, boundary drought, current run rate); batter setness (balls faced this innings so far, runs so far); new-batter flag; bowler class (pace/spin, style); handedness matchup; batter/bowler career quality terms (two-pass, prior seasons only within walk-forward); competition, knockout flag, toss outcome; WP₁/WP₂ and leverage (from models fit on training seasons only).

**T2. Delivery outcomes (never features of their own pressure):**
runs off bat, extras by type, wicket + dismissal mode, is_four/is_six, WPA of the ball.

**T3. Post-delivery derived (evidence/reporting only):**
per-match value totals, positive-match share, per-match value distributions, split-half reliabilities.

**T4. Leakage risks and controls:**
- Using the *match's own* venue par → computed from venue×season matches excluding the current match (leave-one-match-out where needed).
- WP trained on all seasons then applied to earlier seasons → walk-forward: season s features use models trained on seasons < s only; final full-data models used only for the final ratings after the validation exercise.
- Two-pass quality terms seeing the current season → pass-1 terms computed on training seasons; for final ratings, terms recomputed on all data but the model's *coefficients* were frozen by walk-forward.
- Target leakage via DLS `revised_target` known before innings 2 starts → legitimate pre-delivery information (announced at innings break); allowed in T1.
- Player names/reputation → never features.

**T5. Unavailable in the supplied data (explicitly out of scope):**
ball-tracking (line/length/impact), fielding positions, pitch reports, weather, player injuries/fatigue, captaincy plans, crowd state, travel/schedule fatigue, contract/salary (and any reputation proxy). `bowl_speed` is 38.7% unrecorded → excluded. No venue *names* (only numeric venue_id + country) → venue par uses venue_id aggregates with shrinkage.

---

## 3. Pressure framework — three competing models

All three produce a continuous `P ∈ [0,1]` per legal delivery; high-pressure set = top-quantile of P (grid: 10/15/20/25/30%). All three are evaluated per Q19; Model C is the pre-registered primary.

### Model A — Transparent rule-based index
Situation-only, no learned components (the theic-alt/submission-5 family, upgraded):
```
S_A = 0.35·z(RRR_gap) + 0.25·z(scarcity) + 0.20·z(collapse) + 0.10·z(death) + 0.10·knockout
```
where `RRR_gap = RRR − venue_era_baseline_RR` (chase) or `CRR − par_projection_RR` (innings 1); `scarcity` = min(1, wickets_lost/8, balls_left_factor); `collapse` = wickets in last 12 legal balls ≥ 2; `death` = phase ≥ 16. z-scores computed within innings-phase strata. P_A = logistic(S_A). Purpose: interpretability anchor and multiverse member. Limitation: static weights (D-risk), no outcome linkage.

### Model B — Win-probability leverage index (primary candidate)
```
B = 1 − |2·WP − 1|                    (match balance)
L = E|ΔWP(state)| / E_allsates|ΔWP|   (leverage index)
S = severity(state)                   (resource stress: wickets+RRR+phase, normalised)
P_B = w_B·B + w_L·logistic(L − 1) + w_S·S
```
with `(w_B, w_L, w_S)` selected from a small grid {0.4–0.6, 0.25–0.35, 0.15–0.25} by validation score (Q19 composite), never by player outcomes (leakage guard). This is the submission-1/2 family (B) with Tango leverage (A).

### Model C — Hybrid (pre-registered primary)
```
P_C = 0.60·P_B + 0.40·P_A
```
Rationale: B captures outcome-linked stakes (better validated); A guarantees situational stress is represented even where WP models are weakest (innings 1). The 60/40 split is pre-registered and ±0.1-perturbed in the multiverse. Categories for interpretation: `Extreme` (top 10%), `High` (10–25%), `Moderate` (25–50%), `Normal` (rest).

---

## 4. Performance metrics

### 4.1 Batter
Ball-level value on HP balls (primary window) and all balls (baseline):
```
BV_i = R_i − xR_i − λ(state_i)·(W_i − xW_i)
```
- `R_i` = runs off the bat credited to the striker (+ any no-ball free-hit continuation handled by counting legal-ball sequence only; extras classification per audit).
- `xR_i` = expected off-bat runs for *this* state with *this* bowler and *this* batter-quality term.
- `W_i` = 1 if striker dismissed (any mode; run-out attribution to batter remains a wicket for the batting side — sensitivity: bowler-wickets-only variant).
- `λ(state)` = dynamic wicket value in runs.

Player metrics (all EB-shrunk where rates): `Level` = shrunk mean BV on HP balls; `Level_LCB` = Level − 0.85·SE(boot); `Uplift` = residual of HP-mean on Normal-mean; `Consistency` = shrunk positive-HP-match share; `Downside` = 5th percentile per-match mean; `Volume` = HP balls, matches, scenario diversity (entropy of HP situations faced).

Composite (weights pre-registered, ±0.1-perturbed):
```
Rating_bat = 0.45·z(Level_LCB) + 0.20·z(Consistency) + 0.20·z(Uplift) + 0.15·z(Volume) − 0.10·z(|Downside|)
```
(WUSA — weights unified, sensitivity-analysed.)

### 4.2 Bowler
```
BoV_i = xRc_i − Rc_i + λ(state_i)·(BW_i − xBW_i)
```
- `Rc_i` = bowler-conceded runs (audit definition), `xRc_i` = expectation for this bowler vs these batters in this state.
- `BW_i` = 1 iff dismissal credited to bowler (audit definition).

Same metric family as batting (Level, LCB, Uplift, Consistency, Downside, Volume) with phase-split evidence (PP/middle/death runs saved) reported in dossiers.

---

## 5. Models & training protocol

| Model | Purpose | Algorithm family | Validation |
|---|---|---|---|
| WP₂ chase | win prob pre-delivery | LightGBM (vs logistic baseline) | walk-forward by season; calibration curve, Brier, AUC |
| Total projection | innings-1 WP₁ + λ(state) | LightGBM quantile/mean regression of final total | walk-forward; MAE of projection; λ sanity (monotone in scarcity) |
| xR | expected off-bat runs | LightGBM regression (count-cap) vs Poisson GLM baseline | walk-forward; calibration by phase |
| xW / xBW | dismissal probabilities | LightGBM classifier (vs logistic) | walk-forward; Brier, calibration |
| Deep sequence model | experimental (§9) | GRU over last-K balls of the innings + state | same walk-forward; only promoted if it beats LightGBM |
| Shrinkage | rate reliability | EB method-of-moments (k from data) | split-half check |

All splits are season walk-forward (`train ≤ s → validate s+1`); **no random ball-level splits**. Seeds fixed in config; every model saved to `outputs/models/` with its config hash.

---

## 6. Player baseline methodology

Player baselines are the model's expectation *for that player* (quality terms), not raw career averages. "Normal performance" = mean BV on non-HP balls with the same estimator. Raw historical rates (SR/Econ/Avg on HP balls) are reported as evidence alongside, never used directly for ranking (D: raw sample averages without uncertainty).

## 7. Eligibility methodology

Floors (grid-tested, never single fixed): HP balls ≥ {40, 60, 80}; HP matches ≥ 6/8/10; both batteries applied; primary = 60 balls & 8 matches (matches the data scale: ~38k HP balls). Sensitivity table shows all grid members' top-10s. Additionally a diversity floor (HP situations span ≥ 2 phases) is tested but not enforced by default.

## 8. Competition & opposition adjustments

Competition fixed effects in all expectation models (IPL/T20I); venue-era par with shrinkage (venues with <10 matches pulled to global); two-pass opponent quality (Q15). No named-country tiers (D).

## 9. Deep-learning evaluation protocol (experiment, not assumption)

GRU (2×64) over last-30-ball sequence embeddings + static state, predicting (a) WP₂ and (b) next-ball xR/xW. Compared on identical walk-forward folds vs LightGBM/logistic. Decision rule pre-registered: promote to production only if it improves test Brier/RMSE by >2% relative on ≥ 2 folds; otherwise documented as evaluated-but-not-adopted (with learning curves). Expected outcome (honest prior): tabular boosting likely wins on 200k samples with structured features; the experiment exists to verify, not to decorate.

## 10. Final ranking architecture

1. Reconstruct leak-free states (Phase 2 of pipeline).
2. Fit WP₂/WP₁/xR/xW/λ walk-forward; produce per-ball P, leverage, BV/BoV.
3. Choose pressure model via Q19 battery (A vs B vs C); produce HP sets for quantile grid.
4. Player aggregates + EB shrinkage + composite ratings (bat & bowl).
5. Match-cluster bootstrap (400 draws): rank CIs, P(top-10).
6. Multiverse: thresholds × eligibility × weights × competition scope → stability table.
7. Final Top 10s = composite ranking at primary settings, with top-10 probability and rank CI published next to each pick.

## 11. Limitations (pre-registered)

- Innings-1 WP is a projection proxy, not true match-win probability (no DLS par mid-innings).
- No ball-tracking/fielding data → xR cannot separate shot quality from delivery quality fully.
- Clutch persistence in sport is modest; ratings are evidence-weighted but small-sample players near the eligibility floor remain uncertain (CI published).
- 2026 season data may post-date some T20I cycles unevenly; recency effects inherent.
- Run-outs are batting-side wickets under the primary definition (sensitivity variant provided).
- Opposition quality adjustment is two-pass approximation, not a full hierarchical plus-minus.
- Pressure definition is necessarily normative; the multiverse + validation battery is the defence, not a claim of uniqueness.

---

## 12. Decision table

| # | Decision | Choice | Alternative(s) rejected | Rationale (cat.) |
|---|---|---|---|---|
| 1 | Pressure representation | Continuous index + decile categories | Binary rules only | Preserves information; categories for humans (A,B,C) |
| 2 | Pressure models | 3 candidates (A rules, B WP-leverage, C hybrid), C primary | Single assertion | Pre-registered comparison avoids post hoc choice (C) |
| 3 | HP threshold | Quantile grid 10–30%, primary 20% | Fixed RRR≥10 etc. | Thresholds must be tested, not assumed (C; anti-B/D) |
| 4 | WP models | LightGBM chase WP + innings-1 total projection, calibrated | Single WP for both innings | 1st-innings "win prob" is ill-posed mid-innings (B experience) |
| 5 | Leverage | Tango-style LI from empirical ΔWP | Raw ΔWP only | Scale-free across states (A) |
| 6 | Value currency | Context-adjusted runs (BV/BoV) with dynamic λ | Raw SR/Econ, pure WPA | WPA alone punishes/rewards teammates' context (A,B,C) |
| 7 | Wicket value | λ(state) from total-projection derivatives | Fixed run values | Wickets' worth varies hugely by state (A,B) |
| 8 | Baselines | Model expectation incl. quality terms (two-pass) | Career averages | Opponent/venue/context adjustment (A,C; anti-D tiers) |
| 9 | Clutch uplift | Residual of HP on Normal regression, EB-shrunk | Raw HP − Normal | Removes mechanical correlation (B idea, C implementation) |
| 10 | Sample size | EB shrinkage (k from data) + eligibility grid + bootstrap CIs | Score × n/(n+k) | Shrinking to pool mean, not zero (A; anti-D) |
| 11 | Uncertainty | 400-draw match-cluster bootstrap; LCB component in score | Point estimates | Balls within matches correlate (A) |
| 12 | Validation | Season walk-forward only | Random ball split | Serial correlation; near-duplicate states (C) |
| 13 | Consistency | Positive-match share + split-half reliability | Raw variance | Split-half separates signal/noise (B) |
| 14 | Roles | Data-driven from batting_order/overs-bowled | Named-role assumptions | Reproducible; no labels needed (C) |
| 15 | IPL/T20I | Fixed effects + scope sensitivity re-runs | 1.1–1.2x multipliers | Multipliers are asserted constants (anti-D) |
| 16 | Opposition | Two-pass quality terms | Country tier lists | Data-driven; no named lists (C; anti-D) |
| 17 | Deep learning | GRU experiment vs LightGBM; promote only if >2% better | Blind inclusion | Evidence-first (project constraint) |
| 18 | Penalties | Folded into value framework; downside reported | Non-compensatory multipliers | Magic constants encode taste (anti-D) |
| 19 | Eligibility | Grid {40,60,80} balls × {6,8,10} matches | Single hard cutoff | Threshold sensitivity mandatory (project constraint) |
| 20 | Ranking headline | Composite with LCB level, EB consistency, uplift, volume, downside | Any single metric | Multi-evidence composite with uncertainty baked in (B1 idea, C weights) |

---

## 13. Why this beats simpler alternatives

A pure rule index (A) is interpretable but blind to how much a moment actually *matters*; a pure WP index (B) inherits model error exactly where data is thinnest (innings 1). The hybrid (C) uses each where it is strong. Value-above-expectation beats raw rates because it prices context (venue, phase, opponent, match state) that raw SR/Econ conflate. EB shrinkage + cluster bootstrap + LCB scoring beat point estimates because clutch effects are small and the top-10 must survive uncertainty, not just luck. The multiverse + validation battery beats single-setting assertion because every threshold in this problem is contestable — the honest answer is a ranking that is *stable across reasonable choices*, and we measure that stability instead of claiming it.

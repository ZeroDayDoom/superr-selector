# Robustness Report

## Most robust players (tightest bootstrap rank intervals)

### batters
- Mohammad Rizwan: rank P05-P95 = [1, 22], P(top10)=0.85
- Tashinga Musekiwa: rank P05-P95 = [2, 50], P(top10)=0.50
- Reeza Hendricks: rank P05-P95 = [2, 50], P(top10)=0.41
- Shai Hope: rank P05-P95 = [4, 60], P(top10)=0.29
- Glenn Phillips: rank P05-P95 = [5, 60], P(top10)=0.24
- Fakhar Zaman: rank P05-P95 = [3, 63], P(top10)=0.31
- Harry Tector: rank P05-P95 = [3, 64], P(top10)=0.28
- Parvez Emon: rank P05-P95 = [6, 69], P(top10)=0.17
- Afif Hossain: rank P05-P95 = [1, 72], P(top10)=0.57
- Tristan Stubbs: rank P05-P95 = [12, 73], P(top10)=0.04

### bowlers
- Kuldeep Sen: rank P05-P95 = [6, 38], P(top10)=0.24
- Mohammed Shami: rank P05-P95 = [6, 43], P(top10)=0.19
- Fareed Malik: rank P05-P95 = [1, 46], P(top10)=0.68
- Prince Yadav: rank P05-P95 = [3, 54], P(top10)=0.41
- Shakib Al Hasan: rank P05-P95 = [1, 56], P(top10)=0.64
- Blair Tickner: rank P05-P95 = [2, 61], P(top10)=0.46
- Mohsin Khan: rank P05-P95 = [6, 64], P(top10)=0.16
- Harshit Rana: rank P05-P95 = [2, 70], P(top10)=0.46
- Kagiso Rabada: rank P05-P95 = [9, 73], P(top10)=0.07
- Marcus Stoinis: rank P05-P95 = [10, 74], P(top10)=0.06

## Most uncertain players (widest intervals)

### batters
- Wriddhiman Saha: rank P05-P95 = [119, 169], P(top10)=0.00
- Shakib Al Hasan: rank P05-P95 = [36, 169], P(top10)=0.02
- Craig Ervine: rank P05-P95 = [4, 169], P(top10)=0.09
- Will Young: rank P05-P95 = [1, 169], P(top10)=0.18
- Evin Lewis: rank P05-P95 = [115, 169], P(top10)=0.00
- Nazmul Hossain Shanto: rank P05-P95 = [70, 169], P(top10)=0.01
- Mohammad Haris: rank P05-P95 = [133, 169], P(top10)=0.00
- Matthew Short: rank P05-P95 = [50, 168], P(top10)=0.01
- Dion Myers: rank P05-P95 = [8, 168], P(top10)=0.06
- Sherfane Rutherford: rank P05-P95 = [151, 168], P(top10)=0.00

### bowlers
- Sean Williams: rank P05-P95 = [159, 169], P(top10)=0.00
- Dwaine Pretorius: rank P05-P95 = [62, 169], P(top10)=0.00
- Dunith Wellalage: rank P05-P95 = [26, 169], P(top10)=0.01
- Tanzim Sakib: rank P05-P95 = [38, 169], P(top10)=0.01
- Benjamin Lister: rank P05-P95 = [131, 168], P(top10)=0.00
- Daniel Sams: rank P05-P95 = [112, 168], P(top10)=0.00
- Gerald Coetzee: rank P05-P95 = [120, 168], P(top10)=0.00
- Sheldon Cottrell: rank P05-P95 = [91, 167], P(top10)=0.00
- Fabian Allen: rank P05-P95 = [8, 167], P(top10)=0.06
- Mahedi Hasan: rank P05-P95 = [56, 167], P(top10)=0.00

## Interpretation

- A ranking that survives threshold changes, competition scope,
  eligibility tiers, shrinkage strengths, weight variants and
  component ablations with high Spearman stability is robust;
  the numbers above quantify exactly that.
- The final recommendation for the submission is the set of
  players with BOTH a high primary rating AND a high
  P(top-10) across 1000 bootstrap draws — see
  `outputs/top10_probability.csv`.
# E5 Monitor Error Analysis — k=5 (Supp B.9)

- Deterministic regeneration, `seed=42`, task-stratified 5-fold CV (`StratifiedGroupKFold`, no leakage). git `32f0c66`.
- Pooled OOF predictions: **15000** samples (1490 positives / 13510 negatives, prevalence 0.099).
- Combined-model AUC=0.752, AP=0.211.

## Confusion matrix @ operating threshold p ≥ 0.50 (deployed fire_at_p)

|        | pred fail | pred ok |
|--------|-----------|---------|
| **actual fail** | TP=1107 | FN=383 |
| **actual ok**   | FP=4991 | TN=8519 |

Precision=0.182 · Recall=0.743 · Specificity=0.631 · F1=0.292

Youden-J optimal threshold = 0.436 → TP=1259, FP=6250, FN=231, TN=7260 (P=0.168, R=0.845).

## False positives — driving-signal clusters
- Top-20 FP: oscillation=19, drift=1
- All 4991 FP: oscillation=2992, drift=1996, fidelity=3

### 20 highest-confidence false positives
| # | task | turn | score | driving signal | drift | osc | fid | maxdrift |
|---|------|------|-------|----------------|-------|-----|-----|----------|
| 1 | `viol_0179` | 42 | 0.941 | oscillation | 0.303 | 0.750 | 1.000 | 0.695 |
| 2 | `viol_0179` | 43 | 0.938 | oscillation | 0.406 | 0.667 | 1.000 | 0.695 |
| 3 | `viol_0179` | 41 | 0.918 | oscillation | 0.406 | 0.750 | 1.000 | 0.695 |
| 4 | `viol_0179` | 45 | 0.917 | oscillation | 0.431 | 0.750 | 1.000 | 0.695 |
| 5 | `viol_0179` | 44 | 0.916 | oscillation | 0.429 | 0.750 | 1.000 | 0.695 |
| 6 | `viol_0024` | 37 | 0.916 | oscillation | 0.531 | 1.000 | 1.000 | 0.709 |
| 7 | `viol_0018` | 24 | 0.914 | oscillation | 0.370 | 0.750 | 1.000 | 0.686 |
| 8 | `viol_0024` | 36 | 0.910 | oscillation | 0.509 | 1.000 | 1.000 | 0.709 |
| 9 | `viol_0179` | 40 | 0.909 | oscillation | 0.450 | 0.750 | 1.000 | 0.695 |
| 10 | `viol_0179` | 37 | 0.909 | oscillation | 0.433 | 0.500 | 1.000 | 0.695 |
| 11 | `viol_0024` | 38 | 0.908 | oscillation | 0.553 | 1.000 | 1.000 | 0.709 |
| 12 | `viol_0069` | 33 | 0.907 | oscillation | 0.379 | 0.500 | 1.000 | 0.695 |
| 13 | `viol_0179` | 36 | 0.906 | oscillation | 0.436 | 0.667 | 1.000 | 0.695 |
| 14 | `viol_0069` | 34 | 0.905 | drift | 0.413 | 0.250 | 1.000 | 0.695 |
| 15 | `viol_0029` | 38 | 0.904 | oscillation | 0.509 | 0.750 | 1.000 | 0.688 |
| 16 | `viol_0034` | 34 | 0.903 | oscillation | 0.403 | 0.500 | 1.000 | 0.684 |
| 17 | `viol_0021` | 43 | 0.902 | oscillation | 0.400 | 0.667 | 1.000 | 0.682 |
| 18 | `viol_0172` | 29 | 0.902 | oscillation | 0.519 | 0.667 | 1.000 | 0.696 |
| 19 | `viol_0034` | 35 | 0.901 | oscillation | 0.418 | 0.500 | 1.000 | 0.684 |
| 20 | `viol_0179` | 34 | 0.900 | oscillation | 0.470 | 0.667 | 1.000 | 0.695 |

## False negatives — driving-signal clusters
- Top-20 FN: oscillation=11, drift=9
- All 383 FN: oscillation=197, drift=186

### 20 highest-confidence false negatives (lowest predicted P among true failures)
| # | task | turn | score | driving signal | drift | osc | fid | maxdrift |
|---|------|------|-------|----------------|-------|-----|-----|----------|
| 1 | `ctrl_0011` | 48 | 0.129 | oscillation | 0.433 | 0.667 | 1.000 | 0.433 |
| 2 | `ctrl_0011` | 46 | 0.132 | oscillation | 0.408 | 0.667 | 1.000 | 0.423 |
| 3 | `ctrl_0011` | 47 | 0.138 | oscillation | 0.429 | 0.500 | 1.000 | 0.429 |
| 4 | `ctrl_0019` | 48 | 0.144 | oscillation | 0.471 | 0.500 | 1.000 | 0.471 |
| 5 | `ctrl_0020` | 48 | 0.154 | drift | 0.457 | 0.000 | 1.000 | 0.480 |
| 6 | `ctrl_0020` | 47 | 0.157 | drift | 0.475 | 0.000 | 1.000 | 0.480 |
| 7 | `ctrl_0020` | 49 | 0.166 | drift | 0.444 | 0.250 | 1.000 | 0.480 |
| 8 | `ctrl_0034` | 50 | 0.169 | oscillation | 0.451 | 0.600 | 1.000 | 0.468 |
| 9 | `ctrl_0019` | 47 | 0.170 | drift | 0.470 | 0.000 | 1.000 | 0.470 |
| 10 | `ctrl_0076` | 46 | 0.170 | drift | 0.440 | 0.333 | 1.000 | 0.458 |
| 11 | `ctrl_0076` | 47 | 0.179 | oscillation | 0.398 | 0.500 | 1.000 | 0.458 |
| 12 | `ctrl_0020` | 50 | 0.180 | drift | 0.481 | 0.333 | 1.000 | 0.481 |
| 13 | `ctrl_0076` | 50 | 0.180 | oscillation | 0.445 | 0.500 | 1.000 | 0.458 |
| 14 | `ctrl_0034` | 49 | 0.182 | drift | 0.438 | 0.250 | 1.000 | 0.468 |
| 15 | `ctrl_0019` | 49 | 0.184 | oscillation | 0.468 | 0.500 | 1.000 | 0.471 |
| 16 | `ctrl_0019` | 46 | 0.184 | drift | 0.447 | 0.250 | 1.000 | 0.468 |
| 17 | `ctrl_0011` | 49 | 0.187 | oscillation | 0.458 | 1.000 | 1.000 | 0.458 |
| 18 | `ctrl_0034` | 46 | 0.204 | oscillation | 0.466 | 0.750 | 1.000 | 0.468 |
| 19 | `ctrl_0076` | 48 | 0.205 | oscillation | 0.408 | 0.667 | 1.000 | 0.458 |
| 20 | `ctrl_0034` | 48 | 0.206 | drift | 0.413 | 0.250 | 1.000 | 0.468 |

## Failure-mechanism mapping
- **drift** → goal drift — state embedding rotated away from goal (A3 mechanism)
- **oscillation** → action cycling — agent stuck repeating a small action set
- **fidelity** → schema/tool-output infidelity — malformed or invalid tool responses
# Optuna Hyperparameter Optimization Report: Factorized SAC Loss

## Executive Summary
- **Total Completed Trials:** 80
- **Feasible Candidates (Satisfied Rank & Uniformity):** 2
- **Best Feasible Trial:** Trial #49

### Optimal Hyperparameters Found
- **Softmax Temperature (\tau):** `0.8502`
- **Gaussian Bandwidth Scale (\sigma_scale):** `N/A`
- **Overall Alignment (Objective):** `0.8291`
- **Overall Uniformity:** `-0.5520` (Constraint: <= -0.45)
- **Overall Effective Rank:** `2.7784` (Constraint: >= 2.5)

---

## Best Trial Per-Feature-Group Breakdown
| Feature Group | Alignment (\uparrow) | Uniformity (\downarrow) | Effective Rank (\uparrow) | Weight Entropy Ratio |
|---|---|---|---|---|
| **Prosody** | 0.9217 | -0.4398 | 2.6539 | 0.7896 |
| **Vocal_Tract** | 0.8477 | -0.4163 | 2.6471 | 0.9303 |
| **Timbre** | 0.7721 | -0.7187 | 2.9649 | 0.8365 |
| **Voice_Quality** | 0.8413 | -0.4261 | 2.6397 | 0.8692 |
| **Scene** | 0.7626 | -0.7592 | 2.9866 | 0.7914 |

---

## Top 10 Feasible Trials Overview
| Trial # | Tau (\tau) | Sigma Scale (\sigma) | Alignment (\uparrow) | Uniformity (\downarrow) | Effective Rank (\uparrow) | Entropy Ratio |
|---|---|---|---|---|---|---|
| 49 | 0.8502 | nan | 0.8291 | -0.4163 | 2.6397 | 0.8434 |
| 2 | 0.8408 | nan | 0.7775 | -0.3567 | 3.2750 | 0.7750 |

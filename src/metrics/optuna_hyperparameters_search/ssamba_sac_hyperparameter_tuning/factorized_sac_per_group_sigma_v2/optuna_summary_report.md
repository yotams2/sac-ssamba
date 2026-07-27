# Optuna Hyperparameter Optimization Report: Factorized SAC Loss

## Executive Summary
- **Total Completed Trials:** 53
- **Feasible Candidates (Satisfied Rank & Uniformity):** 2
- **Best Feasible Trial:** Trial #47

### Optimal Hyperparameters Found
- **Softmax Temperature (\tau):** `0.8888`
- **Gaussian Bandwidth Scale (\sigma_scale):** `N/A`
- **Overall Alignment (Objective):** `0.7979`
- **Overall Uniformity:** `-0.7574` (Constraint: <= -0.45)
- **Overall Effective Rank:** `3.8762` (Constraint: >= 2.5)

---

## Best Trial Per-Feature-Group Breakdown
| Feature Group | Alignment (\uparrow) | Uniformity (\downarrow) | Effective Rank (\uparrow) | Weight Entropy Ratio |
|---|---|---|---|---|
| **Prosody** | 0.8619 | -0.6966 | 3.4970 | 0.4514 |
| **Vocal_Tract** | 0.7611 | -0.6241 | 3.4703 | 0.1689 |
| **Timbre** | 0.8096 | -0.9223 | 4.4613 | 0.8441 |
| **Voice_Quality** | 0.7654 | -0.5722 | 3.4366 | 0.6314 |
| **Scene** | 0.7915 | -0.9719 | 4.5158 | 0.1656 |

---

## Top 10 Feasible Trials Overview
| Trial # | Tau (\tau) | Sigma Scale (\sigma) | Alignment (\uparrow) | Uniformity (\downarrow) | Effective Rank (\uparrow) | Entropy Ratio |
|---|---|---|---|---|---|---|
| 47 | 0.8888 | nan | 0.7979 | -0.5722 | 3.4366 | 0.4523 |
| 2 | 0.8408 | nan | 0.7471 | -0.6354 | 3.1669 | 0.3825 |

# Optuna Hyperparameter Optimization Report: Factorized SAC Loss

## Executive Summary
- **Total Completed Trials:** 80
- **Feasible Candidates (Satisfied Rank & Uniformity):** 4
- **Best Feasible Trial:** Trial #67

### Optimal Hyperparameters Found
- **Softmax Temperature (\tau):** `0.5034`
- **Gaussian Bandwidth Scale (\sigma_scale):** `N/A`
- **Overall Alignment (Objective):** `0.9005`
- **Overall Uniformity:** `-0.5685` (Constraint: <= -0.45)
- **Overall Effective Rank:** `3.3938` (Constraint: >= 2.5)

---

## Best Trial Per-Feature-Group Breakdown
| Feature Group | Alignment (\uparrow) | Uniformity (\downarrow) | Effective Rank (\uparrow) | Weight Entropy Ratio |
|---|---|---|---|---|
| **Prosody** | 0.9459 | -0.5077 | 3.0155 | 0.7651 |
| **Vocal_Tract** | 0.8942 | -0.4033 | 2.7587 | 0.5650 |
| **Timbre** | 0.9045 | -0.7196 | 4.1523 | 0.8680 |
| **Voice_Quality** | 0.8766 | -0.3664 | 2.6953 | 0.6172 |
| **Scene** | 0.8816 | -0.8457 | 4.3470 | 0.5990 |

---

## Top 10 Feasible Trials Overview
| Trial # | Tau (\tau) | Sigma Scale (\sigma) | Alignment (\uparrow) | Uniformity (\downarrow) | Effective Rank (\uparrow) | Entropy Ratio |
|---|---|---|---|---|---|---|
| 67 | 0.5034 | nan | 0.9005 | -0.3664 | 2.6953 | 0.6829 |
| 4 | 0.4833 | nan | 0.8815 | -0.3301 | 2.9016 | 0.6053 |
| 55 | 0.6539 | nan | 0.8375 | -0.3239 | 3.1991 | 0.5126 |
| 22 | 0.5245 | nan | 0.8304 | -0.3495 | 4.0722 | 0.5917 |

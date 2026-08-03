# Factorized SAC Optuna Study v5: Analysis & Winning Parameters

## 1. Executive Summary

- **Study Name:** `factorized_sac_per_group_sigma_v5`
- **Total Completed Trials:** 80
- **Feasible Candidates (Satisfying Per-Group Uniformity & Rank Limits):** 4
- **Winning Trial:** **Trial #67**

---

## 2. Winning Hyperparameters (Trial #67)

### Softmax Temperature ($\tau$)
$$\mathbf{\tau^* = 0.5034}$$

### Per-Group Bandwidth Multipliers ($\alpha_g^*$) & Exact Effective Sigmas ($\sigma_g^*$)

The effective Gaussian bandwidth for each feature group is computed as $\sigma_g^* = \text{OPTIMAL\_SIGMAS}[64][g] \times \alpha_g^*$:

| Feature Group | Base Anchor ($\sigma_{\text{base}}$) | Winning Multiplier ($\alpha_g^*$) | Exact Effective Sigma ($\sigma_g^*$) |
|---|---|---|---|
| **`Prosody`** (2D) | $0.1931$ | **`1.5376`** | **`0.2969`** |
| **`Vocal_Tract`** (3D) | $0.2223$ | **`0.8916`** | **`0.1982`** |
| **`Timbre`** (5D) | $0.3907$ | **`1.6403`** | **`0.6409`** |
| **`Voice_Quality`** (1D) | $0.0954$ | **`0.9729`** | **`0.0928`** |
| **`Scene`** (6D) | $0.2947$ | **`0.9326`** | **`0.2748`** |

---

## 3. Winning Representation Geometry Metrics

| Feature Group | Alignment ($\uparrow$) | Uniformity ($\downarrow$) | Effective Rank ($\uparrow$) | Weight Entropy Ratio |
|---|---|---|---|---|
| **`Prosody`** | **`0.9459`** | **`-0.5077`** | `3.0155` | `0.7651` |
| **`Vocal_Tract`** | **`0.8942`** | **`-0.4033`** | `2.7587` | `0.5650` |
| **`Timbre`** | **`0.9045`** | **`-0.7196`** | `4.1523` | `0.8680` |
| **`Voice_Quality`** | **`0.8766`** | **`-0.3664`** | `2.6953` | `0.6172` |
| **`Scene`** | **`0.8816`** | **`-0.8457`** | `4.3470` | `0.5990` |
| **Overall Average** | **`0.9005`** | **`-0.5685`** | **`3.3938`** | **`0.6829`** |

---

## 4. Key Insights & Takeaways

1. **Simultaneous Multi-Group Feasibility Achieved**: All 5 feature groups passed their constraint limits simultaneously in Trial #67 without any representation collapse.
2. **`Voice_Quality` & `Vocal_Tract` Tightly Bounded**: Both low-dimensional feature groups required multipliers close to $1.0$ ($\alpha_{\text{Voice\_Quality}} = 0.9729$, $\alpha_{\text{Vocal\_Tract}} = 0.8916$), confirming that their optimal bandwidths are very near their 62.5% Shannon entropy baselines ($\sigma \approx 0.0928$ and $\sigma \approx 0.1982$).
3. **`Prosody` & `Timbre` Benefit from Broader Bandwidths**: Multipliers $\approx 1.5 - 1.64$ allowed `Prosody` and `Timbre` to expand latent rank while maintaining strong alignment ($\ge 0.90$).

---

## 5. Ready-to-Use Pretraining Configuration

To use these winning parameters in full pretraining, update `OPTIMAL_SIGMAS` or pass the per-group sigmas and temperature $\tau = 0.5034$:

```python
# Optimal Per-Group Sigmas (Batch Size 64)
SAC_TEMPERATURE = 0.5034
SAC_OPTIMAL_SIGMAS = {
    'Prosody': 0.2969,
    'Vocal_Tract': 0.1982,
    'Timbre': 0.6409,
    'Voice_Quality': 0.0928,
    'Scene': 0.2748
}
```

"""
Step 2: SSAMBA + SAC Loss Model.

Wraps the SSAMBA bidirectional Mamba encoder with:
  - Mean pooling over sequence tokens → global representation e(X)
  - Projection head g(·): Linear → BatchNorm → ReLU → Linear → latent z
  - SAC (Soft Acoustic Contrastive) loss:
      * Pairwise temperature-scaled cosine similarity s_ij for latent vectors z
      * Gaussian kernel weights w_ij from acoustic feature vectors c_i
      * CWCL (Continuous Weighted Contrastive Learning) objective

Also retains the native SSAMBA masked patch reconstruction (generative) objective.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from random import randrange
import random

# Ensure SSAMBA model imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, '/storage/yotam/ssamba/Vim')
sys.path.insert(0, '/storage/yotam/ssamba/Vim/vim')
sys.path.insert(0, '/storage/yotam/ssamba/Vim/mamba-1p1p1')

from models.both_models import AMBAModel
from sac.sigma_configs import OPTIMAL_SIGMAS


class SSAMBASACModel(nn.Module):
    """
    SSAMBA encoder + SAC loss for single-channel audio SSL pretraining.
    
    During pretraining, this model:
      1. Takes log-Mel spectrogram patches as input
      2. Passes through the bidirectional Mamba encoder (from AMBAModel)
      3. Computes masked patch reconstruction loss (generative objective, from original SSAMBA)
      4. Mean-pools the encoder tokens → global embedding
      5. Projects through g(·) → latent z
      6. Computes SAC loss using latent z and acoustic feature vectors c
      7. Returns total loss = L_recon + λ * L_SAC
    
    Args:
        fshape, tshape: patch shape (frequency, time)
        input_fdim, input_tdim: spectrogram dimensions
        model_size: Mamba model size ('base', etc.)
        embed_dim: embedding dimension of the Mamba encoder
        proj_dim: output dimension of the projection head g(·)
        sac_temperature: temperature τ for cosine similarity scaling
        sac_sigma: bandwidth σ for the Gaussian kernel
        sac_lambda: weight λ for the SAC loss term
        vision_mamba_config: config dict for VisionMamba
    """

    def __init__(
        self,
        fshape=16, tshape=16,
        input_fdim=128, input_tdim=1024,
        model_size='base',
        embed_dim=768,
        depth=24,
        proj_dim=128,
        sac_temperature=0.3,
        sac_sigma=1.0,
        sac_lambda=1.0,
        recon_lambda=1.0,
        classif_lambda=0.0,
        sac_features="f0_mean,hnr,centroid,flux,zcr_mean",
        mask_patch=300,
        vision_mamba_config=None,
        local_sigma_mode='chi2_median',
        use_cross_attention=True,
        num_queries_per_group=4,
        use_checkpointing=False,
    ):
        super(SSAMBASACModel, self).__init__()

        self.sac_temperature = sac_temperature
        self.sac_sigma = sac_sigma
        self.sac_lambda = sac_lambda
        self.recon_lambda = recon_lambda
        self.classif_lambda = classif_lambda
        self.embed_dim = embed_dim
        self.mask_patch = mask_patch
        self.local_sigma_mode = local_sigma_mode
        self.use_cross_attention = use_cross_attention
        self.num_queries_per_group = num_queries_per_group
        self.use_checkpointing = use_checkpointing

        # Default vision mamba config
        if vision_mamba_config is None:
            vision_mamba_config = {}

        # Build the SSAMBA encoder (in pretraining mode)
        self.encoder = AMBAModel(
            fshape=fshape, tshape=tshape,
            fstride=fshape, tstride=tshape,
            input_fdim=input_fdim, input_tdim=input_tdim,
            model_size=model_size,
            pretrain_stage=True,
            load_pretrained_mdl_path=None,
            vision_mamba_config=vision_mamba_config,
        )
        self.encoder.use_checkpointing = use_checkpointing
        if hasattr(self.encoder, 'v'):
            self.encoder.v.use_checkpointing = use_checkpointing

        # The encoder's original_embedding_dim is set during __init__
        actual_embed_dim = self.encoder.original_embedding_dim

        # ---- Feature Groups & Cross-Attention ----
        if self.use_cross_attention:
            from sac.acoustic_features import get_feature_groups
            self.group_indices, self.num_active_groups = get_feature_groups(sac_features)
            
            # [num_active_groups * num_queries_per_group, actual_embed_dim]
            self.group_queries = nn.Parameter(torch.empty(self.num_active_groups * self.num_queries_per_group, actual_embed_dim))
            nn.init.normal_(self.group_queries, std=0.02)
            
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=actual_embed_dim, 
                num_heads=8, 
                batch_first=True
            )

            # ---- Projection Head g(·) for SAC loss ----
            # Maps pooled encoder output to latent space for similarity comparison
            # Architecture: Linear → LayerNorm → ReLU → Linear
            self.projection_head = nn.Sequential(
                nn.Linear(actual_embed_dim, actual_embed_dim),
                nn.LayerNorm(actual_embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(actual_embed_dim, proj_dim),
            )
        else:
            # ---- Projection Head g(·) for SAC loss ----
            # Architecture perfectly matched to the Universal model for clean ablation
            self.projection_head = nn.Sequential(
                nn.Linear(actual_embed_dim, actual_embed_dim),
                nn.LayerNorm(actual_embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(actual_embed_dim, proj_dim),
            )

        # Feature normalization stats (set externally before training)
        self.feature_stats = None

    def _encode_with_mamba(self, x):
        """
        Run the bidirectional Mamba encoder and return all token outputs.
        
        Args:
            x: [B, 1, F, T] spectrogram (as processed by AMBAModel.forward)
            
        Returns:
            hidden_states: [B, num_patches + cls_tokens, embed_dim] encoder output
        """
        B = x.shape[0]
        # Patch embedding
        x_embed = self.encoder.v.patch_embed(x)

        # Add CLS token
        cls_tokens = self.encoder.v.cls_token.expand(B, -1, -1)
        x_embed = torch.cat((cls_tokens, x_embed), dim=1)

        # Add positional embedding
        x_embed = x_embed + self.encoder.v.pos_embed
        x_embed = self.encoder.v.pos_drop(x_embed)

        # Bidirectional Mamba layers
        residual = None
        hidden_states, residual = self.encoder._forward_mamba_layers(x_embed, residual)

        # Final normalization
        if not self.encoder.v.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.encoder.v.drop_path(hidden_states)
            hidden_states = self.encoder.v.norm_f(
                residual.to(dtype=self.encoder.v.norm_f.weight.dtype)
            )
        else:
            try:
                from mamba_ssm.ops.triton.layernorm import rms_norm_fn, layer_norm_fn, RMSNorm
                fused_add_norm_fn = rms_norm_fn if (RMSNorm is not None and isinstance(self.encoder.v.norm_f, RMSNorm)) else layer_norm_fn
                hidden_states = fused_add_norm_fn(
                    self.encoder.v.drop_path(hidden_states),
                    self.encoder.v.norm_f.weight,
                    self.encoder.v.norm_f.bias,
                    eps=self.encoder.v.norm_f.eps,
                    residual=residual,
                    prenorm=False,
                    residual_in_fp32=self.encoder.v.residual_in_fp32,
                )
            except (ImportError, TypeError):
                if residual is None:
                    residual = hidden_states
                else:
                    residual = residual + hidden_states
                hidden_states = self.encoder.v.norm_f(residual)

        return hidden_states

    def _pool_tokens(self, hidden_states):
        """
        Mean-pool over patch tokens (excluding CLS) to get global representation.
        
        Args:
            hidden_states: [B, num_patches + cls_token_num, embed_dim]
            
        Returns:
            pooled: [B, embed_dim]
        """
        cls_token_num = self.encoder.cls_token_num
        # Exclude CLS token(s), mean pool over remaining patch tokens
        patch_tokens = hidden_states[:, cls_token_num:, :]  # [B, num_patches, embed_dim]
        pooled = patch_tokens.mean(dim=1)  # [B, embed_dim]
        return pooled

    def sac_loss_legacy(self, z, c, eps=1e-8, return_diagnostics=False):
        """
        Compute the Soft Acoustic Contrastive (SAC) loss using CWCL (Legacy without cross-attention).
        """
        device = z.device
        B = z.shape[0]

        if B < 2:
            loss = torch.tensor(0.0, device=device, requires_grad=True)
            if return_diagnostics:
                return loss, z, torch.zeros((B, B), device=device), torch.zeros((B, B), device=device)
            return loss

        # 1. Compute pairwise cosine similarity / τ
        z_norm = F.normalize(z, dim=-1)  # [B, proj_dim]
        sim = torch.matmul(z_norm, z_norm.T) / self.sac_temperature  # [B, B]

        # 2. Compute Gaussian kernel weights from acoustic features
        # ||c_i - c_j||² via cdist
        cue_dist = torch.cdist(c, c, p=2)  # [B, B]
        w = torch.exp(-(cue_dist / self.sac_sigma) ** 2)  # [B, B]

        # 3. Mask out self-pairs
        diag_mask = torch.eye(B, device=device, dtype=torch.bool)
        off_diag = ~diag_mask

        # 4. Softmax denominator (sum over k ≠ i)
        exp_sim = torch.exp(sim) * off_diag.float()
        den = exp_sim.sum(dim=1, keepdim=True) + eps  # [B, 1]

        # 5. Log probability
        log_prob = sim - torch.log(den)  # [B, B]

        # 6. Normalize weights (row-wise, excluding self)
        w_masked = w * off_diag.float()
        w_sum = w_masked.sum(dim=1, keepdim=True) + eps  # [B, 1]
        w_norm = w_masked / w_sum  # [B, B]

        # 7. Weighted sum of log-probabilities
        loss_per_sample = -(w_norm * log_prob).sum(dim=1)  # [B]
        loss = loss_per_sample.mean()

        if return_diagnostics:
            return loss, z_norm, sim, w
        return loss

    def sac_loss(self, z_groups, c, eps=1e-8, return_diagnostics=False):
        """
        Compute the Soft Acoustic Contrastive (SAC) loss using CWCL,
        factorized by Acoustic Feature Groups.
        """
        device = z_groups.device
        B = z_groups.shape[0]

        if B < 2:
            loss = torch.tensor(0.0, device=device, requires_grad=True)
            if return_diagnostics:
                return loss, z_groups, torch.zeros((B, B), device=device), torch.zeros((B, B), device=device), {}
            return loss

        total_loss = 0.0
        group_names = list(self.group_indices.keys())
        
        z_norms = []
        sims = []
        ws = []
        entropies = {}
        
        for k in range(self.num_active_groups):
            group_name = group_names[k]
            indices = self.group_indices[group_name]
            
            c_group = c[:, indices]
            z_group = z_groups[:, k, :]
            
            cue_dist = torch.cdist(c_group, c_group, p=2)
            
            import math
            if self.local_sigma_mode == 'static_entropy_optimal':
                assert B in [16, 32, 64], f"static_entropy_optimal requires batch size in [16, 32, 64], got {B}"
                local_sigma = OPTIMAL_SIGMAS[B].get(group_name, self.sac_sigma * math.sqrt(len(indices)))
                
            elif self.local_sigma_mode == 'dynamic_batch_median':
                # Calculate median off-diagonal distance
                diag_mask_med = torch.eye(B, device=device, dtype=torch.bool)
                off_diag_dist = cue_dist[~diag_mask_med]
                median_dist = off_diag_dist.median()
                # Overlook sac_sigma and use raw median scaled by ln(2)
                local_sigma = median_dist / math.sqrt(math.log(2.0))
                if local_sigma < 1e-4:
                    local_sigma = math.sqrt(len(indices))
            
            elif self.local_sigma_mode == 'offline_global_median':
                # Use global median computed over the entire dataset
                if self.feature_stats is not None and 'group_medians' in self.feature_stats:
                    median_dist = self.feature_stats['group_medians'].get(group_name, None)
                    if median_dist is not None:
                        # Overlook sac_sigma and use raw median scaled by ln(2)
                        local_sigma = median_dist / math.sqrt(math.log(2.0))
                    else:
                        local_sigma = self.sac_sigma * math.sqrt(len(indices))
                else:
                    local_sigma = self.sac_sigma * math.sqrt(len(indices))
            
            elif self.local_sigma_mode == 'chi2_median':
                # Theoretical Chi-squared median scaling
                chi2_medians = {1: 0.455, 2: 1.386, 3: 2.366, 4: 3.357, 5: 4.351, 6: 5.348, 7: 6.346}
                D = len(indices)
                median_D = chi2_medians.get(D, D - 2/3)
                local_sigma = self.sac_sigma * math.sqrt(median_D)
            
            else: # 'sqrt_dim'
                local_sigma = self.sac_sigma * math.sqrt(len(indices))
            
            w = torch.exp(-(cue_dist / local_sigma) ** 2)
            z_norm = F.normalize(z_group, dim=-1)
            sim = torch.matmul(z_norm, z_norm.T) / self.sac_temperature
            
            diag_mask = torch.eye(B, device=device, dtype=torch.bool)
            off_diag = ~diag_mask
            
            exp_sim = torch.exp(sim) * off_diag.float()
            den = exp_sim.sum(dim=1, keepdim=True) + eps
            log_prob = sim - torch.log(den)
            
            w_masked = w * off_diag.float()
            w_sum = w_masked.sum(dim=1, keepdim=True) + eps
            w_norm = w_masked / w_sum
            
            entropy = - (w_norm * torch.log(w_norm + eps)).sum(dim=1).mean()
            entropies[group_name] = entropy
            
            loss_per_sample = -(w_norm * log_prob).sum(dim=1)
            loss_group = loss_per_sample.mean()
            
            total_loss += loss_group
            
            if return_diagnostics:
                z_norms.append(z_norm)
                sims.append(sim)
                ws.append(w)
                
        if self.num_active_groups > 0:
            total_loss = total_loss / self.num_active_groups
                
        if return_diagnostics:
            return total_loss, torch.stack(z_norms, dim=1), torch.stack(sims, dim=0), torch.stack(ws, dim=0), entropies
            
        return total_loss

    def forward(self, fbank, task='pretrain_joint', mask_patch=None,
                cluster=True, acoustic_features=None, return_diagnostics=False):
        """
        Forward pass for pretraining.
        
        Args:
            fbank: [B, T_frames, F_bins] log-Mel spectrogram (e.g., [B, 1024, 128])
            task: pretraining task ('pretrain_mpc', 'pretrain_mpg', 'pretrain_joint')
            mask_patch: number of patches to mask
            cluster: whether to use cluster masking
            acoustic_features: [B, 5] pre-computed acoustic feature vectors c_i (optional)
            
        Returns:
            If acoustic_features is provided (SAC mode):
                dict with keys: 'loss_total', 'loss_recon', 'loss_sac', 'acc'
            Otherwise:
                Falls through to standard SSAMBA pretraining behavior
        """
        if mask_patch is None:
            mask_patch = self.mask_patch

        # If no acoustic features, fall back to standard SSAMBA pretraining
        if acoustic_features is None:
            return self.encoder(fbank, task=task, mask_patch=mask_patch, cluster=cluster)

        # ---- Flexible multi-loss pretraining ----
        # Reshape input: [B, T, F] → [B, 1, F, T]
        x = fbank.unsqueeze(1).transpose(2, 3)
        B = x.shape[0]

        # 1. Reconstruction loss (generative objective, MPG)
        if self.recon_lambda > 0:
            loss_recon = self.encoder.mpg(x, mask_patch=mask_patch, cluster=cluster)
        else:
            loss_recon = torch.tensor(0.0, device=x.device)

        # 2. Classification loss (discriminative objective, MPC)
        if self.classif_lambda > 0:
            acc, loss_classif = self.encoder.mpc(x, mask_patch=mask_patch, cluster=cluster)
        else:
            loss_classif = torch.tensor(0.0, device=x.device)
            acc = torch.tensor(0.0, device=x.device)

        # 3. SAC contrastive loss
        if self.sac_lambda > 0 and acoustic_features is not None:
            # Encode FULL (unmasked) spectrogram for the SAC branch
            with torch.no_grad() if not self.training else torch.enable_grad():
                hidden_states = self._encode_with_mamba(x)

            if self.use_cross_attention:
                # Cross-attention to extract group embeddings
                Q = self.group_queries.unsqueeze(0).expand(B, -1, -1)
                cls_token_num = self.encoder.cls_token_num
                attn_output, _ = self.cross_attention(query=Q, key=hidden_states[:, cls_token_num:, :], value=hidden_states[:, cls_token_num:, :])
                attn_output = attn_output.view(B, self.num_active_groups, self.num_queries_per_group, -1)
                attn_pooled = attn_output.mean(dim=2)
                Z_groups = self.projection_head(attn_pooled)

                if return_diagnostics:
                    loss_sac, z_norm, sim, w, entropies = self.sac_loss(Z_groups, acoustic_features, return_diagnostics=True)
                    z = Z_groups.mean(dim=1)
                else:
                    loss_sac = self.sac_loss(Z_groups, acoustic_features)
                    z = None
            else:
                e = self._pool_tokens(hidden_states)
                z = self.projection_head(e)
                if return_diagnostics:
                    loss_sac, z_norm, sim, w = self.sac_loss_legacy(z, acoustic_features, return_diagnostics=True)
                    entropies = {}
                else:
                    loss_sac = self.sac_loss_legacy(z, acoustic_features)
        else:
            loss_sac = torch.tensor(0.0, device=x.device)
            z = None
            z_norm = None
            sim = None
            w = None
            entropies = {}

        # 4. Total weighted loss
        loss_total = (
            self.recon_lambda * loss_recon +
            self.classif_lambda * loss_classif +
            self.sac_lambda * loss_sac
        )

        output = {
            'loss_total': loss_total,
            'loss_recon': loss_recon,
            'loss_classif': loss_classif,
            'loss_sac': loss_sac,
            'acc': acc.detach() if isinstance(acc, torch.Tensor) else acc,
        }

        if return_diagnostics:
            output['z'] = z
            output['z_norm'] = z_norm
            output['sim'] = sim
            output['w'] = w
            output['entropies'] = entropies

        return output


class SSAMBASACModelParallel(nn.Module):
    """
    Thin wrapper to make SSAMBASACModel compatible with nn.DataParallel.
    Handles the dict output and acoustic_features passing.
    """

    def __init__(self, model: SSAMBASACModel):
        super().__init__()
        self.module = model

    def forward(self, fbank, task='pretrain_joint', mask_patch=None,
                cluster=True, acoustic_features=None, return_diagnostics=False):
        return self.module(fbank, task=task, mask_patch=mask_patch,
                          cluster=cluster, acoustic_features=acoustic_features,
                          return_diagnostics=return_diagnostics)

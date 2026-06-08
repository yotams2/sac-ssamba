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
        mask_patch=300,
        vision_mamba_config=None,
    ):
        super(SSAMBASACModel, self).__init__()

        self.sac_temperature = sac_temperature
        self.sac_sigma = sac_sigma
        self.sac_lambda = sac_lambda
        self.embed_dim = embed_dim
        self.mask_patch = mask_patch

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

        # The encoder's original_embedding_dim is set during __init__
        actual_embed_dim = self.encoder.original_embedding_dim

        # ---- Projection Head g(·) for SAC loss ----
        # Maps pooled encoder output to latent space for similarity comparison
        # Architecture: Linear → BatchNorm → ReLU → Linear
        self.projection_head = nn.Sequential(
            nn.Linear(actual_embed_dim, actual_embed_dim),
            nn.BatchNorm1d(actual_embed_dim),
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
        hidden_states = x_embed

        if not self.encoder.v.if_bidirectional:
            for layer in self.encoder.v.layers:
                hidden_states, residual = layer(hidden_states, residual)
        else:
            for i in range(len(self.encoder.v.layers) // 2):
                hidden_states_f, residual_f = self.encoder.v.layers[i * 2](
                    hidden_states, residual
                )
                hidden_states_b, residual_b = self.encoder.v.layers[i * 2 + 1](
                    hidden_states.flip([1]),
                    None if residual is None else residual.flip([1])
                )
                hidden_states = hidden_states_f + hidden_states_b.flip([1])
                residual = residual_f + residual_b.flip([1])

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
                fused_add_norm_fn = rms_norm_fn if isinstance(self.encoder.v.norm_f, RMSNorm) else layer_norm_fn
                hidden_states = fused_add_norm_fn(
                    self.encoder.v.drop_path(hidden_states),
                    self.encoder.v.norm_f.weight,
                    self.encoder.v.norm_f.bias,
                    eps=self.encoder.v.norm_f.eps,
                    residual=residual,
                    prenorm=False,
                    residual_in_fp32=self.encoder.v.residual_in_fp32,
                )
            except ImportError:
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

    def sac_loss(self, z, c, eps=1e-8, return_diagnostics=False):
        """
        Compute the Soft Acoustic Contrastive (SAC) loss using CWCL.
        
        The SAC loss uses continuous similarity weights instead of binary positive/negative
        pairs. Weights w_ij are computed via a Gaussian kernel over the distance between
        acoustic feature vectors c_i and c_j.
        
        L_SAC = -(1/B) Σ_i Σ_{j≠i} w̃_ij * log( exp(s_ij) / Σ_{k≠i} exp(s_ik) )
        
        where:
            s_ij = cos(z_i, z_j) / τ          (temperature-scaled cosine similarity)
            w_ij = exp(- ||c_i - c_j||² / σ²) (Gaussian kernel weight)
            w̃_ij = w_ij / Σ_{k≠i} w_ik       (row-normalized weights)
        
        Args:
            z: [B, proj_dim] projected latent vectors
            c: [B, num_features] acoustic feature vectors (already normalized & clipped)
            eps: numerical stability constant
            
        Returns:
            loss: scalar SAC loss
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

        # ---- SAC-augmented pretraining ----
        # Reshape input: [B, T, F] → [B, 1, F, T]
        x = fbank.unsqueeze(1).transpose(2, 3)
        B = x.shape[0]

        # 1. Compute reconstruction loss (generative objective)
        # Use the encoder's native mpg method
        loss_recon = self.encoder.mpg(x, mask_patch=mask_patch, cluster=cluster)

        # 2. Encode the FULL (unmasked) spectrogram for the SAC branch
        # This gives us the clean global representation for contrastive learning
        with torch.no_grad() if not self.training else torch.enable_grad():
            hidden_states = self._encode_with_mamba(x)

        # 3. Mean pool to get global representation e(X)
        e = self._pool_tokens(hidden_states)  # [B, embed_dim]

        # 4. Project through g(·) to get latent z
        z = self.projection_head(e)  # [B, proj_dim]

        # 5. Compute SAC loss
        if return_diagnostics:
            loss_sac, z_norm, sim, w = self.sac_loss(z, acoustic_features, return_diagnostics=True)
        else:
            loss_sac = self.sac_loss(z, acoustic_features)

        # 6. Total loss
        loss_total = loss_recon + self.sac_lambda * loss_sac

        # 7. Compute accuracy metric (from discriminative objective, if applicable)
        # Use MPC for accuracy tracking — no_grad to avoid retaining a third gradient graph
        with torch.no_grad():
            try:
                acc, _ = self.encoder.mpc(x, mask_patch=mask_patch, cluster=cluster)
            except Exception:
                acc = torch.tensor(0.0, device=x.device)

        output = {
            'loss_total': loss_total,
            'loss_recon': loss_recon,
            'loss_sac': loss_sac,
            'acc': acc.detach() if isinstance(acc, torch.Tensor) else acc,
        }

        if return_diagnostics:
            output['z'] = z
            output['z_norm'] = z_norm
            output['sim'] = sim
            output['w'] = w

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

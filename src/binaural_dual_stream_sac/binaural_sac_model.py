import sys
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure parent directory imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from binaural_dual_stream_sac.binaural_amba import BinauralAMBAEncoder
from sac.acoustic_features import get_feature_groups
from binaural_dual_stream_sac.spatial_features import get_spatial_feature_groups


class BinauralSSAMBASACModel(nn.Module):
    """
    Binaural SSAMBA + Dual-Branch Factorized SAC Model.
    
    Architecture:
    1. Dual Mamba Encoders:
       - Spatial Encoder sees X_spat (same CCSR time-frame mask on both ears).
       - Spectral Encoder sees X_spec (inverse mask on Right ear).
       
    2. Shared FC Reconstruction Decoder:
       - Concat([h_spat, h_spec]) -> predicts Re_L, Im_L for Channel 1 masked time-frames per CCSR Eq. 6.
       
    3. Spectral SAC Branch (attaches ONLY to h_spec):
       - Uses get_feature_groups() directly from sac.acoustic_features.
       - Projection Head -> Cross-Attention Layer (queries for Prosody, Vocal_Tract, Timbre, Voice_Quality, Scene) -> Factorized SAC loss with 15 monaural spectral features.
       
    4. Spatial SAC Branch (attaches ONLY to h_spat):
       - Uses get_spatial_feature_groups() from spatial_features.py (aligned with SAR-SSL).
       - Projection Head (NO CA layer by default) -> Single spatial latent vector z_spat -> CWCL loss with 5 spatial features (TDOA, GCC-PHAT peak, Sub-band Coherence).
    """
    def __init__(
        self,
        fshape=16, tshape=16,
        input_fdim=256, input_tdim=1024,
        model_size='base',
        embed_dim=768,
        proj_dim=128,
        sac_temperature=0.3,
        sac_sigma=1.0,
        sac_spec_lambda=1.0,
        sac_spat_lambda=1.0,
        recon_lambda=1.0,
        spectral_features="f0,hnr,centroid,flux,zcr",
        spatial_features="spatial",
        vision_mamba_config=None,
        local_sigma_mode='offline_global_median', # Default: offline_global_median
        share_encoder_weights=False,
        num_queries_per_group=4,
        use_cross_attention_spat=False, # Default False: single vector projection for spatial stream
    ):
        super(BinauralSSAMBASACModel, self).__init__()
        self.sac_temperature = sac_temperature
        self.sac_sigma = sac_sigma
        self.sac_spec_lambda = sac_spec_lambda
        self.sac_spat_lambda = sac_spat_lambda
        self.recon_lambda = recon_lambda
        self.local_sigma_mode = local_sigma_mode
        self.fshape = fshape
        self.tshape = tshape
        self.embed_dim = embed_dim
        self.num_queries_per_group = num_queries_per_group
        self.use_cross_attention_spat = use_cross_attention_spat

        if vision_mamba_config is None:
            vision_mamba_config = {}

        if not torch.cuda.is_available():
            vision_mamba_config['fused_add_norm'] = False

        encoder_config = {
            'fshape': fshape, 'tshape': tshape,
            'fstride': fshape, 'tstride': tshape,
            'input_fdim': input_fdim, 'input_tdim': input_tdim,
            'model_size': model_size,
            'in_chans': 4,
            'pretrain_stage': True,
            'vision_mamba_config': vision_mamba_config,
        }

        # Dual Encoder (Spatial + Spectral Mamba streams)
        self.dual_encoder = BinauralAMBAEncoder(encoder_config, share_weights=share_encoder_weights)

        # Shared Reconstruction Decoder (CCSR Eq. 6)
        patch_dim = 2 * fshape * tshape # Re_L and Im_L for 1st mic
        self.reconstruction_decoder = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, patch_dim)
        )

        # ---- 1. SPECTRAL SAC BRANCH (Attaches to h_spec) ----
        self.spec_group_indices, self.num_active_spec_groups = get_feature_groups(spectral_features)
        
        self.group_queries_spec = nn.Parameter(
            torch.empty(self.num_active_spec_groups * self.num_queries_per_group, embed_dim)
        )
        nn.init.normal_(self.group_queries_spec, std=0.02)
        
        self.cross_attention_spec = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=8, batch_first=True
        )
        
        self.projection_head_spec = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, proj_dim),
        )

        # ---- 2. SPATIAL SAC BRANCH (Attaches to h_spat) ----
        self.spat_group_indices, self.num_active_spat_groups = get_spatial_feature_groups(spatial_features)
        
        self.projection_head_spat = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, proj_dim),
        )

        if self.use_cross_attention_spat:
            self.group_queries_spat = nn.Parameter(
                torch.empty(self.num_active_spat_groups * self.num_queries_per_group, embed_dim)
            )
            nn.init.normal_(self.group_queries_spat, std=0.02)
            self.cross_attention_spat = nn.MultiheadAttention(
                embed_dim=embed_dim, num_heads=8, batch_first=True
            )

        self.feature_stats = None

    def _reconstruct(self, h_spat, h_spec, X, W):
        """
        CCSR Eq. 6 Reconstruction Loss: MSE over masked time-frames of Channel 1 (Re_L, Im_L).
        """
        B = X.shape[0]
        # Exclude CLS token (index 0)
        h_spat_patch = h_spat[:, 1:, :] # [B, num_patches, D]
        h_spec_patch = h_spec[:, 1:, :] # [B, num_patches, D]
        
        # Concat along embedding dimension
        h_concat = torch.cat([h_spat_patch, h_spec_patch], dim=-1) # [B, num_patches, 2D]
        pred_patches = self.reconstruction_decoder(h_concat) # [B, num_patches, 2 * fshape * tshape]
        
        # Target: unfold Channel 1 (Re_L, Im_L) raw STFT patches
        X_ch1 = X[:, :2, :, :] # [B, 2, F, T]
        unfold = torch.nn.Unfold(kernel_size=(self.fshape, self.tshape), stride=(self.fshape, self.tshape))
        target_patches = unfold(X_ch1).transpose(1, 2) # [B, num_patches, 2 * fshape * tshape]
        
        # Masked region weights (where W == 0)
        masked_weights = (1.0 - W).squeeze(1).squeeze(1) # [B, T]
        patch_w = F.adaptive_avg_pool1d(masked_weights.unsqueeze(1), pred_patches.shape[1]).transpose(1, 2) # [B, num_patches, 1]
        
        # Normalize by total masked element count (masked patches * patch_dim)
        patch_dim = pred_patches.shape[-1]
        denom = patch_w.sum() * patch_dim + 1e-8
        loss_recon = torch.sum(patch_w * (pred_patches - target_patches) ** 2) / denom
        return loss_recon

    def _compute_factorized_sac(self, z_groups, c_group_indices, num_groups, eps=1e-8, return_diagnostics=False):
        """
        Factorized SAC loss for groups (used for spectral encoder branch).
        """
        device = z_groups.device
        B = z_groups.shape[0]

        if B < 2:
            loss = torch.tensor(0.0, device=device, requires_grad=True)
            if return_diagnostics:
                return loss, z_groups, torch.zeros((B, B), device=device), torch.zeros((B, B), device=device), {}
            return loss

        total_loss = 0.0
        group_names = list(c_group_indices.keys())
        entropies = {}
        z_norms, sims, ws = [], [], []

        for k in range(num_groups):
            group_name = group_names[k]
            indices = c_group_indices[group_name]
            
            c_group = c_group_indices['_raw_c'][:, indices] if '_raw_c' in c_group_indices else c_group_indices[group_name]
            z_group = z_groups[:, k, :]
            
            cue_dist = torch.cdist(c_group, c_group, p=2)
            
            if self.local_sigma_mode == 'offline_global_median':
                if self.feature_stats is not None and 'group_medians' in self.feature_stats:
                    median_dist = self.feature_stats['group_medians'].get(group_name, None)
                    if median_dist is not None:
                        local_sigma = median_dist / math.sqrt(math.log(2.0))
                    else:
                        local_sigma = self.sac_sigma * math.sqrt(len(indices))
                else:
                    local_sigma = self.sac_sigma * math.sqrt(len(indices))
            elif self.local_sigma_mode == 'dynamic_batch_median':
                diag_mask_med = torch.eye(B, device=device, dtype=torch.bool)
                off_diag_dist = cue_dist[~diag_mask_med]
                median_dist = off_diag_dist.median()
                local_sigma = median_dist / math.sqrt(math.log(2.0))
                if local_sigma < 1e-4:
                    local_sigma = math.sqrt(len(indices))
            elif self.local_sigma_mode == 'chi2_median':
                chi2_medians = {1: 0.455, 2: 1.386, 3: 2.366, 4: 3.357, 5: 4.351, 6: 5.348, 7: 6.346}
                D = len(indices)
                median_D = chi2_medians.get(D, D - 2/3)
                local_sigma = self.sac_sigma * math.sqrt(median_D)
            else:
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

        if num_groups > 0:
            total_loss = total_loss / num_groups

        if return_diagnostics:
            return total_loss, torch.stack(z_norms, dim=1), torch.stack(sims, dim=0), torch.stack(ws, dim=0), entropies

        return total_loss

    def _compute_single_vector_sac(self, z_spat, c_spat, eps=1e-8):
        """
        Single-vector continuous weighted contrastive loss for Spatial Branch (5 spatial features).
        """
        device = z_spat.device
        B = z_spat.shape[0]

        if B < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)

        cue_dist = torch.cdist(c_spat, c_spat, p=2)
        
        D = c_spat.shape[1] # D = 5 spatial features
        if self.local_sigma_mode == 'offline_global_median':
            if self.feature_stats is not None and 'group_medians' in self.feature_stats:
                median_dist = self.feature_stats['group_medians'].get('spatial', None)
                if median_dist is not None:
                    local_sigma = median_dist / math.sqrt(math.log(2.0))
                else:
                    local_sigma = self.sac_sigma * math.sqrt(D)
            else:
                local_sigma = self.sac_sigma * math.sqrt(D)
        elif self.local_sigma_mode == 'dynamic_batch_median':
            diag_mask_med = torch.eye(B, device=device, dtype=torch.bool)
            off_diag_dist = cue_dist[~diag_mask_med]
            median_dist = off_diag_dist.median()
            local_sigma = median_dist / math.sqrt(math.log(2.0))
            if local_sigma < 1e-4:
                local_sigma = math.sqrt(D)
        elif self.local_sigma_mode == 'chi2_median':
            chi2_medians = {1: 0.455, 2: 1.386, 3: 2.366, 4: 3.357, 5: 4.351, 6: 5.348, 7: 6.346}
            median_D = chi2_medians.get(D, D - 2/3)
            local_sigma = self.sac_sigma * math.sqrt(median_D)
        else:
            local_sigma = self.sac_sigma * math.sqrt(D)

        w = torch.exp(-(cue_dist / local_sigma) ** 2)
        z_norm = F.normalize(z_spat, dim=-1)
        sim = torch.matmul(z_norm, z_norm.T) / self.sac_temperature
        
        diag_mask = torch.eye(B, device=device, dtype=torch.bool)
        off_diag = ~diag_mask
        
        exp_sim = torch.exp(sim) * off_diag.float()
        den = exp_sim.sum(dim=1, keepdim=True) + eps
        log_prob = sim - torch.log(den)
        
        w_masked = w * off_diag.float()
        w_sum = w_masked.sum(dim=1, keepdim=True) + eps
        w_norm = w_masked / w_sum
        
        loss_per_sample = -(w_norm * log_prob).sum(dim=1)
        return loss_per_sample.mean()

    def forward(self, X, W, c_mono=None, c_spatial=None, return_diagnostics=False):
        """
        Forward pass for pretraining.
        
        Args:
            X: [B, 4, F, T] complex STFT tensor (Re_L, Im_L, Re_R, Im_R)
            W: [B, 1, 1, T] binary time-frame mask (1 = unmasked, 0 = masked)
            c_mono: [B, K_mono] monaural acoustic features
            c_spatial: [B, 5] spatial acoustic features (TDOA, GCC-PHAT peak, Sub-band Coherence)
        """
        B = X.shape[0]
        
        # 1. Dual Encoder Pass
        h_spat, h_spec = self.dual_encoder(X, W)
        
        # 2. Reconstruction Loss (CCSR Eq. 6)
        if self.recon_lambda > 0:
            loss_recon = self._reconstruct(h_spat, h_spec, X, W)
        else:
            loss_recon = torch.tensor(0.0, device=X.device)
        
        # 3. SPECTRAL SAC BRANCH (Attaches to h_spec with CA Layer & sac.acoustic_features)
        if self.sac_spec_lambda > 0 and c_mono is not None:
            Q_spec = self.group_queries_spec.unsqueeze(0).expand(B, -1, -1)
            attn_spec, _ = self.cross_attention_spec(query=Q_spec, key=h_spec[:, 1:, :], value=h_spec[:, 1:, :])
            attn_spec = attn_spec.view(B, self.num_active_spec_groups, self.num_queries_per_group, -1).mean(dim=2)
            Z_spec_groups = self.projection_head_spec(attn_spec)
            
            groups_dict = {g: self.spec_group_indices[g] for g in self.spec_group_indices}
            groups_dict['_raw_c'] = c_mono
            
            if return_diagnostics:
                loss_sac_spec, _, _, _, entropies_spec = self._compute_factorized_sac(
                    Z_spec_groups, groups_dict, self.num_active_spec_groups, return_diagnostics=True
                )
            else:
                loss_sac_spec = self._compute_factorized_sac(Z_spec_groups, groups_dict, self.num_active_spec_groups)
                entropies_spec = {}
        else:
            loss_sac_spec = torch.tensor(0.0, device=X.device)
            entropies_spec = {}

        # 4. SPATIAL SAC BRANCH (Attaches to h_spat with Projection Head & NO CA layer)
        if self.sac_spat_lambda > 0 and c_spatial is not None:
            if not self.use_cross_attention_spat:
                e_spat = h_spat[:, 1:, :].mean(dim=1) # [B, D]
                z_spat = self.projection_head_spat(e_spat) # [B, proj_dim]
                loss_sac_spat = self._compute_single_vector_sac(z_spat, c_spatial)
            else:
                Q_spat = self.group_queries_spat.unsqueeze(0).expand(B, -1, -1)
                attn_spat, _ = self.cross_attention_spat(query=Q_spat, key=h_spat[:, 1:, :], value=h_spat[:, 1:, :])
                attn_spat = attn_spat.view(B, self.num_active_spat_groups, self.num_queries_per_group, -1).mean(dim=2)
                Z_spat_groups = self.projection_head_spat(attn_spat)
                
                groups_dict_spat = {g: self.spat_group_indices[g] for g in self.spat_group_indices}
                groups_dict_spat['_raw_c'] = c_spatial
                loss_sac_spat = self._compute_factorized_sac(Z_spat_groups, groups_dict_spat, self.num_active_spat_groups)
        else:
            loss_sac_spat = torch.tensor(0.0, device=X.device)

        # 5. Total Weighted Loss
        loss_total = (
            self.recon_lambda * loss_recon +
            self.sac_spec_lambda * loss_sac_spec +
            self.sac_spat_lambda * loss_sac_spat
        )
        
        output = {
            'loss_total': loss_total,
            'loss_recon': loss_recon,
            'loss_sac_spec': loss_sac_spec,
            'loss_sac_spat': loss_sac_spat,
            'entropies_spec': entropies_spec,
        }
        
        return output


class BinauralSSAMBASACModelParallel(nn.Module):
    """
    DataParallel wrapper for BinauralSSAMBASACModel.
    Scatters inputs across GPUs and averages scalar loss dictionary elements.
    """
    def __init__(self, model: BinauralSSAMBASACModel):
        super().__init__()
        self.dp = nn.DataParallel(model)
        self.module = model

    def forward(self, X, W, c_mono=None, c_spatial=None, return_diagnostics=False):
        if return_diagnostics:
            return self.module(X, W, c_mono=c_mono, c_spatial=c_spatial, return_diagnostics=True)
        out = self.dp(X, W, c_mono=c_mono, c_spatial=c_spatial)
        res = {}
        for k, v in out.items():
            if isinstance(v, torch.Tensor):
                res[k] = v.mean()
            else:
                res[k] = v
        return res

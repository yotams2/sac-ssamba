import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from binaural_dual_stream_sac.binaural_sac_model import BinauralSSAMBASACModel


class MonauralSpectralUpstreamExpert(nn.Module):
    """
    S3PRL Upstream Expert wrapper for monaural downstream tasks (VoxCeleb SID, IEMOCAP ER).
    Routes representations through the spectral_encoder ONLY to avoid spatial contamination.
    """
    def __init__(self, model_checkpoint_path=None, model_config=None):
        super().__init__()
        if model_config is None:
            model_config = {}

        self.binaural_model = BinauralSSAMBASACModel(**model_config)
        
        if model_checkpoint_path is not None and os.path.exists(model_checkpoint_path):
            sd = torch.load(model_checkpoint_path, map_location='cpu')
            self.binaural_model.load_state_dict(sd, strict=False)

        # Freeze spectral encoder for upstream feature extraction
        self.spectral_encoder = self.binaural_model.dual_encoder.spectral_encoder
        self.spectral_encoder.eval()
        for param in self.spectral_encoder.parameters():
            param.requires_grad = False

    def forward(self, x_mono):
        """
        Args:
            x_mono: [B, 1, F, T] or [B, F, T] mono spectrogram or STFT tensor
        Returns:
            features: [B, embed_dim] mean-pooled patch representation
        """
        if x_mono.dim() == 3:
            x_mono = x_mono.unsqueeze(1)
            
        # Duplicate mono input to 4 channels [B, 4, F, T]
        if x_mono.shape[1] == 1:
            x_4ch = x_mono.repeat(1, 4, 1, 1)
        elif x_mono.shape[1] == 2:
            x_4ch = x_mono.repeat(1, 2, 1, 1)
        else:
            x_4ch = x_mono

        with torch.no_grad():
            hidden_states = self.spectral_encoder._encode_with_mamba(x_4ch)
            
        # Mean pool patch tokens (skipping CLS token at index 0)
        patch_tokens = hidden_states[:, 1:, :]
        pooled_features = patch_tokens.mean(dim=1)
        return pooled_features


class SpatialBinauralUpstreamExpert(nn.Module):
    """
    S3PRL Upstream Expert wrapper for spatial downstream tasks (Speaker Localization, DoA, SELD).
    Routes representations through the spatial_encoder using Attention-Free Bilinear Fusion.
    """
    def __init__(self, model_checkpoint_path=None, model_config=None, embed_dim=768, proj_dim=768):
        super().__init__()
        if model_config is None:
            model_config = {}

        self.binaural_model = BinauralSSAMBASACModel(**model_config)
        
        if model_checkpoint_path is not None and os.path.exists(model_checkpoint_path):
            sd = torch.load(model_checkpoint_path, map_location='cpu')
            self.binaural_model.load_state_dict(sd, strict=False)

        self.spatial_encoder = self.binaural_model.dual_encoder.spatial_encoder
        self.spatial_encoder.eval()
        for param in self.spatial_encoder.parameters():
            param.requires_grad = False

        # Bilinear fusion projection layer: [e_L; e_R; e_L * e_R; |e_L - e_R|] -> 4 * embed_dim -> proj_dim
        self.bilinear_fusion = nn.Sequential(
            nn.Linear(4 * embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, proj_dim)
        )

    def forward(self, X_binaural):
        """
        Args:
            X_binaural: [B, 4, F, T] complex STFT tensor (Re_L, Im_L, Re_R, Im_R)
        Returns:
            fused_spatial_representation: [B, proj_dim]
        """
        # Create Left-ear view [B, 4, F, T] (duplicate Left ear)
        X_Left = X_binaural.clone()
        X_Left[:, 2:, :, :] = X_binaural[:, :2, :, :]
        
        # Create Right-ear view [B, 4, F, T] (duplicate Right ear)
        X_Right = X_binaural.clone()
        X_Right[:, :2, :, :] = X_binaural[:, 2:, :, :]

        with torch.no_grad():
            h_L = self.spatial_encoder._encode_with_mamba(X_Left)[:, 1:, :].mean(dim=1)
            h_R = self.spatial_encoder._encode_with_mamba(X_Right)[:, 1:, :].mean(dim=1)

        # Bilinear Fusion: [e_L; e_R; e_L * e_R; |e_L - e_R|]
        concat_features = torch.cat([h_L, h_R, h_L * h_R, torch.abs(h_L - h_R)], dim=-1)
        fused_features = self.bilinear_fusion(concat_features)
        return fused_features

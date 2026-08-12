import sys
import os
import torch
import torch.nn as nn

# Ensure parent directory imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.both_models import AMBAModel

def apply_ccsr_masking(X, W):
    """
    Pre-embedding channel-conditional time-frame masking per CCSR (Eq. 7 & 8).
    
    Args:
        X: [B, 4, F, T] tensor where dim 1 contains (Re_L, Im_L, Re_R, Im_R)
        W: [B, 1, 1, T] binary time-frame mask (1 = unmasked, 0 = masked)
        
    Returns:
        X_spat: [B, 4, F, T] - Same time-frame mask W applied to both channels (Left * W, Right * W)
        X_spec: [B, 4, F, T] - Target Left channel masked with W (Left * W), Right channel inverse-masked (Right * (1-W))
    """
    # Spatial encoder sees same time-frame mask W on both Left and Right channels
    X_spat = X * W
    
    # Spectral encoder sees Left channel masked with W, Right channel inverse-masked with (1-W)
    X_spec = X.clone()
    X_spec[:, :2, :, :] = X[:, :2, :, :] * W         # Target Left channel masked with W
    X_spec[:, 2:, :, :] = X[:, 2:, :, :] * (1.0 - W) # Reference Right channel inverse-masked with (1 - W)
    
    return X_spat, X_spec


class BinauralAMBAEncoder(nn.Module):
    """
    Dual-stream Mamba encoder for CCSR-style spatial/spectral disentanglement.
    Holds spatial_encoder and spectral_encoder instances.
    """
    def __init__(self, encoder_config, share_weights=False):
        super().__init__()
        self.share_weights = share_weights
        
        # Ensure 4 channels for complex STFT input (Re_L, Im_L, Re_R, Im_R)
        cfg = dict(encoder_config)
        cfg['in_chans'] = 4
        
        self.spatial_encoder = AMBAModel(**cfg)
        if share_weights:
            self.spectral_encoder = self.spatial_encoder
        else:
            self.spectral_encoder = AMBAModel(**cfg)

    def forward(self, X, W):
        """
        Args:
            X: [B, 4, F, T] complex STFT tensor
            W: [B, 1, 1, T] binary time-frame mask
            
        Returns:
            h_spat: [B, num_patches + cls, D] spatial encoder representations
            h_spec: [B, num_patches + cls, D] spectral encoder representations
        """
        X_spat, X_spec = apply_ccsr_masking(X, W)
        h_spat = self.spatial_encoder._encode_with_mamba(X_spat)
        h_spec = self.spectral_encoder._encode_with_mamba(X_spec)
        return h_spat, h_spec

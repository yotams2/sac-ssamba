import os
import sys
import torch
import torch.nn as nn

# Ensure parent directory imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from binaural_dual_stream_sac.binaural_sac_model import BinauralSSAMBASACModel


class SSAMBASpatialDownstreamModel(nn.Module):
    """
    Downstream task model wrapper for SSAMBA (Binaural Dual-Stream SAC).
    Connects spatial and/or spectral encoder representations to downstream MLP heads
    for spatial acoustic prediction tasks (TDOA, DRR, T60, C50, ABS).
    """
    def __init__(
        self,
        fshape=16,
        tshape=16,
        input_fdim=256,
        input_tdim=1024,
        model_size='base',
        embed_dim=768,
        downstream_token='all',
        downstream_head='mlp',
        downstream_embed='spat',
        downstream_dlabel=1,
        load_pretrained_mdl_path=None,
        param_frozen=False,
        device='cuda'
    ):
        super().__init__()
        self.downstream_token = downstream_token
        self.downstream_embed = downstream_embed
        self.downstream_dlabel = downstream_dlabel
        self.device = device

        model_config = {
            'fshape': fshape,
            'tshape': tshape,
            'input_fdim': input_fdim,
            'input_tdim': input_tdim,
            'model_size': model_size,
            'embed_dim': embed_dim,
        }

        # Instantiate full Binaural SSAMBA SAC model
        self.ssamba_model = BinauralSSAMBASACModel(**model_config)

        # Load pre-trained weights if path provided
        if load_pretrained_mdl_path and os.path.exists(load_pretrained_mdl_path):
            self.load_pretrained_weights(load_pretrained_mdl_path)

        # Freeze encoder parameters if running linear evaluation
        if param_frozen:
            for param in self.ssamba_model.dual_encoder.parameters():
                param.requires_grad = False
            print("Spatial/Spectral encoders frozen for linear evaluation.")

        # Determine downstream head input dimension
        if downstream_embed == 'spec_spat':
            dembed_ds = embed_dim * 2
        else:
            dembed_ds = embed_dim

        # Downstream MLP Head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dembed_ds),
            nn.Linear(dembed_ds, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, downstream_dlabel)
        )

    def load_pretrained_weights(self, checkpoint_path):
        """
        Loads pre-trained checkpoint weights into self.ssamba_model.
        Robustly handles DataParallel prefixes ('dp.module.', 'module.')
        """
        print(f"Loading pre-trained SSAMBA weights from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            sd = checkpoint['model']
        elif isinstance(checkpoint, dict):
            sd = checkpoint
        else:
            sd = checkpoint

        new_sd = {}
        for k, v in sd.items():
            key = k
            if key.startswith('dp.module.'):
                key = key.replace('dp.module.', '', 1)
            elif key.startswith('module.'):
                key = key.replace('module.', '', 1)
            new_sd[key] = v

        missing, unexpected = self.ssamba_model.load_state_dict(new_sd, strict=False)
        print(f"Weights loaded cleanly. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

    def forward(self, reim_rebatch):
        """
        Args:
            reim_rebatch: [B, 2, F, T, 2] tensor containing Real & Imag parts for Left and Right channels.
                          Alternatively: [B, 2, F, T] complex STFT tensor.
        Returns:
            pred: [B, 1] predicted downstream scalar target.
            pooled_embed: [B, embed_dim] pooled representation.
            hidden_states: raw encoder outputs.
        """
        # Convert reim_rebatch to [B, 4, F, T] real-valued tensor with (Re_L, Im_L, Re_R, Im_R)
        if reim_rebatch.dim() == 5:
            # reim_rebatch: [B, 2_ch, F, T, 2_reim]
            re_L = reim_rebatch[:, 0, :, :, 0]
            im_L = reim_rebatch[:, 0, :, :, 1]
            re_R = reim_rebatch[:, 1, :, :, 0]
            im_R = reim_rebatch[:, 1, :, :, 1]
            X_4ch = torch.stack([re_L, im_L, re_R, im_R], dim=1) # [B, 4, F, T]
        elif reim_rebatch.is_complex():
            # reim_rebatch: [B, 2_ch, F, T] complex
            re_L = reim_rebatch[:, 0, :, :].real
            im_L = reim_rebatch[:, 0, :, :].imag
            re_R = reim_rebatch[:, 1, :, :].real
            im_R = reim_rebatch[:, 1, :, :].imag
            X_4ch = torch.stack([re_L, im_L, re_R, im_R], dim=1)
        else:
            X_4ch = reim_rebatch

        # Pass through Dual AMBA Encoders (unmasked)
        h_spat, h_spec = self.ssamba_model.dual_encoder(X_4ch, W=None)

        # Select downstream representation
        if self.downstream_embed == 'spec_spat':
            embed = torch.cat([h_spec, h_spat], dim=-1)
        elif self.downstream_embed == 'spec':
            embed = h_spec
        elif self.downstream_embed == 'spat':
            embed = h_spat
        else:
            raise ValueError(f"Unrecognized downstream_embed: {self.downstream_embed}")

        # Pooling across patch dimension
        if self.downstream_token == 'cls':
            pooled_embed = embed[:, 0, :]
        else:
            # 'all': mean pool patch tokens (skipping index 0 if CLS token present)
            if embed.shape[1] > 1:
                pooled_embed = embed[:, 1:, :].mean(dim=1)
            else:
                pooled_embed = embed.mean(dim=1)

        # Downstream MLP prediction
        pred = self.mlp_head(pooled_embed)

        return pred, pooled_embed, [h_spat, h_spec]

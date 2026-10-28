"""
Quick check of the GRAM-T wrapper on random audio (run from sac-ssamba/src):
    python -m gramt_wrapper.smoke_test [model_id]
Checks output shapes / token counts, that the left and right ears are encoded separately,
and that the layer-wise path matches the official HF "raw" representation.
"""
import sys

import torch

from gramt_wrapper import GRAMTEncoder, GRAMTClassifier, DEFAULT_MODEL_ID

model_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL_ID
device = "cuda" if torch.cuda.is_available() else "cpu"

enc = GRAMTEncoder(model_id).to(device).eval()
print(f"{model_id}: in_channels={enc.in_channels} grid=({enc.grid_f},{enc.grid_t}) "
      f"output_dim={enc.output_dim} downsample={enc.downsample_rate}")

wavs = [torch.randn(2, 16000), torch.randn(2, 16000 * 5), torch.randn(16000 * 3)]
with torch.no_grad():
    hidden, n_tokens = enc(wavs)
print("layers:", len(hidden), "shape:", tuple(hidden[-1].shape), "valid tokens:", n_tokens)
assert len(hidden) == enc.num_layers + 1
for wav, n in zip(wavs, n_tokens):
    expected = round(wav.shape[-1] / enc.downsample_rate)
    assert abs(n - expected) <= 1, (n, expected)

# swapping the ears must change the representation of a binaural model
lr = torch.randn(2, 16000 * 2)
lr[1] *= 0.1
with torch.no_grad():
    h_lr = enc([lr])[0][-1]
    h_rl = enc([lr.flip(0)])[0][-1]
diff = (h_lr - h_rl).abs().mean().item()
print("L/R swap feature difference:", diff)
if enc.in_channels == 2:
    assert diff > 1e-4, "binaural model is insensitive to the L/R order"

# match the official HF "raw" strategy on one 2 s chunk
log_mel, _ = enc.wavs_to_logmel([lr])
with torch.no_grad():
    ours = enc.encode_logmel(log_mel[:, :, :enc.chunk_frames])[-1]
    ref = enc.gram._get_segment_representation(log_mel[:, :, :enc.chunk_frames], strategy="raw")
print("max |ours - HF raw|:", (ours - ref).abs().max().item())
assert torch.allclose(ours, ref, atol=1e-4)

clf = GRAMTClassifier(35, model_id=model_id).to(device)
out = clf(torch.randn(4, 2, 16000, device=device), "ft_avgtok")
print("classifier logits:", tuple(out.shape))
assert out.shape == (4, 35)
print("OK")

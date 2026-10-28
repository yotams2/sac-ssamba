"""
GRAM-T (https://github.com/labhamlet/GRAM-T) wrapped as a layer-wise binaural feature encoder.

Preprocessing mirrors the official HF feature extractor (labhamlet/gramt-*):
resample to 32 kHz, joint RMS normalisation to -14 dBFS (one gain for both ears, so ILDs are kept),
128-bin log-mel with a 10 ms hop. Encoding mirrors the "raw" strategy: non-overlapping 2 s chunks,
patch tokens rearranged to [T_tokens, F_patches * D] and concatenated over time.

Unlike the HF `get_audio_representation`, this returns every encoder layer and does not force
`torch.no_grad()`, so it can be used for both frozen (weighted-sum) probing and fine-tuning.
"""
import torch
import torch.nn as nn
import torchaudio
from einops import rearrange

GRAMT_SAMPLE_RATE = 32000
GRAMT_HOP_LENGTH = 320
DEFAULT_MODEL_ID = "labhamlet/gramt-binaural-frame"


def normalize_rms(wav, target_dbfs=-14.0):
    rms = torch.sqrt(torch.mean(wav ** 2))
    if rms == 0:
        return wav
    gain_db = target_dbfs - 20 * torch.log10(rms)
    return wav * (10 ** (gain_db / 20))


class GRAMTEncoder(nn.Module):
    def __init__(self, model_id=DEFAULT_MODEL_ID, input_sample_rate=16000, rms_normalize=True):
        """
        Args:
            model_id: HF repo id (labhamlet/gramt-binaural-frame | gramt-binaural-time | gramt-mono)
                      or a local directory holding a downloaded copy of one.
            input_sample_rate: sample rate of the waveforms passed to `forward`.
            rms_normalize: apply the -14 dBFS RMS normalisation used by the official extractor.
        """
        super().__init__()
        from transformers import AutoModel
        self.gram = AutoModel.from_pretrained(model_id, trust_remote_code=True).model

        self.in_channels = self.gram.in_channels
        self.chunk_frames = self.gram.input_length            # 200 frames = 2 s
        self.grid_f, self.grid_t = self.gram.grid_size        # frame: (8, 25), time: (1, 100)
        self.embed_dim = self.gram.encoder_embedding_dim      # 768
        self.num_layers = len(self.gram.encoder.layers)       # 12
        self.output_dim = self.grid_f * self.embed_dim        # per-token feature size of the "raw" strategy
        self.input_sample_rate = input_sample_rate
        self.rms_normalize = rms_normalize

        chunk_secs = self.chunk_frames * GRAMT_HOP_LENGTH / GRAMT_SAMPLE_RATE
        # input samples per output token (e.g. 1280 = 80 ms @ 16 kHz for the frame model)
        self.downsample_rate = int(round(input_sample_rate * chunk_secs / self.grid_t))

        if input_sample_rate != GRAMT_SAMPLE_RATE:
            self.resample = torchaudio.transforms.Resample(input_sample_rate, GRAMT_SAMPLE_RATE)
        else:
            self.resample = nn.Identity()
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=GRAMT_SAMPLE_RATE,
            n_fft=1024,
            win_length=1024,
            hop_length=GRAMT_HOP_LENGTH,
            f_min=50,
            f_max=GRAMT_SAMPLE_RATE // 2,
            n_mels=self.gram.input_shape[0],
            power=2.0,
        )

    def _match_channels(self, wav):
        """wav: [C, N] -> [in_channels, N]. Mono is duplicated, as in the official extractor."""
        if wav.shape[0] == self.in_channels:
            return wav
        if self.in_channels == 2 and wav.shape[0] == 1:
            return wav.repeat(2, 1)
        if self.in_channels == 1:
            return wav.mean(dim=0, keepdim=True)
        if self.in_channels == 2 and wav.shape[0] > 2:
            return wav[:2]
        raise ValueError(f"Cannot map {wav.shape[0]}-channel audio to a {self.in_channels}-channel GRAM-T")

    def wavs_to_logmel(self, wavs):
        """
        Args:
            wavs: list of [T] or [C, T] waveforms (or a [B, C, T] tensor) at `input_sample_rate`.
        Returns:
            log_mel: [B, in_channels, T_frames_max, n_mels], zero-padded like the HF extractor.
            n_frames: list of valid frame counts per item.
        """
        device = self.melspec.spectrogram.window.device
        feats = []
        # mel power can overflow fp16, so keep feature extraction in fp32 under autocast
        with torch.autocast(device_type=device.type, enabled=False):
            for wav in wavs:
                wav = torch.as_tensor(wav, device=device).float()
                if wav.ndim == 1:
                    wav = wav.unsqueeze(0)
                wav = self._match_channels(wav)
                wav = self.resample(wav)
                if self.rms_normalize:
                    wav = normalize_rms(wav)
                mel = self.melspec(wav).transpose(1, 2)  # [C, T, F]
                feats.append((mel + torch.finfo(mel.dtype).eps).log())
        n_frames = [f.shape[1] for f in feats]
        # pad_sequence pads dim 0, so put time first
        log_mel = torch.nn.utils.rnn.pad_sequence([f.transpose(0, 1) for f in feats], batch_first=True)
        return log_mel.transpose(1, 2), n_frames

    def _encode_layers(self, x):
        """x: [N, C, chunk_frames, F] -> list of num_layers + 1 tensors [N, grid_t, grid_f * D]."""
        gram = self.gram
        B = x.shape[0]
        x = gram.spectrogram_normalize(x.transpose(2, 3))
        tokens = gram.patch_strategy.embed(x, gram.patch_embed)
        n_cls = gram.encoder_cls_token_num
        tokens = tokens + gram.encoder.pos_embedding[:, n_cls:, :]
        cls_token = gram.cls_token.expand(B, -1, -1) + gram.encoder.pos_embedding[:, :1, :]
        h = gram.encoder.dropout(torch.cat((cls_token, tokens), dim=1))

        hidden = [h]
        for i, block in enumerate(gram.encoder.layers):
            h = block(h)
            hidden.append(gram.encoder.ln(h) if i == self.num_layers - 1 else h)
        return [
            rearrange(h[:, n_cls:, :], "b (f t) d -> b t (f d)", f=self.grid_f, d=self.embed_dim)
            for h in hidden
        ]

    def encode_logmel(self, log_mel):
        """
        Args:
            log_mel: [B, C, T_frames, F]
        Returns:
            list of num_layers + 1 tensors [B, T_tokens, grid_f * D], where
            T_tokens = ceil(T_frames / chunk_frames) * grid_t (padding tokens included).
        """
        B, C, T, F = log_mel.shape
        n_chunks = max(1, -(-T // self.chunk_frames))
        pad = n_chunks * self.chunk_frames - T
        if pad > 0:
            log_mel = torch.nn.functional.pad(log_mel, (0, 0, 0, pad))
        # fold the 2 s chunks into the batch: [B * n_chunks, C, chunk_frames, F]
        chunks = log_mel.reshape(B, C, n_chunks, self.chunk_frames, F).permute(0, 2, 1, 3, 4)
        chunks = chunks.reshape(B * n_chunks, C, self.chunk_frames, F)
        layers = self._encode_layers(chunks)
        return [h.reshape(B, n_chunks * self.grid_t, -1) for h in layers]

    def frames_to_tokens(self, n_frames):
        return max(1, -(-n_frames * self.grid_t // self.chunk_frames))

    def forward(self, wavs):
        """
        Returns:
            hidden_states: list of num_layers + 1 tensors [B, T_tokens, grid_f * D]
            n_tokens: list of valid token counts per item
        """
        log_mel, n_frames = self.wavs_to_logmel(wavs)
        hidden_states = self.encode_logmel(log_mel)
        n_tokens = [self.frames_to_tokens(n) for n in n_frames]
        return hidden_states, n_tokens

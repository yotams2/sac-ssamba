# SAC acoustic features: definitions and selection

*2026-11-09, branch `formant-functionals`. Numbers from `scripts/feature_analysis.sbatch`, job 23493999
(`docs/feature_analysis.json`).*

## Summary

- **Definitions follow eGeMAPS** (Eyben et al., 2016) wherever it defines the feature. Each descriptor is computed
  every 10 ms, then summarised per clip by its mean and coefficient of variation (CoV), over voiced frames for voice
  features and over non-silent frames for the rest. That one rule replaces all the earlier ad hoc choices.
- **The old set had real confounds.** MFCC0 and the flux variance track recording level. MFCCs, centroid, ZCR and
  "rhythm" shift when a clip is zero-padded (as every LibriSpeech clip under 11.24 s is in pretraining). In the new
  set, every feature moves by at most 0.03 IQR under both tests.
- **The new set carries more information for all three tasks.** Leave-one-out 1-NN on the whole vector:
  speaker 0.72 → 0.86, word 0.19 → 0.25, emotion 0.44 → 0.48.
- **Proposal:** drop 3 of the 20 candidate values (ZCR, voiced segments per second, unvoiced segment length): two are
  near-duplicates of other features and one carries almost nothing. That leaves 17 values in the same 5 groups.

## Definitions (new set)

All on the pitch track's grid: 10 ms hop, voicing by normalised autocorrelation (Boersma 1993, as in Praat).
Spectral descriptors use 25 ms Hann windows centred on the same frames. Clips with fewer than 5 voiced frames get
neutral fallback values.

| Group | Feature | Definition | Values |
|---|---|---|---|
| Prosody | F0 | mean and CoV over voiced frames, semitones from 27.5 Hz (eGeMAPS) | 2 |
| Prosody | Temporal | voiced segments per second; mean voiced and unvoiced segment length, between the first and last non-silent frame (eGeMAPS) | 3 |
| Vocal tract | Formants | F1-F3 by LPC with Praat's default settings (11 kHz, order 10); mean and CoV over voiced frames (eGeMAPS) | 6 |
| Timbre | MFCC | MFCC 1-4 (26 mel bands, no c0), mean over voiced frames (eGeMAPS) | 4 |
| Voice quality | HNR | mean harmonics-to-noise ratio over voiced frames, dB (Boersma 1993) | 1 |
| Scene | Spectral centroid | mean over non-silent frames | 1 |
| Scene | Spectral flux | L2 distance between consecutive level-normalised spectra; mean and CoV over voiced frames (eGeMAPS) | 2 |
| Scene | ZCR | mean zero-crossing rate over non-silent frames | 1 |

Spectral centroid and ZCR are not in eGeMAPS. Both are standard (MPEG-7, and textbook ZCR) and need no extra
explanation.

## How each feature was scored

No downstream test data is used.

| Column | Meaning | Data |
|---|---|---|
| split | Spearman between the two halves of each clip: is the feature a stable property of the clip? | LibriSpeech dev-clean, 2703 clips |
| spk | rank-based eta², share of variance explained by speaker (0-1) | same, 40 speakers |
| word | same, by word | Speech Commands v2 **train** split, 100 clips × 35 words |
| emo | same, by emotion, after z-scoring each speaker's features | IEMOCAP sessions 2-5 (the fold-1 **training** set), 4446 clips, 4 classes |
| level | median change when the clip is played 20 dB quieter, in IQRs of the feature (0 = level-invariant) | 500 LibriSpeech clips |
| pad | median change when a 4 s excerpt is zero-padded to the 11.24 s pretraining window, in IQRs | same |
| max ρ | largest absolute Spearman with another feature of the same set | LibriSpeech |

LibriSpeech clips are prepared as in mono pretraining (first 11.24 s, zero-padded). Values above about 0.5 IQR in the
level or pad columns are confounds.

## Results: new set

| Feature | split | spk | word | emo | level | pad | max ρ (with) |
|---|---|---|---|---|---|---|---|
| F0 mean | 0.86 | **0.86** | 0.09 | 0.08 | 0.00 | 0.00 | 0.58 (MFCC4) |
| F0 CoV | 0.51 | 0.56 | 0.10 | 0.09 | 0.00 | 0.00 | 0.55 |
| Voiced segments/s | 0.44 | 0.45 | 0.13 | 0.18 | 0.00 | 0.02 | **0.93** (voiced length) |
| Voiced length | 0.47 | 0.53 | 0.26 | 0.20 | 0.00 | 0.01 | **0.93** (segments/s) |
| Unvoiced length | 0.38 | 0.26 | 0.18 | 0.01 | 0.00 | 0.03 | 0.59 |
| F1 mean | 0.64 | 0.72 | 0.29 | 0.03 | 0.01 | 0.00 | 0.62 |
| F2 mean | 0.42 | 0.51 | **0.51** | 0.01 | 0.01 | 0.00 | 0.57 |
| F3 mean | 0.56 | 0.57 | 0.18 | 0.02 | 0.02 | 0.00 | 0.57 |
| F1 CoV | 0.71 | 0.73 | 0.21 | 0.02 | 0.01 | 0.00 | 0.72 (MFCC4) |
| F2 CoV | 0.42 | 0.45 | 0.29 | 0.02 | 0.02 | 0.01 | 0.48 |
| F3 CoV | 0.48 | 0.53 | 0.19 | 0.02 | 0.01 | 0.00 | 0.40 |
| MFCC1 | 0.77 | 0.75 | 0.27 | 0.14 | 0.00 | 0.00 | 0.78 (centroid) |
| MFCC2 | 0.86 | **0.85** | 0.18 | 0.17 | 0.00 | 0.00 | 0.68 |
| MFCC3 | 0.68 | 0.69 | 0.37 | 0.04 | 0.00 | 0.00 | 0.51 |
| MFCC4 | 0.82 | 0.79 | 0.20 | 0.21 | 0.00 | 0.00 | 0.72 (F1 CoV) |
| HNR | 0.71 | 0.70 | 0.23 | 0.03 | 0.00 | 0.00 | 0.71 (voiced length) |
| Centroid | 0.53 | 0.62 | 0.40 | 0.22 | 0.00 | 0.01 | **0.95** (ZCR) |
| Flux mean | 0.51 | 0.53 | 0.13 | 0.08 | 0.00 | 0.00 | 0.44 |
| Flux CoV | 0.67 | 0.67 | 0.14 | 0.01 | 0.00 | 0.00 | 0.63 |
| ZCR mean | 0.47 | 0.55 | 0.36 | **0.23** | 0.00 | 0.01 | **0.95** (centroid) |

## Results: old set (`main`, formants already fixed)

| Feature | split | spk | word | emo | level | pad | max ρ (with) |
|---|---|---|---|---|---|---|---|
| F0 median | 0.87 | 0.87 | 0.05 | 0.08 | 0.00 | 0.00 | 0.43 |
| F0 IQR | 0.41 | 0.41 | 0.06 | 0.01 | 0.00 | 0.00 | 0.35 |
| F1 (10 loudest frames) | 0.56 | 0.57 | 0.32 | 0.10 | 0.00 | 0.00 | 0.52 |
| F2 (10 loudest frames) | 0.29 | 0.34 | 0.53 | 0.01 | 0.00 | 0.00 | 0.37 |
| F3 (10 loudest frames) | 0.35 | 0.31 | 0.22 | 0.03 | 0.00 | 0.00 | 0.37 |
| MFCC0 | 0.68 | 0.20 | 0.04 | 0.32 | **1.18** | **1.39** | 0.88 (centroid) |
| MFCC1 | 0.66 | 0.30 | 0.27 | 0.21 | 0.00 | **1.41** | 0.76 |
| MFCC2 | 0.87 | 0.80 | 0.11 | 0.18 | 0.00 | 0.49 | 0.59 |
| MFCC3 | 0.75 | 0.59 | 0.23 | 0.07 | 0.00 | **0.94** | 0.43 |
| MFCC4 | 0.84 | 0.68 | 0.09 | 0.21 | 0.00 | 0.36 | 0.59 |
| HNR | 0.71 | 0.70 | 0.23 | 0.03 | 0.00 | 0.00 | 0.38 |
| Centroid | 0.60 | 0.19 | 0.22 | 0.23 | 0.00 | **1.26** | 0.98 (ZCR) |
| Flux var | 0.60 | 0.42 | 0.04 | 0.28 | **0.94** | 0.47 | 0.61 |
| ZCR mean | 0.55 | 0.23 | 0.23 | 0.23 | 0.00 | **1.17** | 0.98 (centroid) |
| ZCR var ("rhythm") | 0.25 | 0.42 | n/a* | 0.03 | 0.00 | 0.30 | 0.45 |

\*Undefined (NaN) on clips shorter than about 1 s: 3489 of 3500 SC2 clips and 28 of 4446 IEMOCAP clips. It never
happens on the 11.24 s pretraining window, but it shows the feature was not a measure of rhythm.

## Whole-vector 1-NN accuracy (leave one out, z-scored)

| Set | Speaker (chance 0.025) | Word (chance 0.029) | Emotion (majority 0.305) |
|---|---|---|---|
| Old (15 values) | 0.721 | 0.190 | 0.441 |
| New (20 values) | **0.855** | **0.249** | **0.475** |

## Reading

- **The fixes were needed, and eGeMAPS implies them.** Adopting eGeMAPS definitions is the same change as the bug
  fixes from the review: drop c0, normalise spectra before flux, average over voiced or non-silent frames only,
  replace "rhythm" with segment statistics. The level and pad columns show each of those confounds is gone.
- **Speaker:** F0 mean, MFCC 1-4, F1 mean and CoV, and HNR carry most of it (eta² 0.70-0.86). Under the old
  definitions MFCC1 and centroid looked weak on speaker (0.30, 0.19) because padding and level noise drowned the
  signal; corrected, they reach 0.75 and 0.62.
- **Word (SC2):** F2 mean (0.51), centroid (0.40), MFCC3 (0.37) and ZCR (0.36), i.e. vowel quality and frication.
- **Emotion (ER):** no single feature is strong after speaker normalisation. The best are ZCR, centroid, MFCC4 and the
  voiced-segment statistics (0.18-0.23). Old MFCC0 (0.32) and flux variance (0.28) scored higher because both follow
  loudness, which carries arousal but also recording gain, so it can't be used as is.
- **Redundancy:** centroid and ZCR (ρ 0.95), and voiced segments/s and voiced length (ρ −0.93), are near-duplicates.
  Inside a SAC group a duplicate doubles that factor's weight in the distance.

## Proposed set: 17 values

Drop:
- **ZCR mean:** duplicates centroid. Centroid is equal or better on speaker and word and level on emotion.
- **Voiced segments/s:** duplicates voiced length, which scores higher on every task.
- **Unvoiced length:** weakest feature overall (split 0.38, speaker 0.26, emotion 0.01).

`SAC_FEATURES=f0_mean,f0_var,voiced_len,formant_stats,mfcc,hnr,centroid,flux,flux_cov FEATURE_ALIAS=egemaps17`

## Open choices

- **Loudness CoV.** eGeMAPS has loudness mean and CoV. The mean depends on recording gain, so leave it out. The CoV is
  level-invariant and is the eGeMAPS way to capture loudness dynamics, a known arousal cue that the old MFCC0 and flux
  variance carried by accident. It's one value, and it would need a test before adding.
- **Formant CoV in a speaker-heavy group.** F1-F3 CoV are as speaker-informative as the means, but weak on word and
  emotion. Keep them unless the next run suggests the Vocal_Tract group over-weights speaker.

## Limitations

- Eta² is univariate: it scores each feature alone, and SAC uses the features in groups.
- SC2 clips are 1 s, while pretraining uses about 11 s windows. The word column shows what a feature can encode,
  not what it will encode in pretraining.
- One dataset per task, no seeds. The analysis takes 10 minutes and can be rerun with other data.
- Cost: on CPU (8 threads) a batch of 64 × 11.24 s takes 2.0 s vs 0.9 s for the old set. On the GPU it runs on the
  device; watch peak memory in the first run.

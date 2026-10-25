# Phase 1G: Valence Data-Level Pretraining Audit

Created: 2026-07-14 CST

## Purpose

Test whether external relative or cross-dataset Valence supervision can improve
the Phase 1 full-song Valence teacher without reading the PMEmo final locked
test.

## Current Findings

- MusAV provides public pairwise annotations and evaluation scripts, but its
  2,100 Spotify audio previews are available only by request for non-commercial
  scientific research. The repository alone cannot support immediate training.
- Direct DEAM-to-PMEmo MERT song-polarity transfer failed. A Ridge model selected
  on DEAM validation achieved PMEmo development PCC `0.153755`, CCC `0.037881`,
  and pairwise accuracy `0.558755`; therefore full DEAM MuQ extraction is not
  justified.
- The PMEmo final locked test remains untouched.

## Artifacts

- `musav_public/`: official public MusAV metadata and evaluation repository.
- `scripts/audit_deam_to_pmemo_song_valence_transfer.py`
- `outputs/deam_to_pmemo_song_transfer_20260714/`

## Next Decision

Continue PMEmo locked-development optimization with selective audio feature
ablation. Revisit MusAV ranking pretraining only after research-audio access is
obtained.

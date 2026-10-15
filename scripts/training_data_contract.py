"""Dataset intake contract — synthetic 2D-US elongated training spectra.

Producer (offline generator) -> Consumer (this repo, scripts/prepare_training_data.py).

Each spectrum file (JSON, debug/interchange only; prefer .npz for bulk):
  {
    "data":   [[float,...],...],   # (N_F1, N_F2): each ROW is a full F2 trace at F1 index k
    "labels": [[0/1,...],...],     # (N_F1, N_F2) binary, same shape as data (optional but recommended)
    "meta": {                      # spectrum-level + per-peak metadata (recommended)
      "SW_F1_Hz": float, "SW_F2_Hz": float,
      "N1": int, "N2": int,
      "noise_sd": float,           # must equal A_min per paper p.12
      "r": float,                  # Gamma_F1_pts / Gamma_F2_pts used
      "peaks": [                   # one entry per peak (256 expected)
        {"A": float,
         "Gamma_F1_Hz": float, "Gamma_F2_Hz": float,
         "eta_F1": float, "eta_F2": float,
         "i0": float, "j0": float, # continuous sub-pixel centre
         "group": "isolated|pair|triple|shoulder",
         "d1": float, "d2": float} # separation in FWHM units (pairs), else null
      ]
    }
  }

Orientation contract (mandatory):
  data[k][l] with k = F1 (indirect, SHORT axis) and l = F2 (direct, LONG axis).
  I.e. shape (N_F1, N_F2), e.g. (256, 512). Rows are F2 traces.
  This matches split_2D() convention S[i=F1][j=F2] with NO transpose needed.
  First half of each cross vector (indices 0:64) = F1 (elongated arm).

Bulk format (recommended for the ~1000 spectra):
  data_0000.npz with arrays S (H,W float32), L (H,W float32 {0,1}), plus meta JSON sidecar.
  Naming: split assigned BY FILE (never mix points of one spectrum across train/val).

Acceptance checks (run by scripts/prepare_training_data.py --validate-only):
  1. S.shape[0] < S.shape[1]  (F1 short, e.g. 256; F2 long, e.g. 512)
  2. L.shape == S.shape, values subset of {0,1}
  3. positives/spectrum scaled: ~256 peaks on the 256x512 reference patch
     (band [0.5x, 2x] expected; --min-positives override for debug probes)
  4. no NaN/Inf in S; finite after split (Norm==0 sanitised to 0)
  5. (logged, not fatal) probe FWHM: F1 arm wider than F2 arm on brightest point
  6. noise SD ≈ A_min within tolerance (if meta present)
"""

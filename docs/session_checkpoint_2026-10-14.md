# Checkpoint de sesión — 2026-10-14 (re-entrenamiento 2D-US elongado)

## Objetivo
Re-entrenar/fine-tune del modelo 2D-US (`Trained/Probability_2D/US/DNN_0`, input `(128,1)=64+64`)
para picos elongados en F1 (indirecta), con datos sintéticos pseudo-Voigt generados en frecuencia.

## Estado Fase 1 — COMPLETA
- Dataset: `/Users/abolanos/git/cheminfo/spectrum-generator/dataset` (1000 espectros, 2.5 GB)
  - `spectra/<id>.bin` float32 LE (256,2048)=(F1,F2); `labels/<id>.bin` uint8 binaria
  - `manifest.jsonl` (1000 líneas) + `dataset.json` (split 800/200 explícito, rRange [1,3], aMin 0.001, bandas 70/25/5)
- Validación: 0 errores (256000 picos; bandas 69.9/25.0/5.1; 575835 positivos con dobles sub-píxel)
- Manifiesto: `/Users/abolanos/git/mrai_p3/data/train_2d_elong/manifest.json` (800/200, por referencia memmap)
- Scripts: `scripts/jres_ingest.py`, `scripts/cross_extract.py` (espejo NumPy de `split_2D`, sin TF),
  `scripts/prepare_training_data.py`, `scripts/training_data_contract.py`
- Reqs: `docs/training_dataset_requirements.md` (§1–§10, nomenclatura Γ=FWHM / η=forma / r)
- Plan generador: `/Users/abolanos/git/cheminfo/spectrum-generator/docs/jres-dataset-plan-v3.md`

## Estado Fase 2 — SMOKE EN CURSO
- Script: `scripts/train_model.py` (222 líneas; `build_model()` WNN_2D filter=10, fusión 2*filter,
  BCE+Adam, `BinSequence` memmap + split_2d bajo demanda, EarlyStopping+Checkpoint+CSVLogger,
  `history.npy/txt`, `model.export()`, smoke predict)
- Parches Keras 3 aplicados: Lambda+`keras.ops` (softsign/split/concat/squeeze) y slicing con
  `output_shape` explícita (el `K.split` perdía forma estática → conv colapsaba a 0).
- Entorno TF: `/Users/abolanos/.pyenv/versions/3.13.2/envs/tflow/bin/python` (pyenv `tflow`, TF 2.21.0,
  Keras 3.15.1 con TFSMLayer; activar con `pyenv activate tflow`; usar `python`, no `python3`)
- Smoke: `--init Trained/Probability_2D/US/DNN_0 --lr 1e-4 --batch 65536 --epochs 3 --max-pts 16384
  --out /tmp/smoke_elong`, log `/tmp/retrain_smoke.log`, PID 65262
- Último estado: época 1/3 step ~11/200, acc 0.80, loss 0.694→0.670, ~20 s/step (~1 h/época, CPU, GPUs: [])
- Alertas: (1) auc≈0.50 (posible aprendizaje de clase mayoritaria, bg_ratio=4); (2) fine-tune copió 0/62
  pesos (nombres `conv1d` vs `conv1d_52`, `dense` vs `dense_10`) → entrena desde cero con lr bajo.
- Decisión: dejar terminar (~3 h), luego mapeo de pesos + rebalanceo antes del completo (210 épocas inviable
  en CPU sin optimizar: ~semanas; ref paper 4×A100 ~3 h para 2D).

## Comandos clave
```bash
pyenv activate tflow
python -c "import tensorflow as tf; print(tf.__version__)"  # 2.21.0
tail -f /tmp/retrain_smoke.log
# completo (tras smoke + parches):
nohup python scripts/train_model.py --data /Users/abolanos/git/mrai_p3/data/train_2d_elong \
  --init Trained/Probability_2D/US/DNN_0 --lr 1e-4 --batch 65536 --epochs 210 --patience 20 \
  --out Trained/Probability_2D_Elongated/DNN_0 > /tmp/retrain_fase2.log 2>&1 &
```

## Pendiente (Fase 3)
Evaluación §9 del .md: recall-vs-d por eje y r, F1 estratificado, IP3 en hombros,
base vs nuevo, control r=1. Notas: rRange [1,3] (ampliar a 5–8 en v2 dataset);
clases extra multiplet (578) + 1 demoted (decidir inclusión); rejilla 256×2048 (no parche 512).

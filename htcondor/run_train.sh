#!/usr/bin/env bash
# HTCondor entry point for mrai_p3 training on NMRbox.
#
# Lo llama htcondor/train.sub (arguments = RUN_NAME POOL [extra...]); tambien
# se puede correr a mano. La geometria de negativos duros sale del manifest del
# pool, nunca de flags escritos a mano.
#
# Usage:
#   run_train.sh RUN_NAME [POOL] [extra train_model.py args...]
#
#   RUN_NAME   output dir under Trained/, e.g. out_nmrbox_454510
#   POOL       pool dir under data/, default pre_aniso_f2lo0_454510
#   extra args are appended last, so they override the defaults below
#              (argparse keeps the last occurrence of each flag)
#
# Environment:
#   MRAI_PYTHON   interpreter (default: $HOME/vtflow/bin/python)
#   MRAI_THREADS  TF/OMP threads; keep it equal to request_cpus (default 8)
#   MRAI_STAGE    1 = copy the pool to the node-local scratch first (default 1)
#   MRAI_INIT     SavedModel for --init (default Trained/Probability_2D/US/DNN_0,
#                 empty string = train from scratch)
#   MRAI_PAIR_FRAC  value for --pair-frac (default 0; needs a pool with `pair`)
#   MRAI_DRY_RUN  1 = print the command and exit without training
#
# The hard-negative geometry (ring / band / aniso rectangle / axis quota) is
# read from the pool's manifest.json instead of being passed by hand: the pool
# and the training must use the same geometry or the A/B is invalid.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 RUN_NAME [POOL] [extra train_model.py args...]" >&2
  exit 2
fi

RUN_NAME="$1"; shift
POOL="${1:-${MRAI_POOL:-pre_aniso_f2lo0_454510}}"
if [[ $# -ge 1 ]]; then
  if [[ "$POOL" == -* ]]; then
    echo "ERROR: el segundo argumento es el POOL, no un flag ($POOL)." >&2
    echo "       Pasá el pool explícito si querés agregar flags después." >&2
    exit 2
  fi
  shift
fi

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# --- interpreter -----------------------------------------------------------
# ~/vtflow: Python 3.12 + TF 2.21 / Keras 3 (el antiguo ~/venvs/mrai se eliminó).
# MRAI_PYTHON puede ser un path absoluto o un nombre en el PATH; como el .sub
# usa getenv=True, exportarlo antes de condor_submit alcanza para cambiarlo.
PY="${MRAI_PYTHON:-$HOME/vtflow/bin/python}"
if [[ "$PY" != */* ]] && command -v "$PY" >/dev/null 2>&1; then
  PY="$(command -v "$PY")"
fi
if [[ ! -x "$PY" ]]; then
  if [[ -n "${MRAI_PYTHON:-}" ]]; then
    echo "ERROR: MRAI_PYTHON=$MRAI_PYTHON no es un ejecutable válido" >&2
    exit 1
  fi
  if command -v python3 >/dev/null 2>&1; then
    echo "WARN: no existe $PY; uso $(command -v python3) (¿tiene TensorFlow?)" >&2
    PY="$(command -v python3)"
  else
    echo "ERROR: no encuentro $PY ni python3 en el PATH" >&2
    exit 1
  fi
fi

# --- threads: must match request_cpus in the .sub ---------------------------
THREADS="${MRAI_THREADS:-${OMP_NUM_THREADS:-8}}"
export OMP_NUM_THREADS="$THREADS"
export TF_NUM_INTRAOP_THREADS="$THREADS"
export TF_NUM_INTEROP_THREADS=1
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-1}"

# --- pool (optionally staged to the node-local execute dir) -----------------
SRC_POOL="$REPO/data/$POOL"
if [[ ! -f "$SRC_POOL/manifest.json" ]]; then
  echo "ERROR: no existe $SRC_POOL/manifest.json" >&2
  exit 1
fi

SCRATCH_BASE="${_CONDOR_SCRATCH_DIR:-/tmp}"
DATA_DIR="$SRC_POOL"
STAGE="${MRAI_STAGE:-1}"

# getenv=True hereda el TMPDIR de la VM que hace submit (acá /scratch/...),
# que en el worker no existe. Si no es escribible, usamos el scratch del job.
if [[ -z "${TMPDIR:-}" || ! -d "$TMPDIR" || ! -w "$TMPDIR" ]]; then
  export TMPDIR="$SCRATCH_BASE"
  mkdir -p "$TMPDIR"
fi

is_zero() { [[ -z "$1" || "$1" == "0" || "$1" == "0.0" ]]; }

if [[ "$STAGE" != "0" && -n "${_CONDOR_SCRATCH_DIR:-}" ]]; then
  STAGE_POOL="$SCRATCH_BASE/pool_$POOL"
  need_kb=$(du -sk "$SRC_POOL" | awk '{print $1}') || true
  avail_kb=$(df -Pk "$SCRATCH_BASE" | awk 'NR==2{print $4}') || true
  need_kb="${need_kb:-0}"
  avail_kb="${avail_kb:-0}"
  if (( avail_kb > need_kb + 1048576 )); then
    if [[ ! -f "$STAGE_POOL/manifest.json" ]]; then
      echo "[stage] copiando $POOL ($((need_kb / 1024)) MB) a $STAGE_POOL"
      # copia a un temporal y rename: si el job muere a mitad de camino no
      # queda un pool incompleto que el próximo intento tomaría por bueno
      tmp="$SCRATCH_BASE/pool_$POOL.tmp.$$"
      rm -rf "$tmp"
      mkdir -p "$tmp"
      cp -a "$SRC_POOL/." "$tmp/"
      rm -rf "$STAGE_POOL"
      mv "$tmp" "$STAGE_POOL"
    fi
    DATA_DIR="$STAGE_POOL"
  else
    echo "[stage] sin espacio en $SCRATCH_BASE ($((avail_kb / 1024)) MB libres)" >&2
    echo "[stage] leo el pool desde \$HOME (más lento, pero correcto)" >&2
  fi
fi

# --- geometry, straight from the pool manifest ------------------------------
# El cupo del eje va ANTES de HARD_ANISO: la última variable de `read` absorbe
# el resto de la línea (la geometría anisótropa son 4 ints).
read -r HARD_FRAC HARD_PX HARD_FRAC2 HARD_BAND2 HARD_AXIS_FRAC HARD_ANISO < <(
  "$PY" - "$SRC_POOL/manifest.json" <<'PY'
import json
import sys

m = json.load(open(sys.argv[1]))
aniso = m.get("hard_aniso") or []
print(
    m.get("hard_frac", 0),
    m.get("hard_px", 2),
    m.get("hard_frac2", 0),
    m.get("hard_band2", 0),
    m.get("hard_axis_frac", 0),
    " ".join(str(x) for x in aniso) if aniso else "-",
)
PY
)

if [[ -z "$HARD_FRAC" || -z "$HARD_PX" || -z "$HARD_FRAC2" || \
      -z "$HARD_BAND2" || -z "$HARD_AXIS_FRAC" || -z "$HARD_ANISO" ]]; then
  echo "ERROR: no pude leer la geometría de $SRC_POOL/manifest.json" >&2
  exit 1
fi

if [[ "$HARD_ANISO" != "-" ]] && is_zero "$HARD_FRAC2"; then
  echo "ERROR: manifest con hard_aniso pero hard_frac2=0; pool inconsistente" >&2
  exit 1
fi
# --- output dir, resume and .done marker ------------------------------------
OUT="$REPO/Trained/$RUN_NAME"
mkdir -p "$OUT" "$REPO/htcondor/logs"

if [[ -f "$OUT/.done" ]]; then
  echo "ERROR: $OUT ya terminó (tiene .done). Elegí otro RUN_NAME." >&2
  exit 1
fi

ARGS=(--data "$DATA_DIR" --preextracted --out "Trained/$RUN_NAME"
      --batch 65536 --bg-ratio 4 --epochs 210 --patience 20 --sync-every 5)

if ! is_zero "$HARD_FRAC"; then
  ARGS+=(--hard-frac "$HARD_FRAC" --hard-px "$HARD_PX")
fi
if ! is_zero "$HARD_FRAC2"; then
  if [[ "$HARD_ANISO" != "-" ]]; then
    # shellcheck disable=SC2086  # 4 ints separados por espacios
    ARGS+=(--hard-aniso $HARD_ANISO --hard-frac2 "$HARD_FRAC2")
  elif [[ "$HARD_BAND2" != "0" ]]; then
    ARGS+=(--hard-band2 "$HARD_BAND2" --hard-frac2 "$HARD_FRAC2")
  fi
fi
# Cupo del eje F1 dentro de la banda (0 = sin cupo). Sale del manifest para que
# el entrenamiento no pueda pedir una relación eje:diagonal distinta a la que
# tiene el pool: train_model.py corta si no coinciden.
if ! is_zero "$HARD_AXIS_FRAC"; then
  ARGS+=(--hard-axis-frac "$HARD_AXIS_FRAC")
fi
if [[ "${MRAI_PAIR_FRAC:-0}" != "0" ]]; then
  ARGS+=(--pair-frac "$MRAI_PAIR_FRAC")
fi

# checkpoints en disco local del nodo, sincronizados a Trained/$RUN_NAME
SCRATCH_RUN="$SCRATCH_BASE/run_$RUN_NAME"
mkdir -p "$SCRATCH_RUN"
# conservar el log de una corrida anterior: CSVLogger agrega si el archivo
# existe, así el sync no pisa la historia al reanudar
if [[ -f "$OUT/training.log" && ! -f "$SCRATCH_RUN/training.log" ]]; then
  cp "$OUT/training.log" "$SCRATCH_RUN/training.log"
fi
ARGS+=(--scratch "$SCRATCH_RUN")

if [[ -f "$OUT/ckpt.keras" ]]; then
  echo "[resume] $OUT/ckpt.keras"
  ARGS+=(--resume "$OUT/ckpt.keras")
else
  INIT="${MRAI_INIT-Trained/Probability_2D/US/DNN_0}"
  if [[ -n "$INIT" && -d "$INIT" ]]; then
    ARGS+=(--init "$INIT")
  elif [[ -n "$INIT" ]]; then
    echo "WARN: --init $INIT no existe; entreno desde cero" >&2
  fi
fi

ARGS+=("$@")

echo "== mrai_p3 training =="
echo "run:     $RUN_NAME"
echo "pool:    $POOL"
echo "data:    $DATA_DIR"
echo "out:     $OUT"
echo "threads: $THREADS"
echo "cmd:     $PY scripts/train_model.py ${ARGS[*]}"

if [[ "${MRAI_DRY_RUN:-0}" != "0" ]]; then
  echo "[dry-run] no se entrena"
  exit 0
fi

if "$PY" scripts/train_model.py "${ARGS[@]}"; then
  touch "$OUT/.done"
else
  status=$?
  if [[ -f "$OUT/ckpt.keras" ]]; then
    echo "FALLÓ (exit $status); reenviá el job y reanuda desde $OUT/ckpt.keras" >&2
  else
    echo "FALLÓ (exit $status); no hay checkpoint, el reenvío arranca de cero" >&2
  fi
  exit "$status"
fi

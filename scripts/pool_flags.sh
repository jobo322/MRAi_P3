#!/usr/bin/env bash
# Imprime los flags de negativos duros de un pool, listos para train_model.py.
#
#   scripts/pool_flags.sh data/pre_aniso_f2lo0_3268_f2hi4_a20
#   -> --hard-frac 0.32 --hard-px 2 --hard-aniso 3 8 0 4 --hard-frac2 0.68 --hard-axis-frac 0.2
#
# Por qué
# ------
# La geometría (anillo, banda anisótropa y cupo del eje) vive en el
# manifest.json del pool, que es su única fuente de verdad. El entrenamiento ya
# la recibe de ahí (`htcondor/run_train.sh`, que es lo que corre `train.sub`),
# así que este helper es para los dos casos donde el manifest todavía no existe
# o no se está usando:
#
#   * crear un pool: los flags van a `preextract.py`, que es quien escribe el
#     manifest (ver htcondor/preextract.sub);
#   * correr `train_model.py` a mano, sin el wrapper.
#
#   htcondor/submit.sh preextract.sub RunName=a25 Pool=data/<pool> \
#       Extra="--pool-k 24 $(scripts/pool_flags.sh data/<otro-pool>)"
#
# `train.sub` NO necesita el helper: `run_train.sh` lee la geometría del pool.
#
# Qué imprime, en este orden (solo lo que está encendido):
#   --hard-frac F --hard-px P            si F > 0
#   --hard-aniso a b c d --hard-frac2 G  si hay banda anisótropa
#   --hard-band2 B --hard-frac2 G        si la banda es isotrópica
#   --hard-axis-frac A                   si A > 0 (cupo del eje F1, d2 == 0)
#
# Con el pool clásico (sin banda) imprime solo `--hard-frac F --hard-px P`, y con
# un pool sin geometría no imprime nada (el entrenamiento cae a los defaults).
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "uso: $0 <dir-del-pool>" >&2
  exit 2
fi

POOL="$1"
MAN="$POOL/manifest.json"
if [[ ! -f "$MAN" ]]; then
  echo "ERROR: no existe $MAN" >&2
  exit 1
fi

PY="${MRAI_PYTHON:-python3}"
if [[ ! -x "$PY" ]] && command -v "$PY" >/dev/null 2>&1; then
  PY="$(command -v "$PY")"
fi
if [[ ! -x "$PY" ]]; then
  echo "ERROR: no encuentro un intérprete de Python ($PY)" >&2
  exit 1
fi

"$PY" - "$MAN" <<'PY'
import json
import sys

m = json.load(open(sys.argv[1]))
flags = []

hard_frac = float(m.get("hard_frac", 0) or 0)
hard_px = m.get("hard_px", 2)
if hard_frac > 0:
    flags += ["--hard-frac", f"{hard_frac:g}", "--hard-px", str(hard_px)]

frac2 = float(m.get("hard_frac2", 0) or 0)
aniso = m.get("hard_aniso") or []
band2 = int(m.get("hard_band2", 0) or 0)
if frac2 > 0:
    if aniso:
        flags += ["--hard-aniso"] + [str(int(x)) for x in aniso]
        flags += ["--hard-frac2", f"{frac2:g}"]
    elif band2 > 0:
        flags += ["--hard-band2", str(band2), "--hard-frac2", f"{frac2:g}"]

# El cupo del eje: train_model.py corta si no coincide con el del pool.
axis_frac = float(m.get("hard_axis_frac", 0) or 0)
if axis_frac > 0:
    if not aniso:
        sys.exit("ERROR: el pool tiene hard_axis_frac pero no hard_aniso; "
                 "pool inconsistente")
    if int(aniso[2]) != 0:
        sys.exit(f"ERROR: el pool tiene hard_axis_frac {axis_frac:g} con "
                 f"F2LO={aniso[2]}; el cupo necesita F2LO = 0")
    flags += ["--hard-axis-frac", f"{axis_frac:g}"]

print(" ".join(flags))
PY

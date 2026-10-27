#!/usr/bin/env bash
# Envia un .sub a HTCondor con overrides, dejando una copia en htcondor/runs/.
#
#   htcondor/submit.sh <archivo.sub> RunName=<nombre> [VAR=valor ...]
#
# - RunName es obligatorio (nombra logs, pool, out y la copia del .sub).
# - Los VAR=valor se insertan justo ANTES de `queue`, asi pisan los valores
#   por defecto del template (HTCondor expande $(VAR) al crear el job).
# - El job corre con cwd = raiz del repo, que es donde vive el codigo.
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "uso: $0 <archivo.sub> RunName=<nombre> [VAR=valor ...]" >&2
  exit 2
fi

sub=$1
shift
here=$(cd "$(dirname "$0")" && pwd)
repo=$(dirname "$here")

case $sub in
  */*) path=$sub ;;
  *)   path=$here/$sub ;;
esac
[ -f "$path" ] || { echo "no existe el submit file: $path" >&2; exit 2; }

run=""
ov_text=""
for kv in "$@"; do
  case $kv in
    *=*) ;;
    *) echo "override invalido (falta '='): $kv" >&2; exit 2 ;;
  esac
  key=${kv%%=*}
  val=${kv#*=}
  [ "$key" = "RunName" ] && run=$val
  ov_text+="$key = $val"$'\n'
done

if [ -z "$run" ]; then
  echo "falta RunName=<nombre> (alfanumerico, '_', '-', '.')" >&2
  exit 2
fi
if ! [[ $run =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "RunName invalido: '$run' (solo letras, numeros, '_', '-', '.')" >&2
  exit 2
fi

if ! command -v condor_submit >/dev/null 2>&1; then
  echo "condor_submit no esta instalado en esta maquina." >&2
  echo "Ver htcondor/README.md seccion 1 (solicitud a soporte)." >&2
  exit 127
fi

mkdir -p "$here/runs" "$here/logs"
out=$here/runs/$run.sub

awk -v ov="$ov_text" '
  BEGIN { n = split(ov, lines, "\n") }
  /^[[:space:]]*queue([[:space:]]|$)/ {
    for (i = 1; i <= n; i++) if (lines[i] != "") print lines[i]
  }
  { print }
' "$path" > "$out"

echo "==> $out"
cd "$repo"
exec condor_submit "$out"

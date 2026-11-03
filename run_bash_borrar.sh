   #!/bin/bash

   # 1. Activar el entorno (ajustar según tu configuración)
   # pyenv activate tflow

   OUT_DIR="Trained"
   PL_THE=0.6
   MSEP="3 1"

   echo "🚀 Iniciando recuperación de métricas con nuevos offsets (2, 3, 6, 8, 10, 12)..."
   echo "---------------------------------------------------------------------------"

   # Buscar todas las carpetas que coincidan con el patrón de inferencia aniso
   for dir in $(find $OUT_DIR -maxdepth 1 -type d -name "infer_out_aniso_*"); do
       echo "Processing: $dir"

       # Ejecutar el script de recuperación
       python scripts/plotting/recover_peaks.py \
           --out "$dir" \
           --pl-the $PL_THE \
           --msep $MSEP

       if [ $? -eq 0 ]; then
           echo "✅ Done: $dir"
       else
           echo "❌ Error processing $dir"
       fi
       echo "---------------------------------------------------------------------------"
   done

   echo "✨ Todos los reportes han sido actualizados."

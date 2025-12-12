@echo off
echo ========================================================
echo INICIANDO PROCESSAMENTO L2 (COMPORTAMENTAL) - MES DE AGOSTO - Janela 60 minutos
echo ========================================================

REM --- MES 1 (Agosto) ---
REM Nota: A primeiras duas semana pode ter Z-Scores estranhos por falta de historico previo (Cold Start), é normal.

echo.
echo [4/13] L2: A processar semana de 29-Ago a 03-Set...(ATUALIZADO V2.0)
python .\extracao-metricas-tpr-layer-02.py --minutes 60 --start-date "2025-08-29T07:00:00" --end-date "2025-09-03T00:00:00"


echo.
echo ========================================================
echo PROCESSAMENTO L2 CONCLUIDO!
echo ========================================================
pause
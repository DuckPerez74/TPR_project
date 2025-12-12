@echo off
echo ========================================================
echo INICIANDO PROCESSAMENTO L2 (COMPORTAMENTAL) - MES DE AGOSTO - Janela 10 minutos
echo ========================================================

REM --- MES 1 (Agosto) ---
REM Nota: A primeiras duas semana pode ter Z-Scores estranhos por falta de historico previo (Cold Start), é normal.

echo.
echo [X/XX] L2: A processar semana de 06-Ago a 18-Ago...
python .\extracao-metricas-tpr-layer-02.py --minutes 10 --start-date "2025-08-06T00:00:00" --end-date "2025-08-18T00:00:00"



echo.
echo ========================================================
echo PROCESSAMENTO L2 CONCLUIDO!
echo ========================================================
pause
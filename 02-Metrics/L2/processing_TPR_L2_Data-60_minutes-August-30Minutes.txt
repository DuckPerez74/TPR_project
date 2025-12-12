@echo off
echo ========================================================
echo INICIANDO PROCESSAMENTO L2 (COMPORTAMENTAL) - MES DE AGOSTO - Janela 30 minutos
echo ========================================================

REM --- MES 1 (Agosto) ---
REM Nota: A primeiras duas semana pode ter Z-Scores estranhos por falta de historico previo (Cold Start), é normal.

echo.
echo [1/13] L2: A processar semana de 06-Ago a 13-Ago...
python .\extracao-metricas-tpr-layer-02.py --minutes 30 --start-date "2025-08-06T00:00:00" --end-date "2025-08-13T00:00:00"

echo.
echo [2/13] L2: A processar semana de 13-Ago a 20-Ago...
python .\extracao-metricas-tpr-layer-02.py --minutes 30 --start-date "2025-08-13T00:00:00" --end-date "2025-08-20T00:00:00"

echo.
echo [3/13] L2: A processar semana de 20-Ago a 27-Ago...
python .\extracao-metricas-tpr-layer-02.py --minutes 30 --start-date "2025-08-20T00:00:00" --end-date "2025-08-27T00:00:00"

echo.
echo [4/13] L2: A processar semana de 27-Ago a 03-Set...
python .\extracao-metricas-tpr-layer-02.py --minutes 30 --start-date "2025-08-27T00:00:00" --end-date "2025-09-03T00:00:00"


echo.
echo ========================================================
echo PROCESSAMENTO L2 CONCLUIDO!
echo ========================================================
pause
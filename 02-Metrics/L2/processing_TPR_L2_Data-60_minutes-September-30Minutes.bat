@echo off
echo ========================================================
echo INICIANDO PROCESSAMENTO L2 (COMPORTAMENTAL) - MES DE SETEMBRO - Janela 30 minutos
echo ========================================================



REM --- MES 2 (Setembro) ---

echo.
echo [5/13] L2: A processar semana de 03-Set a 10-Set...
python .\extracao-metricas-tpr-layer-02.py --minutes 30 --start-date "2025-09-03T00:00:00" --end-date "2025-09-10T00:00:00"

echo.
echo [6/13] L2: A processar semana de 10-Set a 17-Set...
python .\extracao-metricas-tpr-layer-02.py --minutes 30 --start-date "2025-09-10T00:00:00" --end-date "2025-09-17T00:00:00"

echo.
echo [7/13] L2: A processar semana de 17-Set a 24-Set...
python .\extracao-metricas-tpr-layer-02.py --minutes 30 --start-date "2025-09-17T00:00:00" --end-date "2025-09-24T00:00:00"

echo.
echo [8/13] L2: A processar semana de 24-Set a 01-Out...
python .\extracao-metricas-tpr-layer-02.py --minutes 30 --start-date "2025-09-24T00:00:00" --end-date "2025-10-01T00:00:00"

echo.
echo ========================================================
echo PROCESSAMENTO L2 CONCLUIDO!
echo ========================================================
pause
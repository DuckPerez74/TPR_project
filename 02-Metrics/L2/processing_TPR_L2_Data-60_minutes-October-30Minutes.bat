@echo off
echo ========================================================
echo INICIANDO PROCESSAMENTO L2 (COMPORTAMENTAL) - MES DE OUTUBRO - Janela 30 minutos
echo ========================================================

REM --- MES 3 (Outubro ate inicio Nov) ---

echo.
echo [9/13] L2: A processar semana de 01-Out a 08-Out...
python .\extracao-metricas-tpr-layer-02.py --minutes 30 --start-date "2025-10-01T00:00:00" --end-date "2025-10-08T00:00:00"

echo.
echo [10/13] L2: A processar semana de 08-Out a 15-Out...
python .\extracao-metricas-tpr-layer-02.py --minutes 30 --start-date "2025-10-08T00:00:00" --end-date "2025-10-15T00:00:00"

echo.
echo [11/13] L2: A processar semana de 15-Out a 22-Out...
python .\extracao-metricas-tpr-layer-02.py --minutes 30 --start-date "2025-10-15T00:00:00" --end-date "2025-10-22T00:00:00"

echo.
echo [12/13] L2: A processar semana de 22-Out a 29-Out...
python .\extracao-metricas-tpr-layer-02.py --minutes 30 --start-date "2025-10-22T00:00:00" --end-date "2025-10-29T00:00:00"

echo.
echo [13/13] L2: A processar semana de 29-Out a 06-Nov...
python .\extracao-metricas-tpr-layer-02.py --minutes 30 --start-date "2025-10-29T00:00:00" --end-date "2025-11-06T00:00:00"

echo.
echo ========================================================
echo PROCESSAMENTO L2 CONCLUIDO!
echo ========================================================
pause
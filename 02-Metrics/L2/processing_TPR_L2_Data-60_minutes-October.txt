@echo off
echo ========================================================
echo INICIANDO PROCESSAMENTO L2 (COMPORTAMENTAL) - MES DE OUTUBRO - Janela 60 minutos
echo ========================================================

REM --- MES 3 (Outubro ate inicio Nov) ---

echo.
echo [12/13] L2: A processar semana de 23-Out a 29-Out...(ATUALIZADO V2.0)
python .\extracao-metricas-tpr-layer-02.py --minutes 60 --start-date "2025-10-23T13:00:00" --end-date "2025-10-29T00:00:00"

echo.
echo [13/13] L2: A processar semana de 29-Out a 06-Nov...
python .\extracao-metricas-tpr-layer-02.py --minutes 60 --start-date "2025-10-29T00:00:00" --end-date "2025-11-06T00:00:00"

echo.
echo ========================================================
echo PROCESSAMENTO L2 CONCLUIDO!
echo ========================================================
pause
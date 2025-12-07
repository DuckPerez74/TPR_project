@echo off
echo ========================================================
echo INICIANDO PROCESSAMENTO DE 3 MESES DE DADOS (L1 - 30m)
echo De: 2025-08-06
echo Ate: 2025-11-06
echo ========================================================


REM --- MES 3 (Outubro ate inicio Nov) ---

echo.
echo [10/13] A processar semana de 14-Out a 15-Out...(ATUALIZADO V6.0)
python .\extracao-metricas-tpr.py --minutes 30 --start-date "2025-10-14T18:00:00" --end-date "2025-10-15T00:00:00"

echo.
echo [11/13] A processar semana de 15-Out a 22-Out...
python .\extracao-metricas-tpr.py --minutes 30 --start-date "2025-10-15T00:00:00" --end-date "2025-10-22T00:00:00"

echo.
echo [12/13] A processar semana de 22-Out a 29-Out...
python .\extracao-metricas-tpr.py --minutes 30 --start-date "2025-10-22T00:00:00" --end-date "2025-10-29T00:00:00"

echo.
echo [13/13] A processar semana de 29-Out a 06-Nov...
python .\extracao-metricas-tpr.py --minutes 30 --start-date "2025-10-29T00:00:00" --end-date "2025-11-06T00:00:00"

echo.
echo ========================================================
echo PROCESSAMENTO CONCLUIDO COM SUCESSO!
echo ========================================================
pause

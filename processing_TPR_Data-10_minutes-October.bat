@echo off
echo ========================================================
echo INICIANDO PROCESSAMENTO DE 3 MESES DE DADOS (L1 - 10m)
echo De: 2025-08-06
echo Ate: 2025-11-06
echo ========================================================

REM --- MES 3 (Outubro ate inicio Nov) ---

echo.
echo [12/13] A processar semana de 26-Out a 29-Out...(ATUALIZADO V2.0)
python .\extracao-metricas-tpr-BackUP_Version.py --minutes 10 --start-date "2025-10-26T00:00:00" --end-date "2025-10-29T00:00:00"

echo.
echo [13/13] A processar semana de 29-Out a 06-Nov...
python .\extracao-metricas-tpr-BackUP_Version.py --minutes 10 --start-date "2025-10-29T00:00:00" --end-date "2025-11-06T00:00:00"

echo.
echo ========================================================
echo PROCESSAMENTO CONCLUIDO COM SUCESSO!
echo ========================================================
pause
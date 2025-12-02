@echo off
echo ========================================================
echo INICIANDO PROCESSAMENTO DE 3 MESES DE DADOS (L1 - 10m)
echo De: 2025-08-06
echo Ate: 2025-11-06
echo ========================================================


REM --- MES 2 (Setembro) ---

echo.
echo [6/13] A processar semana de 15-Set a 17-Set...(ATUALIZADO V2.0)
python .\extracao-metricas-tpr-BackUP_Version.py --minutes 10 --start-date "2025-09-15T15:00:00" --end-date "2025-09-17T00:00:00"

echo.
echo [7/13] A processar semana de 17-Set a 24-Set...
python .\extracao-metricas-tpr-BackUP_Version.py --minutes 10 --start-date "2025-09-17T00:00:00" --end-date "2025-09-24T00:00:00"


echo.
echo ========================================================
echo PROCESSAMENTO CONCLUIDO COM SUCESSO!
echo ========================================================
pause
@echo off
echo ========================================================
echo INICIANDO PROCESSAMENTO DE 3 MESES DE DADOS (L1 - 10m)
echo De: 2025-08-06
echo Ate: 2025-11-06
echo ========================================================


REM --- MES 2 (Setembro) ---

echo.
echo [5/13] A processar semana de 03-Set a 10-Set...(MODIFICADO PARA FAZER A PRIMEIRA QUINZENA DE SETEMBRO)
python .\extracao-metricas-tpr-BackUP_Version.py --minutes 10 --start-date "2025-09-03T00:00:00" --end-date "2025-09-10T00:00:00"

echo.
echo [6/13] A processar semana de 10-Set a 17-Set...
python .\extracao-metricas-tpr-BackUP_Version.py --minutes 10 --start-date "2025-09-10T00:00:00" --end-date "2025-09-17T00:00:00"

echo.
echo [7/13] A processar semana de 17-Set a 24-Set...
python .\extracao-metricas-tpr-BackUP_Version.py --minutes 10 --start-date "2025-09-17T00:00:00" --end-date "2025-09-24T00:00:00"


echo.
echo ========================================================
echo PROCESSAMENTO CONCLUIDO COM SUCESSO!
echo ========================================================
pause
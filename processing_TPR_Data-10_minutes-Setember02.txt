@echo off
echo ========================================================
echo INICIANDO PROCESSAMENTO DE 3 MESES DE DADOS (L1 - 10m)
echo De: 2025-08-06
echo Ate: 2025-11-06
echo ========================================================

REM --- MES 2 (Setembro) ---


echo.
echo [8/13] A processar semana de 24-Set a 01-Out...(MODIFICADO PARA FAZER A SEGUNDA QUINZENA DE SETEMBRO)
python .\extracao-metricas-tpr-BackUP_Version.py --minutes 10 --start-date "2025-09-24T00:00:00" --end-date "2025-10-01T00:00:00"


REM --- MES 3 (Outubro ate inicio Nov) ---

echo.
echo [9/13] A processar semana de 01-Out a 08-Out...
python .\extracao-metricas-tpr-BackUP_Version.py --minutes 10 --start-date "2025-10-01T00:00:00" --end-date "2025-10-08T00:00:00"

echo.
echo [10/13] A processar semana de 08-Out a 15-Out...
python .\extracao-metricas-tpr-BackUP_Version.py --minutes 10 --start-date "2025-10-08T00:00:00" --end-date "2025-10-15T00:00:00"

echo.
echo ========================================================
echo PROCESSAMENTO CONCLUIDO COM SUCESSO!
echo ========================================================
pause
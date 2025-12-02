@echo off
echo ========================================================
echo INICIANDO PROCESSAMENTO DE 3 MESES DE DADOS (L1 - 10m)
echo De: 2025-08-06
echo Ate: 2025-11-06
echo ========================================================

REM --- MES 3 (Outubro ate inicio Nov) ---

echo.
echo [9/13] A processar semana de 03-Out a 08-Out...ATUALIZADO V3.0 - FINAL!!!)
python .\extracao-metricas-tpr-BackUP_Version.py --minutes 10 --start-date "2025-10-03T00:00:00" --end-date "2025-10-08T00:00:00"

echo.
echo ========================================================
echo PROCESSAMENTO CONCLUIDO COM SUCESSO!
echo ========================================================
pause
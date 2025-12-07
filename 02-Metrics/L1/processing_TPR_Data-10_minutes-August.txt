@echo off
echo ========================================================
echo INICIANDO PROCESSAMENTO DE 3 MESES DE DADOS (L1 - 10m)
echo De: 2025-08-06
echo Ate: 2025-11-06
echo ========================================================

REM --- MES 1 (Agosto) ---

echo.
echo [4/13] A processar semana de 02-Set a 03-Set...(ATUALIZADO PARTE DE OUTUBRO FINAL)
python .\extracao-metricas-tpr-BackUP_Version.py --minutes 10 --start-date "2025-08-27T00:00:00" --end-date "2025-09-03T00:00:00"


echo.
echo ========================================================
echo PROCESSAMENTO CONCLUIDO COM SUCESSO!
echo ========================================================
pause
@echo off
echo ========================================================
echo INICIANDO PROCESSAMENTO DE 3 MESES DE DADOS (L1 - 60m)
echo De: 2025-08-06
echo Ate: 2025-11-06
echo ========================================================

REM --- MES 3 (Outubro ate inicio Nov) ---

echo.
echo [13/13] A processar semana de 01-Nov a 06-Nov... (Atualizado 7)
python .\extracao-metricas-tpr.py --minutes 60 --start-date "2025-11-01T00:00:00" --end-date "2025-11-06T00:00:00"

echo.
echo ========================================================
echo PROCESSAMENTO CONCLUIDO COM SUCESSO!
echo ========================================================
pause
@echo off
echo ========================================================
echo INICIANDO PROCESSAMENTO L2 (COMPORTAMENTAL) - MES DE OUTUBRO - Janela 10 minutos
echo ========================================================

REM --- MES 3 (Outubro ate inicio Nov) ---

echo.
echo [X/XX] L2: A processar semana de 02-Out a 10-Out...
python .\extracao-metricas-tpr-layer-02.py --minutes 10 --start-date "2025-10-03T01:00:00" --end-date "2025-10-17T00:00:00"

echo.
echo ========================================================
echo PROCESSAMENTO L2 CONCLUIDO!
echo ========================================================
pause
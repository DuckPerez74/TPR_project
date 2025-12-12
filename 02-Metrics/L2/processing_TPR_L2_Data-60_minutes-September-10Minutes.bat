@echo off
echo ========================================================
echo INICIANDO PROCESSAMENTO L2 (COMPORTAMENTAL) - MES DE SETEMBRO - Janela 10 minutos
echo ========================================================



REM --- MES 2 (Setembro) ---

echo.
echo [X/XX] L2: A processar semana de 09-Set a 17-Set...
python .\extracao-metricas-tpr-layer-02.py --minutes 10 --start-date "2025-09-09T00:00:00" --end-date "2025-09-17T00:00:00"


echo.
echo ========================================================
echo PROCESSAMENTO L2 CONCLUIDO!
echo ========================================================
pause
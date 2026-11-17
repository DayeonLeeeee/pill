@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (echo Run setup first. & pause & exit /b 1)
".venv\Scripts\python.exe" predict.py --inputs examples/inputs.json --device cpu --search --output results/example_cpu.jsonl
if errorlevel 1 (echo Inference failed. & pause & exit /b 1)
echo Saved: results/example_cpu.jsonl
pause

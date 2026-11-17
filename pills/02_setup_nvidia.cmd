@echo off
chcp 65001 >nul
cd /d "%~dp0"
py -3.12 bootstrap.py --device cuda
if errorlevel 1 (echo Installation failed. Python 3.12 64-bit and internet access are required. & pause & exit /b 1)
pause

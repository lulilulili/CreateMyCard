@echo off
setlocal
cd /d "%~dp0.."
python -m training.build_3b_dataset %*
if errorlevel 1 pause

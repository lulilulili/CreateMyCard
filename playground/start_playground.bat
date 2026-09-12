@echo off
title GenUI 3B Playground Launcher
cd /d "%~dp0.."
set "SYS32=%SystemRoot%\System32"

echo ==============================================
echo   GenUI ��Ƭ���������� - һ������
echo ==============================================

rem ---- [1/3] ȷ�� Ollama �����У����Ựͬʱ���п��� ----
set OLLAMA_ORIGINS=*
"%SYS32%\curl.exe" -s --max-time 3 http://localhost:11434/api/version >nul 2>&1
if errorlevel 1 (
    echo [1/3] Ollama δ���У���������...
    start "" "%LOCALAPPDATA%\Programs\Ollama\ollama app.exe"
    "%SYS32%\ping.exe" -n 7 127.0.0.1 >nul
) else (
    echo [1/3] Ollama ��������
)

rem ---- ���Ĭ��ģ���Ƿ�����ȡ ----
"%SYS32%\curl.exe" -s --max-time 5 http://localhost:11434/api/tags | "%SYS32%\findstr.exe" /C:"qwen2.5:3b" >nul 2>&1
if errorlevel 1 (
    echo [��ʾ] δ��⵽ qwen2.5:3b������Ĭ��ģ������ִ��: ollama pull qwen2.5:3b
) else (
    echo       ģ�� qwen2.5:3b �Ѿ���
)

rem ---- [2/3] ��������ҳ����� ----
"%SYS32%\netstat.exe" -ano | "%SYS32%\findstr.exe" /C:":8630 " | "%SYS32%\findstr.exe" LISTENING >nul 2>&1
if errorlevel 1 (
    echo [2/3] ����ҳ����� http://localhost:8630 ...
    start "GenUI Playground Server" /min cmd /c "cd /d "%~dp0.." && python -m http.server 8630"
    "%SYS32%\ping.exe" -n 3 127.0.0.1 >nul
) else (
    echo [2/3] ҳ�������������
)

rem ---- [3/3] ������� ----
echo [3/3] �������...
start "" "http://localhost:8630/playground/genui_3b_playground.html"

echo.
echo ��ɡ��رձ����ڲ�Ӱ��ʹ�ã�ҳ����񴰿�����С����
echo ����ʹ��ʱ�ص�����Ϊ "GenUI Playground Server" �Ĵ��ڼ��ɡ�
"%SYS32%\ping.exe" -n 6 127.0.0.1 >nul
exit

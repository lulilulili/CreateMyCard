@echo off
title GenUI 3B Playground Launcher
cd /d "%~dp0"
set "SYS32=%SystemRoot%\System32"

echo ==============================================
echo   GenUI 卡片生成演练场 - 一键启动
echo ==============================================

rem ---- [1/3] 确保 Ollama 在运行（本会话同时放行跨域） ----
set OLLAMA_ORIGINS=*
"%SYS32%\curl.exe" -s --max-time 3 http://localhost:11434/api/version >nul 2>&1
if errorlevel 1 (
    echo [1/3] Ollama 未运行，正在启动...
    start "" "%LOCALAPPDATA%\Programs\Ollama\ollama app.exe"
    "%SYS32%\ping.exe" -n 7 127.0.0.1 >nul
) else (
    echo [1/3] Ollama 已在运行
)

rem ---- 检查默认模型是否已拉取 ----
"%SYS32%\curl.exe" -s --max-time 5 http://localhost:11434/api/tags | "%SYS32%\findstr.exe" /C:"qwen2.5:3b" >nul 2>&1
if errorlevel 1 (
    echo [提示] 未检测到 qwen2.5:3b，如需默认模型请先执行: ollama pull qwen2.5:3b
) else (
    echo       模型 qwen2.5:3b 已就绪
)

rem ---- [2/3] 启动本地页面服务 ----
"%SYS32%\netstat.exe" -ano | "%SYS32%\findstr.exe" /C:":8630 " | "%SYS32%\findstr.exe" LISTENING >nul 2>&1
if errorlevel 1 (
    echo [2/3] 启动页面服务 http://localhost:8630 ...
    start "GenUI Playground Server" /min cmd /c "cd /d "%~dp0" && python -m http.server 8630"
    "%SYS32%\ping.exe" -n 3 127.0.0.1 >nul
) else (
    echo [2/3] 页面服务已在运行
)

rem ---- [3/3] 打开浏览器 ----
echo [3/3] 打开浏览器...
start "" "http://localhost:8630/genui_3b_playground.html"

echo.
echo 完成。关闭本窗口不影响使用；页面服务窗口已最小化，
echo 不再使用时关掉标题为 "GenUI Playground Server" 的窗口即可。
"%SYS32%\ping.exe" -n 6 127.0.0.1 >nul
exit

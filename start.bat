@echo off
cd /d "%~dp0"
title XtuAgent 控制台

where uv >nul 2>nul
if %errorlevel%==0 (
    set "UV=uv"
) else (
    if exist "%USERPROFILE%\.local\bin\uv.exe" (
        set "UV=%USERPROFILE%\.local\bin\uv.exe"
    ) else (
        echo [错误] 未找到 uv，请先安装: https://docs.astral.sh/uv/
        pause
        exit /b 1
    )
)

:menu
cls
echo ==============================================
echo    XtuAgent - 湘潭大学智能学业助手 v2
echo    基于 RAG 架构的私域知识库问答系统
echo ==============================================
echo.
echo    1) 启动 Web 服务并打开浏览器
echo    2) 重建知识库索引
echo    3) 爬取官网文档
echo    4) 系统自检 (doctor)
echo    5) 运行测试 (pytest)
echo    q) 退出
echo.
choice /c 12345q /n /m "请选择 [1-5,q]: "

if errorlevel 6 exit /b 0
if errorlevel 5 goto tests
if errorlevel 4 goto doctor
if errorlevel 3 goto crawl
if errorlevel 2 goto build
if errorlevel 1 goto serve
goto menu

:serve
cls
echo 正在启动 Web 服务...
echo 启动后浏览器将自动打开 http://127.0.0.1:8000
echo 按 Ctrl+C 停止服务
echo.
"%UV%" run python scripts/serve.py
pause
goto menu

:build
cls
echo 正在重建知识库索引...
echo 模型加载约需 1 分钟，嵌入计算时间取决于文本量
echo.
"%UV%" run python scripts/build_index.py
pause
goto menu

:crawl
cls
echo 正在爬取湘潭大学官网文档...
echo 结果保存在 data\texts\ 目录
echo.
"%UV%" run python scripts/crawl.py
pause
goto menu

:doctor
cls
echo 正在执行系统自检...
echo.
"%UV%" run python scripts/doctor.py
pause
goto menu

:tests
cls
echo 正在运行单元测试...
echo.
"%UV%" run pytest tests/ -v
pause
goto menu

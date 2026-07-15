@echo off
title 湘潭大学智能学业问答助手 - XtuAgent
cd /d "%~dp0"

set UV_EXE=C:\Users\23314\.local\bin\uv.exe
if not exist "%UV_EXE%" (
    echo [错误] 找不到 uv.exe，请确认路径: %UV_EXE%
    pause
    exit /b 1
)

:env
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set TOKENIZERS_PARALLELISM=false
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1

:menu
cls
echo ============================================
echo   湘潭大学智能学业问答助手 (XtuAgent)
echo   基于 RAG 架构的私域知识库问答系统
echo ============================================
echo.
echo  [1] 启动 Web 界面 (Streamlit)
echo  [2] 启动 Web 界面 (Gradio)
echo  [3] 爬取文档
echo  [4] 文档入库 (构建向量库)
echo  [5] 运行测试
echo  [6] 查看系统状态
echo  [q] 退出
echo.
set /p choice="请选择: "

if "%choice%"=="1" goto streamlit
if "%choice%"=="2" goto gradio
if "%choice%"=="3" goto crawl
if "%choice%"=="4" goto ingest
if "%choice%"=="5" goto test
if "%choice%"=="6" goto status
if /i "%choice%"=="q" goto end
goto menu

:streamlit
cls
echo 正在启动 Web 界面...
echo 启动后请访问 http://localhost:8501
echo 按 Ctrl+C 停止服务后返回菜单
echo.
"%UV_EXE%" run streamlit run src/xtuagent/app_streamlit.py --server.port 8501 --server.fileWatcherType none
echo.
echo Streamlit 已退出。
pause
goto menu

:gradio
cls
echo 正在启动 Gradio 界面...
echo 按 Ctrl+C 停止服务后返回菜单
echo.
"%UV_EXE%" run python src/xtuagent/app_gradio.py
echo.
pause
goto menu

:crawl
cls
echo 正在爬取湘潭大学官方文档...
echo 爬取结果保存在 data/texts/ 目录
echo.
"%UV_EXE%" run scripts/run_crawler.py --max-depth 2 --delay 0.3
echo.
pause
goto menu

:ingest
cls
echo 正在将文档向量化入库...
echo 模型加载约需 3-5 分钟，请耐心等待
echo.
"%UV_EXE%" run scripts/run_ingest.py --chunk-size 800
echo.
pause
goto menu

:test
cls
echo 运行测试...
echo.
"%UV_EXE%" run python tests/test_qa.py
echo.
pause
goto menu

:status
cls
echo 系统状态...
"%UV_EXE%" run python scripts/run_status.py
echo.
pause
goto menu

:end
exit /b 0

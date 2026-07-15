@echo off
setlocal enabledelayedexpansion
title 湘潭大学智能学业问答助手

set PATH=%PATH%;C:\Users\23314\.local\bin

:menu
cls
echo ============================================
echo   湘潭大学智能学业问答助手 (XtuAgent)
echo   基于 RAG 架构的私域知识库问答系统
echo ============================================
echo.
echo   [1] 启动 Web 界面 (Streamlit)
echo   [2] 启动 Web 界面 (Gradio)
echo   [3] 爬取文档
echo   [4] 文档入库 (构建向量库)
echo   [5] 运行测试
echo   [6] 查看系统状态
echo   [q] 退出
echo.
set /p choice="请选择: "

if "%choice%"=="1" goto streamlit
if "%choice%"=="2" goto gradio
if "%choice%"=="3" goto crawl
if "%choice%"=="4" goto ingest
if "%choice%"=="5" goto test
if "%choice%"=="6" goto status
if /i "%choice%"=="q" exit
goto menu

:streamlit
cls
echo 正在启动 Streamlit 界面...
echo 启动后请访问 http://localhost:8501
echo.
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set TOKENIZERS_PARALLELISM=false
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
uv run streamlit run src/xtuagent/app_streamlit.py --server.port 8501 --server.fileWatcherType none
echo.
pause
goto menu

:gradio
cls
echo 正在启动 Gradio 界面...
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set TOKENIZERS_PARALLELISM=false
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
uv run python src/xtuagent/app_gradio.py
echo.
pause
goto menu

:crawl
cls
echo 正在爬取湘潭大学官方文档...
echo 可在 data/texts/ 目录查看爬取结果
echo.
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
uv run scripts/run_crawler.py --max-depth 2 --delay 0.3
echo.
pause
goto menu

:ingest
cls
echo 正在将文档向量化入库...
echo 正在嵌入 bge-base-zh 模型，首次加载约 3-5 分钟
echo.
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set TOKENIZERS_PARALLELISM=false
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
uv run scripts/run_ingest.py --chunk-size 800
echo.
pause
goto menu

:test
cls
echo 运行测试...
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set TOKENIZERS_PARALLELISM=false
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
uv run python tests/test_qa.py
echo.
pause
goto menu

:status
cls
echo 正在获取系统状态...
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set TOKENIZERS_PARALLELISM=false
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
uv run python scripts/run_status.py
echo.
pause
goto menu

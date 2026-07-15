@echo off
chcp 65001 >nul
title 湘潭大学智能学业问答助手

set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set TOKENIZERS_PARALLELISM=false
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1

echo ============================================
echo   湘潭大学智能学业问答助手 (XtuAgent)
echo   基于 RAG 架构的私域知识库问答系统
echo ============================================
echo.
echo [1] 启动 Web 界面 (Streamlit)
echo [2] 启动 Web 界面 (Gradio)
echo [3] 爬取文档
echo [4] 文档入库 (构建向量库)
echo [5] 运行测试
echo [q] 退出
echo.
set /p choice="请选择: "

if "%choice%"=="1" goto streamlit
if "%choice%"=="2" goto gradio
if "%choice%"=="3" goto crawl
if "%choice%"=="4" goto ingest
if "%choice%"=="5" goto test
if "%choice%"=="q" goto end
if "%choice%"=="Q" goto end
echo 无效选择
pause
goto end

:streamlit
echo 启动 Streamlit 界面...
uv run streamlit run src/xtuagent/app_streamlit.py --server.port 8501 --server.fileWatcherType none
goto end

:gradio
echo 启动 Gradio 界面...
uv run python src/xtuagent/app_gradio.py
goto end

:crawl
echo 爬取湘潭大学官方文档...
uv run scripts/run_crawler.py --max-depth 2 --delay 0.3
pause
goto end

:ingest
echo 文档向量化入库...
uv run scripts/run_ingest.py --chunk-size 800
pause
goto end

:test
echo 运行测试...
uv run python tests/test_qa.py
pause
goto end

:end

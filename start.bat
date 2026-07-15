@echo off
cd /d "%~dp0"

set UV_EXE=C:\Users\23314\.local\bin\uv.exe
if not exist "%UV_EXE%" (
    echo ERROR: uv.exe not found at %UV_EXE%
    pause
    exit /b 1
)

set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set TOKENIZERS_PARALLELISM=false
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1

:menu
cls
echo ==========================================
echo   XtuAgent - Academic Q&A Assistant
echo   Xiangtan University
echo ==========================================
echo.
echo   1) Web UI (Streamlit)  - port 8501
echo   2) Web UI (Gradio)
echo   3) Crawl websites
echo   4) Build vector store
echo   5) Run tests
echo   6) System status
echo   q) Exit
echo.
choice /c 123456q /n /m "Select [1-6,q]: "

if errorlevel 7 exit /b 0
if errorlevel 6 goto status
if errorlevel 5 goto test
if errorlevel 4 goto ingest
if errorlevel 3 goto crawl
if errorlevel 2 goto gradio
if errorlevel 1 goto streamlit
goto menu

:streamlit
cls
echo Starting Streamlit... Access http://localhost:8501
echo Press Ctrl+C to stop.
echo.
"%UV_EXE%" run streamlit run src/xtuagent/app_streamlit.py --server.port 8501 --server.fileWatcherType none
pause
goto menu

:gradio
cls
echo Starting Gradio...
"%UV_EXE%" run python src/xtuagent/app_gradio.py
pause
goto menu

:crawl
cls
echo Crawling XTU websites...
"%UV_EXE%" run scripts/run_crawler.py --max-depth 2 --delay 0.3
pause
goto menu

:ingest
cls
echo Building FAISS vector store (may take a few minutes)...
"%UV_EXE%" run scripts/run_ingest.py --chunk-size 800
pause
goto menu

:test
cls
echo Running tests...
"%UV_EXE%" run python tests/test_qa.py
pause
goto menu

:status
cls
"%UV_EXE%" run python scripts/run_status.py
echo.
pause
goto menu

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
echo    XtuAgent - 校园智能学业问答系统 v2
echo    基于 RAG 架构的私域知识库问答系统
echo ==============================================
echo.
echo    1) 启动 Web 服务并打开浏览器
echo    2) 重建知识库索引
echo    3) 爬取官网文档
echo    4) 系统自检 (doctor)
echo    5) 运行测试 (pytest)
echo    6) 离线重抽文本 (改了抽取规则后用，无需联网)
echo    7) 灌入外部渠道 (微信/站外/本地文件)
echo    8) 采集微信公众号文章 (可绕过 403 的学院站)
echo    q) 退出
echo.
choice /c 12345678q /n /m "请选择 [1-8,q]: "

if errorlevel 9 exit /b 0
if errorlevel 8 goto wechat
if errorlevel 7 goto ingest
if errorlevel 6 goto extract
if errorlevel 5 goto tests
if errorlevel 4 goto doctor
if errorlevel 3 goto crawl
if errorlevel 2 goto build
if errorlevel 1 goto serve
goto menu

:wechat
cls
echo 正在经搜狗微信搜索采集公众号文章...
echo 覆盖被 403 的 13 个学院/部门，落盘到 data\raw_html\
echo 完成后请执行「2) 重建知识库索引」使之生效。
echo 提示：触发验证码会自动停止，加大间隔重跑即可，已抓内容不会丢。
echo.
"%UV%" run python scripts/crawl_wechat.py
pause
goto menu

:extract
cls
echo 正在从 data\raw_html\ 缓存重新抽取正文（不联网）...
echo.
"%UV%" run python scripts/extract_texts.py
pause
goto menu

:ingest
cls
echo 把外部 URL / 本地文件灌入语料库
echo 用法示例：
echo   ingest_urls.py urls.txt --source wechat
echo   ingest_urls.py https://mp.weixin.qq.com/s/xxxx --source wechat
echo   ingest_urls.py .\docs\*.pdf --source handbook
echo.
set /p TARGETS="请输入 目标(URL/文件/列表文件，多个用空格分隔): "
set /p SOURCE="请输入 来源标记 (如 wechat/zhihu/handbook，直接回车用 external): "
if "%SOURCE%"=="" set "SOURCE=external"
"%UV%" run python scripts/ingest_urls.py %TARGETS% --source %SOURCE%
echo.
echo 完成后请执行「2) 重建知识库索引」使之生效。
pause
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

@echo off
setlocal enabledelayedexpansion
call "D:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" > nul
cd /d D:\kd_forest_v2\src
cl /O2 /EHsc /arch:AVX2 /openmp /std:c++17 /I. bench_full_pipeline.cpp 2>&1
echo.
if %ERRORLEVEL% EQU 0 (
    echo Compilation OK. Running from D:\kd_forest_v2...
    cd /d D:\kd_forest_v2
    .\src\bench_full_pipeline.exe
) else (
    echo COMPILATION FAILED (code %ERRORLEVEL%)
)

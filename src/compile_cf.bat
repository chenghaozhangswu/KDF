@echo off
cd /d D:\kd_forest_v2_gh\src
call "D:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cl /O2 /EHsc /arch:AVX2 /std:c++17 /I. bench_coarse_fine.cpp /Fe:bench_coarse_fine.exe > D:\kd_forest_v2_gh\src\build_log.txt 2>&1
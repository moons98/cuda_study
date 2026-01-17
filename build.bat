@echo off
REM CUDA Benchmark Framework - Windows Build Script
REM Requirements: CMake 3.18+, CUDA Toolkit, Visual Studio with C++ support

setlocal enabledelayedexpansion

REM Configuration
set BUILD_TYPE=Release
set BUILD_DIR=build

REM Parse arguments
:parse_args
if "%~1"=="" goto :done_parsing
if /i "%~1"=="debug" set BUILD_TYPE=Debug
if /i "%~1"=="release" set BUILD_TYPE=Release
if /i "%~1"=="clean" goto :clean
if /i "%~1"=="rebuild" goto :rebuild
if /i "%~1"=="help" goto :help
shift
goto :parse_args
:done_parsing

REM Check for required tools
where cmake >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Error: CMake not found in PATH
    echo Please install CMake 3.18 or later
    exit /b 1
)

where nvcc >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Error: NVCC not found in PATH
    echo Please install CUDA Toolkit and add to PATH
    exit /b 1
)

REM Create build directory
if not exist %BUILD_DIR% mkdir %BUILD_DIR%

REM Configure
echo.
echo ============================================================
echo Configuring CUDA Benchmark Framework (%BUILD_TYPE%)
echo ============================================================
echo.

cd %BUILD_DIR%
cmake -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=%BUILD_TYPE% ..
if %ERRORLEVEL% neq 0 (
    echo.
    echo Configuration failed!
    cd ..
    exit /b 1
)

REM Build
echo.
echo ============================================================
echo Building...
echo ============================================================
echo.

cmake --build . --config %BUILD_TYPE% --parallel
if %ERRORLEVEL% neq 0 (
    echo.
    echo Build failed!
    cd ..
    exit /b 1
)

cd ..

echo.
echo ============================================================
echo Build successful!
echo ============================================================
echo.
echo Executable: %BUILD_DIR%\bin\%BUILD_TYPE%\benchmark.exe
echo.
echo Usage examples:
echo   %BUILD_DIR%\bin\%BUILD_TYPE%\benchmark.exe --list
echo   %BUILD_DIR%\bin\%BUILD_TYPE%\benchmark.exe --single --kernel=naive --size=1024
echo   %BUILD_DIR%\bin\%BUILD_TYPE%\benchmark.exe --compare --sizes=512,1024,2048
echo   %BUILD_DIR%\bin\%BUILD_TYPE%\benchmark.exe --all
echo.
goto :eof

:clean
echo Cleaning build directory...
if exist %BUILD_DIR% rmdir /s /q %BUILD_DIR%
echo Done.
goto :eof

:rebuild
echo Rebuilding...
if exist %BUILD_DIR% rmdir /s /q %BUILD_DIR%
goto :done_parsing

:help
echo.
echo CUDA Benchmark Framework - Build Script
echo.
echo Usage: build.bat [options]
echo.
echo Options:
echo   debug     Build in Debug mode
echo   release   Build in Release mode (default)
echo   clean     Remove build directory
echo   rebuild   Clean and rebuild
echo   help      Show this help message
echo.
goto :eof

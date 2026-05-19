@echo off
REM Build a single-file Windows .exe with PyInstaller.
REM Bundles the face_landmarker.task model + all mediapipe data/binaries.
REM Output: dist\face-mesh-stream.exe

setlocal
cd /d "%~dp0"

if not exist face_landmarker.task (
    echo [build] face_landmarker.task missing — run "python detect.py" once to auto-download it, then re-run this script.
    exit /b 1
)

echo [build] cleaning previous build...
if exist build rmdir /S /Q build
if exist dist rmdir /S /Q dist
if exist face-mesh-stream.spec del /Q face-mesh-stream.spec

echo [build] running pyinstaller...
python -m PyInstaller ^
    --onefile ^
    --console ^
    --noconfirm ^
    --name face-mesh-stream ^
    --collect-all mediapipe ^
    --add-data "face_landmarker.task;." ^
    --hidden-import mediapipe.framework.formats.landmark_pb2 ^
    detect.py

if errorlevel 1 (
    echo [build] FAILED
    exit /b 1
)

echo.
echo [build] done -^> dist\face-mesh-stream.exe
dir dist\face-mesh-stream.exe
endlocal

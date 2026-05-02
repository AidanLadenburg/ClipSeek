@echo off
setlocal
cd /d "%~dp0"

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\install_clipseek.ps1"
set "EXITCODE=%ERRORLEVEL%"

echo.
if not "%EXITCODE%"=="0" (
  echo ClipSeek install failed. See the messages above.
) else (
  echo ClipSeek install completed successfully.
)
echo.
pause
exit /b %EXITCODE%

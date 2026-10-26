@echo off
rem HybridRIP launcher. ASCII only on purpose: cmd.exe misreads UTF-8 .bat files.
cd /d "%~dp0"

rem 1) Your own path to python.exe (optional), e.g.:
rem set PY="C:\Users\Name\AppData\Local\Programs\Python\Python312\python.exe"
set PY=

rem 2) py launcher
if not defined PY py -3 --version >nul 2>nul && set PY=py -3

rem 3) python in PATH (skip the Microsoft Store stub in WindowsApps)
if not defined PY call :from_path

rem 4) usual install folders
if not defined PY call :probe "%LocalAppData%\Programs\Python"
if not defined PY call :probe "%ProgramFiles%"
if not defined PY call :probe "C:\"
if not defined PY if exist "%UserProfile%\anaconda3\python.exe" set PY="%UserProfile%\anaconda3\python.exe"
if not defined PY if exist "%UserProfile%\miniconda3\python.exe" set PY="%UserProfile%\miniconda3\python.exe"
if not defined PY if exist "%ProgramData%\anaconda3\python.exe" set PY="%ProgramData%\anaconda3\python.exe"
if not defined PY if exist "%ProgramData%\miniconda3\python.exe" set PY="%ProgramData%\miniconda3\python.exe"

if not defined PY (
  echo.
  echo Python not found.
  echo Install Python 3.12 from https://www.python.org/downloads/windows/
  echo and tick "Add python.exe to PATH", or put the path to python.exe
  echo into the "set PY=" line at the top of this file.
  echo.
  pause
  exit /b 1
)

echo Python: %PY%
%PY% launch.py
if errorlevel 1 pause
exit /b

:from_path
for /f "delims=" %%P in ('where python 2^>nul') do (
  echo %%P | find /i "WindowsApps" >nul
  if errorlevel 1 if not defined PY set PY="%%P"
)
exit /b

:probe
for /d %%V in ("%~1\Python3*") do (
  if exist "%%~V\python.exe" set PY="%%~V\python.exe"
)
exit /b

@echo off
rem HybridRIP: install all Python libraries. ASCII only on purpose: cmd.exe misreads UTF-8 .bat files.
cd /d "%~dp0"

rem Your own path to python.exe (optional), e.g.:
rem set PY="C:\Users\Name\AppData\Local\Programs\Python\Python312\python.exe"
set PY=

if not defined PY py -3 --version >nul 2>nul && set PY=py -3
if not defined PY call :from_path
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
%PY% --version
echo.
echo Updating pip...
%PY% -m pip install --user --upgrade pip --disable-pip-version-check
echo.
echo Installing libraries from requirements.txt ...
%PY% -m pip install --user --upgrade --disable-pip-version-check -r requirements.txt
if errorlevel 1 (
  echo.
  echo *** pip reported an error, see the messages above. ***
  pause
  exit /b 1
)
echo.
%PY% launch.py --check
echo.
pause
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

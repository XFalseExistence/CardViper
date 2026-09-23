@echo off
setlocal
set GRADLE_VERSION=9.6.0
where gradle >nul 2>nul
if errorlevel 1 (
  echo Install Gradle %GRADLE_VERSION% or use GitHub Actions for CardViper builds.
  exit /b 1
)
gradle %*

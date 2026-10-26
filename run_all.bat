@echo off
REM =============================================================================
REM IEEE SANER 2027 Replication Package - One-Click Reproduction Runner (Windows)
REM =============================================================================
echo =============================================================================
echo Starting IEEE SANER 2027 Replication Package Verification...
echo =============================================================================
python scripts\reproduce_all.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Replication script failed with error code %ERRORLEVEL%!
    exit /b %ERRORLEVEL%
)
echo.
echo [SUCCESS] Full replication suite completed successfully.

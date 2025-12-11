@echo off
echo ============================================================
echo Pushing AT-PRMD Project to GitHub
echo ============================================================

echo.
echo [*] Adding remote repository...
git remote add origin https://github.com/AimeeAyat/AT-PRMD.git

echo.
echo [*] Checking current branch...
git branch -M main

echo.
echo [*] Adding all files...
git add .

echo.
echo [*] Creating initial commit...
git commit -m "Initial commit: AT-PRMD implementation with dual-approach support"

echo.
echo [*] Pushing to GitHub...
git push -u origin main

echo.
echo ============================================================
echo [OK] Push complete!
echo ============================================================
echo.
echo Repository: https://github.com/AimeeAyat/AT-PRMD
echo.
pause

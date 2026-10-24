@echo off
setlocal
cd /d "%~dp0.."
set "RUNTIME=%~dp0"
if not exist "%RUNTIME%llama-server.exe" (
  echo llama-server.exe was not found.
  echo Download the official Windows x64 CPU release and copy the executable plus its DLLs beside this batch file.
  pause
  exit /b 1
)
if not exist "model\ravan-sft-Q4_0.gguf" (
  echo The model is missing: model\ravan-sft-Q4_0.gguf
  pause
  exit /b 1
)
start "Ravan local model" "%RUNTIME%llama-server.exe" -m "model\ravan-sft-Q4_0.gguf" --host 127.0.0.1 --port 8080 --ctx-size 128 --threads 2 --parallel 1 --path "app"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$deadline=(Get-Date).AddSeconds(45); do { try { $r=Invoke-RestMethod 'http://127.0.0.1:8080/health' -TimeoutSec 2; if ($r.status -eq 'ok') { exit 0 } } catch {}; Start-Sleep -Seconds 1 } while ((Get-Date) -lt $deadline); exit 1"
if errorlevel 1 (
  echo The local model server did not become ready within 45 seconds.
  echo Check the Ravan local model window for an error.
  pause
  exit /b 1
)
start "" "http://127.0.0.1:8080"
endlocal

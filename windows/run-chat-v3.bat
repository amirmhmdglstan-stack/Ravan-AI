@echo off
setlocal
cd /d "%~dp0.."
set "RUNTIME=%~dp0"

rem --- llama-server runtime: use bundled zip if exe missing ---
if not exist "%RUNTIME%llama-server.exe" (
  if exist "%RUNTIME%llama-b-win-cpu-x64.zip" (
    echo Extracting llama.cpp runtime...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "Expand-Archive -Force '%RUNTIME%llama-b-win-cpu-x64.zip' '%RUNTIME%'"
  )
)
if not exist "%RUNTIME%llama-server.exe" (
  echo llama-server.exe not found.
  echo Download llama-b-win-cpu-x64.zip from the "assets-v3" release of this
  echo repository and place it next to this batch file, then run again.
  pause
  exit /b 1
)

rem --- pick the first available model (priority order) ---
set "MODEL="
if exist "model\qwen3-1.7b-q4_k_m.gguf"      set "MODEL=model\qwen3-1.7b-q4_k_m.gguf"
if not defined MODEL if exist "model\qwen3-0.6b-q4_k_m.gguf" set "MODEL=model\qwen3-0.6b-q4_k_m.gguf"
if not defined MODEL if exist "model\qwen2.5-1.5b-instruct-q4_k_m.gguf" set "MODEL=model\qwen2.5-1.5b-instruct-q4_k_m.gguf"
if not defined MODEL if exist "model\gemma-3-1b-it-q4_k_m.gguf" set "MODEL=model\gemma-3-1b-it-q4_k_m.gguf"
if not defined MODEL (
  echo No model found. Download one GGUF from the "assets-v3" release and put it
  echo in the model\ folder. Recommended: qwen3-1.7b-q4_k_m.gguf
  pause
  exit /b 1
)

echo Starting Ravan v3 with %MODEL%
start "Ravan v3" "%RUNTIME%llama-server.exe" -m "%MODEL%" --host 127.0.0.1 --port 8080 --ctx-size 2048 --threads 2 --parallel 1 --path "app"

powershell -NoProfile -ExecutionPolicy Bypass -Command "$deadline=(Get-Date).AddSeconds(45); do { try { $r=Invoke-RestMethod 'http://127.0.0.1:8080/health' -TimeoutSec 2; if ($r.status -eq 'ok') { exit 0 } } catch {}; Start-Sleep -Seconds 1 } while ((Get-Date) -lt $deadline); exit 1"
if errorlevel 1 (
  echo The model server did not become ready within 45 seconds.
  echo Check the "Ravan v3" window for errors. On a tight 4 GB system, try the
  echo smaller qwen3-0.6b model instead.
  pause
  exit /b 1
)
start "" "http://127.0.0.1:8080"
endlocal

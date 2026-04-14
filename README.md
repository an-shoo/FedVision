# Assistive Navigation System

Real-time assistive guidance system for visually impaired navigation:

1. Webcam stream
2. YOLOv8 object detection
3. Proximity + direction heuristics
4. Structured scene JSON
5. LLM reasoning via Ollama (phi3 mini)
6. Optional text-to-speech (offline, `pyttsx3`)

## Current Status

- YOLO runs continuously with FPS-focused settings (`yolov8n`, configurable image size and frame-skip).
- LLM is scene-aware (triggered on scene change with min/max timing windows).
- Guidance is normalized to short human-friendly hazard + action output.
- TTS pipeline is integrated and configurable from `.env.example` (copy to `.env` locally).
- Default runtime profile uses `phi3:mini` for lower latency.

## Setup (Windows / PowerShell)

1. Create virtual environment:
   - `python -m venv .venv`
2. Activate it:
   - `.venv\Scripts\Activate.ps1`
3. Install dependencies:
   - `pip install -r requirements.txt`
4. Install Ollama from [https://ollama.com](https://ollama.com)
5. Pull the default fast model:
   - `ollama pull phi3:mini`
6. Start Ollama server:
   - `ollama serve`

## Run

Create local config from the template first:

- `Copy-Item .env.example .env`

Then run:

- `.\.venv\Scripts\python -m app.main --camera 0 --show`

Stop the app with `q` or `Esc` in the OpenCV window.

## Configuration

Push-safe defaults are stored in `.env.example`.
Use `.env.example` as the source of truth and keep `.env` local/untracked.

Current recommended defaults in `.env.example`:

- `OLLAMA_BASE_URL=http://127.0.0.1:11434`
- `OLLAMA_MODEL=phi3:mini`
- `LLM_MIN_INTERVAL_SEC=1.5`
- `LLM_MAX_INTERVAL_SEC=4.0`
- `LLM_NUM_PREDICT=28`
- `ENABLE_TTS=true`
- `YOLO_MODEL=yolov8n.pt`
- `YOLO_IMGSZ=416`
- `DETECT_EVERY_N_FRAMES=2`
- `YOLO_DEVICE=auto`

You can override any of these in your local `.env` without changing code.

## Troubleshooting

- If you see `LLM unavailable`, verify Ollama:
  - `ollama ps`
  - `curl http://127.0.0.1:11434/api/tags` (or equivalent request check)
- If TTS does not speak, keep output short and ensure `ENABLE_TTS=true`.
- If latency is high, use smaller model (`phi3:mini`) and lower `LLM_NUM_PREDICT`.

## PS
- If the command doesn't work try:
   - `$env:OLLAMA_BASE_URL='http://127.0.0.1:11434'; $env:OLLAMA_MODEL='phi3:mini'; $env:YOLO_DEVICE='auto'; $env:DETECT_EVERY_N_FRAMES='2'; $env:YOLO_IMGSZ='416'; $env:ENABLE_TTS='true'; $env:LLM_NUM_PREDICT='28'; $env:LLM_MIN_INTERVAL_SEC='1.0'; $env:LLM_MAX_INTERVAL_SEC='4.0'; .\.venv\Scripts\python -m app.main --camera 0 --show`

import os
from dataclasses import dataclass


def _as_bool(value: str, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct-q4_K_M")
    llm_interval_sec: float = float(os.getenv("LLM_INTERVAL_SEC", "2.5"))
    llm_min_interval_sec: float = float(os.getenv("LLM_MIN_INTERVAL_SEC", "1.0"))
    llm_max_interval_sec: float = float(os.getenv("LLM_MAX_INTERVAL_SEC", "3.0"))
    llm_num_predict: int = int(os.getenv("LLM_NUM_PREDICT", "56"))
    max_objects: int = int(os.getenv("MAX_OBJECTS", "5"))
    enable_tts: bool = _as_bool(os.getenv("ENABLE_TTS"), default=False)
    tts_rate: int = int(os.getenv("TTS_RATE", "180"))
    yolo_model: str = os.getenv("YOLO_MODEL", "yolov8n.pt")
    confidence_threshold: float = float(os.getenv("CONF_THRESHOLD", "0.35"))
    yolo_imgsz: int = int(os.getenv("YOLO_IMGSZ", "416"))
    detect_every_n_frames: int = int(os.getenv("DETECT_EVERY_N_FRAMES", "2"))
    yolo_device: str = os.getenv("YOLO_DEVICE", "auto")

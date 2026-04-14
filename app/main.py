import argparse
import json
import queue
import re
import threading
import time

import cv2
from dotenv import load_dotenv

from .config import Settings
from .detection import Detector
from .llm import OllamaReasoner
from .scene import build_scene, prioritize_detections
from .tts import SpeechManager


def _scene_signature(scene: dict) -> tuple:
    objects = scene.get("objects", []) or []
    primary = objects[0] if objects else {}
    return (
        scene.get("risk_hint", ""),
        primary.get("label", ""),
        primary.get("zone", ""),
        primary.get("distance", ""),
    )


def _sanitize_text(text: str) -> str:
    cleaned = text or ""
    cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"[*_`#>]+", " ", cleaned)
    cleaned = re.sub(r"\[[^\]]*\]\([^)]+\)", " ", cleaned)
    cleaned = re.sub(r"[\[\]\(\)\{\}]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _wrap_text_to_pixels(text: str, max_width: int, font, font_scale: float, thickness: int) -> list[str]:
    words = (text or "").strip().split()
    if not words:
        return []

    lines: list[str] = []
    current = words[0]

    for word in words[1:]:
        candidate = f"{current} {word}"
        candidate_w = cv2.getTextSize(candidate, font, font_scale, thickness)[0][0]
        if candidate_w <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _draw_overlay(frame, detections, guidance: str) -> None:
    h, w = frame.shape[:2]
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det.bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 220, 60), 2)
        label = f"{det.label} {det.confidence:.2f}"
        cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 220, 60), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    title_scale = 0.58
    text_scale = 0.52
    title_thickness = 1
    text_thickness = 1
    x_margin = 12
    max_text_width = max(120, w - (2 * x_margin) - 8)

    wrapped = _wrap_text_to_pixels(_sanitize_text(guidance), max_text_width, font, text_scale, text_thickness)
    if not wrapped:
        wrapped = ["Starting..."]
    wrapped = wrapped[:5]

    line_height = 20
    top_padding = 16
    bottom_padding = 12
    title_height = 24
    box_height = top_padding + title_height + (line_height * len(wrapped)) + bottom_padding
    y_top = max(0, h - box_height)

    cv2.rectangle(frame, (0, y_top), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, "Guidance:", (x_margin, y_top + 24), font, title_scale, (240, 240, 240), title_thickness)
    base_y = y_top + 24 + line_height
    for i, line in enumerate(wrapped):
        cv2.putText(frame, line, (x_margin, base_y + (i * line_height)), font, text_scale, (80, 255, 80), text_thickness)


def run(camera_index: int, show_window: bool) -> None:
    load_dotenv()
    settings = Settings()
    detector = Detector(
        settings.yolo_model,
        settings.confidence_threshold,
        imgsz=settings.yolo_imgsz,
        device=settings.yolo_device,
    )
    reasoner = OllamaReasoner(
        settings.ollama_base_url,
        settings.ollama_model,
        num_predict=settings.llm_num_predict,
    )
    speech = SpeechManager(rate=settings.tts_rate) if settings.enable_tts else None

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")

    last_llm_time = 0.0
    last_scene_signature = ("", "", "", "")
    latest_guidance = "Starting detection..."
    frame_idx = 0
    cached_top = []
    cached_scene = {"objects": [], "zone_summary": {}, "risk_hint": "no_significant_objects"}
    detect_every_n = max(1, settings.detect_every_n_frames)
    llm_request_queue: queue.Queue = queue.Queue(maxsize=1)
    stop_event = threading.Event()
    guidance_lock = threading.Lock()

    def llm_worker() -> None:
        nonlocal latest_guidance
        while not stop_event.is_set():
            try:
                item = llm_request_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                llm_request_queue.task_done()
                break

            local_frame_idx, local_scene = item
            try:
                next_guidance = reasoner.reason(local_scene)
            except Exception as exc:
                next_guidance = f"LLM unavailable. Continue cautiously. ({exc.__class__.__name__})"

            cleaned_guidance = _sanitize_text(next_guidance)
            with guidance_lock:
                latest_guidance = cleaned_guidance
            print(json.dumps({"frame": local_frame_idx, "scene": local_scene, "guidance": cleaned_guidance}, ensure_ascii=True))
            if speech:
                speech.process(cleaned_guidance)
            llm_request_queue.task_done()

    worker = threading.Thread(target=llm_worker, daemon=True)
    worker.start()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            if frame_idx % detect_every_n == 0:
                detections = detector.detect(frame)
                cached_top = prioritize_detections(detections, max_objects=settings.max_objects)
                cached_scene = build_scene(cached_top)
            top = cached_top
            scene = cached_scene

            now = time.time()
            scene_sig = _scene_signature(scene)
            time_since_last = now - last_llm_time
            scene_changed = scene_sig != last_scene_signature
            should_call_llm = (
                (scene_changed and time_since_last >= settings.llm_min_interval_sec)
                or (time_since_last >= settings.llm_max_interval_sec)
            )
            if should_call_llm:
                last_llm_time = now
                last_scene_signature = scene_sig
                if llm_request_queue.full():
                    try:
                        llm_request_queue.get_nowait()
                        llm_request_queue.task_done()
                    except queue.Empty:
                        pass
                llm_request_queue.put_nowait((frame_idx, scene))

            if show_window:
                with guidance_lock:
                    guidance_for_frame = latest_guidance
                _draw_overlay(frame, top, guidance_for_frame)
                cv2.imshow("Assistive Navigation", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
    finally:
        stop_event.set()
        try:
            llm_request_queue.put_nowait(None)
        except queue.Full:
            pass
        worker.join(timeout=1.5)
        cap.release()
        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="Assistive navigation pipeline")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument("--show", action="store_true", help="Show annotated live window")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(camera_index=args.camera, show_window=args.show)

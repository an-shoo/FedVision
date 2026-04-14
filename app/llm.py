import json
import re
import time
from typing import Any

import requests


SYSTEM_PROMPT = """You are an assistive navigation guide for a visually impaired user.
Given a structured scene, respond in natural spoken English.

Strict rules:
- Output exactly 2 short spoken sentences.
- Sentence 1: hazard and proximity/position.
- Sentence 2: immediate movement instruction.
- No markdown, no labels, no bullets.
- Max 8 words per sentence.
- Do not use these phrases: keep an eye out, watch out, look out, posing.
- Avoid vision-centric words: see, look, watch.
- Keep concise and action-oriented.
"""


def build_user_prompt(scene: dict[str, Any]) -> str:
    return f"Structured scene JSON:\\n{json.dumps(scene, ensure_ascii=True)}"


def _clean_text(text: str) -> str:
    cleaned = text or ""
    cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"[*_`#>]+", " ", cleaned)
    cleaned = re.sub(r"\[[^\]]*\]\([^)]+\)", " ", cleaned)
    cleaned = re.sub(r"[\[\]\(\)\{\}]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _sentence_fits(text: str, max_words: int = 15) -> bool:
    words = (text or "").strip().split()
    return 1 <= len(words) <= max_words


def _has_hazard_details(text: str) -> bool:
    t = (text or "").lower()
    hazard_terms = {"close", "near", "front", "left", "right", "ahead", "obstacle", "person", "car", "chair", "bike"}
    return any(term in t for term in hazard_terms)


def _has_action_verb(text: str) -> bool:
    t = (text or "").lower()
    action_terms = {"stop", "move", "step", "turn", "shift", "continue", "wait", "slow"}
    return any(term in t for term in action_terms)


def _finalize_sentence(text: str) -> str:
    s = (text or "").strip().rstrip(".,;:")
    if not s:
        return ""
    return s + "."


def _scene_guardrail(scene: dict[str, Any]) -> tuple[str, str]:
    risk = (scene or {}).get("risk_hint", "")
    objects = (scene or {}).get("objects", []) or []
    primary = objects[0] if objects else {}
    label = primary.get("label", "obstacle")
    zone = primary.get("zone", "center")
    distance = primary.get("distance", "mid")

    if risk == "high_risk_front_obstacle":
        return (
            f"{label.capitalize()} is very close in front.",
            "Stop now and move only when the path clears.",
        )
    if risk == "side_obstacle_nearby":
        if zone == "left":
            return (
                f"{label.capitalize()} is close on your left side.",
                "Shift slightly right and continue slowly.",
            )
        if zone == "right":
            return (
                f"{label.capitalize()} is close on your right side.",
                "Shift slightly left and continue slowly.",
            )
        return (
            f"{label.capitalize()} is close nearby.",
            "Slow down and keep a safe side distance.",
        )
    if objects and distance in {"mid", "far"}:
        return (
            f"{label.capitalize()} detected ahead at a safer distance.",
            "Move forward carefully and keep scanning.",
        )
    return ("No immediate obstacle ahead.", "Move forward cautiously and keep scanning.")


def _format_guidance(content: str, scene: dict[str, Any]) -> str:
    cleaned = _clean_text(content)
    parts = [p.strip() for p in re.split(r"[.!?]+", cleaned) if p.strip()]

    guard_safety, guard_action = _scene_guardrail(scene)
    safety = parts[0] if len(parts) > 0 else guard_safety
    action = parts[1] if len(parts) > 1 else guard_action

    # Normalize old label-style outputs if model drifts.
    safety = re.sub(r"^\s*(safety|warning)\s*:\s*", "", safety, flags=re.I)
    action = re.sub(r"^\s*(action|movement suggestion)\s*:\s*", "", action, flags=re.I)

    # Guardrail against incorrect "low risk" wording when scene says otherwise.
    risk = (scene or {}).get("risk_hint", "")
    if risk in {"high_risk_front_obstacle", "side_obstacle_nearby"} and "low risk" in safety.lower():
        safety = guard_safety
    if risk == "high_risk_front_obstacle" and not any(k in action.lower() for k in ["stop", "wait"]):
        action = guard_action

    # Deterministic validation: ensure both required information types are present.
    if not _has_hazard_details(safety):
        safety = guard_safety
    if not _has_action_verb(action):
        action = guard_action

    # Avoid awkward clipping: if sentence is too long, use guardrail sentence instead.
    if not _sentence_fits(safety, max_words=15):
        safety = guard_safety
    if not _sentence_fits(action, max_words=15):
        action = guard_action

    safety = _finalize_sentence(safety)
    action = _finalize_sentence(action)
    return f"{safety} {action}"


class OllamaReasoner:
    def __init__(self, base_url: str, model: str, timeout_sec: int = 35, num_predict: int = 56) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_sec = timeout_sec
        self.num_predict = num_predict

    def reason(self, scene: dict[str, Any]) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(scene)},
            ],
            "options": {
                "temperature": 0.2,
                "num_predict": self.num_predict,
            },
        }
        url = f"{self.base_url}/api/chat"
        last_error: Exception | None = None
        body = None
        for attempt in range(3):
            try:
                response = requests.post(url, json=payload, timeout=self.timeout_sec)
                response.raise_for_status()
                body = response.json()
                break
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(0.35)
                    continue
                raise

        if body is None:
            raise RuntimeError(f"Ollama response unavailable: {last_error}")
        message = body.get("message", {})
        text = message.get("content", "").strip()
        if not text:
            guard_safety, guard_action = _scene_guardrail(scene)
            return f"{guard_safety} {guard_action}"
        return _format_guidance(text, scene)

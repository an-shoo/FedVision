import threading
import time
import re

import pyttsx3


class SpeechManager:
    def __init__(
        self,
        cooldown_sec: float = 2.0,
        rate: int = 180,
        max_chars: int = 160,
        repeat_after_sec: float = 8.0,
    ) -> None:
        self.cooldown_sec = cooldown_sec
        self.max_chars = max_chars
        self.repeat_after_sec = repeat_after_sec
        self.last_spoken_text = ""
        self.last_spoken_time = 0.0

        self._lock = threading.Lock()
        self._rate = rate

    def _normalize(self, text: str) -> str:
        return " ".join((text or "").strip().lower().split())

    def _sanitize(self, text: str) -> str:
        cleaned = text or ""
        cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)
        cleaned = re.sub(r"[*_`#>]+", " ", cleaned)
        cleaned = re.sub(r"\[[^\]]*\]\([^)]+\)", " ", cleaned)
        cleaned = re.sub(r"[\[\]\(\)\{\}]", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _shorten(self, text: str) -> str:
        text = self._sanitize(text)
        if len(text) <= self.max_chars:
            return text
        cut = text.rfind(".", 0, self.max_chars)
        if cut > 30:
            return text[: cut + 1]
        return text[: self.max_chars].rstrip() + "..."

    def should_speak(self, text: str) -> bool:
        candidate_raw = self._shorten(text)
        candidate = self._normalize(candidate_raw)
        if not candidate:
            print("[TTS] skip: empty candidate")
            return False

        with self._lock:
            now = time.time()
            # Exact repeats are blocked briefly but allowed again after a reminder interval.
            if candidate == self.last_spoken_text and (now - self.last_spoken_time) < self.repeat_after_sec:
                print("[TTS] skip: recent repeat")
                return False
            if (now - self.last_spoken_time) < self.cooldown_sec:
                print("[TTS] skip: cooldown")
                return False
            print("[TTS] accept")
            return True

    def speak(self, text: str) -> None:
        text_to_speak = self._shorten(text)
        normalized = self._normalize(text_to_speak)
        if not normalized:
            return

        with self._lock:
            self.last_spoken_text = normalized
            self.last_spoken_time = time.time()
            try:
                # Fresh engine per utterance avoids occasional pyttsx3 state lockups on Windows.
                engine = pyttsx3.init()
                engine.setProperty("rate", self._rate)
                engine.say(text_to_speak)
                engine.runAndWait()
                engine.stop()
            except Exception:
                # Avoid crashing the main pipeline on TTS runtime errors.
                print("[TTS] error: runtime failure")
                return

    def process(self, text: str) -> None:
        if self.should_speak(text):
            self.speak(text)

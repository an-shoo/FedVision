from dataclasses import dataclass
from typing import List

import torch
from ultralytics import YOLO


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: tuple[float, float, float, float]
    area_ratio: float
    center_x: float
    center_y: float


class Detector:
    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        conf_threshold: float = 0.35,
        imgsz: int = 416,
        device: str = "auto",
    ) -> None:
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self.device = self._resolve_device(device)

    def _resolve_device(self, requested: str) -> str:
        if requested and requested.lower() != "auto":
            return requested
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    def detect(self, frame) -> List[Detection]:
        h, w = frame.shape[:2]
        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        output: List[Detection] = []

        for result in results:
            if result.boxes is None:
                continue

            boxes = result.boxes
            for box in boxes:
                cls_idx = int(box.cls.item())
                label = self.model.names.get(cls_idx, str(cls_idx))
                conf = float(box.conf.item())
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                box_area = max(0.0, (x2 - x1) * (y2 - y1))
                area_ratio = box_area / float(w * h)
                cx = (x1 + x2) / 2.0 / w
                cy = (y1 + y2) / 2.0 / h
                output.append(
                    Detection(
                        label=label,
                        confidence=conf,
                        bbox=(x1, y1, x2, y2),
                        area_ratio=area_ratio,
                        center_x=cx,
                        center_y=cy,
                    )
                )

        return output

from collections import defaultdict
from typing import Any

from .detection import Detection


def _zone_from_center_x(cx: float) -> str:
    if cx < 0.33:
        return "left"
    if cx > 0.66:
        return "right"
    return "center"


def _distance_bucket(area_ratio: float) -> str:
    if area_ratio > 0.18:
        return "very_close"
    if area_ratio > 0.08:
        return "close"
    if area_ratio > 0.03:
        return "mid"
    return "far"


def prioritize_detections(detections: list[Detection], max_objects: int = 5) -> list[Detection]:
    # Score by confidence and apparent proximity (larger area means closer obstacle).
    return sorted(
        detections,
        key=lambda d: (d.area_ratio * 1.8) + (d.confidence * 1.2),
        reverse=True,
    )[:max_objects]


def build_scene(detections: list[Detection]) -> dict[str, Any]:
    by_zone = defaultdict(list)
    objects = []

    for det in detections:
        zone = _zone_from_center_x(det.center_x)
        distance = _distance_bucket(det.area_ratio)
        by_zone[zone].append(det.label)
        objects.append(
            {
                "label": det.label,
                "confidence": round(det.confidence, 3),
                "zone": zone,
                "distance": distance,
                "area_ratio": round(det.area_ratio, 4),
            }
        )

    return {
        "objects": objects,
        "zone_summary": {k: v for k, v in by_zone.items()},
        "risk_hint": _risk_hint(objects),
    }


def _risk_hint(objects: list[dict[str, Any]]) -> str:
    close_center = [o for o in objects if o["zone"] == "center" and o["distance"] in {"very_close", "close"}]
    if close_center:
        return "high_risk_front_obstacle"
    if any(o["distance"] in {"very_close", "close"} for o in objects):
        return "side_obstacle_nearby"
    if objects:
        return "path_likely_clear_with_attention"
    return "no_significant_objects"

"""
events.py – Compute the 7 proctoring features from the server-side event log.
Client-supplied totals are IGNORED (anti-tamper). The server computes everything.
"""
import statistics
from typing import Any, Dict, List


CHEAT_FEATURES = [
    "tab_switches", "face_not_detected_count", "multiple_faces_count",
    "avg_time_per_question", "copy_paste_events", "mouse_idle_time",
    "answer_speed_variance",
]


def build_session_features(
    events: List[Dict[str, Any]],
    seconds_spent_list: List[float],
    face_state: Dict[str, int],
) -> Dict[str, Any]:
    """
    Compute the 7 features from the server-side event log.
    `events` is the list of dicts from DB (each has `type`, `duration_ms`, `t_server`).
    `seconds_spent_list` is the per-question time in seconds (from DB answers table).
    `face_state` is {'face_not_detected_count': int, 'multiple_faces_count': int}.
    """
    # -- tab_switches: count server-side "away" events
    tab_switches = sum(1 for e in events if e.get("type") == "away")

    # -- copy_paste_events: count copy + cut + paste events
    copy_paste_events = sum(1 for e in events if e.get("type") in ("copy", "cut", "paste"))

    # -- mouse_idle_time: sum of "idle" event durations (seconds)
    mouse_idle_time = sum(
        max(0, (e.get("duration_ms") or 0)) / 1000.0
        for e in events if e.get("type") == "idle"
    )

    # -- avg_time_per_question and answer_speed_variance
    # Handle edge cases: empty list, negative values, single question
    seconds_spent_list = [max(0.0, float(s)) for s in seconds_spent_list]
    if not seconds_spent_list:
        avg_time = 0.0
        variance = 0.0
    else:
        avg_time = statistics.mean(seconds_spent_list)
        variance = statistics.pstdev(seconds_spent_list) if len(seconds_spent_list) > 1 else 0.0

    return {
        "tab_switches": int(tab_switches),
        "face_not_detected_count": int(face_state.get("face_not_detected_count", 0)),
        "multiple_faces_count": int(face_state.get("multiple_faces_count", 0)),
        "avg_time_per_question": round(avg_time, 4),
        "copy_paste_events": int(copy_paste_events),
        "mouse_idle_time": round(mouse_idle_time, 4),
        "answer_speed_variance": round(variance, 4),
    }


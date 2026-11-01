"""
test_events.py – Test event aggregation and server-side feature derivation (anti-tamper).
"""
from server.events import build_session_features

def test_events_aggregation():
    events = [
        {"type": "away", "duration_ms": 1200},
        {"type": "away", "duration_ms": 3400},
        {"type": "away", "duration_ms": 800},
        {"type": "copy"},
        {"type": "paste"},
        {"type": "idle", "duration_ms": 5000},
        {"type": "idle", "duration_ms": 10000},
        {"type": "heartbeat"},
        {"type": "return"},
    ]
    seconds_spent = [20.0, 30.0, 40.0]
    face_state = {"face_not_detected_count": 1, "multiple_faces_count": 0}
    
    feats = build_session_features(events, seconds_spent, face_state)
    
    assert feats["tab_switches"] == 3
    assert feats["copy_paste_events"] == 2
    assert feats["mouse_idle_time"] == 15.0  # (5000 + 10000) / 1000
    assert feats["avg_time_per_question"] == 30.0
    assert feats["face_not_detected_count"] == 1
    assert feats["multiple_faces_count"] == 0
    assert feats["answer_speed_variance"] > 0

def test_single_question_edge_case():
    events = []
    seconds_spent = [45.0]
    face_state = {"face_not_detected_count": 0, "multiple_faces_count": 0}
    
    feats = build_session_features(events, seconds_spent, face_state)
    assert feats["avg_time_per_question"] == 45.0
    assert feats["answer_speed_variance"] == 0.0

def test_empty_answers_edge_case():
    events = []
    seconds_spent = []
    face_state = {"face_not_detected_count": 0, "multiple_faces_count": 0}
    
    feats = build_session_features(events, seconds_spent, face_state)
    assert feats["avg_time_per_question"] == 0.0
    assert feats["answer_speed_variance"] == 0.0

def test_anti_tamper_client_totals_ignored():
    # If malicious client tries to send fake totals inside events list
    # Server only counts genuine typed events
    tampered_events = [
        {"type": "away"},
        {"type": "away"},
        {"type": "fake_client_total", "tab_switches": 0},
    ]
    feats = build_session_features(tampered_events, [10.0, 10.0], {})
    assert feats["tab_switches"] == 2

"""
test_agent.py – Test ExamAgent methods, decisions, guardrails, and contracts.
"""
import os
import pytest

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from agent import ExamAgent

@pytest.fixture(scope="module")
def agent():
    return ExamAgent()

def test_agent_initialization(agent):
    assert agent.cheat_model is not None
    assert agent.grader_model is not None
    assert agent.tfidf is not None
    assert agent.embedder is not None
    assert len(agent.cheat_features) == 7
    assert agent.low == 0.25
    assert agent.high == 0.60

def test_decision_logic(agent):
    # Below low threshold
    assert agent.decide(0.10) == "RELEASE"
    # Between low and high
    assert agent.decide(0.40) == "RELEASE_WITH_WARNING"
    # Above high
    assert agent.decide(0.85) == "HOLD_FOR_REVIEW"
    
    # Guardrail override: if risk is low but guardrail is hit -> never silently release
    assert agent.decide(0.10, guard_hits=["many tab switches"]) == "RELEASE_WITH_WARNING"

def test_run_contract_keys(agent):
    ref = "Overfitting is when a model learns the training noise."
    ans = "The algorithm fits the training samples too specifically and generalizes poorly."
    sess = {
        "tab_switches": 1,
        "face_not_detected_count": 0,
        "multiple_faces_count": 0,
        "avg_time_per_question": 50.0,
        "copy_paste_events": 0,
        "mouse_idle_time": 40.0,
        "answer_speed_variance": 5.0
    }
    res = agent.run("TEST_STUDENT", ref, ans, sess)
    
    expected_keys = {"student_id", "predicted_score", "risk_percent", "decision", "final_grade", "reason"}
    for k in expected_keys:
        assert k in res
    
    assert res["student_id"] == "TEST_STUDENT"
    assert 0.0 <= res["predicted_score"] <= 5.0
    assert 0.0 <= res["risk_percent"] <= 100.0
    assert res["decision"] in {"RELEASE", "RELEASE_WITH_WARNING", "HOLD_FOR_REVIEW"}
    if res["decision"] == "HOLD_FOR_REVIEW":
        assert res["final_grade"] is None
    else:
        assert res["final_grade"] is not None

def test_cheating_script_flags_hold_for_review(agent):
    ref = "Gradient descent updates parameters iteratively to minimize loss."
    ans = "Parameters are adjusted according to gradient of loss."
    cheating_sess = {
        "tab_switches": 25,
        "face_not_detected_count": 8,
        "multiple_faces_count": 2,
        "avg_time_per_question": 15.0,
        "copy_paste_events": 8,
        "mouse_idle_time": 400.0,
        "answer_speed_variance": 40.0
    }
    res = agent.run("CHEATER_01", ref, ans, cheating_sess)
    assert res["decision"] == "HOLD_FOR_REVIEW"
    assert res["final_grade"] is None
    assert res["risk_percent"] > 60.0

def test_bad_camera_honest_student_not_held(agent):
    ref = "Gradient descent updates parameters iteratively to minimize loss."
    ans = "Gradient descent iteratively minimizes loss function."
    bad_cam_sess = {
        "tab_switches": 1,
        "face_not_detected_count": 6,
        "multiple_faces_count": 0,
        "avg_time_per_question": 60.0,
        "copy_paste_events": 0,
        "mouse_idle_time": 80.0,
        "answer_speed_variance": 10.0
    }
    res = agent.run("HONEST_BADCAME", ref, ans, bad_cam_sess)
    # Bad camera alone without tab switches or paste events shouldn't trigger HOLD_FOR_REVIEW
    assert res["decision"] != "HOLD_FOR_REVIEW"

def test_clipping_bounds(agent):
    # Provide extreme out-of-bounds telemetry
    extreme_sess = {
        "tab_switches": 999,
        "face_not_detected_count": 999,
        "multiple_faces_count": 999,
        "avg_time_per_question": -50.0,
        "copy_paste_events": 999,
        "mouse_idle_time": 9999.0,
        "answer_speed_variance": 999.0
    }
    clean, notes, hits = agent._prepare(extreme_sess)
    # Ensure all values were clipped to valid ranges
    assert clean["tab_switches"] <= agent.bounds["tab_switches"][1]
    assert clean["avg_time_per_question"] >= agent.bounds["avg_time_per_question"][0]
    assert len(notes) > 0
    assert len(hits) > 0

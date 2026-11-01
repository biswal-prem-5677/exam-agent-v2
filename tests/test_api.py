"""
test_api.py – Test FastAPI endpoints, security contracts, timer enforcement, and audit logs.
"""
import os
import uuid
import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from server.main import app
from server.db import init_db, get_agent_log

client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_db():
    init_db()

def test_health_check():
    response = client.get("/api/v1/site/check")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_student_cannot_see_reference_answers():
    sid = f"S_SEC_{uuid.uuid4().hex[:6]}"
    # Start exam
    response = client.get(f"/api/start?student_id={sid}")
    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert "questions" in data
    
    # CRITICAL SECURITY CONTRACT: Reference answers must NEVER be sent to students
    for q in data["questions"]:
        assert "reference_answer" not in q
        assert "qid" in q
        assert "question" in q

def test_one_attempt_rule():
    sid = f"S_ONE_{uuid.uuid4().hex[:6]}"
    res_start = client.get(f"/api/start?student_id={sid}")
    assert res_start.status_code == 200
    sess_id = res_start.json()["session_id"]
    
    # Submit exam
    submit_payload = {
        "session_id": sess_id,
        "student_id": sid,
        "answers": [{"qid": "q1", "answer": "Sample answer", "seconds_spent": 30}],
        "camera_available": True
    }
    res_sub = client.post("/api/submit", json=submit_payload)
    assert res_sub.status_code == 200
    
    # Try starting second attempt with same student_id
    res_second = client.get(f"/api/start?student_id={sid}")
    assert res_second.status_code == 409
    assert "already submitted" in res_second.json()["detail"].lower()

def test_reject_unknown_session():
    fake_session = "00000000-0000-0000-0000-000000000000"
    res = client.post("/api/event", json={"session_id": fake_session, "events": []})
    assert res.status_code == 404

def test_full_scripted_cheating_pipeline():
    student_id = f"S_CHEAT_{uuid.uuid4().hex[:6]}"
    start_resp = client.get(f"/api/start?student_id={student_id}")
    assert start_resp.status_code == 200
    session_id = start_resp.json()["session_id"]
    
    # Send cheating events: 15 tab away episodes, 5 pastes, long idle
    cheating_events = [{"type": "away"} for _ in range(15)]
    cheating_events += [{"type": "paste"} for _ in range(5)]
    cheating_events += [{"type": "idle", "duration_ms": 300000}]
    
    client.post("/api/event", json={"session_id": session_id, "events": cheating_events})
    
    # Submit
    res_submit = client.post("/api/submit", json={
        "session_id": session_id,
        "student_id": student_id,
        "answers": [{"qid": start_resp.json()["questions"][0]["qid"], "answer": "Test answer", "seconds_spent": 10}],
        "camera_available": True
    })
    assert res_submit.status_code == 200
    sub_data = res_submit.json()
    assert sub_data["decision"] == "HOLD_FOR_REVIEW"
    
    # Assert agent_log entries were created
    logs = get_agent_log(session_id)
    assert len(logs) >= 5
    step_names = [l["step"] for l in logs]
    assert "observe" in step_names
    assert "grade" in step_names
    assert "assess_risk" in step_names
    assert "decide" in step_names

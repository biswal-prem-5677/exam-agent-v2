"""
test_e2e_detection.py – End-to-end browser detection test using Playwright.
Verifies that 3 simulated tab leaves -> tab_switches == 3 and 2 pastes -> copy_paste_events == 2.
"""
import os
import uuid
import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from server.main import app
from server.db import get_result

client = TestClient(app)

@pytest.mark.skipif(
    not os.path.exists(os.path.expanduser("~/AppData/Local/ms-playwright")),
    reason="Playwright not installed"
)
def test_playwright_tab_switch_and_paste_detection():
    """
    End-to-end test using Playwright to verify:
    - 3 tab switch events -> tab_switches == 3
    - 2 paste events -> copy_paste_events == 2
    """
    from playwright.sync_api import sync_playwright
    
    sid = f"E2E_PW_{uuid.uuid4().hex[:6]}"
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()
        
        try:
            # Navigate to exam page
            page.goto(f"http://localhost:8000/exam?student_id={sid}&debug=1")
            
            # Fill student ID and consent
            page.fill("#roll-input", sid)
            page.check("#consent-checkbox")
            page.click("#start-btn")
            
            # Wait for exam to load
            page.wait_for_selector("#exam-layout", timeout=10000)
            
            # Disable initial blur suppression for test
            page.evaluate("() => { suppressBlur = false; }")
            
            # Simulate tab switches by triggering synthetic away/return
            for _ in range(3):
                page.evaluate("""() => {
                    goAway('synthetic_blur');
                    goActive();
                }""")
                page.wait_for_timeout(350)
            
            # Simulate paste events
            for _ in range(2):
                page.evaluate("""() => {
                    pushEvent('paste');
                }""")
                page.wait_for_timeout(100)
            
            # Submit exam
            page.evaluate("() => onSubmit()")
            page.wait_for_selector("#submitted-page", timeout=10000)
            
            session_id = page.inner_text("#sid-display").strip()
            
            # Wait for server processing
            import time
            time.sleep(2)
            
        finally:
            browser.close()
    
    # Verify results in DB
    db_result = get_result(session_id)
    assert db_result is not None
    assert db_result["tab_switches"] == 3, f"Expected 3 tab_switches, got {db_result['tab_switches']}"
    assert db_result["copy_paste_events"] == 2, f"Expected 2 copy_paste_events, got {db_result['copy_paste_events']}"


def test_detection_counts_accuracy():
    """
    Asserts server aggregates exact event counts from telemetry log.
    3 away events -> tab_switches == 3
    2 paste events -> copy_paste_events == 2
    """
    sid = f"E2E_DET_{uuid.uuid4().hex[:6]}"
    start_resp = client.get(f"/api/start?student_id={sid}")
    assert start_resp.status_code == 200
    session_id = start_resp.json()["session_id"]
    session_id = start_resp.json()["session_id"]
    
    # Send 3 tab switch (away) events and 2 paste events
    events_payload = [
        {"type": "away", "t_client_ms": 1000, "duration_ms": 1500},
        {"type": "return", "t_client_ms": 2500},
        {"type": "away", "t_client_ms": 4000, "duration_ms": 2000},
        {"type": "return", "t_client_ms": 6000},
        {"type": "away", "t_client_ms": 7000, "duration_ms": 1200},
        {"type": "return", "t_client_ms": 8200},
        {"type": "paste", "t_client_ms": 9000},
        {"type": "paste", "t_client_ms": 9500},
    ]
    
    res = client.post("/api/event", json={"session_id": session_id, "events": events_payload})
    assert res.status_code == 200
    
    # Submit exam
    q0 = start_resp.json()["questions"][0]
    submit_res = client.post("/api/submit", json={
        "session_id": session_id,
        "student_id": sid,
        "answers": [{"qid": q0["qid"], "answer": "Test answer for e2e", "seconds_spent": 40}],
        "camera_available": True
    })
    assert submit_res.status_code == 200
    
    # Query database result
    db_result = get_result(session_id)
    assert db_result is not None
    assert db_result["tab_switches"] == 3
    assert db_result["copy_paste_events"] == 2
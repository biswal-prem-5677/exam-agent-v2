"""
main.py – FastAPI server for the Autonomous Exam Proctoring & Grading Agent.

Routes:
  GET  /                          → exam.html (redirect with ?exam_id=DEMO)
  GET  /exam                      → exam.html
  GET  /api/exams                 → list exams
  POST /api/exams                 → create exam
  GET  /api/start                 → start session (student, exam_id)
  POST /api/event                 → batch event ingestion (server timestamps)
  POST /api/frame                 → camera frame (OpenCV Haar, never stored)
  POST /api/quality               → camera quality check (pre-exam)
  POST /api/autosave              → autosave answers (restore after refresh)
  POST /api/submit                → submit exam → agent pipeline → result
  GET  /api/results               → all results (examiner only)
  GET  /api/session/{sid}         → live session counters (examiner)
  POST /api/review                → human review action
  GET  /api/collusion/{exam_id}   → collusion flags
  GET  /api/export/csv            → CSV of all results
  DELETE /api/session/{sid}       → delete student data (GDPR)
  POST /api/appeal/{sid}          → student appeal → creates review item
  GET  /api/v1/site/check         → health check
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

# Force offline mode for sentence-transformers
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))

from agent import ExamAgent
from server.db import (
    init_db, new_exam, get_exam, list_exams, new_session, get_session, session_exists,
    already_submitted, update_session_status, insert_event, insert_events_batch, upsert_answer,
    get_answers, get_events, save_result, get_result, get_all_results, get_active_sessions,
    insert_review, get_reviews, get_held_sessions, log_agent_step, get_agent_log,
    save_collusion_flag, get_collusion_flags, delete_session_data, get_all_agent_logs,
)
from server.events import build_session_features
from server.vision import process_frame, check_image_quality
from server.explain import explain_grading_multi, explain_risk
from server.collusion import detect_collusion

# ── App setup ──────────────────────────────────────────────────────────────

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    _seed_demo()
    yield

app = FastAPI(title="Exam Proctoring Agent", version="2.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")

# Initialise DB
init_db()

# Seed demo exam on startup
_DEMO_SEEDED = False

def _seed_demo():
    global _DEMO_SEEDED
    if _DEMO_SEEDED:
        return
    try:
        answers_df = pd.read_csv(BASE / "data" / "answers.csv")
        bank = answers_df.drop_duplicates("question")[["question", "reference_answer"]].reset_index(drop=True)
        import uuid
        questions = [
            {"qid": str(uuid.uuid4())[:8], "question": row["question"], "reference_answer": row["reference_answer"]}
            for _, row in bank.head(6).iterrows()
        ]
        existing = [e for e in list_exams() if e["title"] == "Demo Exam"]
        if not existing:
            eid = new_exam("Demo Exam", 1200, questions)
            print(f"[server] Demo exam seeded: exam_id={eid}")
        _DEMO_SEEDED = True
    except Exception as exc:
        print(f"[server] Demo seed warning: {exc}")


# Load agent (singleton)
_agent: Optional[ExamAgent] = None


def get_agent() -> ExamAgent:
    global _agent
    if _agent is None:
        _agent = ExamAgent()
    return _agent


# In-memory session face states (not persisted)
_face_states: Dict[str, Dict] = {}

# ── Helpers ────────────────────────────────────────────────────────────────

def _face_state_for(session_id: str) -> Dict:
    if session_id not in _face_states:
        _face_states[session_id] = {
            "face_not_detected_count": 0,
            "multiple_faces_count": 0,
            "consecutive_no_face": 0,
            "consecutive_multi": 0,
        }
    return _face_states[session_id]


def _exam_elapsed(session: Dict) -> float:
    """Seconds since the session started."""
    try:
        started = datetime.fromisoformat(session["started_at"].replace("Z", "+00:00"))
        return (datetime.utcnow() - started.replace(tzinfo=None)).total_seconds()
    except Exception:
        return 0.0


# ── Routes ─────────────────────────────────────────────────────────────────




@app.get("/", response_class=HTMLResponse)
async def root():
    return (BASE / "static" / "exam.html").read_text(encoding="utf-8", errors="replace")


@app.get("/exam", response_class=HTMLResponse)
async def exam_page():
    return (BASE / "static" / "exam.html").read_text(encoding="utf-8", errors="replace")


@app.get("/api/v1/site/check")
@app.post("/api/v1/site/check")
async def health():
    return {"status": "ok", "version": "2.0"}


# ── Exam management ─────────────────────────────────────────────────────────

@app.get("/api/exams")
async def list_all_exams():
    return list_exams()


@app.post("/api/exams")
async def create_exam(request: Request):
    data = await request.json()
    title = data.get("title", "Untitled Exam")
    duration_s = int(data.get("duration_s", 1200))
    questions = data.get("questions", [])
    if not questions:
        raise HTTPException(400, "No questions provided")
    import uuid
    for q in questions:
        q.setdefault("qid", str(uuid.uuid4())[:8])
    eid = new_exam(title, duration_s, questions)
    return {"exam_id": eid, "title": title}


@app.get("/api/exams/{exam_id}")
async def get_exam_info(exam_id: str):
    exam = get_exam(exam_id)
    if not exam:
        raise HTTPException(404, "Exam not found")
    # Strip reference_answers (never sent to students via this route)
    safe = {k: v for k, v in exam.items() if k != "questions"}
    safe["n_questions"] = len(exam["questions"])
    return safe


# ── Session start ────────────────────────────────────────────────────────────

@app.get("/api/start")
async def start_exam(student_id: str, exam_id: str = None):
    # Resolve exam
    if exam_id:
        exam = get_exam(exam_id)
        if not exam:
            raise HTTPException(404, "Exam not found")
    else:
        # Use demo exam
        exams = list_exams()
        if not exams:
            _seed_demo()
            exams = list_exams()
        exam = get_exam(exams[0]["exam_id"])
        exam_id = exam["exam_id"]

    if already_submitted(exam_id, student_id):
        raise HTTPException(409, "You have already submitted this exam. Only one attempt is allowed.")

    sid = new_session(exam_id, student_id)

    # Return questions WITHOUT reference_answer (security)
    student_questions = [
        {"qid": q.get("qid", f"q_{i+1}"), "question": q["question"]}
        for i, q in enumerate(exam["questions"])
    ]
    return {
        "session_id": sid,
        "exam_id": exam_id,
        "duration_s": exam["duration_s"],
        "questions": student_questions,
    }


# ── Event ingestion (anti-tamper: server ignores client totals) ─────────────

@app.post("/api/event")
async def ingest_events(request: Request):
    data = await request.json()
    session_id = data.get("session_id")
    events = data.get("events", [])

    if not session_exists(session_id):
        raise HTTPException(404, "Session not found")

    insert_events_batch(session_id, events)
    return {"ok": True, "count": len(events)}


# ── Camera frame (never stored) ──────────────────────────────────────────────

@app.post("/api/frame")
async def process_camera_frame(request: Request):
    data = await request.json()
    session_id = data.get("session_id")
    image_b64 = data.get("image", "")

    if not session_exists(session_id):
        raise HTTPException(404, "Session not found")

    face_state = _face_state_for(session_id)
    face_state, n_faces = process_frame(image_b64, face_state)

    return {
        "face_not_detected_count": face_state["face_not_detected_count"],
        "multiple_faces_count": face_state["multiple_faces_count"],
        "n_faces": n_faces,
    }


# ── Camera quality check ─────────────────────────────────────────────────────

@app.post("/api/quality")
async def camera_quality(request: Request):
    data = await request.json()
    image_b64 = data.get("image", "")
    return check_image_quality(image_b64)


# ── Autosave ─────────────────────────────────────────────────────────────────

@app.post("/api/autosave")
async def autosave(request: Request):
    data = await request.json()
    session_id = data.get("session_id")
    if not session_exists(session_id):
        raise HTTPException(404, "Session not found")
    answers = data.get("answers", [])
    for ans in answers:
        upsert_answer(session_id, ans["qid"], ans.get("answer", ""), ans.get("seconds_spent", 0))
    return {"ok": True}


@app.get("/api/autosave/{session_id}")
async def get_autosave(session_id: str):
    if not session_exists(session_id):
        raise HTTPException(404, "Session not found")
    answers = get_answers(session_id)
    return {"answers": answers}


# ── Submit exam ──────────────────────────────────────────────────────────────

@app.post("/api/submit")
async def submit_exam(request: Request):
    data = await request.json()
    session_id = data.get("session_id")
    student_id = data.get("student_id")
    answers_raw = data.get("answers", [])
    camera_available = data.get("camera_available", True)

    # Validate session
    session = get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    if session["status"] == "submitted":
        raise HTTPException(409, "Already submitted")

    exam_id = session["exam_id"]
    exam = get_exam(exam_id)

    # Timer enforcement: reject if past deadline
    elapsed = _exam_elapsed(session)
    if exam and elapsed > exam["duration_s"] + 30:  # 30 s grace
        update_session_status(session_id, "expired")
        raise HTTPException(403, "Exam time has expired")

    agent = get_agent()
    log_agent_step(session_id, "observe", f"student={student_id}, n_answers={len(answers_raw)}", "")

    # Build answers from DB (server authoritative) + submitted payload
    for ans in answers_raw:
        upsert_answer(session_id, ans["qid"], ans.get("answer", ""), ans.get("seconds_spent", 0))

    db_answers = {a["qid"]: a for a in get_answers(session_id)}

    # Build qa_pairs from exam questions + student answers
    qa_pairs = []
    seconds_spent_list = []
    if exam:
        for q in exam["questions"]:
            qid = q["qid"]
            stu_ans = db_answers.get(qid, {}).get("student_answer", "")
            secs = db_answers.get(qid, {}).get("seconds_spent") or 0
            qa_pairs.append({
                "qid": qid,
                "reference_answer": q["reference_answer"],
                "student_answer": stu_ans,
            })
            seconds_spent_list.append(float(secs))

    # Compute session features from SERVER-SIDE event log (anti-tamper)
    events_log = get_events(session_id)
    if camera_available:
        face_state = _face_state_for(session_id)
    else:
        face_state = {"face_not_detected_count": 0, "multiple_faces_count": 0}

    session_features = build_session_features(events_log, seconds_spent_list, face_state)
    log_agent_step(session_id, "features", json.dumps(session_features), "server-events")

    # Grade each question
    grading_pairs = [(q["reference_answer"], q["student_answer"]) for q in qa_pairs]
    log_agent_step(session_id, "grade", f"{len(grading_pairs)} questions", agent.meta.get("grader_model","RF"))

    # Explainable grading
    grading_detail_list = explain_grading_multi(
        [(q["reference_answer"], q["student_answer"]) for q in qa_pairs],
        agent.embedder
    )
    log_agent_step(session_id, "explain_grade", "grading detail computed", "explain.py")

    # Run agent pipeline
    result = agent.run_exam(student_id, grading_pairs, session_features)
    result["session_id"] = session_id
    result["exam_id"] = exam_id
    result["timestamp"] = datetime.utcnow().isoformat()
    result["notes"] = "" if camera_available else "face signals unavailable"
    for k, v in session_features.items():
        result[k] = v

    # Explainable risk
    risk_detail = explain_risk(session_features, agent.cheat_model, agent.cheat_features)
    result["grading_detail"] = {
        "per_question": [
            {
                "qid": q["qid"],
                "score": result["per_question_scores"][i] if i < len(result.get("per_question_scores", [])) else 0,
                "covered": grading_detail_list[i]["covered"],
                "missing": grading_detail_list[i]["missing"],
                "feedback": grading_detail_list[i]["feedback"],
            }
            for i, q in enumerate(qa_pairs)
        ]
    }
    result["risk_detail"] = risk_detail

    log_agent_step(session_id, "assess_risk", f"risk={result['risk_percent']}%", agent.meta.get("cheat_model","GB"))
    log_agent_step(session_id, "decide", result["decision"], "policy")

    # Collusion detection (async-friendly: do it per exam after each submit)
    try:
        exam_submissions = []
        for q in qa_pairs:
            exam_submissions.append({
                "session_id": session_id,
                "qid": q["qid"],
                "student_answer": q["student_answer"],
                "reference_answer": q["reference_answer"],
            })
        # Fetch other submitted sessions for this exam
        all_results = get_all_results()
        for r in all_results:
            if r.get("exam_id") == exam_id and r.get("session_id") != session_id:
                other_session_id = r["session_id"]
                other_answers = get_answers(other_session_id)
                if exam:
                    for q in exam["questions"]:
                        oa = next((a for a in other_answers if a["qid"] == q["qid"]), None)
                        exam_submissions.append({
                            "session_id": other_session_id,
                            "qid": q["qid"],
                            "student_answer": oa["student_answer"] if oa else "",
                            "reference_answer": q["reference_answer"],
                        })
        flags = detect_collusion(exam_submissions, agent.embedder)
        for flag in flags:
            save_collusion_flag(exam_id, flag["qid"], flag["session_a"], flag["session_b"], flag["similarity"])
            insert_event(flag["session_a"], "possible_collusion", meta={"with_session": flag["session_b"], "qid": flag["qid"], "similarity": flag["similarity"]})
            insert_event(flag["session_b"], "possible_collusion", meta={"with_session": flag["session_a"], "qid": flag["qid"], "similarity": flag["similarity"]})
        if flags:
            log_agent_step(session_id, "collusion", f"{len(flags)} pair(s) flagged", "collusion.py")
    except Exception as cex:
        print(f"[collusion] warning: {cex}")

    log_agent_step(session_id, "act", f"decision={result['decision']}, final_grade={result.get('final_grade')}", "")

    # Save result
    save_result(result)
    update_session_status(session_id, "submitted", camera_available)

    # Return result WITHOUT reference answers or risk details (student-facing)
    student_result = {
        "session_id": session_id,
        "decision": result["decision"],
        "message": {
            "RELEASE": "Your exam has been submitted and graded.",
            "RELEASE_WITH_WARNING": "Your exam has been submitted.",
            "HOLD_FOR_REVIEW": "Your exam has been submitted and is under review.",
        }.get(result["decision"], "Submitted."),
    }
    return student_result


# ── Examiner API ─────────────────────────────────────────────────────────────

@app.get("/api/results")
async def all_results():
    return get_all_results()


@app.get("/api/session/{session_id}")
async def session_detail(session_id: str):
    session = get_session(session_id)
    if not session:
        raise HTTPException(404)
    events = get_events(session_id)
    face_state = _face_state_for(session_id)
    # Compute provisional features
    db_answers = get_answers(session_id)
    seconds_list = [float(a.get("seconds_spent") or 0) for a in db_answers]
    provisional = build_session_features(events, seconds_list, face_state)
    result = get_result(session_id)
    log = get_agent_log(session_id)
    reviews = get_reviews(session_id)
    return {
        "session": dict(session),
        "provisional_features": provisional,
        "result": result,
        "events": events[-50:],  # last 50
        "agent_log": log,
        "reviews": reviews,
    }


@app.get("/api/sessions/active")
async def active_sessions():
    return get_active_sessions()


@app.get("/api/sessions/held")
async def held_sessions():
    return get_held_sessions()


@app.post("/api/review")
async def submit_review(request: Request):
    data = await request.json()
    session_id = data.get("session_id")
    action = data.get("action")  # approve | override | mark_cheat | dismiss
    note = data.get("note", "")
    override_grade = data.get("override_grade")
    reviewer = data.get("reviewer", "examiner")

    if not session_exists(session_id):
        raise HTTPException(404, "Session not found")
    if not note.strip():
        raise HTTPException(400, "A note is required for reviews")
    if action not in ("approve", "override", "mark_cheat", "dismiss"):
        raise HTTPException(400, f"Unknown action: {action}")

    insert_review(session_id, action, note, override_grade, reviewer)
    log_agent_step(session_id, "review", f"action={action} by {reviewer}", "human")
    return {"ok": True}


@app.get("/api/collusion/{exam_id}")
async def collusion_flags(exam_id: str):
    return get_collusion_flags(exam_id)


@app.get("/api/agent_log/{session_id}")
async def agent_log(session_id: str):
    if not session_exists(session_id):
        raise HTTPException(404)
    return get_agent_log(session_id)


@app.get("/api/export/csv")
async def export_csv():
    results = get_all_results()
    if not results:
        return {"error": "No results"}
    df = pd.DataFrame(results)
    csv_bytes = df.to_csv(index=False).encode("utf-8")

    def stream():
        yield csv_bytes

    return StreamingResponse(
        stream(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=results.csv"},
    )


@app.get("/api/export/audit_csv")
async def export_audit_csv():
    logs = get_all_agent_logs()
    if not logs:
        return {"error": "No logs recorded"}
    df = pd.DataFrame(logs)
    csv_bytes = df.to_csv(index=False).encode("utf-8")

    def stream():
        yield csv_bytes

    return StreamingResponse(
        stream(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=audit_log.csv"},
    )


@app.delete("/api/session/{session_id}")
async def delete_data(session_id: str):
    if not session_exists(session_id):
        raise HTTPException(404)
    delete_session_data(session_id)
    if session_id in _face_states:
        del _face_states[session_id]
    return {"ok": True, "message": "All data for this session has been deleted"}


@app.post("/api/appeal/{session_id}")
async def student_appeal(session_id: str, request: Request):
    data = await request.json()
    reason = data.get("reason", "Student appeal")
    if not session_exists(session_id):
        raise HTTPException(404)
    insert_review(session_id, "appeal", reason, reviewer="student")
    return {"ok": True, "message": "Your appeal has been submitted and will be reviewed by the examiner."}


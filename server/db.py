"""
db.py – SQLite schema + helper functions for the Exam Proctoring Agent.
All DB access goes through this module.
"""
import json
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE = Path(__file__).parent.parent
DB_PATH = BASE / "results.db"


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS exams (
            exam_id     TEXT PRIMARY KEY,
            title       TEXT NOT NULL,
            duration_s  INTEGER NOT NULL DEFAULT 1200,
            created_at  TEXT NOT NULL,
            questions   TEXT NOT NULL  -- JSON list of {qid, question, reference_answer}
        );

        CREATE TABLE IF NOT EXISTS sessions (
            session_id      TEXT PRIMARY KEY,
            exam_id         TEXT REFERENCES exams(exam_id),
            student_id      TEXT NOT NULL,
            started_at      TEXT NOT NULL,
            submitted_at    TEXT,
            status          TEXT NOT NULL DEFAULT 'active',
            camera_ok       INTEGER NOT NULL DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS events (
            event_id    INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT NOT NULL REFERENCES sessions(session_id),
            type        TEXT NOT NULL,
            t_server    TEXT NOT NULL,
            t_client_ms INTEGER,
            duration_ms INTEGER,
            meta        TEXT
        );

        CREATE TABLE IF NOT EXISTS answers (
            answer_id       INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id      TEXT NOT NULL REFERENCES sessions(session_id),
            qid             TEXT NOT NULL,
            student_answer  TEXT,
            seconds_spent   REAL,
            saved_at        TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS results (
            result_id           INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id          TEXT NOT NULL REFERENCES sessions(session_id),
            student_id          TEXT NOT NULL,
            exam_id             TEXT,
            timestamp           TEXT NOT NULL,
            tab_switches        INTEGER,
            face_not_detected_count INTEGER,
            multiple_faces_count    INTEGER,
            avg_time_per_question   REAL,
            copy_paste_events   INTEGER,
            mouse_idle_time     REAL,
            answer_speed_variance REAL,
            predicted_score     REAL,
            risk_percent        REAL,
            decision            TEXT,
            final_grade         REAL,
            reason              TEXT,
            notes               TEXT,
            per_question_scores TEXT,
            grading_detail      TEXT,
            risk_detail         TEXT
        );

        CREATE TABLE IF NOT EXISTS reviews (
            review_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT NOT NULL REFERENCES sessions(session_id),
            action      TEXT NOT NULL,
            override_grade REAL,
            note        TEXT NOT NULL,
            reviewer    TEXT NOT NULL DEFAULT 'examiner',
            reviewed_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS agent_log (
            log_id      INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT NOT NULL REFERENCES sessions(session_id),
            step        TEXT NOT NULL,
            detail      TEXT,
            model_ver   TEXT,
            ts          TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS collusion_flags (
            flag_id     INTEGER PRIMARY KEY AUTOINCREMENT,
            exam_id     TEXT NOT NULL,
            qid         TEXT NOT NULL,
            session_a   TEXT NOT NULL,
            session_b   TEXT NOT NULL,
            similarity  REAL NOT NULL,
            flagged_at  TEXT NOT NULL
        );
        """)


# ── helpers ─────────────────────────────────────────────────────────────────

def new_exam(title: str, duration_s: int, questions: List[Dict]) -> str:
    eid = str(uuid.uuid4())[:8].upper()
    for i, q in enumerate(questions):
        if not q.get("qid"):
            q["qid"] = f"q_{i+1}_{str(uuid.uuid4())[:6]}"
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO exams VALUES (?,?,?,datetime('now'),?)",
            (eid, title, duration_s, json.dumps(questions))
        )
    return eid


def get_exam(exam_id: str) -> Optional[Dict]:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM exams WHERE exam_id=?", (exam_id,)).fetchone()
    if not row:
        return None
    d = dict(row)
    qs = json.loads(d["questions"])
    modified = False
    for i, q in enumerate(qs):
        if not q.get("qid"):
            q["qid"] = f"q_{i+1}_{exam_id[:4]}"
            modified = True
    if modified:
        with get_conn() as conn:
            conn.execute("UPDATE exams SET questions=? WHERE exam_id=?", (json.dumps(qs), exam_id))
    d["questions"] = qs
    return d


def list_exams() -> List[Dict]:
    with get_conn() as conn:
        rows = conn.execute("SELECT exam_id, title, duration_s, created_at FROM exams ORDER BY created_at DESC").fetchall()
    return [dict(r) for r in rows]


def new_session(exam_id: str, student_id: str) -> str:
    sid = str(uuid.uuid4())
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO sessions(session_id,exam_id,student_id,started_at,status) VALUES(?,?,?,datetime('now'),'active')",
            (sid, exam_id, student_id)
        )
    return sid


def get_session(session_id: str) -> Optional[Dict]:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM sessions WHERE session_id=?", (session_id,)).fetchone()
    return dict(row) if row else None


def session_exists(session_id: str) -> bool:
    with get_conn() as conn:
        row = conn.execute("SELECT 1 FROM sessions WHERE session_id=?", (session_id,)).fetchone()
    return row is not None


def already_submitted(exam_id: str, student_id: str) -> bool:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT 1 FROM sessions WHERE exam_id=? AND student_id=? AND status='submitted'",
            (exam_id, student_id)
        ).fetchone()
    return row is not None


def update_session_status(session_id: str, status: str, camera_ok: bool = True):
    with get_conn() as conn:
        if status == "submitted":
            conn.execute(
                "UPDATE sessions SET status=?, submitted_at=datetime('now'), camera_ok=? WHERE session_id=?",
                (status, int(camera_ok), session_id)
            )
        else:
            conn.execute(
                "UPDATE sessions SET status=?, camera_ok=? WHERE session_id=?",
                (status, int(camera_ok), session_id)
            )


def insert_event(session_id: str, etype: str, t_client_ms: Optional[int] = None,
                 duration_ms: Optional[int] = None, meta: Optional[Dict] = None):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO events(session_id,type,t_server,t_client_ms,duration_ms,meta) VALUES(?,?,datetime('now'),?,?,?)",
            (session_id, etype, t_client_ms, duration_ms, json.dumps(meta) if meta else None)
        )


def insert_events_batch(session_id: str, events: List[Dict]):
    with get_conn() as conn:
        for ev in events:
            conn.execute(
                "INSERT INTO events(session_id,type,t_server,t_client_ms,duration_ms,meta) VALUES(?,?,datetime('now'),?,?,?)",
                (session_id, ev.get("type","unknown"), ev.get("t_client_ms"), ev.get("duration_ms"),
                 json.dumps(ev.get("meta")) if ev.get("meta") else None)
            )


def upsert_answer(session_id: str, qid: str, student_answer: str, seconds_spent: float):
    with get_conn() as conn:
        existing = conn.execute(
            "SELECT answer_id FROM answers WHERE session_id=? AND qid=?", (session_id, qid)
        ).fetchone()
        if existing:
            conn.execute(
                "UPDATE answers SET student_answer=?, seconds_spent=?, saved_at=datetime('now') WHERE session_id=? AND qid=?",
                (student_answer, seconds_spent, session_id, qid)
            )
        else:
            conn.execute(
                "INSERT INTO answers(session_id,qid,student_answer,seconds_spent,saved_at) VALUES(?,?,?,?,datetime('now'))",
                (session_id, qid, student_answer, seconds_spent)
            )


def get_answers(session_id: str) -> List[Dict]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM answers WHERE session_id=?", (session_id,)).fetchall()
    return [dict(r) for r in rows]


def get_events(session_id: str) -> List[Dict]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM events WHERE session_id=? ORDER BY t_server", (session_id,)).fetchall()
    return [dict(r) for r in rows]


def save_result(data: Dict[str, Any]):
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO results(
                session_id, student_id, exam_id, timestamp,
                tab_switches, face_not_detected_count, multiple_faces_count,
                avg_time_per_question, copy_paste_events, mouse_idle_time,
                answer_speed_variance, predicted_score, risk_percent,
                decision, final_grade, reason, notes,
                per_question_scores, grading_detail, risk_detail
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            data.get("session_id"), data["student_id"], data.get("exam_id"),
            data.get("timestamp"), data.get("tab_switches"), data.get("face_not_detected_count"),
            data.get("multiple_faces_count"), data.get("avg_time_per_question"),
            data.get("copy_paste_events"), data.get("mouse_idle_time"),
            data.get("answer_speed_variance"), data["predicted_score"], data["risk_percent"],
            data["decision"], data.get("final_grade"), data["reason"], data.get("notes",""),
            json.dumps(data.get("per_question_scores",[])),
            json.dumps(data.get("grading_detail",{})),
            json.dumps(data.get("risk_detail",{}))
        ))


def get_result(session_id: str) -> Optional[Dict]:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM results WHERE session_id=?", (session_id,)).fetchone()
    if not row:
        return None
    d = dict(row)
    for k in ("per_question_scores","grading_detail","risk_detail"):
        try:
            d[k] = json.loads(d.get(k) or "null")
        except Exception:
            d[k] = None
    return d


def get_all_results() -> List[Dict]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM results ORDER BY timestamp DESC").fetchall()
    out = []
    for row in rows:
        d = dict(row)
        for k in ("per_question_scores","grading_detail","risk_detail"):
            try:
                d[k] = json.loads(d.get(k) or "null")
            except Exception:
                d[k] = None
        out.append(d)
    return out


def get_active_sessions() -> List[Dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT s.*, r.risk_percent, r.decision FROM sessions s "
            "LEFT JOIN results r ON s.session_id=r.session_id "
            "ORDER BY s.started_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def insert_review(session_id: str, action: str, note: str,
                  override_grade: Optional[float] = None, reviewer: str = "examiner"):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO reviews(session_id,action,override_grade,note,reviewer,reviewed_at) VALUES(?,?,?,?,?,datetime('now'))",
            (session_id, action, override_grade, note, reviewer)
        )
    # update result final_grade if override
    if action == "override" and override_grade is not None:
        with get_conn() as conn:
            conn.execute(
                "UPDATE results SET final_grade=? WHERE session_id=?",
                (override_grade, session_id)
            )
    elif action == "approve":
        # set final_grade = predicted_score if it was held
        with get_conn() as conn:
            conn.execute(
                "UPDATE results SET final_grade=predicted_score, decision='RELEASE' WHERE session_id=? AND final_grade IS NULL",
                (session_id,)
            )


def get_reviews(session_id: str = None) -> List[Dict]:
    with get_conn() as conn:
        if session_id:
            rows = conn.execute("SELECT * FROM reviews WHERE session_id=? ORDER BY reviewed_at DESC", (session_id,)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM reviews ORDER BY reviewed_at DESC").fetchall()
    return [dict(r) for r in rows]


def get_held_sessions() -> List[Dict]:
    """Return sessions that are HOLD_FOR_REVIEW and not yet reviewed."""
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT r.*, s.student_id as sid FROM results r
            JOIN sessions s ON r.session_id=s.session_id
            WHERE r.decision='HOLD_FOR_REVIEW'
              AND r.session_id NOT IN (SELECT session_id FROM reviews)
            ORDER BY r.timestamp DESC
        """).fetchall()
    out = []
    for row in rows:
        d = dict(row)
        for k in ("per_question_scores","grading_detail","risk_detail"):
            try:
                d[k] = json.loads(d.get(k) or "null")
            except Exception:
                d[k] = None
        out.append(d)
    return out


def log_agent_step(session_id: str, step: str, detail: str = "", model_ver: str = ""):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO agent_log(session_id,step,detail,model_ver,ts) VALUES(?,?,?,?,datetime('now'))",
            (session_id, step, detail, model_ver)
        )


def get_agent_log(session_id: str) -> List[Dict]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM agent_log WHERE session_id=? ORDER BY ts", (session_id,)).fetchall()
    return [dict(r) for r in rows]


def get_all_agent_logs() -> List[Dict]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM agent_log ORDER BY ts DESC").fetchall()
    return [dict(r) for r in rows]


def save_collusion_flag(exam_id: str, qid: str, session_a: str, session_b: str, similarity: float):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO collusion_flags(exam_id,qid,session_a,session_b,similarity,flagged_at) VALUES(?,?,?,?,?,datetime('now'))",
            (exam_id, qid, session_a, session_b, similarity)
        )


def get_collusion_flags(exam_id: str = None) -> List[Dict]:
    with get_conn() as conn:
        if exam_id:
            rows = conn.execute("SELECT * FROM collusion_flags WHERE exam_id=? ORDER BY flagged_at DESC", (exam_id,)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM collusion_flags ORDER BY flagged_at DESC").fetchall()
    return [dict(r) for r in rows]


def delete_session_data(session_id: str):
    """GDPR-style: delete all data for a session."""
    with get_conn() as conn:
        for tbl in ("events","answers","reviews","agent_log","results"):
            conn.execute(f"DELETE FROM {tbl} WHERE session_id=?", (session_id,))
        conn.execute("DELETE FROM sessions WHERE session_id=?", (session_id,))


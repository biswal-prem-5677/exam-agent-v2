# Autonomous Exam Proctoring & Grading Agent 2.0

An intelligent, multi-tool autonomous proctoring and grading platform.  
Evaluates student exam answers using semantic text embeddings and lexical regression models.  
Monitors exam integrity via server-authoritative telemetry and calibrated cheating risk models.

---

## Architecture Diagram

```
+---------------------------------------------------------------------------------+
|                                STUDENT BROWSER                                  |
|   +-------------------------------------------------------------------------+   |
|   |  FastAPI Standalone Exam Portal (http://localhost:8000/exam)           |   |
|   |  - Debounced State Machine (ACTIVE <-> AWAY, 300ms)                     |   |
|   |  - VisibilityChange + Window Blur/Focus + 500ms Active Poll            |   |
|   |  - Shortcut Blocker + Fullscreen Manager + 10s Autosave Sync            |   |
|   |  - Live Webcam Frame Capture (Sent to /api/frame every 3s)              |   |
|   +-------------------------------------------------------------------------+   |
+----------------------------------------|----------------------------------------+
                                         | REST API (Events, Frames, Submits)
                                         v
+---------------------------------------------------------------------------------+
|                                FASTAPI SERVER                                   |
|   +-----------------------+ +--------------------+ +------------------------+   |
|   |   server/events.py    | |   server/vision.py | |   server/collusion.py  |   |
|   |   Anti-Tamper Log     | |   In-Memory Haar   | |   Cosine Similarity    |   |
|   |   Feature Derivation  | |   Episodes (>=2)   | |   Pairs (>= 0.92)      |   |
|   +-----------------------+ +--------------------+ +------------------------+   |
|                                        |                                        |
|                                        v                                        |
|   +-------------------------------------------------------------------------+   |
|   |                     Autonomous Agent (agent.py)                         |   |
|   |  - Tool 1 (Grader): Random Forest (9 Lexical + SentenceTransformer Embed)|  |
|   |  - Tool 2 (Proctor): Calibrated Monotonic Gradient Boosting (7 Signals) |   |
|   |  - Explainability: Key idea coverage + What-if Risk Attribution delta   |   |
|   |  - Decision Policy: Low (0.25) / High (0.60) + Guardrail Floor Overrides|   |
|   +-------------------------------------------------------------------------+   |
|                                        |                                        |
|                                        v                                        |
|                           SQLite Storage (results.db)                           |
+----------------------------------------|----------------------------------------+
                                         | Real-time Queries & Review Actions
                                         v
+---------------------------------------------------------------------------------+
|                           EXAMINER STREAMLIT CONSOLE                            |
|                          (http://localhost:8501)                                |
|   - Live Monitor: Active candidates, provisional risk meter, event timeline    |
|   - Review Queue: Human-in-the-loop dispute disposition (Approve/Override/Note)|
|   - Explainability: Covered vs missing key ideas + Risk attribution chart       |
|   - Collusion Audit: Flags suspected student-to-student copycat answers         |
|   - Cohort Analytics: Risk distribution histogram + decision breakdown donut   |
|   - Exam Authoring: Create custom exams with timers & reference answers         |
+---------------------------------------------------------------------------------+
```

---

## Quick Start

### 1. Prerequisites & Installation
Ensure you have Python 3.10+ installed:
```bash
pip install -r requirements.txt
```

### 2. Start Both Services with One Command
```bash
python run.py
```
This automatically starts:
- **Student Exam Portal**: `http://localhost:8000/exam`
- **Examiner Console**: `http://localhost:8501`

Press `Ctrl+C` in the terminal to cleanly terminate both background services.

---

## Behavioral Signals Detection Matrix

The proctoring model accepts **EXACTLY 7 features**, derived strictly on the server from raw atomic event logs:

| Feature Name | Client / Browser Detection Mechanism | Server-Side Derivation Logic |
|---|---|---|
| `tab_switches` | State machine: `visibilitychange`, `blur`/`focus`, `pagehide`, and a 500 ms active focus poll. Debounced by 300 ms. Suppressed during the first 5 s and while camera permission prompt is open. | Server counts atomic `"away"` episodes logged in the `events` table. Client totals are ignored. |
| `copy_paste_events` | DOM listeners on `copy`, `cut`, and `paste` events. | Server sums count of `"copy"`, `"cut"`, and `"paste"` events in the session event log. |
| `mouse_idle_time` | Inactivity timer tracking lack of `mousemove`, `mousedown`, `keydown`, `scroll`, and `touchstart` after an initial 10 s grace period, active only while window is focused. | Server sums duration (in seconds) of reported `"idle"` intervals from the database log. |
| `face_not_detected_count` | Client streams raw JPEG canvas frames every 3 s via `POST /api/frame`. | Server executes OpenCV Haar Cascade (`haarcascade_frontalface_default.xml`). Counts episodes (>=2 consecutive frames with 0 faces). |
| `multiple_faces_count` | Same 3 s webcam frame stream. | Server counts episodes (>=2 consecutive frames with >1 detected face). |
| `avg_time_per_question` | Client records active question time via question switch timestamps. | Mean seconds calculated across all submitted question responses (`statistics.mean`). |
| `answer_speed_variance` | Time deltas per answered question. | Population standard deviation of seconds spent per question (`statistics.pstdev`). Returns 0.0 for 1 question. |

---

## Decision Policy & Thresholds

Thresholds and guardrail policies are loaded dynamically from `models/meta.json`:
- **Risk < 25% (`thresholds.low`)** → `RELEASE` (Grade released automatically).
- **25% ≤ Risk ≤ 60% (`thresholds.high`)** → `RELEASE_WITH_WARNING` (Grade released with warning notes).
- **Risk > 60%** → `HOLD_FOR_REVIEW` (Grade withheld pending examiner disposition).

### Plain-Rule Safety Guardrails:
An examinee is **never silently released** if extreme single counts exceed honest maximums:
- `tab_switches > 11.0`
- `face_not_detected_count > 11.0`
- `multiple_faces_count > 1.0`
- `copy_paste_events > 3.0`

Any guardrail hit automatically elevates a `RELEASE` decision to at least `RELEASE_WITH_WARNING`.

---

## Privacy & Responsible AI Statement

1. **Zero Image Storage**: Webcam frames are analyzed in-memory on the server and immediately garbage-collected. No images or video streams are ever written to disk or logged.
2. **Fairness Guarantee**: Lighting issues or low-quality webcams never cause automatic test failure. The agent flags ambiguity as `HOLD_FOR_REVIEW` or `RELEASE_WITH_WARNING` for human review.
3. **Student Right to Appeal**: Students can submit technical dispute appeals directly through the portal, which creates a priority review item in the examiner queue.
4. **GDPR / Data Deletion**: The examiner console provides a "Delete Candidate's Records" function that permanently wipes all answers, logs, and events for a session.

---

## Manual Test Checklist

1. **Tab Switch Detection**:
   - Open `http://localhost:8000/exam?student_id=S101&debug=1`
   - Switch tabs (`Ctrl+Tab` or click another window) 3 times.
   - Observe live counter in debug panel incrementing to `3`.
   - Submit and verify `results.db` records `tab_switches == 3`.
2. **Copy / Paste Detection**:
   - Copy text from outside or within the page and paste into the answer box twice.
   - Observe counter incrementing to `2`.
3. **Mouse Idle Time**:
   - Keep hands off keyboard/mouse for 30 seconds.
   - Observe idle duration incrementing ~20 s (accounting for 10 s grace period).
4. **Camera Fallback**:
   - Block or deny camera permissions.
   - Observe "Camera unavailable" notice. Note is added: `"face signals unavailable"` without failing student.
5. **Human Review Disposition**:
   - Take exam with high cheating activity (`15` tab switches, multiple pastes).
   - Check Streamlit console under "Review Queue" (`http://localhost:8501`).
   - Fill in mandatory justification note and click "Approve" or "Override".

---

## 3-Minute Demo Script

1. **Load Demo Class**: Open `http://localhost:8501` → Navigate to **Class Analytics** → Click **"🚀 Load 12-Student Demo Class"**.
2. **Inspect Cohort Analytics**: Observe the risk distribution histogram and decision donut breakdown across 12 realistic candidates (honest, nervous, bad-camera, blatant cheater, stealth, colluding pair).
3. **Live Exam Simulation**: Open `http://localhost:8000/exam?student_id=DEMO_USER&debug=1`. Enter consent, start exam, Alt+Tab away 3 times, paste text twice, and click "Submit".
4. **Live Monitor & Hold Decision**: Return to `http://localhost:8501` → **Live Monitor**. Notice `DEMO_USER` flagged as `HOLD_FOR_REVIEW`.
5. **Human-in-the-Loop Review**: Go to **Review Queue**, view telemetry details, enter a justification note, and apply a grade override.
6. **Cross-Candidate Collusion**: Open **Collusion Detection** tab to show `S008_colluder_A` and `S009_colluder_B` automatically flagged with 96% semantic cosine similarity.
7. **Explainability**: Open **Results & Reports**, select a candidate, and view the **Covered vs Missing Key Ideas** breakdown and **What-If Risk Attribution** waterfall chart.

---

## Known Limitations

- **Synthetic Behavioural Training Data**: Behavioral risk weights were calibrated on simulated interaction distributions from `data/sessions.csv`. Real exam cohorts may have varying baseline focus patterns.
- **OpenCV Haar Cascade Simplicity**: Classical frontal-face Haar cascades are sensitive to extreme head tilt, dark environments, or backlighting. Pre-exam camera quality warnings help mitigate false negatives.
- **Stealth / Peripheral Cheating**: Off-screen second monitors or hidden secondary mobile devices cannot be caught by browser DOM events alone unless reflected in extended idle times or face absence.
- **Exam Length Calibration**: Shorter exams (e.g. 5 minutes) will naturally exhibit lower raw event counts than a standard 20-minute exam. (Optional feature scaling can be enabled).

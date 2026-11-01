/**
 * exam.js – Autonomous Exam Proctoring & Grading Agent
 * Phase-2 tab detection state machine + all required signals.
 *
 * Design:
 *  - State machine: ACTIVE ↔ AWAY. One episode per leave (debounce 300 ms).
 *  - Detects: visibilitychange, blur/focus, pagehide, 500 ms poll.
 *  - Suppress blur for first 5 s and while camera-permission modal is open.
 *  - Events sent immediately via POST /api/event (batched every 2 s).
 *  - sendBeacon on pagehide for final flush.
 *  - Mouse idle: seconds with no mouse/key/scroll/touch AFTER 10 s grace, only while active.
 *  - Client totals are NOT trusted by the server (anti-tamper). We send typed events.
 *  - Autosave every 10 s to /api/autosave.
 *  - Fullscreen required at start; re-enter button shown on exit.
 *  - Blocked shortcuts (Ctrl+C/V/X/A/P/S/U, F12, PrintScreen, Alt+Tab, Ctrl+Tab).
 *  - Debug panel on ?debug=1.
 */

"use strict";

// ── Config ─────────────────────────────────────────────────────────────────
const EVENT_BATCH_MS    = 2000;
const IDLE_GRACE_MS     = 10000;
const IDLE_POLL_MS      = 1000;
const TAB_DEBOUNCE_MS   = 300;
const SUPPRESS_BLUR_MS  = 5000;  // ignore blur events for first 5 s
const AWAY_POLL_MS      = 500;
const AUTOSAVE_MS       = 10000;
const FACE_INTERVAL_MS  = 3000;
const HEARTBEAT_MS      = 10000;
const HEARTBEAT_GAP_MS  = 15000;

// ── State ──────────────────────────────────────────────────────────────────
let sessionId      = null;
let examId         = null;
let studentId      = "";
let questions      = [];        // [{qid, question}]
let qaFull         = [];        // with reference_answer (from exam start response)
let currentIdx     = 0;
let answers        = {};        // {qid: {answer, seconds_spent, qStartMs}}
let cameraStream   = null;
let cameraAvailable = false;
let examStartMs    = null;
let durationMs     = 1200000;
let timerInterval  = null;
let submitted      = false;
let suppressBlur   = true;
let cameraPromptOpen = false;
let debugMode      = new URLSearchParams(location.search).get("debug") === "1";

// Tab detection state machine: ACTIVE | AWAY
let pageState      = "ACTIVE";
let awayStartMs    = null;
let debounceTimer  = null;

// Counters (for debug display; server recomputes from events)
let _dbgTabSwitches   = 0;
let _dbgCopyPaste     = 0;
let _dbgIdleSec       = 0;
let _dbgFaceOk        = true;

// Idle tracking
let lastActivityMs = Date.now();
let idleAccumMs    = 0;
let idlePollTimer  = null;
let idleGracePassed = false;

// Event queue
let eventQueue = [];
let batchTimer = null;

// Heartbeat
let lastHeartbeatMs = Date.now();
let heartbeatTimer  = null;

// Face state (mirrored from server)
let faceState = { face_not_detected_count: 0, multiple_faces_count: 0 };

// ── DOM shortcuts ──────────────────────────────────────────────────────────
const $  = id => document.getElementById(id);
const show = id => $(id).classList.remove("hidden");
const hide = id => $(id).classList.add("hidden");
const announce = msg => { $("live-announce").textContent = ""; setTimeout(() => { $("live-announce").textContent = msg; }, 50); };

// ── Boot ───────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  const params = new URLSearchParams(location.search);
  if (params.get("student_id")) $("roll-input").value = params.get("student_id");
  if (params.get("exam_id"))    $("exam-code-input").value = params.get("exam_id").toUpperCase();
  if (debugMode) $("debug-panel").style.display = "block";

  $("roll-input").addEventListener("input", updateStartBtn);
  $("consent-checkbox").addEventListener("change", updateStartBtn);
  $("start-btn").addEventListener("click", onStartClick);
  $("back-btn").addEventListener("click", onBack);
  $("next-btn").addEventListener("click", onNext);
  $("submit-btn").addEventListener("click", onSubmit);
  if ($("sidebar-submit-btn")) $("sidebar-submit-btn").addEventListener("click", onSubmit);
  if ($("fs-btn")) $("fs-btn").addEventListener("click", enterFullscreen);
  if ($("fs-reentry-btn")) $("fs-reentry-btn").addEventListener("click", enterFullscreen);
  if ($("fs-start-btn")) $("fs-start-btn").addEventListener("click", enterFullscreen);
  $("appeal-btn").addEventListener("click", onAppeal);
  $("appeal-post-btn").addEventListener("click", onAppeal);
  $("answer-area").addEventListener("input", onAnswerInput);

  installActivityListeners();
  installTabDetection();
  installClipboardListeners();
  installShortcutBlocker();
  installFullscreenListener();

  updateStartBtn();
});

function updateStartBtn() {
  const roll = $("roll-input").value.trim();
  const ok = roll && $("consent-checkbox").checked;
  $("start-btn").disabled = !ok;
}

// ── Start flow ─────────────────────────────────────────────────────────────
async function onStartClick() {
  const roll = $("roll-input").value.trim();
  if (!roll) { $("roll-error").style.display = "block"; return; }
  $("roll-error").style.display = "none";

  // Enter fullscreen synchronously on direct user gesture
  enterFullscreen();

  studentId = roll;
  examId    = $("exam-code-input").value.trim().toUpperCase() || null;

  $("start-btn").disabled = true;
  $("start-btn").textContent = "Starting…";

  try {
    await doStartExam();
  } catch (err) {
    $("start-btn").disabled = false;
    $("start-btn").textContent = "Start Exam";
    alert("Could not start exam: " + err.message);
  }
}

async function doStartExam() {
  let url = `/api/start?student_id=${encodeURIComponent(studentId)}`;
  if (examId) url += `&exam_id=${encodeURIComponent(examId)}`;

  const resp = await fetch(url);
  if (!resp.ok) {
    const body = await resp.json().catch(() => ({}));
    throw new Error(body.detail || `Server error ${resp.status}`);
  }
  const data = await resp.json();

  sessionId  = data.session_id;
  examId     = data.exam_id;
  durationMs = (data.duration_s || 1200) * 1000;
  questions  = data.questions;          // no reference_answer
  examStartMs = Date.now();

  // Try to restore autosaved answers
  await restoreAutosave();

  // Detect page reload
  try {
    const navEntries = performance.getEntriesByType("navigation");
    if (navEntries.length > 0 && navEntries[0].type === "reload") {
      pushEvent("page_reload");
    }
  } catch (e) {}

  // Show exam UI
  hide("consent-screen");
  show("topbar");
  show("progress-wrap");
  show("exam-layout");

  $("topbar-title").textContent = "📝 Exam";
  buildNavBar();
  showQuestion(0);

  // Start timers
  startTimer();
  startHeartbeat();
  startBatchFlush();
  startIdleTracking();
  startAutosave();

  // Camera
  cameraPromptOpen = true;
  await startCamera();
  cameraPromptOpen = false;

  // Suppress blur for 5 s after start
  suppressBlur = true;
  setTimeout(() => { suppressBlur = false; }, SUPPRESS_BLUR_MS);

  // Request fullscreen (non-blocking)
  enterFullscreen();
}

// ── Timer ──────────────────────────────────────────────────────────────────
function startTimer() {
  timerInterval = setInterval(() => {
    const elapsed = Date.now() - examStartMs;
    const remaining = Math.max(0, durationMs - elapsed);
    const mins = Math.floor(remaining / 60000);
    const secs = Math.floor((remaining % 60000) / 1000);
    const txt = `${String(mins).padStart(2,"0")}:${String(secs).padStart(2,"0")}`;
    $("timer-chip").textContent = "⏱ " + txt;
    $("timer-chip").classList.toggle("urgent", remaining < 120000);

    if (remaining <= 0 && !submitted) {
      clearInterval(timerInterval);
      announce("Time is up. Auto-submitting your exam.");
      onSubmit();
    }
  }, 500);
}

// ── Question navigation ────────────────────────────────────────────────────
function buildNavBar() {
  const nav = $("q-nav");
  nav.innerHTML = "";
  questions.forEach((q, i) => {
    const btn = document.createElement("button");
    btn.className = "q-nav-btn";
    btn.textContent = `Q${i+1}`;
    btn.setAttribute("aria-label", `Question ${i+1}`);
    btn.addEventListener("click", () => { saveCurrentAnswer(); showQuestion(i); });
    nav.appendChild(btn);
  });
}

function updateNav() {
  const btns = $("q-nav").querySelectorAll(".q-nav-btn");
  btns.forEach((btn, i) => {
    btn.className = "q-nav-btn" +
      (i === currentIdx ? " current" : answers[questions[i]?.qid]?.answer?.trim() ? " answered" : "");
  });
  // Progress bar
  const answered = Object.values(answers).filter(a => a.answer?.trim()).length;
  $("progress-bar").style.width = `${Math.round(100 * answered / Math.max(questions.length,1))}%`;
}

function showQuestion(idx) {
  currentIdx = Math.max(0, Math.min(idx, questions.length - 1));
  const q = questions[currentIdx];
  if (!q) return;

  $("q-counter").textContent = `Question ${currentIdx + 1} of ${questions.length}`;
  $("q-text").textContent = q.question;

  // Restore saved answer
  $("answer-area").value = answers[q.qid]?.answer || "";
  updateWordCount();

  // Start timing this question
  if (!answers[q.qid]) answers[q.qid] = { qid: q.qid, answer: "", seconds_spent: 0, qStartMs: Date.now() };
  else answers[q.qid].qStartMs = Date.now();

  // Buttons
  $("back-btn").disabled = currentIdx === 0;
  const last = currentIdx === questions.length - 1;
  $("next-btn").classList.toggle("hidden", last);
  $("submit-btn").classList.toggle("hidden", !last);

  updateNav();
  $("answer-area").focus();
  announce(`Question ${currentIdx + 1} of ${questions.length}: ${q.question}`);
}

function saveCurrentAnswer() {
  const q = questions[currentIdx];
  if (!q) return;
  const text = $("answer-area").value;
  const elapsed = (Date.now() - (answers[q.qid]?.qStartMs || Date.now())) / 1000;
  if (!answers[q.qid]) answers[q.qid] = { qid: q.qid, answer: text, seconds_spent: 0, qStartMs: Date.now() };
  answers[q.qid].answer = text;
  answers[q.qid].seconds_spent = (answers[q.qid].seconds_spent || 0) + elapsed;
  answers[q.qid].qStartMs = Date.now();
}

function onBack() { saveCurrentAnswer(); showQuestion(currentIdx - 1); }
function onNext() { saveCurrentAnswer(); showQuestion(currentIdx + 1); }

function onAnswerInput() {
  const q = questions[currentIdx];
  if (q) { if (!answers[q.qid]) answers[q.qid] = {qid:q.qid,answer:"",seconds_spent:0,qStartMs:Date.now()}; answers[q.qid].answer = $("answer-area").value; }
  updateWordCount();
  updateNav();
}

function updateWordCount() {
  const words = $("answer-area").value.trim().split(/\s+/).filter(Boolean).length;
  $("word-count").textContent = words;
}

// ── Submit ─────────────────────────────────────────────────────────────────
let isSubmitting = false;

async function onSubmit() {
  if (submitted || isSubmitting) return;
  isSubmitting = true;

  // Stop timer immediately to prevent duplicate triggers
  clearInterval(timerInterval);

  // Disable buttons while submitting
  if ($("submit-btn")) {
    $("submit-btn").disabled = true;
    $("submit-btn").textContent = "Submitting…";
  }
  if ($("sidebar-submit-btn")) {
    $("sidebar-submit-btn").disabled = true;
    $("sidebar-submit-btn").textContent = "Submitting…";
  }

  saveCurrentAnswer();

  // Final flush of events
  try {
    await flushEvents();
  } catch (e) {}

  const payload = {
    session_id: sessionId,
    student_id: studentId,
    answers: Object.values(answers).map(a => ({
      qid: a.qid,
      answer: a.answer || "",
      seconds_spent: a.seconds_spent || 0,
    })),
    camera_available: cameraAvailable,
  };

  try {
    const resp = await fetch("/api/submit", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    // Both HTTP 200 (success) and HTTP 409 (already recorded) are treated as submitted
    if (!resp.ok && resp.status !== 409) {
      const body = await resp.json().catch(() => ({}));
      throw new Error(body.detail || `Submit failed ${resp.status}`);
    }

    submitted = true;
    hide("exam-layout");
    hide("topbar");
    hide("progress-wrap");
    hide("warning-banner");
    hide("fs-reentry");
    show("submitted-page");
    $("sid-display").textContent = sessionId;
    stopCamera();

    // Cleanly exit fullscreen on submitted screen
    if (document.fullscreenElement || document.webkitFullscreenElement) {
      const exitFn = document.exitFullscreen || document.webkitExitFullscreen || document.mozCancelFullScreen;
      exitFn?.call(document).catch(() => {});
    }
  } catch (err) {
    isSubmitting = false;
    if ($("submit-btn")) {
      $("submit-btn").disabled = false;
      $("submit-btn").textContent = "Submit Exam ✓";
    }
    if ($("sidebar-submit-btn")) {
      $("sidebar-submit-btn").disabled = false;
      $("sidebar-submit-btn").textContent = "Finish & Submit Exam";
    }
    alert("Submission error: " + err.message + "\nPlease try again or contact the examiner.");
  }
}

// ── Appeal ─────────────────────────────────────────────────────────────────
async function onAppeal() {
  const reason = prompt("Describe the technical problem or reason for appeal:");
  if (!reason) return;
  if (!sessionId) { alert("No active session."); return; }
  try {
    await fetch(`/api/appeal/${sessionId}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ reason }),
    });
    alert("Appeal submitted. The examiner has been notified.");
  } catch { alert("Could not submit appeal – please contact the examiner directly."); }
}

// ── Camera ─────────────────────────────────────────────────────────────────
async function startCamera() {
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({ video: { width: 320, height: 240 } });
    cameraAvailable = true;
    $("cam-chip").textContent = "🎥 Camera ON";
    $("cam-chip").className = "chip active";

    // Pre-exam quality check
    await doCameraQualityCheck();

    // Start face detection loop
    startFaceDetection();
  } catch {
    cameraAvailable = false;
    $("cam-chip").textContent = "🚫 Camera OFF";
    $("cam-chip").className = "chip away";
    $("camera-warning").style.display = "block";
    pushEvent("camera_denied");
  }
}

async function doCameraQualityCheck() {
  if (!cameraStream) return;
  try {
    const frame = await captureFrame();
    if (!frame) return;
    show("quality-check");
    $("quality-msg").textContent = "Checking…";
    const resp = await fetch("/api/quality", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: frame }),
    });
    const result = await resp.json().catch(() => null);
    if (result && result.warnings?.length) {
      $("quality-msg").textContent = "⚠ " + result.warnings.join("; ");
      $("quality-check").style.borderLeft = "4px solid var(--amber)";
    } else if (result) {
      $("quality-msg").textContent = "✓ Camera looks good";
    }
  } catch { /* non-fatal */ }
}

function startFaceDetection() {
  if (!cameraStream || !sessionId) return;
  const video = document.createElement("video");
  video.srcObject = cameraStream;
  video.playsInline = true;
  video.muted = true;
  video.onloadedmetadata = () => {
    video.play();
    setInterval(async () => {
      if (submitted || !sessionId) return;
      const frame = await captureFrame(video);
      if (!frame) return;
      try {
        const resp = await fetch("/api/frame", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ session_id: sessionId, image: frame }),
        });
        const d = await resp.json().catch(() => null);
        if (d) { faceState = d; if (debugMode) updateDebugPanel(); }
      } catch { /* ignore */ }
    }, FACE_INTERVAL_MS);
  };
}

async function captureFrame(videoEl) {
  try {
    let video = videoEl;
    if (!video) {
      video = document.createElement("video");
      video.srcObject = cameraStream;
      video.playsInline = true;
      video.muted = true;
      await new Promise(r => { video.onloadedmetadata = r; video.play(); });
    }
    const canvas = document.createElement("canvas");
    canvas.width = 320; canvas.height = 240;
    canvas.getContext("2d").drawImage(video, 0, 0, 320, 240);
    const dataUrl = canvas.toDataURL("image/jpeg", 0.6);
    return dataUrl.split(",")[1];
  } catch { return null; }
}

function stopCamera() {
  if (cameraStream) { cameraStream.getTracks().forEach(t => t.stop()); cameraStream = null; }
}

// ── Tab detection state machine ────────────────────────────────────────────
function goAway(reason) {
  if (pageState === "AWAY") return;
  if (submitted) return;
  pageState = "AWAY";
  awayStartMs = Date.now();
  _dbgTabSwitches++;
  pushEvent("away", null, { reason });
  $("warning-banner").style.display = "block";
  announce("Please return to the exam window to continue.");
  if (debugMode) updateDebugPanel();
}

function goActive() {
  if (pageState === "ACTIVE") return;
  const durMs = Date.now() - (awayStartMs || Date.now());
  pageState = "ACTIVE";
  awayStartMs = null;
  pushEvent("return", durMs);
  $("warning-banner").style.display = "none";
  if (debugMode) updateDebugPanel();
}

function installTabDetection() {
  // 1. visibilitychange
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) scheduleAway("visibility"); else goActive();
  });

  // 2. window blur/focus (with suppress logic)
  window.addEventListener("blur", () => {
    if (suppressBlur || cameraPromptOpen) return;
    scheduleAway("blur");
  });
  window.addEventListener("focus", () => {
    clearTimeout(debounceTimer);
    goActive();
  });

  // 3. pagehide
  window.addEventListener("pagehide", () => { pushEvent("page_hide"); flushEvents(true); });

  // 4. 500 ms poll
  setInterval(() => {
    const away = document.hidden || !document.hasFocus();
    if (away && pageState === "ACTIVE" && !suppressBlur && !cameraPromptOpen) {
      scheduleAway("poll");
    } else if (!away && pageState === "AWAY") {
      goActive();
    }
  }, AWAY_POLL_MS);
}

function scheduleAway(reason) {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => goAway(reason), TAB_DEBOUNCE_MS);
}

// ── Clipboard ──────────────────────────────────────────────────────────────
function installClipboardListeners() {
  ["copy","cut","paste"].forEach(type => {
    document.addEventListener(type, e => {
      _dbgCopyPaste++;
      pushEvent(type);
      if (debugMode) updateDebugPanel();
    });
  });
}

// ── Shortcut blocker ──────────────────────────────────────────────────────
function installShortcutBlocker() {
  const BLOCKED = new Set(["c","v","x","a","p","s","u"]);
  document.addEventListener("keydown", e => {
    // Log right-click attempt (handled via contextmenu)
    if (e.key === "PrintScreen") { pushEvent("blocked_shortcut", null, { key:"PrintScreen" }); e.preventDefault(); }
    if (e.key === "F12") { pushEvent("blocked_shortcut", null, { key:"F12" }); e.preventDefault(); }
    if (e.ctrlKey && BLOCKED.has(e.key.toLowerCase())) {
      pushEvent("blocked_shortcut", null, { key: `Ctrl+${e.key.toUpperCase()}` });
      // Allow Ctrl+A/V for legitimate editing — just log, don't block
    }
    // Alt+Tab and Ctrl+Tab keydown (browser may handle, just log)
    if (e.altKey && e.key === "Tab") pushEvent("blocked_shortcut", null, { key:"Alt+Tab" });
    if (e.ctrlKey && e.key === "Tab") pushEvent("blocked_shortcut", null, { key:"Ctrl+Tab" });
  });

  document.addEventListener("contextmenu", e => {
    pushEvent("right_click");
    e.preventDefault();
  });
}

// ── Fullscreen ─────────────────────────────────────────────────────────────
function enterFullscreen() {
  const el = document.documentElement;
  const reqFn = el.requestFullscreen || el.webkitRequestFullscreen || el.mozRequestFullScreen || el.msRequestFullscreen;
  if (reqFn) {
    reqFn.call(el).then(() => {
      hide("fs-reentry");
      if ($("fs-btn")) hide("fs-btn");
    }).catch(err => {
      console.warn("Fullscreen permission / user-activation notice:", err);
    });
  }
}

function installFullscreenListener() {
  const onFsChange = () => {
    const inFs = !!(document.fullscreenElement || document.webkitFullscreenElement || document.mozFullScreenElement || document.msFullscreenElement);
    if ($("fs-chip")) {
      $("fs-chip").style.display = "inline-flex";
      if (inFs) {
        $("fs-chip").textContent = "⛶ Fullscreen";
        $("fs-chip").className = "chip active";
        hide("fs-reentry");
        if ($("fs-btn")) hide("fs-btn");
      } else {
        $("fs-chip").textContent = "⛶ Exited";
        $("fs-chip").className = "chip away";
        if (!submitted && examStartMs) {
          show("fs-reentry");
          if ($("fs-btn")) show("fs-btn");
          pushEvent("fullscreen_exit");
        }
      }
    }
  };

  document.addEventListener("fullscreenchange", onFsChange);
  document.addEventListener("webkitfullscreenchange", onFsChange);
  document.addEventListener("mozfullscreenchange", onFsChange);
  document.addEventListener("MSFullscreenChange", onFsChange);
}

// ── Activity / idle tracking ───────────────────────────────────────────────
function installActivityListeners() {
  ["mousemove","mousedown","keydown","scroll","touchstart"].forEach(evt => {
    document.addEventListener(evt, () => { lastActivityMs = Date.now(); }, { passive: true });
  });
}

function startIdleTracking() {
  // Grace period: ignore first 10 s
  setTimeout(() => {
    idleGracePassed = true;
    idlePollTimer = setInterval(() => {
      if (submitted || pageState === "AWAY") return;
      const now = Date.now();
      if (now - lastActivityMs >= 1000) {
        idleAccumMs += IDLE_POLL_MS;
        _dbgIdleSec = Math.floor(idleAccumMs / 1000);
        // Push idle event every 5 s of accumulated idle
        if (idleAccumMs % 5000 < IDLE_POLL_MS) {
          pushEvent("idle", 5000, { total_s: _dbgIdleSec });
        }
        if (debugMode) updateDebugPanel();
      } else {
        // Reset on activity
      }
    }, IDLE_POLL_MS);
  }, IDLE_GRACE_MS);
}

// ── Heartbeat ──────────────────────────────────────────────────────────────
function startHeartbeat() {
  heartbeatTimer = setInterval(async () => {
    if (submitted) return;
    const now = Date.now();
    if (now - lastHeartbeatMs > HEARTBEAT_GAP_MS) {
      pushEvent("heartbeat_gap", now - lastHeartbeatMs);
    }
    lastHeartbeatMs = now;
    pushEvent("heartbeat");
    await flushEvents();
  }, HEARTBEAT_MS);
}

// ── Event queue ────────────────────────────────────────────────────────────
function pushEvent(type, durationMs, meta) {
  eventQueue.push({
    type,
    t_client_ms: Date.now(),
    duration_ms: durationMs ?? undefined,
    meta: meta ?? undefined,
  });
}

function startBatchFlush() {
  batchTimer = setInterval(flushEvents, EVENT_BATCH_MS);
}

async function flushEvents(useBeacon) {
  if (!sessionId || eventQueue.length === 0) return;
  const batch = eventQueue.splice(0, eventQueue.length);
  const payload = JSON.stringify({ session_id: sessionId, events: batch });
  if (useBeacon) {
    navigator.sendBeacon?.("/api/event", new Blob([payload], { type: "application/json" }));
  } else {
    try {
      await fetch("/api/event", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: payload,
        keepalive: true,
      });
    } catch { eventQueue.unshift(...batch); /* retry */ }
  }
}

// ── Autosave ───────────────────────────────────────────────────────────────
function startAutosave() {
  setInterval(async () => {
    if (!sessionId || submitted) return;
    saveCurrentAnswer();
    const payload = {
      session_id: sessionId,
      answers: Object.values(answers).map(a => ({ qid: a.qid, answer: a.answer || "", seconds_spent: a.seconds_spent || 0 })),
    };
    fetch("/api/autosave", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }).catch(() => {});
  }, AUTOSAVE_MS);
}

async function restoreAutosave() {
  if (!sessionId) return;
  try {
    const resp = await fetch(`/api/autosave/${sessionId}`);
    if (!resp.ok) return;
    const data = await resp.json();
    (data.answers || []).forEach(a => {
      answers[a.qid] = { qid: a.qid, answer: a.student_answer || "", seconds_spent: a.seconds_spent || 0, qStartMs: Date.now() };
    });
  } catch { /* non-fatal */ }
}

// ── Debug panel ────────────────────────────────────────────────────────────
function updateDebugPanel() {
  $("dbg-content").innerHTML = [
    `Session: ${sessionId?.substring(0,8) ?? "none"}`,
    `State: ${pageState}`,
    `Tab switches: ${_dbgTabSwitches}`,
    `Copy/paste: ${_dbgCopyPaste}`,
    `Idle: ${_dbgIdleSec}s`,
    `Face ok: ${faceState.face_not_detected_count === 0 ? "✓" : "✗ ("+faceState.face_not_detected_count+")"}`,
    `Multi-face: ${faceState.multiple_faces_count}`,
    `Queue: ${eventQueue.length} pending`,
  ].join("<br>");
}

// Debug simulation buttons (called from HTML onclick)
function dbgSimTabSwitch() {
  goAway("debug"); setTimeout(() => goActive(), 800);
}
function dbgSimPaste() {
  _dbgCopyPaste++; pushEvent("paste");
  if (debugMode) updateDebugPanel();
}
function dbgSimIdle() {
  idleAccumMs += 10000; _dbgIdleSec = Math.floor(idleAccumMs/1000);
  pushEvent("idle", 10000, { total_s: _dbgIdleSec });
  if (debugMode) updateDebugPanel();
}


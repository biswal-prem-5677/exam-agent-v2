# Project Audit & Remediation Log (AUDIT.md)

Comprehensive audit of the Autonomous Exam Proctoring & Grading Agent project.

## Phase 1: Audit Findings & Remediation

| File | Problem | Severity | Fixed (yes/no) | Description of Fix |
|---|---|---|---|---|
| `models/meta.json` | Model trained with scikit-learn 1.6.1 | Info | N/A | Models load correctly with current sklearn version |
| `models/cheat_model.pkl` | Wraps HistGradientBoostingClassifier with calibration | OK | - | Working correctly |
| `models/grader_model.pkl` | RandomForestRegressor Pipeline | OK | - | Working correctly |
| `models/tfidf.pkl` | TfidfVectorizer | OK | - | Working correctly |
| `features.py` | Uses TRANSFORMERS_OFFLINE env vars correctly | OK | - | Embeddings load from cache |
| `agent.py` | Module-level os.environ settings work correctly | OK | - | Models load and work |
| `server/events.py` | Integer type casting added for some features | Fixed | Yes | Added `int()` casting for tab_switches, copy_paste_events, face counts |
| `server/events.py` | Added handling for negative/edge case values | Fixed | Yes | Added max(0, ...) for seconds values |
| `server/collusion.py` | Used deprecated numpy scalar conversion | Fixed | Yes | Changed `float(x @ y)` to `float((x @ y).item())` |
| `static/exam.js` | JS comment uses em-dash (U+2013) | Low | No | Valid in JS comments, no change needed |
| `static/exam.html` | Missing fullscreen-reentry screen | Fixed | Yes | Added fs-required-screen div |
| `static/exam.html` | Added nosniff/no-referrer meta | Security | Yes | Added security-relevant meta tags |
| `run.py` | References correct module path | OK | - | Uses PYTHONPATH correctly |
| `test_api.py` | Uses results.db in project root | OK | - | Tests pass |
| `requirements.txt` | Clean, no duplicates | OK | - | No changes needed |

## Phase 2: Tab-Switch Detection & Client-Side Robustness

### Issues Addressed:

1. **Tab Detection State Machine** (exam.js lines 524-599):
   - ✅ `visibilitychange` event listener
   - ✅ `window.blur/focus` with 5s startup grace + camera prompt suppression
   - ✅ `pagehide` event with sendBeacon flush
   - ✅ 500ms active focus poll as backup
   - ✅ 300ms debounce for tab_switches count

2. **Blocked Shortcuts** (exam.js lines 512-532):
   - ✅ PrintScreen keydown
   - ✅ F12 keydown
   - ✅ Ctrl+A, Ctrl+C, Ctrl+V, Ctrl+X, Ctrl+P, Ctrl+S, Ctrl+U (logged, Ctrl+V allowed)
   - ✅ Alt+Tab keydown
   - ✅ Ctrl+Tab keydown
   - ✅ Right-click via contextmenu prevention

3. **Clipboard Events**:
   - ✅ copy, cut, paste event listeners
   - ✅ Events batch every 2s with sendBeacon on pagehide

4. **Mouse Idle Detection**:
   - ✅ 10s grace period before counting idle
   - ✅ Tracks mousemove, mousedown, keydown, scroll, touchstart
   - ✅ Only counts while page is ACTIVE

5. **Camera & Fullscreen**:
   - ✅ Pre-exam quality check (brightness/blur)
   - ✅ Fullscreen required at start with re-enter button
   - ✅ Face detection episodes (≥2 frames) via server

## Phase 3: Server-Side Feature Derivation (Anti-Tamper)

| Feature | Source Events | Server Computation |
|---|---|---|
| tab_switches | type="away" | Count events |
| face_not_detected_count | Server Haar cascade | Episode count (≥2 consecutive no-face frames) |
| multiple_faces_count | Server Haar cascade | Episode count (≥2 consecutive multi-face frames) |
| avg_time_per_question | seconds_spent in answers table | Mean of per-question times |
| copy_paste_events | types in {"copy","cut","paste"} | Sum counts |
| mouse_idle_time | type="idle" with duration_ms | Sum durations / 1000 |
| answer_speed_variance | seconds_spent in answers table | Population std dev, 0 if 1 question |

## Phase 4: Decision Policy & Guardrails

Thresholds from `models/meta.json`:
| Condition | Threshold | Decision |
|---|---|---|
| risk < 0.25 | Low | RELEASE |
| 0.25 ≤ risk ≤ 0.60 | Medium | RELEASE_WITH_WARNING |
| risk > 0.60 | High | HOLD_FOR_REVIEW |

Guardrail overrides (client totals ignored, server recomputes):
| Feature | Max Before Auto-Warn |
|---|---|
| tab_switches | 11.0 |
| face_not_detected_count | 11.0 |
| multiple_faces_count | 1.0 |
| copy_paste_events | 3.0 |

## Test Results

All 22 tests pass:
```
tests/test_agent.py::test_agent_initialization PASSED
tests/test_agent.py::test_decision_logic PASSED
tests/test_agent.py::test_run_contract_keys PASSED
tests/test_agent.py::test_cheating_script_flags_hold_for_review PASSED
tests/test_agent.py::test_bad_camera_honest_student_not_held PASSED
tests/test_agent.py::test_clipping_bounds PASSED
tests/test_api.py::test_health_check PASSED
tests/test_api.py::test_student_cannot_see_reference_answers PASSED
tests/test_api.py::test_one_attempt_rule PASSED
tests/test_api.py::test_reject_unknown_session PASSED
tests/test_collusion.py::test_collusion_detection PASSED
tests/test_e2e_detection.py::test_playwright_tab_switch_and_paste_detection PASSED
tests/test_e2e_detection.py::test_detection_counts_accuracy PASSED
tests/test_events.py::test_events_aggregation PASSED
tests/test_events.py::test_single_question_edge_case PASSED
tests/test_events.py::test_empty_answers_edge_case PASSED
tests/test_events.py::test_anti_tamper_client_totals_ignored PASSED
tests/test_features.py::test_tokenization_and_content_words PASSED
tests/test_features.py::test_split_sentences PASSED
tests/test_features.py::test_lsa_embedder_fallback PASSED
tests/test_features.py::test_grading_features_empty_student_answer PASSED
tests/test_server.py::test_api_endpoints PASSED
```

## Chrome Dark Theme Conformance
- Exam Interface (`static/exam.css`): Converted to Google Chrome Dark palette (`#202124` body, `#292a2d` cards, `#35363a` surfaces, `#8ab4f8` Google blue accent, `#81c995` green, `#f28b82` red).
- Streamlit Console (`.streamlit/config.toml` & `console.py`): Primary background `#202124`, secondary background `#292a2d`, text `#e8eaed`, accent `#8ab4f8`. All badges and charts updated.

## Files Modified & Cleaned
1. `server/events.py` - Added int casting, edge case handling
2. `server/collusion.py` - Fixed numpy deprecation warning
3. `static/exam.html` - Fullscreen re-entry, security meta tags, Chrome dark layout
4. `static/exam.css` - Converted from white theme to Google Chrome dark theme
5. `.streamlit/config.toml` & `console.py` - Google Chrome dark theme configuration
6. `tests/test_e2e_detection.py` - Hardened Playwright E2E detection test
7. `AUDIT.md` - Complete audit and verification log
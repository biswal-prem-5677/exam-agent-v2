"""
console.py – Autonomous Exam Proctoring & Grading Agent – Examiner Console
Streamlit app covering all operational, analytical, explainability and review tasks.
"""
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure offline operation for sentence transformers
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))

from agent import ExamAgent
from server.db import (
    get_all_results, get_active_sessions, get_held_sessions,
    get_agent_log, get_collusion_flags, get_events, get_answers,
    list_exams, new_exam, delete_session_data, save_result, new_session,
    update_session_status, save_collusion_flag, get_all_agent_logs
)
from server.review import process_review_action
from server.events import build_session_features

st.set_page_config(
    page_title="Examiner Console | Autonomous Proctoring Agent",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize ExamAgent singleton
@st.cache_resource
def load_agent():
    return ExamAgent()

agent = load_agent()

# Color badges & styles (Chrome Dark Mode palette)
DECISION_COLORS = {
    "RELEASE": "#81c995",
    "RELEASE_WITH_WARNING": "#fdd663",
    "HOLD_FOR_REVIEW": "#f28b82"
}

def decision_badge(decision: str) -> str:
    color = DECISION_COLORS.get(decision, "#9aa0a6")
    return f'<span style="background-color: {color}22; color: {color}; border: 1px solid {color}66; padding: 4px 10px; border-radius: 9999px; font-weight: 600; font-size: 0.82rem; letter-spacing: 0.02em;">{decision}</span>'

# Chrome Dark Theme UI Enhancements & Metric Card Boxes
st.markdown("""
<style>
/* Stat / Metric KPI Cards with Boxes */
[data-testid="stMetric"] {
    background-color: #292a2d !important;
    border: 1px solid #3c4043 !important;
    border-radius: 12px !important;
    padding: 16px 18px !important;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.35) !important;
    transition: transform 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease !important;
}
[data-testid="stMetric"]:hover {
    border-color: #5f6368 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5) !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    color: #9aa0a6 !important;
    letter-spacing: 0.03em !important;
    margin-bottom: 6px !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.75rem !important;
    font-weight: 700 !important;
    color: #e8eaed !important;
    font-feature-settings: "tnum" !important;
}
[data-testid="stMetricDelta"] {
    font-size: 0.84rem !important;
    font-weight: 600 !important;
}

/* Individual Metric Box Top-Color Accents */
div[data-testid="column"]:nth-of-type(1) [data-testid="stMetric"] { border-top: 3px solid #8ab4f8 !important; }
div[data-testid="column"]:nth-of-type(2) [data-testid="stMetric"] { border-top: 3px solid #81c995 !important; }
div[data-testid="column"]:nth-of-type(3) [data-testid="stMetric"] { border-top: 3px solid #f28b82 !important; }
div[data-testid="column"]:nth-of-type(4) [data-testid="stMetric"] { border-top: 3px solid #c58af9 !important; }
div[data-testid="column"]:nth-of-type(5) [data-testid="stMetric"] { border-top: 3px solid #fdd663 !important; }

/* Session Cards Border Wrapper */
[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #292a2d !important;
    border: 1px solid #3c4043 !important;
    border-radius: 12px !important;
    padding: 14px 18px !important;
    margin-bottom: 12px !important;
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.25) !important;
    transition: border-color 0.15s ease !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: #5f6368 !important;
}

/* Badges & Chips */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.82rem;
    font-weight: 600;
}
.pill-active { background: #81c99520; color: #81c995; border: 1px solid #81c99555; }
.pill-submitted { background: #8ab4f820; color: #8ab4f8; border: 1px solid #8ab4f855; }

/* Decision Guide Banner */
.decision-guide-box {
    display: flex;
    flex-wrap: wrap;
    gap: 1.2rem;
    align-items: center;
    background: #252629;
    border: 1px solid #3c4043;
    border-radius: 10px;
    padding: 10px 16px;
    margin: 8px 0 20px 0;
    font-size: 0.84rem;
    color: #9aa0a6;
}
.guide-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
}
.guide-dot {
    width: 9px;
    height: 9px;
    border-radius: 50%;
    display: inline-block;
}
</style>
""", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("🛡️ Examiner Console")
st.sidebar.caption("Autonomous Proctoring & Multi-Tool Agent 2.0")

nav_choice = st.sidebar.radio(
    "Navigation",
    ["🔴 Live Monitor", "📋 Review Queue", "📊 Results & Reports", "👥 Collusion Detection", "📈 Class Analytics", "✏️ Exam Builder", "🧪 Simulation Sandbox"]
)

st.sidebar.divider()
st.sidebar.subheader("Responsible AI & Governance")
st.sidebar.info(
    "🔒 **Privacy Guarantee**: Video frames are analyzed server-side with zero disk writes. Images are discarded in-memory immediately after Haar cascade evaluation.\n\n"
    "⚖️ **Fairness Guardrail**: Camera/lighting artifacts never trigger automatic failure. Decisions above threshold produce HOLD_FOR_REVIEW for human discretion."
)

# -----------------------------------------------------------------------------
# Top KPI Header
# -----------------------------------------------------------------------------
all_res = get_all_results()
active_sess = get_active_sessions()
held_sess = get_held_sessions()

total_students = len(set([s.get("student_id") for s in active_sess] + [r.get("student_id") for r in all_res]))
submitted_count = len(all_res)
held_count = len(held_sess)
avg_score = np.mean([r["predicted_score"] for r in all_res]) if all_res else 0.0
avg_risk = np.mean([r["risk_percent"] for r in all_res]) if all_res else 0.0

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("👥 Total Candidates", total_students)
col2.metric("📝 Submitted Exams", submitted_count)
col3.metric("⚠️ Pending Reviews", held_count, delta=f"{held_count} hold" if held_count > 0 else "Clear", delta_color="inverse")
col4.metric("🎯 Avg Predicted Score", f"{avg_score:.2f} / 5.0")
col5.metric("🛡️ Avg Cheating Risk", f"{avg_risk:.1f}%")

# Quick Visual Decision Guide
st.markdown("""
<div class="decision-guide-box">
  <span style="font-weight: 600; color: #e8eaed;">💡 Decision Guide:</span>
  <span class="guide-badge"><span class="guide-dot" style="background:#81c995;"></span><b>RELEASE</b> (Risk &lt; 25%): Honest behaviour, grade auto-released</span>
  <span class="guide-badge"><span class="guide-dot" style="background:#fdd663;"></span><b>WARNING</b> (25% - 60%): Low-level anomalies detected</span>
  <span class="guide-badge"><span class="guide-dot" style="background:#f28b82;"></span><b>HOLD FOR REVIEW</b> (&gt; 60%): Guardrail hit, human decision required</span>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# 1. LIVE MONITOR
# =============================================================================
if nav_choice == "🔴 Live Monitor":
    st.subheader("🔴 Live Examination Monitor")
    st.caption("Active sessions refresh continuously. Shows real-time activity and provisional risk.")
    
    auto_refresh = st.checkbox("Auto-refresh (every 3 seconds)", value=False)
    if auto_refresh:
        time.sleep(3)
        st.rerun()

    if not active_sess:
        st.info("No active or historical exam sessions found.")
    else:
        df_active = pd.DataFrame(active_sess)
        
        # Display session status cards
        for idx, row in df_active.iterrows():
            sid = row["session_id"]
            status = row.get("status", "active")
            
            # Retrieve recent events for live counter
            evs = get_events(sid)
            tabs = sum(1 for e in evs if e.get("type") == "away")
            pastes = sum(1 for e in evs if e.get("type") in ("copy", "cut", "paste"))
            idle_s = sum((e.get("duration_ms") or 0) / 1000.0 for e in evs if e.get("type") == "idle")
            
            with st.container(border=True):
                c1, c2, c3, c4, c5 = st.columns([2, 1.4, 2.2, 2, 1.4])
                c1.markdown(f"**👤 Student**: `{row['student_id']}`<br><span style='color:#9aa0a6;font-size:0.8rem;'>Session: <code>{sid[:8]}...</code></span>", unsafe_allow_html=True)
                if status == 'active':
                    c2.markdown("<span class='status-pill pill-active'>🟢 In Progress</span>", unsafe_allow_html=True)
                else:
                    c2.markdown("<span class='status-pill pill-submitted'>🏁 Submitted</span>", unsafe_allow_html=True)
                
                c3.markdown(
                    f"<div style='font-size:0.84rem;line-height:1.45;color:#e8eaed;'>"
                    f"📑 Tab leaves: <b>{tabs}</b><br>"
                    f"📋 Copy/Paste: <b>{pastes}</b><br>"
                    f"⏱️ Mouse Idle: <b>{idle_s:.0f}s</b>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                
                # Provisional or Final Risk
                if status == "submitted":
                    r_pct = row.get("risk_percent", 0.0)
                    dec = row.get("decision", "RELEASE")
                    c4.markdown(f"<div style='font-size:0.84rem;'><b>Final Risk</b>: <span style='font-size:1.1rem;font-weight:700;'>{r_pct}%</span><div style='margin-top:4px;'>{decision_badge(dec)}</div></div>", unsafe_allow_html=True)
                else:
                    # Provisional computation
                    ans = get_answers(sid)
                    sec_list = [float(a.get("seconds_spent") or 0) for a in ans]
                    prov_feats = build_session_features(evs, sec_list, {"face_not_detected_count": 0, "multiple_faces_count": 0})
                    prov_risk = float(agent.cheat_model.predict_proba(pd.DataFrame([prov_feats])[agent.cheat_features])[0, 1]) * 100
                    c4.markdown(f"<div style='font-size:0.84rem;'><b>Provisional Risk</b> ⚠️:<br><span style='font-size:1.2rem;font-weight:700;color:#fdd663;'>{prov_risk:.1f}%</span><br><span style='color:#9aa0a6;font-size:0.75rem;'>(live estimate)</span></div>", unsafe_allow_html=True)
                
                with c5:
                    show_timeline = st.button("Timeline", key=f"tl_btn_{sid}", use_container_width=True)
                
                if show_timeline:
                    st.write("---")
                    st.write(f"### Event Timeline for `{row['student_id']}`")
                    if not evs:
                        st.caption("No browser events logged yet.")
                    else:
                        ev_df = pd.DataFrame(evs)[["t_server", "type", "duration_ms", "meta"]]
                        st.dataframe(ev_df, use_container_width=True)

# =============================================================================
# 2. REVIEW QUEUE
# =============================================================================
elif nav_choice == "📋 Review Queue":
    st.subheader("📋 Human-in-the-Loop Review Queue")
    st.caption("Sessions held for human inspection (HOLD_FOR_REVIEW) or student appeals.")
    
    if not held_sess:
        st.success("✅ Clean Queue: No exam sessions currently require human intervention.")
    else:
        for item in held_sess:
            sid = item["session_id"]
            with st.expander(f"🚩 HOLD: {item['student_id']} (Risk: {item['risk_percent']}%, Pred: {item['predicted_score']}/5.0)", expanded=True):
                rc1, rc2 = st.columns([1, 1])
                with rc1:
                    st.write("**Signals Flagged**:")
                    st.markdown(f"*{item.get('reason', 'None')}*")
                    st.write("**Recorded Behavioral Telemetry**:")
                    st.markdown("""
                    <div style="background:#202124;border:1px solid #3c4043;border-radius:8px;padding:12px;font-size:0.85rem;margin-top:6px;">
                      <table style="width:100%;border-collapse:collapse;color:#e8eaed;">
                        <tr style="border-bottom:1px solid #2d3034;"><td style="padding:5px 0;color:#9aa0a6;">📑 Tab Switches</td><td style="text-align:right;font-weight:700;">{tabs}</td></tr>
                        <tr style="border-bottom:1px solid #2d3034;"><td style="padding:5px 0;color:#9aa0a6;">👤 Face Not Detected</td><td style="text-align:right;font-weight:700;">{no_face}</td></tr>
                        <tr style="border-bottom:1px solid #2d3034;"><td style="padding:5px 0;color:#9aa0a6;">👥 Multiple Faces</td><td style="text-align:right;font-weight:700;">{multi_face}</td></tr>
                        <tr style="border-bottom:1px solid #2d3034;"><td style="padding:5px 0;color:#9aa0a6;">⏱️ Avg Time / Question</td><td style="text-align:right;font-weight:700;">{avg_t}s</td></tr>
                        <tr style="border-bottom:1px solid #2d3034;"><td style="padding:5px 0;color:#9aa0a6;">📋 Copy/Paste Events</td><td style="text-align:right;font-weight:700;">{pastes}</td></tr>
                        <tr style="border-bottom:1px solid #2d3034;"><td style="padding:5px 0;color:#9aa0a6;">🛑 Mouse Idle Duration</td><td style="text-align:right;font-weight:700;">{idle}s</td></tr>
                        <tr><td style="padding:5px 0;color:#9aa0a6;">📊 Speed Variance</td><td style="text-align:right;font-weight:700;">{std}</td></tr>
                      </table>
                    </div>
                    """.format(
                        tabs=item.get("tab_switches", 0),
                        no_face=item.get("face_not_detected_count", 0),
                        multi_face=item.get("multiple_faces_count", 0),
                        avg_t=f"{float(item.get('avg_time_per_question') or 0):.1f}",
                        pastes=item.get("copy_paste_events", 0),
                        idle=f"{float(item.get('mouse_idle_time') or 0):.1f}",
                        std=f"{float(item.get('answer_speed_variance') or 0):.2f}"
                    ), unsafe_allow_html=True)
                
                with rc2:
                    st.write("**Review Decision & Disposition**:")
                    action = st.radio("Action", ["approve", "override", "mark_cheat", "dismiss"], key=f"act_{sid}",
                                      format_func=lambda x: {
                                          "approve": "✅ Approve System Grade",
                                          "override": "✏️ Override Grade",
                                          "mark_cheat": "🚫 Confirm Cheating (Grade = 0.0)",
                                          "dismiss": "❌ Dismiss Submission"
                                      }[x])
                    
                    override_val = None
                    if action == "override":
                        override_val = st.number_input("Override Final Grade (0.0 - 5.0)", min_value=0.0, max_value=5.0, value=float(item.get("predicted_score") or 3.0), step=0.1, key=f"ov_{sid}")
                    
                    note = st.text_area("Mandatory Examiner Note / Justification *", placeholder="Explain reason for review decision...", key=f"note_{sid}")
                    
                    if st.button("Submit Human Review Verdict", type="primary", key=f"sub_rev_{sid}"):
                        if not note.strip():
                            st.error("Error: A mandatory justification note is required to resolve review items.")
                        else:
                            try:
                                grade_to_apply = 0.0 if action == "mark_cheat" else override_val
                                if action == "mark_cheat":
                                    action = "override"
                                    grade_to_apply = 0.0
                                process_review_action(sid, action, note, grade_to_apply, reviewer="Lead Examiner")
                                st.success("Verdict saved and logged to audit trail.")
                                st.rerun()
                            except Exception as ex:
                                st.error(f"Failed to submit review: {ex}")

# =============================================================================
# 3. RESULTS & REPORTS
# =============================================================================
elif nav_choice == "📊 Results & Reports":
    st.subheader("📊 Examination Results & Explainable AI Reports")
    
    if not all_res:
        st.info("No exam results available in the database.")
    else:
        df_results = pd.DataFrame(all_res)
        summary_cols = ["student_id", "timestamp", "predicted_score", "final_grade", "risk_percent", "decision"]
        st.dataframe(
            df_results[summary_cols].rename(columns={
                "student_id": "Candidate",
                "timestamp": "Submitted At",
                "predicted_score": "Model Score",
                "final_grade": "Final Grade",
                "risk_percent": "Risk %",
                "decision": "Decision"
            }),
            use_container_width=True
        )
        
        # Download CSVs
        csv_data = df_results.to_csv(index=False).encode("utf-8")
        all_logs = get_all_agent_logs()
        audit_csv_data = pd.DataFrame(all_logs).to_csv(index=False).encode("utf-8") if all_logs else b""
        
        c_dl1, c_dl2 = st.columns(2)
        c_dl1.download_button("📥 Download All Results (CSV)", csv_data, "exam_results.csv", "text/csv")
        if all_logs:
            c_dl2.download_button("📋 Export Full Agent Audit Trail (CSV)", audit_csv_data, "agent_audit_trail.csv", "text/csv")
        
        st.divider()
        st.subheader("Candidate In-Depth Inspection & Explainability Drawer")
        student_opts = {f"{r['student_id']} (Session: {r['session_id'][:8]}...)": r['session_id'] for r in all_res}
        sel_label = st.selectbox("Select Candidate to Inspect", list(student_opts.keys()))
        sel_sid = student_opts[sel_label]
        sel_res = next(r for r in all_res if r["session_id"] == sel_sid)
        
        tab_explain, tab_timeline, tab_audit, tab_report = st.tabs(["💡 Explainability (Grading & Risk)", "⏱️ Timeline & Telemetry", "📝 Audit Feed", "📄 Printable Report"])
        
        # TAB: Explainability
        with tab_explain:
            ex_c1, ex_c2 = st.columns(2)
            
            with ex_c1:
                st.write("#### 🧠 Explainable Grading")
                g_detail = sel_res.get("grading_detail") or {}
                per_q = g_detail.get("per_question") or []
                if not per_q:
                    st.caption("Detailed question breakdown not recorded for this session.")
                else:
                    for i, qd in enumerate(per_q):
                        with st.expander(f"Question {i+1} – Score: {qd.get('score', 'N/A')}/5.0", expanded=(i==0)):
                            st.write(f"**Feedback**: {qd.get('feedback', '')}")
                            if qd.get("covered"):
                                st.markdown("**✅ Covered Key Ideas:**")
                                for c in qd["covered"]:
                                    st.markdown(f"- *{c}*")
                            if qd.get("missing"):
                                st.markdown("**❌ Missing Key Ideas:**")
                                for m in qd["missing"]:
                                    st.markdown(f"- *{m}*")
            
            with ex_c2:
                st.write("#### 🛡️ Explainable Risk (What-If Contributions)")
                r_detail = sel_res.get("risk_detail") or {}
                sorted_contribs = r_detail.get("sorted_contributions") or []
                if not sorted_contribs:
                    st.caption("Detailed feature attribution not calculated.")
                else:
                    st.write(f"Baseline Proctoring Risk: **{r_detail.get('base_risk', 0.0)}%**")
                    contrib_df = pd.DataFrame(sorted_contribs)
                    chart = alt.Chart(contrib_df).mark_bar().encode(
                        x=alt.X("contribution:Q", title="Risk Delta vs Honest Baseline (%)"),
                        y=alt.Y("feature:N", sort="-x", title="Feature"),
                        color=alt.condition(
                            alt.datum.contribution > 0,
                            alt.value("#f28b82"),
                            alt.value("#81c995")
                        )
                    ).properties(height=280)
                    st.altair_chart(chart, use_container_width=True)
                    st.caption("Positive (red) indicates behavioral feature pushed risk higher than normal baseline.")

        # TAB: Timeline
        with tab_timeline:
            evs = get_events(sel_sid)
            if not evs:
                st.caption("No browser events logged.")
            else:
                st.dataframe(pd.DataFrame(evs)[["t_server", "type", "duration_ms", "meta"]], use_container_width=True)

        # TAB: Audit Feed
        with tab_audit:
            log_entries = get_agent_log(sel_sid)
            if not log_entries:
                st.caption("No agent pipeline logs recorded.")
            else:
                for entry in log_entries:
                    st.markdown(f"- **`{entry['ts']}`** | `[{entry['step'].upper()}]` – {entry['detail']} *(tool: {entry['model_ver'] or 'agent'})*")

        # TAB: Printable HTML Report
        with tab_report:
            html_report = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Proctoring Report - {sel_res['student_id']}</title>
                <style>
                    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; padding: 24px; color: #1e293b; }}
                    h1, h2 {{ color: #0f172a; }}
                    .badge {{ display: inline-block; padding: 4px 10px; border-radius: 9999px; font-weight: bold; font-size: 14px; }}
                    table {{ width: 100%; border-collapse: collapse; margin-top: 16px; }}
                    th, td {{ border: 1px solid #cbd5e1; padding: 8px 12px; text-align: left; }}
                    th {{ background: #f1f5f9; }}
                </style>
            </head>
            <body>
                <h1>Official Proctored Exam Report</h1>
                <p><strong>Candidate ID:</strong> {sel_res['student_id']} | <strong>Session:</strong> {sel_res['session_id']}</p>
                <p><strong>Date & Time:</strong> {sel_res['timestamp']}</p>
                <hr>
                <h2>Performance Summary</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Predicted Grade</td><td>{sel_res['predicted_score']} / 5.0</td></tr>
                    <tr><td>Final Grade Released</td><td>{sel_res.get('final_grade', 'HELD FOR REVIEW')}</td></tr>
                    <tr><td>Cheating Probability</td><td>{sel_res['risk_percent']}%</td></tr>
                    <tr><td>Agent Decision</td><td>{sel_res['decision']}</td></tr>
                </table>
                <h2>Behavioral Telemetry Logged</h2>
                <table>
                    <tr><th>Metric</th><th>Recorded Value</th></tr>
                    <tr><td>Tab Switches</td><td>{sel_res.get('tab_switches', 0)}</td></tr>
                    <tr><td>Copy / Paste Events</td><td>{sel_res.get('copy_paste_events', 0)}</td></tr>
                    <tr><td>Mouse Idle Duration</td><td>{sel_res.get('mouse_idle_time', 0.0)} s</td></tr>
                    <tr><td>Face Not Detected Count</td><td>{sel_res.get('face_not_detected_count', 0)}</td></tr>
                    <tr><td>Multiple Faces Count</td><td>{sel_res.get('multiple_faces_count', 0)}</td></tr>
                </table>
                <h2>Agent Rationale & Findings</h2>
                <p><em>{sel_res.get('reason', '')}</em></p>
                <hr>
                <small>Generated autonomously by Exam Proctoring & Grading Agent 2.0</small>
            </body>
            </html>
            """
            st.download_button(
                "🖨️ Download Official HTML Report (Print to PDF)",
                html_report,
                file_name=f"exam_report_{sel_res['student_id']}.html",
                mime="text/html"
            )

        # Danger Zone: GDPR delete
        with st.expander("⚠️ Data Privacy & GDPR Administration"):
            st.warning("Per privacy regulations, candidates or administrators may request immediate total deletion of telemetry.")
            if st.button("Delete Candidate's Records Permanently", type="secondary", key=f"del_{sel_sid}"):
                delete_session_data(sel_sid)
                st.success("Candidate record completely scrubbed.")
                st.rerun()

# =============================================================================
# 4. COLLUSION DETECTION
# =============================================================================
elif nav_choice == "👥 Collusion Detection":
    st.subheader("👥 Cross-Candidate Collusion Detector")
    st.caption("Analyzes semantic text embedding overlap (cosine similarity >= 0.92) across identical exam questions.")
    
    exams = list_exams()
    if not exams:
        st.info("No exams recorded.")
    else:
        exam_titles = {e["title"]: e["exam_id"] for e in exams}
        chosen_exam_title = st.selectbox("Select Exam to Audit", list(exam_titles.keys()))
        chosen_eid = exam_titles[chosen_exam_title]
        
        flags = get_collusion_flags(chosen_eid)
        if not flags:
            st.success("✅ No suspicious text collusion or copycat clusters identified for this examination.")
        else:
            st.warning(f"⚠️ Flagged **{len(flags)}** suspected pair(s) with high semantic similarity.")
            flag_df = pd.DataFrame(flags)
            st.dataframe(flag_df[["qid", "session_a", "session_b", "similarity", "flagged_at"]], use_container_width=True)

# =============================================================================
# 5. CLASS ANALYTICS
# =============================================================================
elif nav_choice == "📈 Class Analytics":
    st.subheader("📈 Cohort & Class Analytics")
    
    # Load Demo Class Button
    with st.container(border=True):
        st.write("#### 🧪 Seed Demonstration Cohort")
        st.caption("Injects a realistic 12-candidate cohort (honest, nervous, bad-camera, blatant cheater, stealth, and colluding pair) for live demonstrations.")
        if st.button("🚀 Load 12-Student Demo Class", type="primary"):
            demo_eid = new_exam("Machine Learning Midterm", 1200, [
                {"qid": "q1", "question": "What is overfitting?", "reference_answer": "Overfitting happens when a model learns training noise and fails to generalize."},
                {"qid": "q2", "question": "What is gradient descent?", "reference_answer": "An optimization algorithm to minimize loss by iteratively moving toward the gradient."}
            ])
            
            # 12 candidate archetypes
            archetypes = [
                ("S001_honest", 1, 0, 0, 58.0, 0, 40.0, 10.0, 4.8, 4.2, "RELEASE"),
                ("S002_honest_fast", 2, 0, 0, 32.0, 0, 20.0, 8.0, 4.5, 9.8, "RELEASE"),
                ("S003_nervous", 3, 1, 0, 75.0, 1, 80.0, 14.0, 4.0, 18.2, "RELEASE"),
                ("S004_bad_camera", 2, 8, 0, 64.0, 0, 70.0, 12.0, 3.8, 22.4, "RELEASE_WITH_WARNING"),
                ("S005_stealth_cheater", 6, 2, 0, 45.0, 3, 160.0, 25.0, 3.2, 52.0, "RELEASE_WITH_WARNING"),
                ("S006_blatant_cheater", 16, 6, 3, 20.0, 7, 340.0, 38.0, 4.6, 98.5, "HOLD_FOR_REVIEW"),
                ("S007_second_device", 4, 12, 1, 82.0, 0, 290.0, 19.0, 2.9, 88.0, "HOLD_FOR_REVIEW"),
                ("S008_colluder_A", 1, 0, 0, 42.0, 1, 30.0, 11.0, 4.7, 8.1, "RELEASE"),
                ("S009_colluder_B", 2, 0, 0, 41.0, 1, 35.0, 10.5, 4.7, 9.0, "RELEASE"),
                ("S010_slow_thinker", 0, 0, 0, 115.0, 0, 210.0, 15.0, 3.5, 12.0, "RELEASE"),
                ("S011_copy_paster", 5, 1, 0, 35.0, 5, 60.0, 28.0, 4.4, 78.4, "HOLD_FOR_REVIEW"),
                ("S012_impostor", 0, 0, 4, 48.0, 0, 50.0, 12.0, 3.0, 94.2, "HOLD_FOR_REVIEW")
            ]
            
            for sid, tab_sw, f_not, f_mul, avg_t, cp, idle_t, spd_var, score, risk, dec in archetypes:
                sess_uuid = new_session(demo_eid, sid)
                update_session_status(sess_uuid, "submitted")
                res_data = {
                    "session_id": sess_uuid,
                    "student_id": sid,
                    "exam_id": demo_eid,
                    "timestamp": datetime.utcnow().isoformat(),
                    "tab_switches": tab_sw,
                    "face_not_detected_count": f_not,
                    "multiple_faces_count": f_mul,
                    "avg_time_per_question": avg_t,
                    "copy_paste_events": cp,
                    "mouse_idle_time": idle_t,
                    "answer_speed_variance": spd_var,
                    "predicted_score": score,
                    "risk_percent": risk,
                    "decision": dec,
                    "final_grade": score if dec != "HOLD_FOR_REVIEW" else None,
                    "reason": "Demo simulated evaluation record."
                }
                save_result(res_data)
            
            # Collusion flag for A and B
            save_collusion_flag(demo_eid, "q1", archetypes[7][0], archetypes[8][0], 0.96)
            st.success("Successfully generated 12-candidate demonstration cohort!")
            st.rerun()

    if all_res:
        df_all = pd.DataFrame(all_res)
        an1, an2 = st.columns(2)
        with an1:
            st.write("#### 🛡️ Cheating Risk Distribution")
            chart_hist = alt.Chart(df_all).mark_bar(color="#8ab4f8").encode(
                alt.X("risk_percent:Q", bin=alt.Bin(maxbins=10), title="Risk %"),
                y="count()"
            ).properties(height=260)
            st.altair_chart(chart_hist, use_container_width=True)

        with an2:
            st.write("#### ⚖️ Decision Breakdown")
            chart_donut = alt.Chart(df_all).mark_arc(innerRadius=50).encode(
                theta=alt.Theta(field="decision", type="nominal", aggregate="count"),
                color=alt.Color(field="decision", type="nominal", scale=alt.Scale(
                    domain=["RELEASE", "RELEASE_WITH_WARNING", "HOLD_FOR_REVIEW"],
                    range=["#81c995", "#fdd663", "#f28b82"]
                ))
            ).properties(height=260)
            st.altair_chart(chart_donut, use_container_width=True)

        # Question-level analytics
        q_scores_map = {}
        for r in all_res:
            pqs = r.get("per_question_scores")
            if pqs and isinstance(pqs, list):
                for i, score in enumerate(pqs):
                    q_scores_map.setdefault(f"Q{i+1}", []).append(float(score))

        st.write("---")
        st.write("#### 📝 Question Difficulty & Performance Analysis")
        if q_scores_map:
            avg_q_data = [{"Question": k, "Avg Score": np.mean(v), "Attempts": len(v)} for k, v in q_scores_map.items()]
            df_q = pd.DataFrame(avg_q_data).sort_values("Avg Score")
            hardest_q = df_q.iloc[0]
            
            q_col1, q_col2 = st.columns([1, 2])
            with q_col1:
                st.metric("🔥 Hardest Question", hardest_q["Question"], f"Avg: {hardest_q['Avg Score']:.2f} / 5.0", delta_color="inverse")
                st.caption("Question with lowest cohort average score across submissions.")
            with q_col2:
                q_chart = alt.Chart(df_q).mark_bar(color="#8ab4f8").encode(
                    x=alt.X("Question:N", title="Question"),
                    y=alt.Y("Avg Score:Q", scale=alt.Scale(domain=[0, 5]), title="Average Score (0-5)"),
                    tooltip=["Question", "Avg Score", "Attempts"]
                ).properties(height=200)
                st.altair_chart(q_chart, use_container_width=True)
        else:
            st.info("Cohort question breakdown will populate as multi-question exams are submitted.")

# =============================================================================
# 6. EXAM BUILDER
# =============================================================================
elif nav_choice == "✏️ Exam Builder":
    st.subheader("✏️ Exam Authoring & Management")
    st.caption("Create new proctored exam codes with specific timers and reference answers.")
    
    exam_mode = st.radio("Creation Mode", ["✍️ Manual Entry", "📂 Upload Question CSV"], horizontal=True)
    
    if exam_mode == "📂 Upload Question CSV":
        st.write("Upload a CSV file containing columns `question` and `reference_answer`:")
        csv_file = st.file_uploader("Upload CSV", type=["csv"], key="exam_csv_upload")
        b_title = st.text_input("Exam Title", "Midterm Assessment", key="csv_title")
        b_dur = st.number_input("Duration (minutes)", min_value=1, max_value=180, value=20, key="csv_dur")
        
        if st.button("Publish Exam from CSV", type="primary", disabled=(csv_file is None)):
            try:
                uploaded_df = pd.read_csv(csv_file)
                if "question" not in uploaded_df.columns or "reference_answer" not in uploaded_df.columns:
                    st.error("CSV must contain 'question' and 'reference_answer' columns.")
                else:
                    csv_qs = uploaded_df[["question", "reference_answer"]].to_dict(orient="records")
                    new_eid = new_exam(b_title, int(b_dur * 60), csv_qs)
                    st.success(f"🎉 Exam created successfully from CSV! Exam Code: `{new_eid}` ({len(csv_qs)} questions)")
            except Exception as e:
                st.error(f"Error parsing CSV: {e}")
    else:
        with st.form("exam_builder_form"):
            title = st.text_input("Exam Title", "Midterm Assessment")
            duration_mins = st.number_input("Duration (minutes)", min_value=1, max_value=180, value=20)
            
            st.write("---")
            st.write("#### Questions & Reference Answer Rows")
            n_q = st.number_input("Number of Questions", min_value=1, max_value=10, value=2)
            
            q_rows = []
            for i in range(int(n_q)):
                st.markdown(f"**Question {i+1}**")
                q_text = st.text_input(f"Question Prompt #{i+1}", key=f"qb_q_{i}")
                ref_text = st.text_area(f"Reference Key Answer #{i+1}", key=f"qb_ref_{i}")
                q_rows.append({"question": q_text, "reference_answer": ref_text})
            
            submitted_exam = st.form_submit_button("Publish & Generate Exam Code", type="primary")
            if submitted_exam:
                valid_qs = [q for q in q_rows if q["question"].strip() and q["reference_answer"].strip()]
                if not valid_qs:
                    st.error("Please provide at least one valid question and reference answer.")
                else:
                    new_eid = new_exam(title, int(duration_mins * 60), valid_qs)
                    st.success(f"🎉 Exam created successfully! Share this Exam Code with students: `{new_eid}`")

# =============================================================================
# 7. SIMULATION SANDBOX
# =============================================================================
elif nav_choice == "🧪 Simulation Sandbox":
    st.subheader("🧪 Single Answer & Behavioral Simulation Sandbox")
    st.caption("Test the ML grading and risk models interactively.")
    
    bank_df = pd.read_csv("data/answers.csv").drop_duplicates("question").reset_index(drop=True)
    
    sim_left, sim_right = st.columns(2)
    with sim_left:
        st.write("#### 1. Question & Student Answer")
        q_sel = st.selectbox("Select Question from Benchmark", bank_df["question"])
        ref_ans = bank_df.loc[bank_df["question"] == q_sel, "reference_answer"].iloc[0]
        st.caption(f"**Reference Answer**: {ref_ans}")
        sim_ans = st.text_area("Candidate Answer", "The model overfits the training distribution and fails on unseen test data.", height=120)

    with sim_right:
        st.write("#### 2. Simulated Session Telemetry")
        sc1, sc2 = st.columns(2)
        sim_tabs = sc1.number_input("Tab Switches", 0, 50, 1)
        sim_pastes = sc2.number_input("Copy/Paste Events", 0, 50, 0)
        sim_f_not = sc1.number_input("Face Absent Count", 0, 50, 0)
        sim_f_mul = sc2.number_input("Multiple Faces Count", 0, 50, 0)
        sim_avg_t = sc1.number_input("Avg Time / Q (s)", 1.0, 300.0, 60.0)
        sim_idle = sc2.number_input("Mouse Idle Time (s)", 0.0, 1000.0, 80.0)
        sim_spd_var = st.number_input("Speed Variance (std)", 0.0, 100.0, 12.0)
    
    if st.button("▶ Run Full Agent Pipeline", type="primary"):
        sim_sess = {
            "tab_switches": sim_tabs,
            "face_not_detected_count": sim_f_not,
            "multiple_faces_count": sim_f_mul,
            "avg_time_per_question": sim_avg_t,
            "copy_paste_events": sim_pastes,
            "mouse_idle_time": sim_idle,
            "answer_speed_variance": sim_spd_var
        }
        res = agent.run("SIM_USER", ref_ans, sim_ans, sim_sess)
        
        st.write("---")
        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted Score", f"{res['predicted_score']} / 5.0")
        m2.metric("Cheating Risk", f"{res['risk_percent']}%")
        m3.markdown(f"**Decision**:<br>{decision_badge(res['decision'])}", unsafe_allow_html=True)
        st.info(f"**Agent Decision Rationale**: {res['reason']}")

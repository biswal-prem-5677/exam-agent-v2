"""
agent.py (v2): the autonomous exam proctoring & grading agent.

Tools:   grade_answer()  -> predicted score 0-5        (Model B, embeddings + lexical features)
         assess_risk()   -> cheating probability 0-1   (Model A, calibrated + monotonic)
Policy:  decide()        -> RELEASE / RELEASE_WITH_WARNING / HOLD_FOR_REVIEW
Safety:  input values are clipped to the trained range, and plain-rule guardrails
         prevent an extreme count from ever being auto-released without a warning.
Drop-in replacement for the v1 agent: same class name, same methods, same result dict.
"""
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from features import grading_features, load_embedder, GRADER_FEATURES  # same code as used in training

BASE = Path(__file__).parent
MODELS = BASE / "models"

DIRECTION = {  # +1: higher is suspicious, -1: lower is suspicious
    "tab_switches": 1, "face_not_detected_count": 1, "multiple_faces_count": 1,
    "avg_time_per_question": -1, "copy_paste_events": 1, "mouse_idle_time": 1,
    "answer_speed_variance": 1,
}
LABELS = {
    "tab_switches": "many tab switches",
    "face_not_detected_count": "face often not detected",
    "multiple_faces_count": "multiple faces detected",
    "avg_time_per_question": "unusually fast answering",
    "copy_paste_events": "copy-paste activity",
    "mouse_idle_time": "long mouse idle time",
    "answer_speed_variance": "erratic answer speed",
}


class ExamAgent:
    def __init__(self, models_dir=MODELS):
        models_dir = Path(models_dir)
        self.meta = json.load(open(models_dir / "meta.json"))
        self._check_versions()
        self.cheat_model = joblib.load(models_dir / "cheat_model.pkl")
        self.grader_model = joblib.load(models_dir / "grader_model.pkl")
        self.tfidf = joblib.load(models_dir / "tfidf.pkl")
        self.embedder = load_embedder(self.meta, models_dir)
        self.cheat_features = self.meta["cheat_features"]
        self.low = self.meta["thresholds"]["low"]
        self.high = self.meta["thresholds"]["high"]
        self.bounds = self.meta.get("clip_bounds", {})
        self.guardrails = self.meta.get("guardrails", {})

    def _check_versions(self):
        import sklearn
        trained = self.meta.get("versions", {}).get("scikit-learn")
        if trained and trained != sklearn.__version__:
            warnings.warn(f"Models were trained with scikit-learn {trained} but {sklearn.__version__} is installed. "
                          f"If loading fails or results look odd run: pip install scikit-learn=={trained}")

    # ---------------- Tool 1: grading ----------------
    def grade_answer(self, reference_answer, student_answer):
        f = grading_features(reference_answer, student_answer, self.tfidf, self.embedder)
        pred = self.grader_model.predict(pd.DataFrame([f], columns=GRADER_FEATURES))[0]
        return float(np.clip(pred, 0, 5))

    # ---------------- Tool 2: proctoring ----------------
    def _prepare(self, session):
        """Clip values to the trained range and detect guardrail hits. Returns (clean, notes, guard_hits)."""
        clean, notes, hits = {}, [], []
        for feat in self.cheat_features:
            v = float(session[feat])
            lo, hi = self.bounds.get(feat, [-np.inf, np.inf])
            if v < lo or v > hi:
                notes.append(f"{feat}={v:g} is outside the trained range [{lo:g}, {hi:g}] and was clipped")
                v = min(max(v, lo), hi)
            clean[feat] = v
            if feat in self.guardrails and float(session[feat]) > self.guardrails[feat]:
                hits.append(f"{LABELS[feat]} ({feat}={session[feat]} exceeds the honest maximum {self.guardrails[feat]:g})")
        return clean, notes, hits

    def assess_risk_detailed(self, session):
        clean, notes, hits = self._prepare(session)
        risk = float(self.cheat_model.predict_proba(pd.DataFrame([clean])[self.cheat_features])[0, 1])
        reasons = []
        for feat in self.cheat_features:
            mean, std = self.meta["normal_mean"][feat], max(self.meta["normal_std"][feat], 1e-6)
            z = DIRECTION[feat] * (float(session[feat]) - mean) / std
            if z > 2:
                reasons.append((z, f"{LABELS[feat]} ({feat}={session[feat]})"))
        reasons = [r for _, r in sorted(reasons, reverse=True)]
        return {"risk": risk, "reasons": reasons, "notes": notes, "guard_hits": hits}

    def assess_risk(self, session):
        d = self.assess_risk_detailed(session)
        return d["risk"], d["reasons"]

    # ---------------- Decision policy ----------------
    def decide(self, risk, guard_hits=None):
        if risk < self.low:
            decision = "RELEASE"
        elif risk <= self.high:
            decision = "RELEASE_WITH_WARNING"
        else:
            decision = "HOLD_FOR_REVIEW"
        if guard_hits and decision == "RELEASE":
            decision = "RELEASE_WITH_WARNING"          # guardrail: extreme counts are never silently released
        return decision

    def _finish(self, student_id, score, a):
        risk, reasons = a["risk"], a["reasons"]
        decision = self.decide(risk, a["guard_hits"])
        signals = reasons + [h for h in a["guard_hits"] if not any(h.split(" (")[0] in r for r in reasons)]
        sig = ("Signals: " + "; ".join(signals) + ". ") if signals else ""
        note = ("Note: " + "; ".join(a["notes"]) + ".") if a["notes"] else ""
        if decision == "RELEASE":
            text, final = "Behaviour looks normal; grade released automatically. ", round(score, 2)
        elif decision == "RELEASE_WITH_WARNING":
            text, final = "Some unusual behaviour detected; grade released with a warning. " + sig, round(score, 2)
        else:
            text, final = "High cheating risk; grade held for human review. " + sig, None
        return {"student_id": student_id, "predicted_score": round(score, 2), "risk_percent": round(risk * 100, 1),
                "decision": decision, "final_grade": final, "reason": (text + note).strip()}

    # ---------------- Full autonomous runs ----------------
    def run(self, student_id, reference_answer, student_answer, session):
        score = self.grade_answer(reference_answer, student_answer)
        return self._finish(student_id, score, self.assess_risk_detailed(session))

    def run_exam(self, student_id, qa_pairs, session):
        """qa_pairs: list of (reference_answer, student_answer). Grade each, average, one risk check for the session."""
        scores = [self.grade_answer(r, s) for r, s in qa_pairs]
        res = self._finish(student_id, float(np.mean(scores)) if scores else 0.0, self.assess_risk_detailed(session))
        res["per_question_scores"] = [round(s, 2) for s in scores]
        return res

    def run_batch(self, df):
        out = [self.run(r["student_id"], r["reference_answer"], r["student_answer"],
                        {c: r[c] for c in self.cheat_features}) for _, r in df.iterrows()]
        return pd.DataFrame(out)


if __name__ == "__main__":
    agent = ExamAgent()
    ref = "The model learns the training data too closely. It also learns the noise in the data. As a result it performs very well on training data. But it performs poorly on new unseen data."
    calm = dict(tab_switches=2, face_not_detected_count=1, multiple_faces_count=0, avg_time_per_question=62,
                copy_paste_events=0, mouse_idle_time=110, answer_speed_variance=12)
    shady = dict(tab_switches=14, face_not_detected_count=7, multiple_faces_count=2, avg_time_per_question=28,
                 copy_paste_events=6, mouse_idle_time=300, answer_speed_variance=30)
    ans = "When a model gets too attached to its training examples and the noise in them, it scores great on that data but is bad on new data."
    for sid, sess in [("S001", calm), ("S002", shady)]:
        print(agent.run(sid, ref, ans, sess))

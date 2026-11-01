"""
explain.py – Explainable grading (covered/missing ideas) and explainable risk (per-feature contributions).
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

BASE = Path(__file__).parent.parent
_meta = None


def _get_meta() -> Dict:
    global _meta
    if _meta is None:
        _meta = json.load(open(BASE / "models" / "meta.json"))
    return _meta


# ── Grading explainability ──────────────────────────────────────────────────

def explain_grading(reference: str, student: str, embedder, threshold: float = 0.55) -> Dict:
    """
    Return which reference sentence ideas are covered and which are missing.
    Uses per-reference-sentence best-match similarity from the embedder.
    """
    from features import split_sentences
    ref_sentences = split_sentences(reference)
    if not student.strip():
        return {
            "covered": [],
            "missing": ref_sentences,
            "feedback": "No answer provided.",
            "coverage_pct": 0,
        }

    stu_sentences = split_sentences(student)
    all_texts = [reference, student] + ref_sentences + stu_sentences
    E = embedder.encode(all_texts)
    R = E[2: 2 + len(ref_sentences)]
    S = E[2 + len(ref_sentences):]

    sims = np.clip(R @ S.T, 0, 1)  # (n_ref, n_stu)
    best_per_ref = sims.max(axis=1) if S.shape[0] > 0 else np.zeros(len(ref_sentences))

    covered = [s for s, score in zip(ref_sentences, best_per_ref) if score >= threshold]
    missing = [s for s, score in zip(ref_sentences, best_per_ref) if score < threshold]

    n_covered = len(covered)
    n_total = len(ref_sentences)
    coverage_pct = round(100 * n_covered / max(n_total, 1), 1)

    if n_covered == n_total:
        feedback = "All key ideas covered."
    elif n_covered == 0:
        feedback = "No key ideas from the reference were clearly addressed."
    else:
        feedback = f"{n_covered}/{n_total} key ideas covered. Missing: {'; '.join(missing[:2])}{'...' if len(missing)>2 else ''}."

    return {
        "covered": covered,
        "missing": missing,
        "feedback": feedback,
        "coverage_pct": coverage_pct,
    }


def explain_grading_multi(qa_pairs: List[Tuple[str, str]], embedder) -> List[Dict]:
    """Run explain_grading for each (reference, student) pair."""
    return [explain_grading(ref, stu, embedder) for ref, stu in qa_pairs]


# ── Risk explainability ─────────────────────────────────────────────────────

def explain_risk(session: Dict[str, Any], cheat_model, cheat_features: List[str]) -> Dict:
    """
    For each of the 7 features, compute:
        contribution = risk(actual) - risk(session with that feature replaced by its honest median)
    Returns a dict with feature contributions and a bar chart data list sorted by |contribution|.
    """
    import pandas as pd
    meta = _get_meta()
    normal_mean = meta["normal_mean"]

    def _risk(s):
        df = pd.DataFrame([s])[cheat_features]
        return float(cheat_model.predict_proba(df)[0, 1])

    base_risk = _risk(session)
    contributions = {}
    for feat in cheat_features:
        counterfactual = dict(session)
        counterfactual[feat] = normal_mean[feat]
        contrib = base_risk - _risk(counterfactual)
        contributions[feat] = round(contrib, 4)

    # Sort by absolute contribution
    sorted_contribs = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)

    return {
        "base_risk": round(base_risk * 100, 1),
        "contributions": contributions,
        "sorted_contributions": [{"feature": f, "contribution": c * 100} for f, c in sorted_contribs],
    }


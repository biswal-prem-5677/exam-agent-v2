"""
collusion.py – Detect suspiciously similar answers between students in the same exam.
Uses the agent's embedder (cosine similarity).
"""
from typing import List, Dict

import numpy as np

COLLUSION_THRESHOLD = 0.92  # configurable


def detect_collusion(
    submissions: List[Dict],
    embedder,
    threshold: float = COLLUSION_THRESHOLD,
) -> List[Dict]:
    """
    submissions: list of dicts with keys: session_id, qid, student_answer, reference_answer
    Returns: list of flagged pairs with keys session_a, session_b, qid, similarity.
    """
    # Group by qid
    from collections import defaultdict
    by_qid: Dict[str, List[Dict]] = defaultdict(list)
    for sub in submissions:
        by_qid[sub["qid"]].append(sub)

    flags = []
    for qid, subs in by_qid.items():
        if len(subs) < 2:
            continue
        texts = [s["student_answer"] or "" for s in subs]
        refs = [s.get("reference_answer", "") for s in subs]

        # Embed all answers
        E = embedder.encode(texts)
        norms = np.linalg.norm(E, axis=1, keepdims=True)
        E_norm = E / np.where(norms == 0, 1, norms)

        sim_matrix = E_norm @ E_norm.T

        for i in range(len(subs)):
            for j in range(i + 1, len(subs)):
                sim = float(sim_matrix[i, j])
                if sim >= threshold:
                    # Also check it's not identical to the reference
                    ref_text = refs[i] if refs[i] else ""
                    if ref_text:
                        ref_e = embedder.encode([ref_text])
                        ref_e_norm = ref_e / np.linalg.norm(ref_e, keepdims=True)
                        sim_to_ref_a = float((ref_e_norm @ E_norm[i]).item())
                        sim_to_ref_b = float((ref_e_norm @ E_norm[j]).item())
                        # If both are very close to the reference, it might just be a correct answer
                        if sim_to_ref_a > 0.97 and sim_to_ref_b > 0.97:
                            continue
                    flags.append({
                        "session_a": subs[i]["session_id"],
                        "session_b": subs[j]["session_id"],
                        "qid": qid,
                        "similarity": round(sim, 4),
                    })

    return flags


# ── Unit test ────────────────────────────────────────────────────────────────

def _selftest():
    """Quick smoke test without pytest."""
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from features import LSAEmbedder, CachedEmbedder
    embedder = CachedEmbedder(LSAEmbedder(64))
    # Fit on a small corpus
    corpus = [
        "Machine learning is a subset of AI",
        "Deep learning uses neural networks",
        "Machine learning is a subset of artificial intelligence",
        "The sky is blue",
    ]
    embedder.base.fit(corpus)
    submissions = [
        {"session_id": "A", "qid": "q1", "student_answer": corpus[0], "reference_answer": ""},
        {"session_id": "B", "qid": "q1", "student_answer": corpus[2], "reference_answer": ""},
        {"session_id": "C", "qid": "q1", "student_answer": corpus[3], "reference_answer": ""},
    ]
    flags = detect_collusion(submissions, embedder, threshold=0.85)
    assert any(f["session_a"] in ("A","B") and f["session_b"] in ("A","B") for f in flags), \
        f"Expected collusion flag for A&B, got: {flags}"
    assert not any(f["session_a"] == "C" or f["session_b"] == "C" for f in flags), \
        f"C should not be flagged, got: {flags}"
    print("collusion selftest PASSED")


if __name__ == "__main__":
    _selftest()


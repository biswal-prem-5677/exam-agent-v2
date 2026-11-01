"""
test_collusion.py – Unit test for cross-student collusion detector.
"""
import os
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from server.collusion import detect_collusion
from features import CachedEmbedder, LSAEmbedder

def test_collusion_detection():
    # Use near-identical answers
    ans_a = "A neural network consists of layers of interconnected artificial neurons with activation functions."
    ans_b = "A neural network consists of layers of interconnected artificial neurons with activation functions."
    ans_c = "Decision trees split feature nodes based on information gain or Gini impurity."
    
    corpus = [ans_a, ans_b, ans_c]
    base_emb = LSAEmbedder(n_components=2)
    base_emb.fit(corpus)
    embedder = CachedEmbedder(base_emb)
    
    submissions = [
        {"session_id": "STUDENT_A", "qid": "q1", "student_answer": ans_a, "reference_answer": ""},
        {"session_id": "STUDENT_B", "qid": "q1", "student_answer": ans_b, "reference_answer": ""},
        {"session_id": "STUDENT_C", "qid": "q1", "student_answer": ans_c, "reference_answer": ""},
    ]
    
    flags = detect_collusion(submissions, embedder, threshold=0.92)
    assert len(flags) >= 1
    
    flagged_sessions = {flags[0]["session_a"], flags[0]["session_b"]}
    assert flagged_sessions == {"STUDENT_A", "STUDENT_B"}
    assert "STUDENT_C" not in flagged_sessions

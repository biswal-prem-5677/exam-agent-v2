"""
test_features.py – Test feature extraction pipelines in features.py.
"""
import os
import pytest
import numpy as np

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from features import (
    tokens, content_words, split_sentences, grading_features,
    GRADER_FEATURES, LSAEmbedder, CachedEmbedder
)

def test_tokenization_and_content_words():
    text = "The model learns the training data too closely! Overfitting occurred?"
    toks = tokens(text)
    assert "model" in toks
    assert "overfitting" in toks
    cw = content_words(text)
    assert "the" not in cw
    assert "model" in cw

def test_split_sentences():
    text = "First idea here. Second idea here! Is this third? Yes."
    parts = split_sentences(text)
    assert len(parts) == 4
    assert parts[0] == "First idea here."
    
    empty_parts = split_sentences("")
    assert len(empty_parts) == 1

def test_lsa_embedder_fallback():
    corpus = [
        "Machine learning models need training data to learn patterns effectively.",
        "Deep learning uses artificial neural networks with multiple hidden layers.",
        "Overfitting happens when variance is too high and model memorizes noise.",
        "Supervised learning trains on pairs of input features and target labels.",
        "Unsupervised algorithms find hidden structures clusters without annotations.",
        "Regularization methods prevent extreme weights and improve model generalization.",
    ]
    embedder = LSAEmbedder(n_components=4)
    embedder.fit(corpus)
    cached = CachedEmbedder(embedder)
    
    enc = cached.encode(["Machine learning models", "Deep learning"])
    assert enc.shape == (2, 4)
    norm0 = np.linalg.norm(enc[0])
    assert pytest.approx(norm0, 1e-4) == 1.0

def test_grading_features_empty_student_answer():
    ref = "Supervised learning requires labeled input data."
    empty_ans = ""
    feats = grading_features(ref, empty_ans, None, None)
    assert feats == [0.0] * len(GRADER_FEATURES)
    assert len(feats) == 9

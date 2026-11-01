import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import re
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

LEXICAL = ["tfidf_cosine", "keyword_recall", "keyword_precision", "jaccard", "bigram_overlap", "length_ratio"]
EMBED = ["emb_cosine", "emb_cov_mean", "emb_cov_min"]
GRADER_FEATURES = LEXICAL + EMBED


def tokens(text):
    return re.findall(r"[a-z0-9]+", str(text).lower())


def content_words(text):
    return [w for w in tokens(text) if w not in ENGLISH_STOP_WORDS]


def split_sentences(text):
    parts = [p.strip() for p in re.split(r"(?<=[.!?;])\s+", str(text).strip()) if p.strip()]
    return parts if parts else [str(text).strip() or "empty"]


class LSAEmbedder:
    # Offline fallback embedder (TF-IDF + SVD). Much weaker than sentence-transformers.
    def __init__(self, n_components=64):
        self.n_components = n_components
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)

    def fit(self, texts):
        X = self.vec.fit_transform(list(texts))
        self.svd.set_params(n_components=max(2, min(self.n_components, X.shape[1] - 1)))
        self.svd.fit(X)
        return self

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False, batch_size=64, **kw):
        Z = self.svd.transform(self.vec.transform(list(texts)))
        n = np.linalg.norm(Z, axis=1, keepdims=True)
        return Z / np.where(n == 0, 1, n)


class CachedEmbedder:
    # Wraps any embedder and remembers results, so each text is embedded only once.
    def __init__(self, base):
        self.base = base
        self.cache = {}

    def encode(self, texts, **kw):
        texts = [str(t) for t in texts]
        new = [t for t in dict.fromkeys(texts) if t not in self.cache]
        if new:
            E = self.base.encode(new, normalize_embeddings=True, show_progress_bar=False, batch_size=64)
            for t, e in zip(new, E):
                self.cache[t] = np.asarray(e, dtype=float)
        return np.vstack([self.cache[t] for t in texts])


def grading_features(reference, student, tfidf, embedder):
    # Compare a student answer with the reference answer; returns 9 numbers (order = GRADER_FEATURES).
    if not tokens(student):
        return [0.0] * len(GRADER_FEATURES)
    r, s = content_words(reference), content_words(student)
    rs, ss = set(r), set(s)
    inter = rs & ss
    recall = len(inter) / max(len(rs), 1)
    precision = len(inter) / max(len(ss), 1)
    jaccard = len(inter) / max(len(rs | ss), 1)
    rb, sb = set(zip(r, r[1:])), set(zip(s, s[1:]))
    bigram = len(rb & sb) / max(len(rb), 1)
    vec = tfidf.transform([str(reference), str(student)])
    tf_cos = float(vec[0].multiply(vec[1]).sum())
    length_ratio = min(len(tokens(student)) / max(len(tokens(reference)), 1), 2.0)

    ref_s, stu_s = split_sentences(reference), split_sentences(student)
    E = embedder.encode([str(reference), str(student)] + ref_s + stu_s)
    emb_cos = float(np.clip(E[0] @ E[1], 0, 1))
    R, S = E[2:2 + len(ref_s)], E[2 + len(ref_s):]
    best = np.clip((R @ S.T).max(axis=1), 0, 1)
    return [tf_cos, recall, precision, jaccard, bigram, length_ratio, emb_cos, float(best.mean()), float(best.min())]


def load_embedder(meta, models_dir):
    """Rebuild the embedder that was used in training.
    Sets TRANSFORMERS_OFFLINE=1 so the cached model is used without network calls.
    """
    import os
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    e = meta["embedder"]
    if e["type"] == "sentence-transformers":
        from sentence_transformers import SentenceTransformer
        return CachedEmbedder(SentenceTransformer(e["name"], local_files_only=True))
    import joblib
    return CachedEmbedder(joblib.load(os.path.join(str(models_dir), e["file"])))

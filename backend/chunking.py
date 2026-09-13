"""
Chunking strategies, as pure functions.

Nothing here calls an API. Sentence vectors are passed in, which means every
strategy can be unit-tested offline and the evaluation harness can sweep
parameters without paying to re-embed the same text over and over.
"""
import re
import numpy as np

# --- Text preparation -------------------------------------------------------

# Table-of-contents leaders ("......"), page-number runs, and rule characters.
_NOISE = re.compile(r"^[\s.,_\-–—•·|=*]+$")


def is_noise(sentence: str, min_alpha_ratio: float = 0.5) -> bool:
    """True for strings carrying no prose signal.

    PDF front matter is dominated by table-of-contents dot leaders. They pass a
    naive length filter but embed to near-meaningless vectors, so they corrupt
    both the similarity math and the retrieval index.
    """
    s = sentence.strip()
    if not s or _NOISE.match(s):
        return True
    alpha = sum(c.isalpha() for c in s)
    return alpha < len(s) * min_alpha_ratio


def split_sentences(text: str, min_len: int = 10, drop_noise: bool = True) -> list[str]:
    """Flatten PDF line breaks, split on sentence boundaries, drop noise."""
    clean = re.sub(r"\s+", " ", text.replace("\n", " "))
    parts = re.split(r"(?<=[.!?])\s+", clean)
    out = []
    for p in parts:
        p = p.strip()
        if len(p) < min_len:
            continue
        if drop_noise and is_noise(p):
            continue
        out.append(p if p[-1] in ".!?" else p + ".")
    return out


# --- Strategy A: fixed-size (the baseline) ----------------------------------

def chunk_fixed(sentences: list[str], max_chars: int = 1000, overlap: int = 200) -> list[str]:
    """Pack sentences into fixed-width windows with overlap.

    This is the standard approach (what RecursiveCharacterTextSplitter does).
    It is the baseline the semantic strategy has to beat.
    """
    chunks, cur = [], ""
    for s in sentences:
        if cur and len(cur) + len(s) + 1 > max_chars:
            chunks.append(cur)
            tail = cur[-overlap:] if overlap else ""
            cur = (tail + " " + s).strip() if tail else s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        chunks.append(cur)
    return chunks


# --- Strategy B: semantic (the contribution) --------------------------------

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return 0.0 if denom == 0 else float(np.dot(a, b) / denom)


def chunk_semantic(
    sentences: list[str],
    vectors: np.ndarray,
    threshold: float = 0.75,
    max_chars: int | None = 2000,
) -> list[str]:
    """Cut a new chunk where consecutive-sentence similarity drops below `threshold`.

    `max_chars` caps runaway chunks: in uniform passages similarity can stay
    above threshold indefinitely, producing one enormous chunk that retrieves
    poorly regardless of how good the boundary detection is.
    """
    if not sentences:
        return []

    vectors = np.asarray(vectors)
    chunks, cur = [], sentences[0]

    for i in range(1, len(sentences)):
        sim = cosine(vectors[i - 1], vectors[i])
        too_long = max_chars is not None and len(cur) + len(sentences[i]) > max_chars
        if sim > threshold and not too_long:
            cur += " " + sentences[i]
        else:
            chunks.append(cur)
            cur = sentences[i]

    if cur:
        chunks.append(cur)
    return chunks


def boundary_similarities(vectors: np.ndarray) -> np.ndarray:
    """Similarity for every adjacent pair. Used to inspect threshold choice."""
    v = np.asarray(vectors)
    if len(v) < 2:
        return np.array([])
    return np.array([cosine(v[i - 1], v[i]) for i in range(1, len(v))])

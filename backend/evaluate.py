"""
Retrieval evaluation: does semantic chunking actually beat fixed-size chunking?

Builds one FAISS index per chunking configuration, runs the same ground-truth
questions against each, and reports Hit@k and MRR so the strategies can be
compared on numbers rather than intuition.

Cost control:
  * Sentence vectors are embedded once and cached to disk. A threshold sweep
    re-chunks from the cache and costs nothing extra.
  * Question vectors are embedded once and reused across every configuration.
  Only chunk vectors must be recomputed per configuration.

Usage:
  python evaluate.py --offline     # no API calls, verifies the harness
  python evaluate.py               # shows a cost estimate, then asks
  python evaluate.py --yes         # skip the prompt
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np
from dotenv import load_dotenv

import chunking as ck

load_dotenv()
CACHE_DIR = "eval_cache"
BATCH = 48          # Cohere accepts 96 texts/call, but trial keys cap TOKENS/min
PAUSE = 8.0         # seconds between calls
MAX_RETRIES = 6     # exponential backoff on HTTP 429


def log(msg):
    print(msg, flush=True)


# --- Embeddings -------------------------------------------------------------

def get_embedder():
    from langchain_cohere import CohereEmbeddings
    key = os.getenv("COHERE_API_KEY")
    if not key:
        raise SystemExit("COHERE_API_KEY missing from .env")
    return CohereEmbeddings(model="embed-english-v3.0", cohere_api_key=key)


def embed_with_backoff(model, batch):
    """Trial keys enforce a tokens-per-minute ceiling. Back off and retry."""
    delay = PAUSE
    for attempt in range(MAX_RETRIES):
        try:
            return model.embed_documents(batch)
        except Exception as e:
            if "429" not in str(e) and "rate limit" not in str(e).lower():
                raise
            if attempt == MAX_RETRIES - 1:
                raise
            log(f"    rate limited; sleeping {delay:.0f}s "
                f"(retry {attempt + 1}/{MAX_RETRIES - 1})")
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


def embed_batched(model, texts, label="", partial_path=None):
    """Embed in batches, saving progress so a rate-limit abort loses nothing."""
    out = []
    if partial_path and os.path.exists(partial_path):
        out = list(np.load(partial_path))
        log(f"    resuming {label} from {len(out)}/{len(texts)}")

    for i in range(len(out), len(texts), BATCH):
        batch = texts[i:i + BATCH]
        log(f"    embedding {label} {i}-{i + len(batch)} of {len(texts)}")
        out.extend(embed_with_backoff(model, batch))
        if partial_path:
            np.save(partial_path, np.array(out))
        if i + BATCH < len(texts):
            time.sleep(PAUSE)
    return np.array(out)


def cached_vectors(texts, tag, model, offline):
    """Embed once, reuse forever. Cache key is a hash of the exact text."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    digest = hashlib.sha256("||".join(texts).encode("utf8")).hexdigest()[:16]
    path = os.path.join(CACHE_DIR, f"{tag}-{digest}.npy")

    if os.path.exists(path):
        log(f"  cache hit: {path}")
        return np.load(path)

    if offline:
        # Deterministic pseudo-vectors so the harness can be exercised for free.
        rng = np.random.RandomState(abs(hash(tag)) % (2**31))
        v = rng.randn(len(texts), 64)
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    partial = os.path.join(CACHE_DIR, f"{tag}-{digest}.partial.npy")
    v = embed_batched(model, texts, tag, partial_path=partial)
    np.save(path, v)
    if os.path.exists(partial):
        os.remove(partial)
    log(f"  cached -> {path}")
    return v


# --- Metrics ----------------------------------------------------------------

def score(retrieved_texts, must_contain):
    """Rank (1-based) of the first chunk containing every required string."""
    for rank, text in enumerate(retrieved_texts, start=1):
        low = " ".join(text.lower().split())
        if all(m.lower() in low for m in must_contain):
            return rank
    return None


def evaluate_config(name, chunks, questions, q_vectors, model, offline, k):
    if offline:
        # Score by keyword overlap instead of a vector index.
        ranks = []
        for q in questions:
            scored = sorted(
                chunks,
                key=lambda c: -sum(w in c.lower() for w in q["q"].lower().split()),
            )[:k]
            ranks.append(score(scored, q["must_contain"]))
    else:
        from langchain_community.vectorstores import FAISS
        from langchain_core.documents import Document

        # Embed chunks through our throttled path, then hand FAISS the vectors
        # directly -- FAISS.from_documents would otherwise embed unthrottled.
        vecs = cached_vectors(chunks, f"chunks-{name}", model, offline)
        db = FAISS.from_embeddings(list(zip(chunks, vecs)), model)
        ranks = []
        for q, qv in zip(questions, q_vectors):
            docs = db.similarity_search_by_vector(list(qv), k=k)
            ranks.append(score([d.page_content for d in docs], q["must_contain"]))

    hits = [r for r in ranks if r]
    lens = [len(c) for c in chunks]
    return {
        "config": name,
        "chunks": len(chunks),
        "avg_chunk_chars": round(float(np.mean(lens)), 1),
        "hit_rate": round(len(hits) / len(questions), 3),
        "mrr": round(float(np.mean([1 / r if r else 0 for r in ranks])), 3),
        "ranks": ranks,
    }


# --- Main -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="no API calls")
    ap.add_argument("--yes", action="store_true", help="skip cost prompt")
    ap.add_argument("--k", type=int, default=5, help="chunks retrieved per query")
    ap.add_argument("--spec", default="eval_questions.json")
    ap.add_argument("--thresholds", default="0.55,0.65,0.70,0.75,0.80,0.85",
                    help="comma-separated similarity cut points to sweep")
    args = ap.parse_args()

    spec = json.load(open(args.spec, encoding="utf8"))
    questions = spec["questions"]
    lo, hi = spec["slice"]

    log(f"Loading {spec['document']} ...")
    from langchain_community.document_loaders import PyPDFLoader
    text = " ".join(p.page_content for p in PyPDFLoader(spec["document"]).load())

    sentences = ck.split_sentences(text)[lo:hi]
    log(f"{len(sentences)} sentences in slice [{lo}:{hi}]\n")

    thresholds = [float(t) for t in args.thresholds.split(",")]
    if not args.offline and not args.yes:
        est = len(sentences) // BATCH + 1 + len(thresholds) * 3 + 1
        log(f"Estimated API calls: ~{est} (sentence vectors cached after first run)")
        if input("Proceed? [y/N] ").strip().lower() != "y":
            raise SystemExit("aborted")

    model = None if args.offline else get_embedder()

    log("Embedding sentences (cached)...")
    sent_vecs = cached_vectors(sentences, "sentences", model, args.offline)
    log("Embedding questions (cached)...")
    q_vecs = cached_vectors([q["q"] for q in questions], "questions", model, args.offline)

    sims = ck.boundary_similarities(sent_vecs)
    log("" + chr(10) + "Adjacent-sentence similarity: "
        f"median {np.median(sims):.3f}, p90 {np.percentile(sims, 90):.3f}, "
        f"max {sims.max():.3f}")
    for t in thresholds:
        log(f"  threshold {t:.2f} sits at percentile "
            f"{(sims < t).mean() * 100:.0f} -> merges {(sims > t).mean() * 100:.0f}% of boundaries")

    configs = []
    for size in (500, 1000, 2000):
        configs.append((f"fixed-{size}", ck.chunk_fixed(sentences, size, size // 5)))
    for t in thresholds:
        configs.append((f"semantic-{t}", ck.chunk_semantic(sentences, sent_vecs, t)))

    results = []
    for name, chunks in configs:
        log(f"\nEvaluating {name} ({len(chunks)} chunks)")
        results.append(evaluate_config(name, chunks, questions, q_vecs, model, args.offline, args.k))
        if not args.offline:
            time.sleep(PAUSE)

    log("\n" + "=" * 74)
    log(f"{'config':<16}{'chunks':>8}{'avg chars':>11}{'Hit@%d' % args.k:>9}{'MRR':>8}")
    log("-" * 74)
    for r in sorted(results, key=lambda r: -r["mrr"]):
        log(f"{r['config']:<16}{r['chunks']:>8}{r['avg_chunk_chars']:>11}"
            f"{r['hit_rate']:>9.2f}{r['mrr']:>8.3f}")
    log("=" * 74)

    best_fix = max((r for r in results if r["config"].startswith("fixed")), key=lambda r: r["mrr"])
    best_sem = max((r for r in results if r["config"].startswith("semantic")), key=lambda r: r["mrr"])
    log(f"\nbest fixed:    {best_fix['config']}  MRR {best_fix['mrr']:.3f}")
    log(f"best semantic: {best_sem['config']}  MRR {best_sem['mrr']:.3f}")
    delta = best_sem["mrr"] - best_fix["mrr"]
    verdict = "semantic wins" if delta > 0 else ("tie" if delta == 0 else "fixed wins")
    log(f"delta: {delta:+.3f}  ->  {verdict}")
    if args.offline:
        log("\n(--offline: random vectors and keyword scoring. Numbers are NOT meaningful.)")

    out = "eval_results.json"
    json.dump({"k": args.k, "n_questions": len(questions),
               "offline": args.offline, "results": results},
              open(out, "w"), indent=2)
    log(f"\nwrote {out}")


if __name__ == "__main__":
    main()

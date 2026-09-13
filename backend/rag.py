import os
import numpy as np
import chunking as ck
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv, find_dotenv

# --- DEBUGGING SCRIPT ---
env_path = find_dotenv()
print(f"--- SYSTEM CHECK: Found .env file at: {env_path} ---")
load_dotenv(env_path)

key_status = "SUCCESS (Key Found)" if os.getenv("COHERE_API_KEY") else "FAILED (Key is None)"
print(f"--- SYSTEM CHECK: API Key Status: {key_status} ---")
# ------------------------

# Cut a new chunk where adjacent-sentence similarity falls in the bottom
# BOUNDARY_PERCENTILE of this document's distribution.
#
# p10 is the value evaluate.py selected on the ArtSci calendar: the winning
# configuration there (threshold 0.32) sat exactly at that corpus's 10th
# percentile, and retrieval MRR improved monotonically with chunk size across
# the whole sweep. Validated on one corpus only -- re-run evaluate.py before
# trusting it on a new document type.
BOUNDARY_PERCENTILE = 10
MAX_CHUNK_CHARS = 1500


# --- FUNCTION 1: The Brain (Math, Chunking & Batching) ---
def create_semantic_chunks_and_store(text: str):
    print("1. Cleaning PDF formatting and splitting text...")
    
    # Explicitly pass the API key so LangChain doesn't get confused
    embeddings_model = CohereEmbeddings(
        model="embed-english-v3.0",
        cohere_api_key=os.getenv("COHERE_API_KEY")
    )
    
    # NEW: Strip out the messy PDF line breaks so the math works better
    clean_text = text.replace("\n", " ")
    
    # Get the sentences
    sentences = [s.strip() + "." for s in clean_text.split(". ") if len(s.strip()) > 10]
    
    # Process the first 500 sentences to capture actual chapter content
    target_sentences = sentences[:500] 
    
    print(f"2. Generating embeddings for {len(target_sentences)} sentences in batches...")
    
    vectors = []
    # ENTERPRISE FEATURE: Batch processing to bypass API limits (max 96 per call)
    for i in range(0, len(target_sentences), 90):
        batch = target_sentences[i:i+90]
        print(f"   -> Processing batch {i} to {i+len(batch)}...")
        batch_vectors = embeddings_model.embed_documents(batch)
        vectors.extend(batch_vectors)
    
    print("3. Calculating Cosine Similarities...")
    vectors = np.array(vectors)

    # Derive the cut point from THIS document's own similarity distribution.
    #
    # A hardcoded threshold does not transfer: adjacent-sentence similarity
    # under embed-english-v3.0 has a median of ~0.38 on narrative prose and
    # ~0.50 on structured reference text. The old fixed 0.75 sat above the 99th
    # percentile of both, so it merged almost nothing and emitted roughly one
    # sentence per chunk -- the algorithm was a no-op.
    sims = ck.boundary_similarities(vectors)
    threshold = float(np.percentile(sims, BOUNDARY_PERCENTILE))
    print(f"   distribution: median {np.median(sims):.3f}, p90 {np.percentile(sims, 90):.3f}")
    print(f"   threshold (p{BOUNDARY_PERCENTILE}): {threshold:.3f} "
          f"-> merges {(sims > threshold).mean() * 100:.0f}% of boundaries")

    chunk_texts = ck.chunk_semantic(target_sentences, vectors, threshold, max_chars=MAX_CHUNK_CHARS)
    chunks = [Document(page_content=c) for c in chunk_texts]

    avg = sum(len(c) for c in chunk_texts) / max(len(chunk_texts), 1)
    print(f"Algorithm finished: Compressed {len(target_sentences)} sentences into "
          f"{len(chunks)} semantic chunks (avg {avg:.0f} chars).")
    
    print("4. Building FAISS Vector Database...")
    vector_db = FAISS.from_documents(chunks, embeddings_model)
    vector_db.save_local("faiss_index")
    
    return len(chunks) 

# --- FUNCTION 2: The Chat Interface ---
def ask_question(query: str):
    print(f"1. Searching FAISS database for: '{query}'")
    
    embeddings_model = CohereEmbeddings(
        model="embed-english-v3.0",
        cohere_api_key=os.getenv("COHERE_API_KEY")
    )
    
    vector_db = FAISS.load_local(
        "faiss_index", 
        embeddings_model, 
        allow_dangerous_deserialization=True 
    )
    
    # NEW: Pull 15 chunks instead of 2 so it can read past the Table of Contents
    relevant_docs = vector_db.similarity_search(query, k=15)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    print("2. Generating answer with Cohere Command A...")
    llm = ChatCohere(
        model="command-a-03-2025",
        cohere_api_key=os.getenv("COHERE_API_KEY")
    )
    
    prompt = f"""
    You are an expert document assistant. Use ONLY the following Context to answer the User's Question. 
    Explain the answer clearly and in plain English. If the answer is not in the context, say "I don't have enough information to answer that."
    
    Context:
    {context}
    
    Question:
    {query}
    """
    
    response = llm.invoke(prompt)
    
    return response.content, context
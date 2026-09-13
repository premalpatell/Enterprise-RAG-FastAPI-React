# 🧠 Enterprise RAG Translator

A full-stack, enterprise-grade Retrieval-Augmented Generation (RAG) system engineered to process, vectorize, and query large, unstructured PDF documents. 

Unlike standard AI wrapper applications, this system abandons naive character-count splitting in favor of a **custom mathematical semantic chunking algorithm**. It includes a dynamic batch-processing backend to handle production-scale rate limits and a strict-typed React frontend.

## 🚀 The Engineering Problem & Solutions

### 1. The "Naive Chunking" Problem
Standard RAG pipelines rely on fixed character-count splitters (e.g., slicing every 1,000 characters). This brutally cuts sentences and concepts in half, destroying contextual meaning before the LLM ever reads it.
*   **The Solution:** This engine embeds text sentence-by-sentence and calculates the **cosine similarity** between consecutive vectors. Chunk boundaries are placed dynamically only where the adjacent-sentence similarity falls into the bottom decile of the document's own distribution.

### 2. The API Rate Limit Crisis
Processing hundreds of sentences individually to calculate semantic distance triggers strict enterprise API payload limits (e.g., `429 RESOURCE_EXHAUSTED`).
*   **The Solution:** Implemented a custom batch-processing architecture that safely intercepts large payloads, slices them into optimized batches (90 items), processes them sequentially, and reconstructs the vectors seamlessly to guarantee stable ingestion of full-length documents.

### 3. Mitigating Hallucinations via Deep Retrieval
*   **The Solution:** PDF formatting artifacts (like rogue line breaks) are programmatically stripped prior to vectorization. The retrieval window was expanded and tuned (`k=15`) to bypass "Narrow Retrieval" failures (such as the AI only reading the Table of Contents of a novel). If the answer is not strictly within the retrieved semantic chunks, the LLM is hard-prompted to refuse to answer.

## 💻 Tech Stack
*   **Backend:** Python 3.12, FastAPI, Uvicorn
*   **AI & NLP:** Cohere (`command-a-03-2025` & `embed-english-v3.0`), LangChain
*   **Vector Database:** FAISS (Facebook AI Similarity Search)
*   **Frontend:** React, TypeScript, Vite, Tailwind CSS (enforcing strict ESLint standards)

## 📊 Evaluation & Benchmarking
The repository includes custom evaluation scripts (`evaluate.py`) that benchmark the performance of the semantic cosine-similarity chunking against standard fixed-size chunking, utilizing cached `.npy` and `.json` data to validate retrieval accuracy and context coherence.

## 🏃‍♂️ How to Run Locally

### 1. Backend Setup
Navigate to the backend directory, initialize your virtual environment, and install dependencies:
```bash
cd backend
python -m venv venv

# On Windows:
.\venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt

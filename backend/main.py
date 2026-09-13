from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import PyPDFLoader
from pydantic import BaseModel
import os
import shutil

# Import our custom RAG functions
from rag import create_semantic_chunks_and_store, ask_question

# 1. Initialize the App (This is what was missing!)
app = FastAPI(
    title="Enterprise RAG Translator API",
    description="Backend for semantic chunking and document translation"
)

# 2. Configure Security (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

# 3. Health Check
@app.get("/")
def health_check():
    return {"status": "AI Engine is running smoothly."}

# 4. Upload Endpoint
@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        return {"error": "Invalid file type. Please upload a PDF."}

    file_path = os.path.join(TEMP_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    
    full_text = " ".join([page.page_content for page in pages])

    # Run the custom math algorithm and build the database
    total_chunks = create_semantic_chunks_and_store(full_text)

    return {
        "message": "File parsed and FAISS database built successfully.",
        "filename": file.filename,
        "semantic_chunks_created": total_chunks
    }

# 5. Query Endpoint Schema
class QueryRequest(BaseModel):
    question: str

# 6. Query Endpoint
@app.post("/api/documents/query")
def query_document(request: QueryRequest):
    # Pass the question to our RAG engine in rag.py
    answer, context_used = ask_question(request.question)
    
    return {
        "answer": answer,
        "context_used": context_used
    }
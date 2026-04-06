import os
import uuid
import json
import logging
import asyncio
import fitz
import faiss
import torch
import numpy as np
import re
import io
import sqlite3
import time
from functools import lru_cache
from PIL import Image
from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from fastapi.staticfiles import StaticFiles

from sentence_transformers import SentenceTransformer

from google.cloud import documentai
from google.oauth2 import service_account
from google.api_core.client_options import ClientOptions

import google.generativeai as genai

import firebase_admin
from firebase_admin import credentials, auth

# =========================
# LOAD ENVIRONMENT VARIABLES
# =========================
# Loads variables from a .env file in the project root (for local dev).
# In production, these should be set as real system environment variables.
load_dotenv()

# =========================
# CONFIG  ── all secrets and paths are now read from environment variables
# =========================

PROJECT_ID      = os.getenv("GCP_PROJECT_ID", "aksharalokam-doc-ai")
LOCATION        = os.getenv("GCP_LOCATION", "us")
PROCESSOR_ID    = os.getenv("GCP_PROCESSOR_ID", "e743d606d75cfc9")

DOC_AI_CREDENTIAL  = os.getenv("DOC_AI_CREDENTIAL_PATH",  "credentials/aksharalokam-doc-ai.json")
FIREBASE_CREDENTIAL = os.getenv("FIREBASE_CREDENTIAL_PATH", "credentials/firebase_admin.json")

GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable is not set. Aborting startup.")

DOC_FOLDER = os.getenv("DOC_FOLDER", "documents")
LOG_FOLDER = os.getenv("LOG_FOLDER", "logs")
DB_PATH    = os.getenv("DB_PATH",    "aksharalokam_database.db")

os.makedirs(DOC_FOLDER, exist_ok=True)
os.makedirs(LOG_FOLDER, exist_ok=True)

# =========================
# SECURITY ── Safe doc_id pattern (UUID v4 only, no path traversal)
# =========================
# Matches exactly the format produced by uuid.uuid4(): 8-4-4-4-12 hex groups
DOC_ID_PATTERN = re.compile(
    r'^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$',
    re.IGNORECASE
)

def validate_doc_id(doc_id: str) -> str:
    """Raise 400 if doc_id doesn't match UUID-v4; otherwise return it unchanged."""
    if not DOC_ID_PATTERN.match(doc_id):
        raise HTTPException(status_code=400, detail="Invalid document ID format.")
    return doc_id

# =========================
# LOGGING
# =========================

logging.basicConfig(
    filename=f"{LOG_FOLDER}/server.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logging.info("Server started - backend_server.py:90")

# =========================
# DATABASE INIT (SQLite)
# =========================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS chats
                 (id TEXT PRIMARY KEY, uid TEXT, title TEXT, messages TEXT, docId TEXT, uploadedFileName TEXT, uploadedFileUrl TEXT, updatedAt INTEGER)''')

    # Safely try to add isPinned if it doesn't exist yet
    try:
        c.execute("ALTER TABLE chats ADD COLUMN isPinned INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass  # Column already exists

    c.execute('''CREATE TABLE IF NOT EXISTS materials
                 (id TEXT PRIMARY KEY, classNum INTEGER, subject TEXT, url TEXT, title TEXT, uploadedBy TEXT, uploaderUid TEXT, createdAt INTEGER)''')
    conn.commit()
    conn.close()

init_db()

# =========================
# FIREBASE AUTH
# =========================

cred = credentials.Certificate(FIREBASE_CREDENTIAL)

if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

def verify_user(token):
    try:
        decoded = auth.verify_id_token(token)
        return decoded["uid"]
    except Exception:
        raise HTTPException(401, "Invalid Firebase token")

# =========================
# MODELS
# =========================

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device: - backend_server.py:135", device)
torch.set_float32_matmul_precision("high")

# Using Multilingual Embeddings (Great for Malayalam & Fast)
embedding_model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    device=device
)

# =========================
# PERFORMANCE ── LRU-cached encode wrapper
# =========================
# lru_cache works on hashable arguments only; we wrap a tuple of strings so
# repeated identical queries hit RAM instead of re-running the neural network.
@lru_cache(maxsize=512)
def _cached_encode_single(text: str) -> np.ndarray:
    """Return the embedding for a single string, cached by exact text."""
    return embedding_model.encode(text, convert_to_numpy=True)

def get_query_embedding(text: str) -> np.ndarray:
    """Public helper used by the /api/ask endpoint."""
    return _cached_encode_single(text)

# =========================
# GENAI SETUP
# =========================

genai.configure(api_key=GEMINI_API_KEY)
# Using Gemini 2.5 Flash for Lightning Fast Multimodal Responses
gemini = genai.GenerativeModel("gemini-2.5-flash")

# =========================
# DOCUMENT AI
# =========================

credentials_doc = service_account.Credentials.from_service_account_file(
    DOC_AI_CREDENTIAL
)

opts = ClientOptions(
    api_endpoint=f"{LOCATION}-documentai.googleapis.com"
)

docai_client = documentai.DocumentProcessorServiceClient(
    credentials=credentials_doc,
    client_options=opts
)

# =========================
# FASTAPI & IN-MEMORY CACHES (FOR SPEED & CONTEXT)
# =========================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Mount the documents folder to serve PDFs
app.mount("/documents", StaticFiles(directory=DOC_FOLDER), name="documents")

# RAM CACHE: Stores FAISS indices and text chunks in RAM to avoid slow hard-drive reads.
# Schema: { doc_id: (faiss_index, list[{"text": str, "page": int}]) }
DOCUMENT_CACHE: Dict[str, Any] = {}

# =========================
# REQUEST MODELS
# =========================

class HistoryItem(BaseModel):
    role: str
    text: str

class AskRequest(BaseModel):
    docId: Optional[str] = None
    question: str
    history: Optional[List[HistoryItem]] = []

class ChatSaveRequest(BaseModel):
    id: str
    title: str
    messages: List[Dict[str, Any]]
    docId: Optional[str] = None
    uploadedFileName: Optional[str] = None
    uploadedFileUrl: Optional[str] = None
    isPinned: bool = False

# =========================
# CHAT DATABASE ENDPOINTS
# =========================

@app.post("/api/chat/save")
async def save_chat(req: ChatSaveRequest, authorization: str = Header(...)):
    uid = verify_user(authorization)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    updatedAt = int(time.time() * 1000)
    is_pinned_val = 1 if req.isPinned else 0
    c.execute('''INSERT OR REPLACE INTO chats (id, uid, title, messages, docId, uploadedFileName, uploadedFileUrl, updatedAt, isPinned)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
              (req.id, uid, req.title, json.dumps(req.messages), req.docId, req.uploadedFileName, req.uploadedFileUrl, updatedAt, is_pinned_val))
    conn.commit()
    conn.close()
    return {"status": "success"}

@app.get("/api/chat/history")
async def get_chat_history(authorization: str = Header(...)):
    uid = verify_user(authorization)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, title, messages, docId, uploadedFileName, uploadedFileUrl, updatedAt, isPinned FROM chats WHERE uid=? ORDER BY isPinned DESC, updatedAt DESC", (uid,))
    rows = c.fetchall()
    conn.close()
    chats = []
    for r in rows:
        chats.append({
            "id": r[0], "title": r[1], "messages": json.loads(r[2]),
            "docId": r[3], "uploadedFileName": r[4], "uploadedFileUrl": r[5], "updatedAt": r[6],
            "isPinned": bool(r[7] if len(r) > 7 else 0)
        })
    return chats

@app.post("/api/chat/pin/{chat_id}")
async def toggle_pin_chat(chat_id: str, isPinned: bool, authorization: str = Header(...)):
    uid = verify_user(authorization)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    pin_val = 1 if isPinned else 0
    c.execute("UPDATE chats SET isPinned=? WHERE id=? AND uid=?", (pin_val, chat_id, uid))
    conn.commit()
    conn.close()
    return {"status": "success"}

@app.delete("/api/chat/{chat_id}")
async def delete_chat(chat_id: str, authorization: str = Header(...)):
    uid = verify_user(authorization)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM chats WHERE id=? AND uid=?", (chat_id, uid))
    conn.commit()
    conn.close()
    return {"status": "deleted"}

# =========================
# MATERIALS ENDPOINTS
# =========================

@app.get("/api/materials")
async def get_materials(classNum: int, subject: str, authorization: str = Header(...)):
    verify_user(authorization)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT id, classNum, subject, url, title, uploadedBy, uploaderUid, createdAt FROM materials WHERE classNum=? AND subject=? ORDER BY createdAt DESC",
        (classNum, subject)
    )
    rows = c.fetchall()
    conn.close()
    return [
        {
            "id": r[0], "classNum": r[1], "subject": r[2], "url": r[3],
            "title": r[4], "uploadedBy": r[5], "uploaderUid": r[6], "createdAt": r[7]
        }
        for r in rows
    ]

@app.delete("/api/materials/{material_id}")
async def delete_material(material_id: str, authorization: str = Header(...)):
    uid = verify_user(authorization)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Only allow the uploader to delete their own material
    c.execute("DELETE FROM materials WHERE id=? AND uploaderUid=?", (material_id, uid))
    conn.commit()
    conn.close()
    return {"status": "deleted"}

# =========================
# OCR & HELPERS
# =========================

def perform_ocr(page):
    pix = page.get_pixmap()
    raw_document = documentai.RawDocument(
        content=pix.tobytes("png"),
        mime_type="image/png"
    )
    name = docai_client.processor_path(
        PROJECT_ID,
        LOCATION,
        PROCESSOR_ID
    )
    request = documentai.ProcessRequest(
        name=name,
        raw_document=raw_document
    )
    result = docai_client.process_document(request=request)
    return result.document.text

def chunk_text(text, size=800, overlap=150):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks

def save_json_file(data, path):
    with open(path, "w", encoding="utf8") as f:
        json.dump(data, f)

def extract_page_images(pdf_path, pages):
    page_images = []
    try:
        pdf_doc = fitz.open(pdf_path)
        # Cap images to 15 max to avoid huge payload on large summary requests
        for p_num in list(pages)[:15]:
            if 1 <= p_num <= len(pdf_doc):
                # 150 DPI is highly optimized for speed while keeping diagrams readable
                pix = pdf_doc[p_num - 1].get_pixmap(dpi=150)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                page_images.append(img)
        pdf_doc.close()  # Free up RAM immediately after capturing image
    except Exception as e:
        logging.error(f"Error loading images for Gemini: {e} - backend_server.py:363")
    return page_images

# =========================
# UPLOAD DOCUMENT
# =========================

@app.post("/api/upload")
async def upload(
        file: UploadFile = File(...),
        authorization: str = Header(...),
        classNum: int = Form(None),
        subject: str = Form(None),
        title: str = Form(None),
        uploadedBy: str = Form(None)
):
    uid = verify_user(authorization)
    file_bytes = await file.read()
    doc_id = str(uuid.uuid4())
    doc_path = f"{DOC_FOLDER}/{doc_id}"
    os.makedirs(doc_path)

    pdf_path = f"{doc_path}/source.pdf"
    with open(pdf_path, "wb") as f:
        f.write(file_bytes)

    logging.info(f"{uid} uploaded document {doc_id} - backend_server.py:389")

    doc = fitz.open(stream=file_bytes, filetype="pdf")
    texts = []
    for page in doc:
        text = await asyncio.to_thread(perform_ocr, page)
        texts.append(text)

    all_chunks = []
    meta = []
    for i, page_text in enumerate(texts):
        chunks = chunk_text(page_text)
        for c in chunks:
            all_chunks.append(c)
            meta.append({"page": i + 1})

    # Offload heavy embedding to thread (batch encode – not cached since this is bulk upload)
    embeddings = await asyncio.to_thread(
        embedding_model.encode,
        all_chunks,
        batch_size=64,
        convert_to_numpy=True
    )

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # Offload disk IO to thread
    await asyncio.to_thread(faiss.write_index, index, f"{doc_path}/text_index.faiss")

    final_chunks_data = [{"text": c, "page": meta[i]["page"]} for i, c in enumerate(all_chunks)]

    # Offload JSON dumping
    await asyncio.to_thread(save_json_file, final_chunks_data, f"{doc_path}/text_chunks.json")

    # Pre-load into RAM Cache so the very first question is lightning fast
    DOCUMENT_CACHE[doc_id] = (index, final_chunks_data)

    # Save material metadata to the materials table so all users can access it
    if classNum is not None and subject and title:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            '''INSERT OR REPLACE INTO materials (id, classNum, subject, url, title, uploadedBy, uploaderUid, createdAt)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
            (doc_id, classNum, subject, f"/documents/{doc_id}/source.pdf",
             title, uploadedBy or "", uid, int(time.time() * 1000))
        )
        conn.commit()
        conn.close()

    return {"docId": doc_id}

# =========================
# ASK QUESTION (MULTIMODAL & CACHED)
# =========================

@app.post("/api/ask")
async def ask(
        req: AskRequest,
        authorization: str = Header(...)
):
    uid = verify_user(authorization)

    # ------------------------------------------------------------------
    # CONTEXTUAL MEMORY ── Build a rich history block from the conversation
    # so the model understands follow-up questions like "Explain that again"
    # or "Give me more detail about the second point."
    # ------------------------------------------------------------------
    history_text = ""
    if req.history:
        history_text = (
            "\n--- PREVIOUS CONVERSATION CONTEXT ---\n"
            "Use this history to understand what the student is referring to in follow-up "
            "questions (e.g. 'Explain that again', 'What did you mean by the second point?', "
            "'Give an example of that'). Always resolve pronouns and references against this "
            "history before answering.\n"
        )
        for msg in req.history:
            # Strip markdown links from history so the AI isn't confused by raw URLs
            clean_text = re.sub(r'\[.*?\]\(.*?\)', '', msg.text).strip()
            history_text += f"{msg.role.upper()}: {clean_text}\n"
        history_text += "--- END OF HISTORY ---\n"

    # ========================================================
    # SCENARIO 1: NO DOCUMENT UPLOADED (GENERAL KNOWLEDGE)
    # ========================================================
    if not req.docId or req.docId == "null":
        prompt = f"""
You are Aksharalokam AI, an expert teacher developed by the Aksharalokam team.

CRITICAL LANGUAGE INSTRUCTIONS:
- Your DEFAULT language for all responses is Malayalam.
- Even if the student asks the question in English or another language, you MUST answer entirely in Malayalam.
- EXCEPTION: You must answer in another language ONLY IF the student explicitly requests it by naming the language (e.g., "answer in English", "explain in Hindi"). If they explicitly request a language, answer strictly in that requested language without mixing words.

General Instructions:
- Answer accurately and clearly using your general knowledge.
- Identity: If asked who developed you, answer: "I am Aksharalokam AI, developed by the Aksharalokam team." (Translate this phrase into the language of the response).
- Summaries: If they ask for a summary of a chapter or a key concept, provide a well-structured summary with key points and headings.
- Formatting: If the user asks for the answer in points, in a table, or for a specific mark weightage, format your response exactly according to that request.

CONTEXTUAL MEMORY INSTRUCTIONS:
- Carefully read the PREVIOUS CONVERSATION CONTEXT below (if any) before answering.
- If the current question is a follow-up (e.g., "Explain that again", "What did you mean?", "Give an example"), refer back to your last answer in the history and elaborate on it. Do NOT treat the question as if it were isolated.

{history_text}

Current Student's Request:
{req.question}

Answer:
"""
        response = await gemini.generate_content_async(prompt)
        answer = response.text

        logging.info(f"{uid} asked general knowledge question - backend_server.py:505")

        return {
            "answer": answer,
            "source_pages": [],
            "source_links": []
        }

    # ========================================================
    # SCENARIO 2: DOCUMENT UPLOADED (STRICT PDF + MULTIMODAL)
    # ========================================================

    # Security: Validate doc_id before using it in any file-system path
    validate_doc_id(req.docId)

    doc_path = f"{DOC_FOLDER}/{req.docId}"
    pdf_path = f"{doc_path}/source.pdf"

    if not os.path.exists(doc_path) or not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="Document not found.")

    # 1. READ FROM RAM CACHE FOR INSTANT SPEED (Instead of slow hard drive reads)
    if req.docId not in DOCUMENT_CACHE:
        index = faiss.read_index(f"{doc_path}/text_index.faiss")
        with open(f"{doc_path}/text_chunks.json") as f:
            chunks = json.load(f)
        DOCUMENT_CACHE[req.docId] = (index, chunks)
    else:
        index, chunks = DOCUMENT_CACHE[req.docId]

    # 2. Check if user requested a specific page
    page_match = re.search(r'(?:page|പേജ്|പേജിലെ)\s*(\d+)', req.question, re.IGNORECASE)
    requested_page_text = ""
    requested_pages = []

    if page_match:
        try:
            page_num = int(page_match.group(1))
            page_chunks = [c["text"] for c in chunks if c["page"] == page_num]
            if page_chunks:
                url = f"/source/{req.docId}/{page_num}"
                requested_page_text = f"\n\n--- EXACT TEXT FROM PAGE {page_num} (Link: {url}) ---\n" + "\n\n".join(page_chunks)
                requested_pages.append(page_num)
        except Exception:
            pass

    # 3. Fast FAISS Semantic Search (uses LRU-cached embedding for repeated queries)
    summary_keywords = ['summary', 'summarize', 'chapter', 'ചുരുക്കം', 'സംഗ്രഹം', 'വിശദീകരിക്കുക']
    is_summary_request = any(kw in req.question.lower() for kw in summary_keywords)
    top_k = 30 if is_summary_request else 8

    # Use the LRU-cached encoding for single-query embeddings (fast on repeated questions)
    query_embedding = await asyncio.to_thread(get_query_embedding, req.question)
    D, I = index.search(np.array([query_embedding]).astype("float32"), top_k)
    best_indices = [idx for idx in I[0] if idx != -1]

    faiss_context_parts = []
    for i in best_indices:
        p = chunks[i]["page"]
        url = f"/source/{req.docId}/{p}"
        faiss_context_parts.append(f"--- PAGE {p} (Link: {url}) ---\n{chunks[i]['text']}")

    faiss_context = "\n\n".join(faiss_context_parts)
    faiss_pages = [chunks[i]["page"] for i in best_indices]

    context = faiss_context + requested_page_text
    pages = list(set(faiss_pages + requested_pages))

    # 4. Extract actual images concurrently so the loop isn't blocked
    page_images = await asyncio.to_thread(extract_page_images, pdf_path, pages)

    # 5. STRICT MULTIMODAL PROMPT
    prompt = f"""
You are Aksharalokam AI, an expert teacher developed by the Aksharalokam team.
Read the following textbook context carefully. I have also attached the actual images of these textbook pages to this prompt so you can see the diagrams and pictures.

CRITICAL LANGUAGE INSTRUCTIONS:
- Your DEFAULT language for all responses is Malayalam.
- Even if the student asks the question in English or another language, you MUST answer entirely in Malayalam.
- EXCEPTION: You must answer in another language ONLY IF the student explicitly requests it by naming the language (e.g., "answer in English", "explain in Hindi"). If they explicitly request a language, translate the context perfectly and answer strictly in that requested language without mixing words.

CONTEXTUAL MEMORY INSTRUCTIONS:
- Carefully read the PREVIOUS CONVERSATION CONTEXT below (if any) BEFORE answering the current question.
- If the current question is a follow-up (e.g., "Explain that again", "What is the second step?", "Can you expand on that?"), refer to the last AI answer in the history and elaborate. Do NOT treat the question as isolated.
- Resolve all pronouns and vague references ("that", "this", "it", "the previous answer") against the conversation history before composing your response.

Strict Context Instructions:
1. Answer the student's question accurately based STRICTLY on the provided text context AND the attached page images.
2. Identity: If asked who developed you, answer: "I am Aksharalokam AI, developed by the Aksharalokam team." but reply this if and only if someone tells you who developed you; no need to mention otherwise. (Translate into the language of the response.)
3. VERBATIM STEPS & STAGES (CRITICAL RULE):
   - If the student asks for "steps", "stages", "procedures", "ഘട്ടങ്ങൾ", "പ്രക്രിയ", or any similar phrasing, you MUST copy the relevant text EXACTLY WORD-FOR-WORD from the provided textbook context below.
   - DO NOT paraphrase, summarise, reorder, or rephrase even a single word when answering step/stage/procedure questions.
   - If answering in a language other than the original text, provide a precise, direct translation of the exact textbook text — no added interpretation.
   - Treat any deviation from the original wording as a critical error.
4. DIAGRAMS & IMAGES: If the user asks to label a diagram, explain a picture, or asks a question about a visual, carefully examine the attached page images and answer accurately based on what you see.
5. MATH/SCIENCE: Use equations present in the text/images. IF AND ONLY IF the required equation is missing, you may use standard outside equations, but you MUST explicitly mention that the equation was taken from outside the text.
6. INLINE CITATIONS: Whenever you state a fact, list a step, or provide a point, you MUST add an inline markdown link to the source page at the end of that sentence/point using a paperclip icon.
   Use the EXACT "Link" provided in the text context blocks.
   Example format: "This process involves using a petri dish [Page 2](/source/{req.docId}/2)."
7. OUT OF CONTEXT: For all other factual questions, DO NOT use your outside knowledge. If the answer is not in the text or images, say "Sorry, the answer is not available in the provided textbook context." (Translate to the user's language).
8. PAGE HALLUCINATION FIX: ONLY cite page numbers that are explicitly written in the "PAGE X" headers of the provided context below. NEVER invent, guess, or reference a page number that is not explicitly given to you in the context blocks.

{history_text}

Context from Textbook:
{context}

Current Student's Question:
{req.question}

Answer:
"""

    # Send the prompt PLUS the images to Gemini Flash asynchronously
    content_to_send = [prompt] + page_images
    response = await gemini.generate_content_async(content_to_send)
    answer = response.text

    logging.info(f"{uid} asked document question - backend_server.py:623")

    return {
        "answer": answer,
        "source_pages": pages,
        "source_links": []
    }

# =========================
# SOURCE PAGE VIEW  ──  secured against directory traversal
# =========================

@app.get("/source/{doc_id}/{page}")
def get_source(doc_id: str, page: int):
    # SECURITY: Validate doc_id against UUID-v4 pattern to prevent
    # path traversal attacks such as ../../etc/passwd
    validate_doc_id(doc_id)

    pdf_path = f"{DOC_FOLDER}/{doc_id}/source.pdf"
    if not os.path.exists(pdf_path):
        raise HTTPException(404)

    doc = fitz.open(pdf_path)
    if page > len(doc):
        raise HTTPException(404)

    p = doc[page - 1]
    pix = p.get_pixmap(dpi=200)

    return Response(
        content=pix.tobytes("png"),
        media_type="image/png"
    )
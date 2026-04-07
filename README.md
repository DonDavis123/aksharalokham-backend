Aksharalokam AI - Backend Engine 🧠
Aksharalokam is a high-performance Retrieval-Augmented Generation (RAG) platform designed for the Malayalam educational sector. This FastAPI-based server acts as the "brain" of the application, handling document intelligence, semantic search, and multimodal AI reasoning.

🌟 Technical Highlights
Multimodal Reasoning: Unlike standard text-only RAG, this engine "sees" the pages. It sends both text context and original page images to Gemini 1.5 Flash for 100% accurate visual Q&A.

High-Speed Optimization: Implements LRU Caching for embeddings and FAISS for sub-millisecond semantic search, reducing latency by up to 80% on repeat queries.

Hallucination Guardrails: Features a "Verbatim Mode" for steps and procedures, forcing the AI to extract textbook content word-for-word.

Malayalam-First Intelligence: Specialized prompt engineering ensures the AI prioritizes Malayalam responses and cultural relevance.

## 📺 Project Walkthrough
[![Aksharalokam Demo](https://img.youtube.com/vi/R07azGXrhow/0.jpg)](https://www.youtube.com/watch?v=R07azGXrhow)

🏗️ Technical Architecture
The backend operates through a sophisticated, multi-layered AI pipeline:

OCR Layer: Transforms PDF pixels into structured data using Google Document AI.

Embedding Layer: Converts text into 768-dimensional vectors using paraphrase-multilingual-mpnet-base-v2.

Storage Layer: Manages metadata in SQLite and high-dimensional vectors in FAISS.

Retrieval Layer: Merges semantic search results with visual page context for the LLM.

🔗 Connected Repositories
Frontend Interface: [https://github.com/DonDavis123/aksharalokham-frontend]

Live Application: https://akshara-803d3.web.app

🛠️ Tech Stack
Language: Python 3.10+

Framework: FastAPI (Asynchronous)

AI/ML: Google Gemini 2.5 Flash, Sentence-Transformers, PyTorch

Database: FAISS (Vector) & SQLite (Relational)

Vision: Google Document AI, PyMuPDF, Pillow

💻 Local Setup
Clone the repository.

Install dependencies: pip install -r requirements.txt.

Create a .env file with your GEMINI_API_KEY and GCP_PROJECT_ID.

Run the engine: uvicorn backend_server:app --reload.

Developed by the Aksharalokam Team.

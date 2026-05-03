"""rag_pipeline.py — RAG (Retrieval Augmented Generation) for Xynth Research.

Handles:
- PDF/text document ingestion with LangChain
- Embedding via DashScope text-embedding-v3
- Storage in Supabase with pgvector
- Similarity search at query time

Supabase SQL to run once:
    CREATE EXTENSION IF NOT EXISTS vector;
    CREATE TABLE documents (
        id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
        user_id UUID NOT NULL,
        doc_title TEXT NOT NULL,
        content TEXT NOT NULL,
        embedding VECTOR(1536),
        chunk_index INTEGER DEFAULT 0,
        created_at TIMESTAMPTZ DEFAULT NOW()
    );
    CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
"""
import os
import json
import io
import requests
from typing import Optional

DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
EMBED_MODEL = "text-embedding-v3"
EMBED_URL   = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/embeddings"
CHUNK_SIZE  = 800
CHUNK_OVERLAP = 100


# ── Text extraction ───────────────────────────────────────────────────────────
def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF using LangChain PyPDFLoader."""
    try:
        from langchain_community.document_loaders import PyPDFLoader
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(file_bytes)
            tmp_path = f.name
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        os.unlink(tmp_path)
        return "\n\n".join(p.page_content for p in pages)
    except Exception:
        # Fallback: pdfplumber
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                return "\n\n".join(p.extract_text() or "" for p in pdf.pages)
        except Exception as e:
            return f"[Could not extract PDF text: {e}]"


def extract_text_from_txt(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode("utf-8", errors="ignore")
    except Exception:
        return ""


# ── Chunking ──────────────────────────────────────────────────────────────────
def chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks using LangChain."""
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        return splitter.split_text(text)
    except Exception:
        # Manual fallback
        words = text.split()
        chunks, i = [], 0
        while i < len(words):
            chunk = " ".join(words[i:i + CHUNK_SIZE])
            chunks.append(chunk)
            i += CHUNK_SIZE - CHUNK_OVERLAP
        return chunks


# ── Embeddings ────────────────────────────────────────────────────────────────
def embed_texts(texts: list[str]) -> list[list[float]]:
    """Get embeddings from DashScope text-embedding-v3."""
    if not DASHSCOPE_API_KEY:
        raise ValueError("DASHSCOPE_API_KEY not set")
    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json",
    }
    # DashScope allows up to 25 texts per batch
    all_embeddings = []
    for i in range(0, len(texts), 25):
        batch = texts[i:i+25]
        payload = {"model": EMBED_MODEL, "input": batch, "encoding_format": "float"}
        resp = requests.post(EMBED_URL, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        embeddings = [item["embedding"] for item in data["data"]]
        all_embeddings.extend(embeddings)
    return all_embeddings


def embed_query(query: str) -> list[float]:
    return embed_texts([query])[0]


# ── Supabase storage ──────────────────────────────────────────────────────────
def store_document(user_id: str, doc_title: str, file_bytes: bytes,
                   file_type: str, sb) -> dict:
    """Process and store a document for RAG. Returns stats."""
    # Extract text
    if file_type in ("application/pdf", "pdf"):
        text = extract_text_from_pdf(file_bytes)
    else:
        text = extract_text_from_txt(file_bytes)

    if not text.strip():
        return {"error": "Could not extract text from document"}

    # Chunk
    chunks = chunk_text(text)
    if not chunks:
        return {"error": "No content chunks generated"}

    # Embed all chunks
    try:
        embeddings = embed_texts(chunks)
    except Exception as e:
        return {"error": f"Embedding failed: {e}"}

    # Store in Supabase
    rows = []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        rows.append({
            "user_id":     user_id,
            "doc_title":   doc_title,
            "content":     chunk,
            "embedding":   emb,
            "chunk_index": i,
        })

    # Insert in batches of 20
    inserted = 0
    for i in range(0, len(rows), 20):
        try:
            res = sb.table("documents").insert(rows[i:i+20]).execute()
            inserted += len(res.data)
        except Exception as e:
            print(f"Insert batch error: {e}")

    return {
        "success": True,
        "doc_title": doc_title,
        "chunks": len(chunks),
        "inserted": inserted,
    }


def retrieve_relevant_chunks(query: str, user_id: str, sb,
                              top_k: int = 5) -> list[str]:
    """Find the most relevant document chunks for a query via vector similarity."""
    try:
        query_embedding = embed_query(query)
        # Use Supabase RPC for vector similarity search
        res = sb.rpc("match_documents", {
            "query_embedding": query_embedding,
            "match_user_id":   user_id,
            "match_count":     top_k,
        }).execute()
        if res.data:
            return [f"[From: {r['doc_title']}]\n{r['content']}" for r in res.data]
    except Exception as e:
        print("Retrieve error:", e)
        # Fallback: just get recent chunks without vector search
        try:
            res = sb.table("documents").select("doc_title, content").eq("user_id", user_id).limit(top_k).execute()
            return [f"[From: {r['doc_title']}]\n{r['content']}" for r in res.data]
        except Exception:
            pass
    return []


def list_user_documents(user_id: str, sb) -> list[dict]:
    """List unique documents for a user."""
    try:
        res = sb.table("documents").select("doc_title, chunk_index").eq("user_id", user_id).execute()
        seen = {}
        for r in res.data:
            title = r["doc_title"]
            if title not in seen:
                seen[title] = 0
            seen[title] += 1
        return [{"title": t, "chunks": c} for t, c in seen.items()]
    except Exception as e:
        print("List docs error:", e)
        return []


def delete_user_document(user_id: str, doc_title: str, sb) -> bool:
    """Delete all chunks for a document."""
    try:
        sb.table("documents").delete().eq("user_id", user_id).eq("doc_title", doc_title).execute()
        return True
    except Exception:
        return False

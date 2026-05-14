"""
Medical RAG Service using pgvector on Neon PostgreSQL
Handles document ingestion and semantic search for medical knowledge
"""

import os
import logging
from typing import List, Optional, Dict, Any
import json

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 produces 384-dim vectors


class RAGService:
    def __init__(self, db_url: str):
        self.db_url = db_url
        self.conn = None
        self.model = None
        self._init_model()

    def _init_model(self):
        """Load sentence-transformer embedding model"""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("✅ Embedding model loaded: all-MiniLM-L6-v2")
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            self.model = None

    def connect(self) -> bool:
        """Establish PostgreSQL connection"""
        try:
            import psycopg2
            self.conn = psycopg2.connect(self.db_url)
            self.conn.autocommit = True
            logger.info("✅ Connected to Neon PostgreSQL")
            return True
        except Exception as e:
            logger.error(f"❌ PostgreSQL connection failed: {e}")
            return False

    def init_db(self) -> bool:
        """Create pgvector extension and medical_documents table"""
        try:
            with self.conn.cursor() as cur:
                # Enable pgvector
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

                # Create medical knowledge table
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS medical_documents (
                        id SERIAL PRIMARY KEY,
                        title TEXT,
                        content TEXT NOT NULL,
                        embedding vector({EMBEDDING_DIM}),
                        metadata JSONB DEFAULT '{{}}',
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );
                """)

                # Create index for fast similarity search
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS medical_docs_embedding_idx
                    ON medical_documents
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 50);
                """)

                # Create conversation history table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS conversation_history (
                        id SERIAL PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );
                """)

                logger.info("✅ Database tables initialized")
                return True
        except Exception as e:
            logger.error(f"❌ DB init failed: {e}")
            return False

    def embed(self, text: str) -> Optional[List[float]]:
        """Generate embedding vector for given text"""
        if not self.model:
            return None
        try:
            vector = self.model.encode(text, convert_to_numpy=True).tolist()
            return vector
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return None

    def add_document(self, content: str, title: str = "", metadata: dict = None) -> bool:
        """
        Ingest a document into the vector store.
        Chunks long documents into ~500-word pieces.
        """
        if not self.conn or not self.model:
            return False

        metadata = metadata or {}
        chunks = self._chunk_text(content, chunk_size=500)

        try:
            with self.conn.cursor() as cur:
                for i, chunk in enumerate(chunks):
                    embedding = self.embed(chunk)
                    if embedding is None:
                        continue

                    chunk_metadata = {**metadata, "chunk_index": i, "total_chunks": len(chunks)}
                    cur.execute(
                        """
                        INSERT INTO medical_documents (title, content, embedding, metadata)
                        VALUES (%s, %s, %s::vector, %s)
                        """,
                        (title or f"Document chunk {i+1}", chunk,
                         str(embedding), json.dumps(chunk_metadata))
                    )
            logger.info(f"✅ Added {len(chunks)} chunks for: {title}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to add document: {e}")
            return False

    def search(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        """Semantic similarity search in the medical knowledge base"""
        if not self.conn or not self.model:
            return []

        query_vec = self.embed(query)
        if query_vec is None:
            return []

        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT title, content, metadata,
                           1 - (embedding <=> %s::vector) AS similarity
                    FROM medical_documents
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (str(query_vec), str(query_vec), k)
                )
                rows = cur.fetchall()
                return [
                    {"title": r[0], "content": r[1], "metadata": r[2], "similarity": float(r[3])}
                    for r in rows
                    if float(r[3]) > 0.25  # minimum relevance threshold
                ]
        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            return []

    def get_context(self, query: str) -> str:
        """Build a context string from top relevant documents for the query"""
        results = self.search(query, k=4)
        if not results:
            return ""

        context_parts = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "Medical Reference")
            content = r["content"][:600]  # limit each chunk
            context_parts.append(f"[Reference {i} - {title}]:\n{content}")

        return "\n\n".join(context_parts)

    def get_document_count(self) -> int:
        """Return total number of stored documents"""
        if not self.conn:
            return 0
        try:
            with self.conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM medical_documents;")
                return cur.fetchone()[0]
        except Exception:
            return 0

    def delete_document(self, doc_id: int) -> bool:
        """Delete a document by ID"""
        if not self.conn:
            return False
        try:
            with self.conn.cursor() as cur:
                cur.execute("DELETE FROM medical_documents WHERE id = %s;", (doc_id,))
            return True
        except Exception as e:
            logger.error(f"Delete error: {e}")
            return False

    def list_documents(self) -> List[Dict]:
        """List all documents (without embeddings)"""
        if not self.conn:
            return []
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT id, title, LEFT(content, 150) AS preview, metadata, created_at
                    FROM medical_documents
                    ORDER BY created_at DESC;
                """)
                rows = cur.fetchall()
                return [
                    {"id": r[0], "title": r[1], "preview": r[2],
                     "metadata": r[3], "created_at": str(r[4])}
                    for r in rows
                ]
        except Exception as e:
            logger.error(f"List error: {e}")
            return []

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 500) -> List[str]:
        """Split text into word-based chunks with overlap"""
        words = text.split()
        chunks = []
        overlap = 50  # word overlap between chunks

        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i: i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)

        return chunks if chunks else [text]


# ── Global singleton ────────────────────────────────────────────────────────

_rag_service: Optional[RAGService] = None


def get_rag_service() -> Optional[RAGService]:
    return _rag_service


async def initialize_rag(db_url: str) -> bool:
    global _rag_service
    try:
        _rag_service = RAGService(db_url)
        if not _rag_service.connect():
            return False
        if not _rag_service.init_db():
            return False
        count = _rag_service.get_document_count()
        logger.info(f"✅ RAG service ready — {count} documents in knowledge base")
        return True
    except Exception as e:
        logger.error(f"RAG init error: {e}")
        return False

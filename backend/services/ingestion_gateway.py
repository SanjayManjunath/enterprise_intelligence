import os
import hashlib
import sqlite3
import logging
from typing import Dict, Any

# Phase 1 Upgrades
from llama_parse import LlamaParse
from services.Vector_worker import VectorWorker

logger = logging.getLogger(__name__)

class IngestionGateway:
    def __init__(self, ledger_db_path="../storage/hash_ledger.db"):
        self.ledger_db_path = ledger_db_path
        self._init_ledger()
        
        # Initialize your upgraded Vector Worker
        self.vector_worker = VectorWorker()
        
        # Initialize LlamaParse (Requires LLAMA_CLOUD_API_KEY in .env)
        self.parser = LlamaParse(
            result_type="markdown",
            num_workers=4,
            verbose=True,
            language="en"
            # Note: For full S3 image routing, LlamaParse premium allows
            # parsing images to a cloud bucket. For now, we extract pure Markdown.
        )

    def _init_ledger(self):
        """Creates the cryptographic ledger if it doesn't exist."""
        os.makedirs(os.path.dirname(self.ledger_db_path), exist_ok=True)
        with sqlite3.connect(self.ledger_db_path) as conn:
            conn.execute('''CREATE TABLE IF NOT EXISTS document_ledger (
                                file_hash TEXT PRIMARY KEY, 
                                file_name TEXT, 
                                clearance_level TEXT, 
                                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                            )''')

    def get_file_hash(self, file_bytes: bytes) -> str:
        """Generates the SHA-256 fingerprint of the raw file contents."""
        return hashlib.sha256(file_bytes).hexdigest()

    def is_duplicate(self, file_hash: str) -> bool:
        """Checks the ledger to see if these exact bytes have been vectorized before."""
        with sqlite3.connect(self.ledger_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT file_name FROM document_ledger WHERE file_hash = ?", (file_hash,))
            return cursor.fetchone() is not None

    def record_to_ledger(self, file_hash: str, file_name: str, clearance_level: str):
        """Permanently records the file hash after successful ingestion."""
        with sqlite3.connect(self.ledger_db_path) as conn:
            conn.execute("INSERT INTO document_ledger (file_hash, file_name, clearance_level) VALUES (?, ?, ?)",
                         (file_hash, file_name, clearance_level))

    def process_upload(self, file_bytes: bytes, file_name: str, clearance_level: str = "public") -> Dict[str, Any]:
        """
        The Main Gateway: Hashes, Checks, Parses, and Vectorizes.
        """
        # 1. Cryptographic Hash Check
        file_hash = self.get_file_hash(file_bytes)
        
        if self.is_duplicate(file_hash):
            logger.info(f"🚫 IDEMPOTENCY LOCK: Exact bytes of {file_name} already exist in Pinecone (Hash: {file_hash}). Skipping processing.")
            return {"status": "skipped", "reason": "duplicate", "hash": file_hash}

        logger.info(f"✅ NEW FILE DETECTED: Initializing Phase 1 Pipeline for {file_name}...")

        # 2. Save temporarily for parsing
        temp_dir = "/tmp/enterprise_ingest"
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, file_name)
        
        with open(temp_path, "wb") as f:
            f.write(file_bytes)

        try:
            ext = os.path.splitext(file_name)[1].lower()
            
            # 3. LlamaParse Layout Extraction (PDFs)
            if ext == ".pdf":
                logger.info(f"   ∟ Routing {file_name} through LlamaParse DLA...")
                # LlamaParse converts complex tables and columns into clean Markdown
                parsed_docs = self.parser.load_data(temp_path)
                full_markdown = "\n\n".join([doc.text for doc in parsed_docs])
                
                # Send the clean markdown to your VectorWorker (which now handles RBAC and Header Splitting)
                self.vector_worker.sync_markdown_to_cloud(full_markdown, source_name=file_name)
            
            # 3b. Standard Relational Extraction (CSVs)
            elif ext in [".csv", ".xlsx"]:
                self.vector_worker.sync_csv_to_cloud(temp_path)
                
            else:
                logger.warning(f"Unsupported file type for LlamaParse gateway: {ext}")
                return {"status": "failed", "reason": "unsupported_format"}

            # 4. Record Success in Ledger
            self.record_to_ledger(file_hash, file_name, clearance_level)
            logger.info(f"   ∟ Ingestion complete. Hash {file_hash} recorded to ledger.")
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)

        return {"status": "success", "hash": file_hash}
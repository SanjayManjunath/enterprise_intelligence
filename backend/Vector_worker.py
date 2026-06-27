import os
import time
import json
import base64
import pandas as pd
import numpy as np
import uuid
from odf import text, teletype
from odf.opendocument import load

# --- API OFFLOAD IMPORTS ---
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchAny

# --- LOCAL RERANKER (ONNX INT8) ---
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

from dotenv import load_dotenv
from langchain_text_splitters import MarkdownHeaderTextSplitter

load_dotenv()

class VectorWorker:
    def __init__(self, logs_path="../storage/vector_indices/project_logs"):
        """
        Meta/VC Grade Hybrid Search Architecture.
        - Embeddings: OpenAI text-embedding-3-small (Cost-efficient, lightning fast)
        - Vector DB: Local Qdrant (Infinite free scaling)
        - Reranker: ONNX INT8 Local CPU Engine
        """
        openai_key = os.environ.get("OPENAI_API_KEY")
        if not openai_key:
            raise ValueError("❌ OPENAI_API_KEY is missing from the .env file.")
        
        print("   ∟ [VECTOR] Connecting to OpenAI Embedding API...")
        self.openai_client = OpenAI(api_key=openai_key)
        
        print("   ∟ [VECTOR] Initializing ONNX INT8 Runtime for BGE-Reranker...")
        model_id = "Xenova/bge-reranker-base"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.reranker = ORTModelForSequenceClassification.from_pretrained(model_id, file_name="onnx/model_quantized.onnx")
            
        print("   ∟ [VECTOR] Connecting to Local Qdrant Database...")
        # Connects to the Qdrant container over the Docker internal bridge
        self.qdrant = QdrantClient(url="http://qdrant-db:6333")
        self.collection_name = "enterprise_ai"
        
        # --- DOCKER BOOT RACE CONDITION SHIELD ---
        # Qdrant takes ~5-10 seconds to open its REST API. FastAPI boots in 1 second.
        # This polling loop prevents the backend from crashing by waiting for Qdrant.
        max_retries = 15
        for attempt in range(max_retries):
            try:
                # Provision Qdrant collection if it doesn't exist
                if not self.qdrant.collection_exists(collection_name=self.collection_name):
                    print(f"   ∟ [VECTOR] Provisioning Qdrant collection: '{self.collection_name}'...")
                    self.qdrant.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
                    )
                print("   ∟ [VECTOR] Qdrant connection established and verified.")
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise Exception(f"❌ Qdrant failed to boot in time: {e}")
                print(f"   ∟ [VECTOR] Waiting for Qdrant REST API to initialize (Attempt {attempt+1}/{max_retries})...")
                time.sleep(2)
            
        self.abs_logs_path = os.path.abspath(logs_path)
        self.refresh_index(self.abs_logs_path)

    def _get_embedding(self, text: str) -> list[float]:
        """Fetches dense vector from OpenAI"""
        response = self.openai_client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding

    def _generate_uuid(self, string_id: str) -> str:
        """Deterministic UUID generation for Qdrant"""
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, string_id))

    def sync_csv_to_cloud(self, file_path: str, clearance_level: str = "public"):
        fname = os.path.basename(file_path)
        print(f"   ∟ [VECTOR] Syncing CSV schema for {fname} to Qdrant (Clearance: {clearance_level})...")
        try:
            df = pd.read_csv(file_path)
            content = f"SOURCE: {fname} (Relational Data) | SCHEMA_SIGNATURE: {list(df.columns)}"
            
            embedding = self._get_embedding(content)
            
            point = PointStruct(
                id=self._generate_uuid(f"{fname}_schema"), 
                vector=embedding,
                payload={
                    "text": content, 
                    "filename": fname, 
                    "type": "structured_csv", 
                    "clearance_level": clearance_level
                }
            )
            self.qdrant.upsert(collection_name=self.collection_name, points=[point])
        except Exception as e:
            print(f"   ∟ [VECTOR] Failed to sync CSV schema: {str(e)}")

    def sync_markdown_to_cloud(self, markdown_text: str, source_name: str, clearance_level: str = "public"):
        print(f"   ∟ [VECTOR] Processing Markdown from {source_name} to Qdrant...")
        
        headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
        splits = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on).split_text(markdown_text)
        
        points = []
        for i, doc in enumerate(splits):
            clean_chunk = doc.page_content.strip()
            if not clean_chunk: continue
                
            embedding = self._get_embedding(clean_chunk)
            
            payload = {
                "text": clean_chunk, 
                "source": source_name, 
                "type": "structured_pdf", 
                "clearance_level": clearance_level
            }
            payload.update(doc.metadata) 
            
            points.append(PointStruct(id=self._generate_uuid(f"{source_name}_md_{i}"), vector=embedding, payload=payload))
            
        if points:
            self.qdrant.upsert(collection_name=self.collection_name, points=points)
            print(f"   ∟ [VECTOR] Synced {len(points)} chunks to Qdrant.")

    def _extract_image_context(self, file_path):
        return ""

    def _parse_any_file(self, file_path):
        ext = os.path.splitext(file_path)[1].lower()
        fname = os.path.basename(file_path)
        try:
            if ext in [".txt", ".md", ".log"]:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f: return f"SOURCE: {fname} | CONTENT: {f.read()}"
            elif ext in [".csv", ".xlsx"]:
                df = pd.read_csv(file_path) if ext == ".csv" else pd.read_excel(file_path)
                return f"SOURCE: {fname} (Relational Data) | SCHEMA_SIGNATURE: {list(df.columns)}"
            elif ext == ".odt":
                doc = load(file_path)
                return f"SOURCE: {fname} | DOCUMENT_CONTENT: " + "\n".join([teletype.extractText(p) for p in doc.getElementsByType(text.P)])
            elif ext == ".ipynb":
                with open(file_path, 'r', encoding='utf-8') as f: nb = json.load(f)
                return f"SOURCE: {fname} | NOTEBOOK_CODE: " + " ".join(["".join(c.get('source', [])) for c in nb.get('cells', [])])
            return ""
        except Exception: return ""

    def refresh_index(self, primary_path, secondary_path=None, clearance_level="public"):
        self.documents, self.filenames = [], []
        p_path = os.path.abspath(primary_path) if primary_path else None
        s_path = os.path.abspath(secondary_path) if secondary_path else None
        paths = [p for p in [s_path, p_path] if p and os.path.exists(p)]
        
        for path in paths:
            for filename in os.listdir(path):
                f_path = os.path.join(path, filename)
                if os.path.isdir(f_path): continue
                content = self._parse_any_file(f_path)
                if content:
                    self.documents.append(content[:10000])
                    self.filenames.append(filename)
        
        if self.documents:
            print(f"   ∟ [VECTOR] Syncing {len(self.documents)} documents to local Qdrant...")
            
            points = []
            for i, doc_text in enumerate(self.documents):
                embedding = self._get_embedding(doc_text)
                points.append(PointStruct(
                    id=self._generate_uuid(f"{self.filenames[i]}_{i}"),
                    vector=embedding,
                    payload={"text": doc_text, "filename": self.filenames[i], "clearance_level": clearance_level}
                ))
            
            self.qdrant.upsert(collection_name=self.collection_name, points=points)
            print("   ∟ [VECTOR] Local Qdrant index synchronized successfully.")

    # =========================================================================
    # UPGRADED RETRIEVAL: QDRANT + ONNX INT8 RERANKING
    # =========================================================================
    def search(self, query: str, top_k: int = 5, user_clearance: list = ["public", "tier_1"]):
        """
        Stage 1: Fast Qdrant Cosine Search (OpenAI embeddings)
        Stage 2: Strict ONNX INT8 Cross-Encoder Reranking
        """
        try:
            # 1. Get OpenAI embedding for the query
            query_vector = self._get_embedding(query)
            
            # 2. Pre-Retrieval RBAC Filter for Qdrant
            rbac_filter = Filter(
                must=[FieldCondition(key="clearance_level", match=MatchAny(any=user_clearance))]
            )
            
            # 3. Pull Top 15 broad candidates from Qdrant
            search_result = self.qdrant.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=rbac_filter,
                limit=15
            )
            
            docs = [hit.payload['text'] for hit in search_result if 'text' in hit.payload]
            if not docs: return "No semantic match found."
                
            print(f"   ∟ [VECTOR] Reranking {len(docs)} chunks via ONNX INT8 Engine...")
            
            # 4. ONNX INT8 Reranker Logit Prediction
            inputs = self.tokenizer([query] * len(docs), docs, padding=True, truncation=True, return_tensors="pt")
            logits = self.reranker(**inputs).logits.squeeze(-1).detach().numpy()
            
            probs = 1 / (1 + np.exp(-logits))
            probs = probs.tolist() if isinstance(probs, np.ndarray) else [probs]
            
            scored_chunks = sorted(zip(probs, docs), key=lambda x: x[0], reverse=True)
            
            PRIMARY_THRESHOLD = 0.35
            passed_chunks = [(score, doc) for score, doc in scored_chunks if score >= PRIMARY_THRESHOLD]
            
            if passed_chunks:
                ranked_results = passed_chunks
            else:
                print(f"   ∟ [VECTOR] All chunks failed threshold ({PRIMARY_THRESHOLD}). Fallback active.")
                ranked_results = scored_chunks[:3]
            
            final_docs = [doc for score, doc in ranked_results[:top_k]]
            print(f"   ∟ [VECTOR] ONNX Reranking complete. Surfaced top {len(final_docs)} chunks securely.")
            
            return "\n\n---\n\n".join(final_docs)
            
        except Exception as e:
            print(f"Qdrant vector search failed: {e}")
            return "No semantic match found."
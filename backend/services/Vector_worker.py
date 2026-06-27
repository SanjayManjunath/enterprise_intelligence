import os
import json
import base64
import pandas as pd
import numpy as np
import whisper
from odf import text, teletype
from odf.opendocument import load
from sentence_transformers import SentenceTransformer, CrossEncoder
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from openai import OpenAI
from dotenv import load_dotenv
from langchain_text_splitters import MarkdownHeaderTextSplitter

load_dotenv()

class VectorWorker:
    def __init__(self, logs_path="../storage/vector_indices/project_logs"):
        """
        Principal DS Logic: Phase 2 Hybrid Search & Reranking Engine.
        Combines BM25 Sparse Vectors with Dense Embeddings and a local Cross-Encoder.
        """
        print("   ∟ [VECTOR] Booting Local Embedding Models...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # --- PHASE 2 UPGRADE: Local Cross-Encoder & BM25 ---
        print("   ∟ [VECTOR] Loading BAAI/bge-reranker-base into memory...")
        self.reranker = CrossEncoder('BAAI/bge-reranker-base')
        self.bm25 = BM25Encoder.default()
        
        try:
            self.speech_model = whisper.load_model("base")
        except Exception as e:
            print(f"⚠️ Whisper init failed: {str(e)}")
            self.speech_model = None
            
        groq_key = os.environ.get("GROQ_API_KEY")
        if groq_key:
            self.groq_client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=groq_key)
        else:
            self.groq_client = None
        
        pinecone_key = os.environ.get("PINECONE_API_KEY")
        if not pinecone_key:
            raise ValueError("❌ PINECONE_API_KEY is missing from the .env file.")
            
        self.pc = Pinecone(api_key=pinecone_key)
        self.index_name = "enterprise-ai"
        
        # --- PHASE 2 UPGRADE: Metric changed to 'dotproduct' for Hybrid Search ---
        active_indexes = [index.name for index in self.pc.list_indexes()]
        if self.index_name not in active_indexes:
            print(f"   ∟ [VECTOR] Provisioning new Hybrid Pinecone index: '{self.index_name}'...")
            self.pc.create_index(
                name=self.index_name,
                dimension=384,
                metric="dotproduct", 
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            
        self.index = self.pc.Index(self.index_name)
        self.abs_logs_path = os.path.abspath(logs_path)
        self.refresh_index(self.abs_logs_path)

    def sync_csv_to_cloud(self, file_path: str):
        fname = os.path.basename(file_path)
        print(f"   ∟ [VECTOR] Syncing CSV schema for {fname} to cloud...")
        try:
            df = pd.read_csv(file_path)
            col_list = list(df.columns)
            content = f"SOURCE: {fname} (Relational Data) | SCHEMA_SIGNATURE: {col_list}"
            
            # Hybrid Encoding
            embedding = self.model.encode(content).tolist()
            sparse_dict = self.bm25.encode_documents(content)
            
            self.index.upsert(vectors=[{
                "id": f"{fname}_schema",
                "values": embedding,
                "sparse_values": sparse_dict,
                "metadata": {"text": content, "filename": fname, "type": "structured_csv", "clearance_level": "public"}
            }])
            print(f"   ∟ [VECTOR] Synced CSV schema to Pinecone.")
        except Exception as e:
            print(f"   ∟ [VECTOR] Failed to sync CSV schema: {str(e)}")

    def sync_markdown_to_cloud(self, markdown_text: str, source_name: str):
        print(f"   ∟ [VECTOR] Processing Hybrid Markdown from {source_name}...")
        
        headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        md_header_splits = markdown_splitter.split_text(markdown_text)
        
        cloud_vectors = []
        for i, doc in enumerate(md_header_splits):
            clean_chunk = doc.page_content.strip()
            if not clean_chunk:
                continue
                
            vector_id = f"{source_name}_md_{i}"
            
            # Phase 2: Dual Encoding (Dense + Sparse)
            embedding = self.model.encode(clean_chunk).tolist()
            sparse_dict = self.bm25.encode_documents(clean_chunk)
            
            # Added RBAC clearance level metadata
            metadata = {"text": clean_chunk, "source": source_name, "type": "structured_pdf", "clearance_level": "public"}
            metadata.update(doc.metadata) 
            
            cloud_vectors.append({
                "id": vector_id,
                "values": embedding,
                "sparse_values": sparse_dict,
                "metadata": metadata
            })
            
        if cloud_vectors:
            batch_size = 100
            for i in range(0, len(cloud_vectors), batch_size):
                self.index.upsert(vectors=cloud_vectors[i:i + batch_size])
            print(f"   ∟ [VECTOR] Synced {len(cloud_vectors)} Hybrid chunks to Pinecone.")

    def _extract_image_context(self, file_path):
        if not self.groq_client: return ""
        try:
            with open(file_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            ext = os.path.splitext(file_path)[1].lower()
            mime_type = "image/png" if ext == ".png" else "image/jpeg"
            response = self.groq_client.chat.completions.create(
                model="llama-3.2-11b-vision-preview",
                messages=[{"role": "user", "content": [{"type": "text", "text": "Analyze this image. Extract all visible text, numbers, and describe any charts explicitly."}, {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded_string}"}}]}]
            )
            return response.choices[0].message.content
        except Exception: return ""

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
            elif ext in [".mp3", ".wav"] and self.speech_model:
                return f"SOURCE: {fname} | AUDIO_TRANSCRIPT: {self.speech_model.transcribe(file_path)['text']}"
            elif ext in [".png", ".jpg", ".jpeg"]:
                return f"SOURCE: {fname} | VISUAL_EXTRACTION: {self._extract_image_context(file_path)}"
            return ""
        except Exception: return ""

    def refresh_index(self, primary_path, secondary_path=None):
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
            print(f"   ∟ [VECTOR] Syncing {len(self.documents)} Hybrid documents to Pinecone cloud...")
            embeddings = self.model.encode(self.documents)
            
            cloud_vectors = []
            for i, doc_text in enumerate(self.documents):
                sparse_dict = self.bm25.encode_documents(doc_text)
                cloud_vectors.append({
                    "id": f"{self.filenames[i]}_{i}",
                    "values": embeddings[i].tolist(),
                    "sparse_values": sparse_dict,
                    # Added RBAC clearance level metadata
                    "metadata": {"text": doc_text, "filename": self.filenames[i], "clearance_level": "public"}
                })
            
            batch_size = 100
            for i in range(0, len(cloud_vectors), batch_size):
                self.index.upsert(vectors=cloud_vectors[i:i+batch_size])
            print("   ∟ [VECTOR] Cloud index synchronized successfully.")

    # =========================================================================
    # PHASE 2: CROSS-ENCODER RERANKING ARCHITECTURE
    # =========================================================================
    def search(self, query, top_k=5, clearance_levels=None):
        """
        Executes a 2-stage retrieval pipeline with RBAC and Two-Stage Guardrails.
        """
        alpha = 0.5 
        dense_vec = self.model.encode([query]).astype('float32').tolist()[0]
        sparse_vec = self.bm25.encode_queries(query)
        
        hdense = [v * alpha for v in dense_vec]
        hsparse = {"indices": sparse_vec["indices"], "values": [v * (1 - alpha) for v in sparse_vec["values"]]}
        
        # Build query arguments dynamically based on RBAC clearance
        query_kwargs = {
            "vector": hdense,
            "sparse_vector": hsparse,
            "top_k": 15,
            "include_metadata": True
        }
        if clearance_levels is not None:
            query_kwargs["filter"] = {"clearance_level": {"$in": clearance_levels}}
        
        try:
            res = self.index.query(**query_kwargs)
            
            if not res.get('matches'):
                return "No semantic match found."
                
            docs = [match['metadata']['text'] for match in res['matches'] if 'metadata' in match and 'text' in match['metadata']]
            if not docs:
                return "No semantic match found."
                
            print(f"   ∟ [VECTOR] Reranking {len(docs)} broad candidate chunks...")
            pairs = [[query, doc] for doc in docs]
            raw_scores = self.reranker.predict(pairs)
            
            # Convert raw logits to 0-1 probability scores using Sigmoid
            probs = [1 / (1 + np.exp(-score)) for score in raw_scores]
            
            # Zip documents with their probability scores
            scored_chunks = sorted(zip(probs, docs), key=lambda x: x[0], reverse=True)
            
            # Apply Industry Standard 0.35 Threshold
            PRIMARY_THRESHOLD = 0.35
            passed_chunks = [(score, doc) for score, doc in scored_chunks if score >= PRIMARY_THRESHOLD]
            
            ranked_results = []
            if passed_chunks:
                print(f"   ∟ [VECTOR] {len(passed_chunks)} chunks passed the threshold (> {PRIMARY_THRESHOLD}).")
                ranked_results = passed_chunks
            else:
                # UNIVERSAL SELF-HEALING OVERRIDE
                print(f"   ∟ [VECTOR] All chunks failed the {PRIMARY_THRESHOLD} threshold. Engaging fallback (taking top 3).")
                ranked_results = scored_chunks[:3]
            
            final_docs = [doc for score, doc in ranked_results[:top_k]]
            print(f"   ∟ [VECTOR] Reranking complete. Forwarding top {len(final_docs)} chunks to Supervisor.")
            
            return "\n\n---\n\n".join(final_docs)
            
        except Exception as e:
            print(f"Pinecone hybrid search failed: {e}")
            return "No semantic match found."
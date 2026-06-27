import os
import time
import uuid
import json
import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset
from dotenv import load_dotenv

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

load_dotenv()

class MarcoEvaluator:
    def __init__(self, target_doc_count=500000):
        self.target_doc_count = target_doc_count
        self.collection_name = "eval_ms_marco"
        
        openai_key = os.environ.get("OPENAI_API_KEY")
        if not openai_key: raise ValueError("❌ OPENAI_API_KEY missing.")
        self.openai_client = OpenAI(api_key=openai_key)
        
        self.qdrant = QdrantClient(url="http://qdrant-db:6333")
        
        print("\n[EVAL] Loading ONNX INT8 Reranker into RAM...")
        model_id = "Xenova/bge-reranker-base"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.reranker = ORTModelForSequenceClassification.from_pretrained(model_id, file_name="onnx/model_quantized.onnx")

        if not self.qdrant.collection_exists(collection_name=self.collection_name):
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
            )

    def _get_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        response = self.openai_client.embeddings.create(input=texts, model="text-embedding-3-small")
        return [data.embedding for data in response.data]

    def ingest_ms_marco(self):
        print(f"\n[EVAL] Downloading MS MARCO Dataset (Target: {self.target_doc_count} passages)...")
        dataset = load_dataset("ms_marco", "v1.1", split="train", streaming=True)
        
        unique_passages = {}
        queries_with_answers = []
        
        for row in dataset:
            query = row['query']
            passages = row['passages']
            correct_passage_idx = next((i for i, is_sel in enumerate(passages['is_selected']) if is_sel == 1), None)
            
            if correct_passage_idx is not None:
                correct_text = passages['passage_text'][correct_passage_idx]
                doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, correct_text))
                unique_passages[doc_id] = correct_text
                queries_with_answers.append({"query": query, "correct_doc_id": doc_id})
                
                for idx, text in enumerate(passages['passage_text']):
                    if idx != correct_passage_idx:
                        distractor_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, text))
                        unique_passages[distractor_id] = text
                        
            if len(unique_passages) >= self.target_doc_count: break

        passage_ids = list(unique_passages.keys())
        passage_texts = list(unique_passages.values())
        
        collection_info = self.qdrant.get_collection(self.collection_name)
        if collection_info.points_count >= self.target_doc_count:
            print(f"\n[EVAL] Qdrant already contains {collection_info.points_count} vectors. Skipping ingestion.")
            return queries_with_answers
        elif collection_info.points_count > 0:
            print(f"\n[EVAL] Target is {self.target_doc_count}, but Qdrant only has {collection_info.points_count}. Wiping and re-ingesting clean batch...")
            self.qdrant.delete_collection(collection_name=self.collection_name)
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
            )
            
        print(f"\n[EVAL] Ingesting {len(passage_texts)} documents into Qdrant...")
        batch_size = 1000 
        for i in tqdm(range(0, len(passage_texts), batch_size), desc="Embedding & Upserting"):
            batch_texts = passage_texts[i:i+batch_size]
            batch_ids = passage_ids[i:i+batch_size]
            embeddings = self._get_embeddings_batch(batch_texts)
            points = [
                PointStruct(id=b_id, vector=emb, payload={"text": text, "doc_id": b_id})
                for b_id, emb, text in zip(batch_ids, embeddings, batch_texts)
            ]
            self.qdrant.upsert(collection_name=self.collection_name, points=points)
            
        return queries_with_answers

    def _evaluate_llm_quality(self, query: str, retrieved_context: str):
        answer_res = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Answer the user's question based strictly on the provided context. If the answer is not in the context, say 'I don't know'."},
                {"role": "user", "content": f"Context: {retrieved_context}\n\nQuestion: {query}"}
            ],
            temperature=0.0
        )
        generated_answer = answer_res.choices[0].message.content

        judge_prompt = f"""
        Evaluate the following RAG system output. 
        Question: {query}
        Retrieved Context: {retrieved_context}
        Generated Answer: {generated_answer}
        
        Score the following two metrics from 0.0 to 1.0:
        1. Faithfulness: Is the generated answer completely derived from the retrieved context without hallucinating outside facts? (1.0 = perfect, 0.0 = total hallucination)
        2. Relevance: Does the generated answer directly address the user's question? (1.0 = perfect answer, 0.0 = irrelevant)
        
        Output ONLY valid JSON in this exact format: {{"faithfulness": 1.0, "relevance": 1.0}}
        """
        try:
            grade_res = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                response_format={ "type": "json_object" },
                messages=[{"role": "system", "content": judge_prompt}],
                temperature=0.0
            )
            scores = json.loads(grade_res.choices[0].message.content)
            return scores.get("faithfulness", 0.0), scores.get("relevance", 0.0)
        except Exception: return 0.0, 0.0

    def evaluate_retrieval(self, test_queries, top_k=10, run_llm_eval_on_first_n=50):
        print(f"\n[EVAL] Commencing Retrieval Evaluation on {len(test_queries)} queries...")
        
        mrr_score = 0.0
        recall_at_10_count = 0
        latencies = []
        faithfulness_scores = []
        relevance_scores = []
        
        for idx, item in enumerate(tqdm(test_queries, desc="Evaluating Queries")):
            query = item["query"]
            target_id = item["correct_doc_id"]
            
            start_time = time.time()
            query_vector = self._get_embeddings_batch([query])[0]
            
            # 🎯 FIX: Drop limit from 50 to 15 to rescue CPU latency
            search_result = self.qdrant.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=15
            )
            
            docs = [hit.payload['text'] for hit in search_result]
            doc_ids = [hit.payload['doc_id'] for hit in search_result]
            
            if docs:
                inputs = self.tokenizer([query] * len(docs), docs, padding=True, truncation=True, return_tensors="pt")
                logits = self.reranker(**inputs).logits.squeeze(-1).detach().numpy()
                probs = 1 / (1 + np.exp(-logits))
                probs = probs.tolist() if isinstance(probs, np.ndarray) else [probs]
                
                scored_chunks = sorted(zip(probs, doc_ids, docs), key=lambda x: x[0], reverse=True)
                final_ranked_ids = [chunk[1] for chunk in scored_chunks[:top_k]]
                best_context = "\n---\n".join([chunk[2] for chunk in scored_chunks[:3]])
            else:
                final_ranked_ids = []
                best_context = ""
                
            latencies.append(time.time() - start_time)
            
            if target_id in final_ranked_ids: recall_at_10_count += 1
                
            try:
                rank = final_ranked_ids.index(target_id) + 1
                mrr_score += (1.0 / rank)
            except ValueError: pass 

            if idx < run_llm_eval_on_first_n:
                f_score, r_score = self._evaluate_llm_quality(query, best_context)
                faithfulness_scores.append(f_score)
                relevance_scores.append(r_score)

        total = len(test_queries)
        final_mrr = mrr_score / total
        final_recall = recall_at_10_count / total
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        
        final_faithfulness = np.mean(faithfulness_scores) if faithfulness_scores else 0.0
        final_relevance = np.mean(relevance_scores) if relevance_scores else 0.0
        
        print("\n" + "="*60)
        print("🎯 ENTERPRISE RAG EVALUATION REPORT")
        print("="*60)
        print(f"Total Documents Indexed : {self.target_doc_count}")
        print(f"Retrieval Queries Run   : {total}")
        print(f"LLM-as-a-Judge Run      : {len(faithfulness_scores)}")
        print("-" * 60)
        print("🔍 STAGE 1 & 2 RETRIEVAL METRICS (Algorithm Accuracy)")
        print(f"MRR@{top_k}             : {final_mrr:.4f} (Industry benchmark > 0.60)")
        print(f"Recall@{top_k}          : {final_recall:.4f} (Industry benchmark > 0.85)")
        print("-" * 60)
        print("🧠 GENERATION METRICS (LLM-as-a-Judge)")
        print(f"Faithfulness       : {final_faithfulness:.4f} (No Hallucinations)")
        print(f"Answer Relevance   : {final_relevance:.4f} (Addresses Query)")
        print("-" * 60)
        print(f"Avg ONNX Latency/Query  : {avg_latency:.3f} seconds")
        print(f"P95 ONNX Latency        : {p95_latency:.3f} seconds")
        print("="*60)

if __name__ == "__main__":
    evaluator = MarcoEvaluator(target_doc_count=500000)
    test_queries = evaluator.ingest_ms_marco()
    sample_queries = np.random.choice(test_queries, size=500, replace=False).tolist()
    evaluator.evaluate_retrieval(sample_queries, top_k=10, run_llm_eval_on_first_n=50)
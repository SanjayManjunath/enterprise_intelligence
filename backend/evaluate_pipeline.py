import os
import asyncio
import pandas as pd
import numpy as np
import json
from dotenv import find_dotenv, load_dotenv
from openai import AsyncOpenAI
import time

from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness, AnswerRelevancy
from ragas.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage

from core_agent import EnterpriseBrain
from services.Vector_worker import VectorWorker  # <-- NEW: Import the VectorWorker

# Force system to load environment keys from the current directory
load_dotenv(find_dotenv())

# --- HYBRID METRIC FUNCTIONS ---
def calculate_mrr(retrieved_items, relevant_items):
    if not relevant_items: return 1.0 
    for i, item in enumerate(retrieved_items):
        if any(rel.lower() in item.lower() for rel in relevant_items):
            return 1.0 / (i + 1)
    return 0.0

def calculate_recall_at_k(retrieved_items, relevant_items, k=5):
    if not relevant_items: return 1.0 
    top_k_retrieved = retrieved_items[:k]
    hits = sum(1 for rel in relevant_items if any(rel.lower() in ret.lower() for ret in top_k_retrieved))
    return hits / len(relevant_items) if relevant_items else 0.0

# --- LOAD GOLDEN DATASET ---
try:
    with open('accuracy_suite.json', 'r') as file:
        EVALUATION_TEST_SUITE = json.load(file)
    print(f"✅ Golden Dataset loaded successfully with {len(EVALUATION_TEST_SUITE)} test cases.")
except FileNotFoundError:
    print("❌ Error: 'golden_dataset.json' not found. Please ensure the file is in the same directory.")
    exit(1)

# NEW: Pass shared_vector_engine into the worker thread
async def process_single_case(test_case, idx, found_db, sem, shared_vector_engine):
    """Worker function to process a single query asynchronously via Threading."""
    async with sem:
        thread_id = f"validation_sim_thread_{idx}_{int(time.time())}"
        question = test_case.get("question", "")
        ground_truth = test_case.get("ground_truth", "")
        category = test_case.get("category", test_case.get("intent_type", "unknown"))
        relevant_targets = test_case.get("relevant_targets", [])
        
        print(f"   ∟ Processing Query: '{question[:45]}...' [Thread: {idx}]")
        
        # ISOLATION FIX: Instantiate a fresh brain, but inject the SINGLETON ML worker
        isolated_brain = EnterpriseBrain(shared_vector_worker=shared_vector_engine)
        isolated_brain.update_context(found_db, "../storage/vector_indices/project_logs")
        
        initial_state = {
            "messages": [HumanMessage(content=question)],
            "question": question,
            "active_documents": [os.path.basename(found_db)], 
            "retries": 0,
            "is_eval": False  # FIX: Set to False so the agent generates the exact production JSON (tables/charts included)
        }
        
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            # Execute LangGraph via the isolated brain
            runtime_state = await asyncio.to_thread(isolated_brain.graph.invoke, initial_state, config)
            raw_output = runtime_state.get("final_output", "")
            
            # --- THE RAGAS JSON INTERCEPTOR ---
            # Unpack the JSON and feed ONLY the natural language summary to the Judge.
            try:
                import json
                parsed_json = json.loads(raw_output)
                # Ensure this key matches exactly what your presentation_node outputs!
                generated_answer = parsed_json.get("executive_summary", raw_output)
            except Exception:
                # Fallback in case the LLM hallucinates plain text instead of JSON
                generated_answer = raw_output
            
            retrieved_contexts = []
            tools_used = []
            for msg in runtime_state.get("messages", []):
                if getattr(msg, "tool_calls", None):
                    tools_used.extend([t["name"] for t in msg.tool_calls])
                if msg.type == "tool":
                    retrieved_contexts.append(str(msg.content))
                    
            if not retrieved_contexts:
                retrieved_contexts = ["No tool context was retrieved during execution."]
                
            mrr_score = 0.0
            recall_score = 0.0
            
            if "Vector" in category or "Semantic" in category:
                mrr_score = calculate_mrr(retrieved_contexts, relevant_targets)
                recall_score = calculate_recall_at_k(retrieved_contexts, relevant_targets, k=5)
            else:
                sql_success = 1 if not any("Error:" in ctx for ctx in retrieved_contexts) else 0
                mrr_score = sql_success 
                recall_score = sql_success

            return {
                "idx": idx,
                "category": category,
                "user_input": question,
                "response": generated_answer,
                "retrieved_contexts": retrieved_contexts,
                "reference": ground_truth,
                "tools_triggered": ", ".join(tools_used) if tools_used else "None",
                "mrr": mrr_score,
                "recall_at_k": recall_score
            }
            
        except Exception as err:
            print(f"⚠️ Simulation failure on record {idx}: {str(err)}")
            return None

async def run_production_validation_async():
    print("🚀 Initializing Hybrid Operational Validation Sandbox...")

    print("🔍 Scanning storage/user_uploads for the live data context...")
    user_uploads_dir = os.path.abspath("../storage/user_uploads")
    if not os.path.exists(user_uploads_dir):
        user_uploads_dir = os.path.abspath("./storage/user_uploads")
        
    found_db = None
    if os.path.exists(user_uploads_dir):
        for root, dirs, files in os.walk(user_uploads_dir):
            for file in files:
                if file.endswith((".db", ".csv")) and not any(k in file for k in ["persistence", "cache", "semantic"]):
                    found_db = os.path.abspath(os.path.join(root, file))
                    break
            if found_db: break

    if not found_db:
        found_db = "/storage/enterprise_data_large/enterprise_erp_large.db"
            
    print(f"✅ Active Context Mapped Successfully: {os.path.basename(found_db)}")

    # NEW: Boot the ML models ONCE globally to prevent HuggingFace deadlocks
    print("🧠 Booting Shared ML Engine (Singleton)...")
    shared_vector_engine = VectorWorker()

    print("⚙️ Harnessing native OpenAI engine for Ragas Generation Judges...")
    openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    ragas_evaluator_llm = llm_factory('gpt-4o-mini', client=openai_client)
    ragas_embeddings = HuggingFaceEmbeddings(model="all-MiniLM-L6-v2")
    
    faithfulness_metric = Faithfulness(llm=ragas_evaluator_llm)
    answer_relevancy_metric = AnswerRelevancy(llm=ragas_evaluator_llm, embeddings=ragas_embeddings)
    
    print(f"🧬 Executing async batch simulation across {len(EVALUATION_TEST_SUITE)} hybrid test records...")
    
    concurrency_limit = 1
    sem = asyncio.Semaphore(concurrency_limit)
    
    # NEW: Pass the shared ML engine into the tasks
    tasks = [
        process_single_case(test_case, idx, found_db, sem, shared_vector_engine) 
        for idx, test_case in enumerate(EVALUATION_TEST_SUITE)
    ]
    
    results = await asyncio.gather(*tasks)
    evaluation_records = [res for res in results if res is not None]

    if not evaluation_records:
        print("❌ Critical Error: No successful evaluation traces gathered. Halting process.")
        return
    
    print("\n⚖️ Commencing RAGAS LLM-as-a-Judge generation scoring...")
    scored_records = []
    
    for record in evaluation_records:
        f_score = 0.0
        ar_score = 0.0
        idx = record["idx"]
        
        try:
            f_score = await faithfulness_metric.ascore(
                user_input=record["user_input"],
                response=record["response"],
                retrieved_contexts=record["retrieved_contexts"]
            )
        except Exception as fe:
            print(f"   ⚠️ Faithfulness bypass row {idx}: {str(fe)}")
            
        try:
            ar_score = await answer_relevancy_metric.ascore(
                user_input=record["user_input"],
                response=record["response"]
            )
        except Exception as are:
            print(f"   ⚠️ Answer Relevancy bypass row {idx}: {str(are)}")
            
        record["faithfulness"] = f_score
        record["answer_relevancy"] = ar_score
        scored_records.append(record)

    report_dataframe = pd.DataFrame(scored_records)
    
    output_df = report_dataframe.drop(columns=['retrieved_contexts', 'idx'])
    output_csv_path = "enterprise_validation_report.csv"
    output_df.to_csv(output_csv_path, index=False)
    
    print("\n==========================================================")
    print("📊 HYBRID ARCHITECTURE VALIDATION MATRIX COMPLETE")
    print("==========================================================")
    print(f"Mean MRR (Vector/SQL)      : {report_dataframe['mrr'].mean():.4f}")
    print(f"Mean Recall@K (Vector/SQL) : {report_dataframe['recall_at_k'].mean():.4f}")
    print(f"Mean Faithfulness Score    : {report_dataframe['faithfulness'].mean():.4f}")
    print(f"Mean Answer Relevancy      : {report_dataframe['answer_relevancy'].mean():.4f}")
    print(f"👉 Complete evaluation summary saved to: '{output_csv_path}'")
    print("==========================================================")

def main():
    asyncio.run(run_production_validation_async())

if __name__ == "__main__":
    main()

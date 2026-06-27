import os
import time
import logging
import re
import json
import redis
from collections import defaultdict
from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core_agent import EnterpriseBrain
from services.ingestion_gateway import IngestionGateway

# --- STEP 4: CELERY DECOUPLED EXECUTION IMPORTS ---
from celery_app import celery_engine
from tasks import execute_agent_audit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Enterprise AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================================
# TIER 1: SYSTEM-WIDE SHARED REDIS L1 BROKER
# =====================================================================
try:
    # decode_responses=True auto-converts Redis raw bytes into clean Python strings
    redis_pool = redis.ConnectionPool(host='redis-broker', port=6379, db=0, decode_responses=True)
    redis_client = redis.Redis(connection_pool=redis_pool)
    redis_client.ping()
    logger.info("⚡ System-Wide Redis L1 Broker successfully attached to worker PID.")
except Exception as e:
    logger.warning(f"Redis L1 Broker unreachable. Running degraded: {e}")
    redis_client = None

# =====================================================================
# NATIVE IN-MEMORY TOKEN SHIELD & IP RATE LIMITER
# =====================================================================
ip_request_history = defaultdict(list)
GLOBAL_TOKEN_LIMIT = 25000
global_token_history = []

def enforce_security_shield(client_ip: str, question: str):
    # 🎟️ VIP IMMUNITY PASSPORT FOR LOAD TESTING
    if client_ip in ["127.0.0.1", "localhost"] or client_ip.startswith("172."):
        return

    now = time.time()
    
    # 1. Rollback expired timestamps
    global global_token_history
    global_token_history = [run for run in global_token_history if now - run[0] <= 60]
    ip_request_history[client_ip] = [t for t in ip_request_history[client_ip] if now - t <= 900]

    # 2. Per-IP Limit Check (Max 10 calls / 15 mins)
    if len(ip_request_history[client_ip]) >= 10:
        oldest_request = ip_request_history[client_ip][0]
        remaining_minutes = max(1, int(900 - (now - oldest_request)) // 60)
        logger.warning(f"🔒 Rate limit blocked IP: {client_ip}. Window resets in {remaining_minutes}m.")
        raise HTTPException(
            status_code=429, 
            detail=f"Rate Limit Exceeded: You can perform 10 database audits every 15 minutes. Please wait {remaining_minutes} minute(s)."
        )

    # 3. Global Token Fuse Check
    estimated_tokens = int(len(question.split()) * 1.3) + 3000
    current_global_tokens = sum(run[1] for run in global_token_history)
    
    if current_global_tokens + estimated_tokens > GLOBAL_TOKEN_LIMIT:
        logger.error(f"🚨 Global Token Fuse Tripped! Volumetric limit reached: {current_global_tokens} tokens/min.")
        raise HTTPException(
            status_code=429,
            detail="Cloud Compute Cooldown Active: Global API throughput threshold reached. Retry in 1 minute."
        )

    ip_request_history[client_ip].append(now)
    global_token_history.append((now, estimated_tokens))

# =====================================================================

STORAGE_BASE = "/storage/user_uploads" 
os.makedirs(STORAGE_BASE, exist_ok=True)

brain = EnterpriseBrain(
    db_path="/storage/enterprise_data_large/enterprise_erp_large.db",
    logs_path="/storage/vector_indices/project_logs"
)
gateway = IngestionGateway()

class ChatRequest(BaseModel):
    question: str
    thread_id: str
    clearance_level: str = "public"

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    clearance_level: str = Form("public")
):
    try:
        file_bytes = await file.read()
        gateway.process_upload(file_bytes, file.filename, clearance_level)
        
        session_dir = os.path.join(STORAGE_BASE, session_id)
        os.makedirs(session_dir, exist_ok=True)
        local_path = os.path.join(session_dir, file.filename)
        
        with open(local_path, "wb") as f:
            f.write(file_bytes)
            
        return {"status": "success", "message": f"Ingested {file.filename} into global and session context."}
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def find_dataset_by_intent(question: str, base_dir: str):
    clean_q = re.sub(r'[^\w\s]', '', question).lower()
    words = clean_q.split()
    stop_words = {"the", "data", "dataset", "table", "file", "csv", "db", "analyze", "give", "me", "show"}
    keywords = [w for w in words if w not in stop_words and len(w) > 3]

    if not keywords:
        return None

    best_match = None
    for root, _, filenames in os.walk(base_dir):
        for f in filenames:
            if f.endswith(('.db', '.csv')):
                f_lower = f.lower()
                if any(kw in f_lower for kw in keywords):
                    best_match = os.path.join(root, f)
                    break
        if best_match:
            break
            
    return best_match

@app.post("/chat")
async def chat_endpoint(request: ChatRequest, fastapi_req: Request):
    try:
        # 0. --- TIER 1: CLUSTER-WIDE REDIS INTERCEPT (<1ms) ---
        cache_key = f"l1:{request.thread_id}:{request.question.strip().lower()}"
        
        if redis_client:
            try:
                cached_raw = redis_client.get(cache_key)
                if cached_raw:
                    logger.info(f"⚡ CLUSTER L1 REDIS HIT! Served PID {os.getpid()} instantly across the network.")
                    return json.loads(cached_raw)
            except Exception as e:
                logger.warning(f"Redis read skipped: {e}")

        # 0.5 --- RE-ENGAGE SECURITY SHIELD ---
        client_ip = fastapi_req.headers.get("x-forwarded-for") or fastapi_req.client.host or "127.0.0.1"
        client_ip = client_ip.split(",")[0].strip()
        enforce_security_shield(client_ip, request.question)

        # 1. DYNAMIC CONTEXT
        session_folder = os.path.join(STORAGE_BASE, request.thread_id)
        active_db = None
        
        if os.path.exists(session_folder):
            files = [f for f in os.listdir(session_folder) if f.endswith(('.db', '.csv'))]
            if files:
                active_db = os.path.join(session_folder, files[0])
        
        # 2. SEMANTIC INTENT DISCOVERY
        if not active_db:
            semantic_match = find_dataset_by_intent(request.question, STORAGE_BASE)
            if semantic_match:
                active_db = semantic_match
                logger.info(f"Semantic Match Found: {active_db}")
        
        # 3. ABSOLUTE FALLBACK
        if not active_db:
            active_db = "/storage/enterprise_data_large/enterprise_erp_large.db"
        
        # 4. PIVOT SQL WORKER
        brain.sql_worker.update_db_path(active_db)

        # 5. UPDATE BRAIN CONTEXT
        brain.update_context(
            db_path=active_db,
            logs_path="/storage/vector_indices/project_logs",
            extra_folder=session_folder if os.path.exists(session_folder) else None
        )
            
        answer = brain.run(question=request.question, thread_id=request.thread_id)
        response_payload = {"response": answer, "thread_id": request.thread_id}
        
        # --- COMMIT RESULT TO SYSTEM-WIDE REDIS BROKER ---
        if redis_client:
            try:
                # Caches for 6 hours (21,600 seconds)
                redis_client.setex(cache_key, 21600, json.dumps(response_payload))
            except Exception as e:
                logger.warning(f"Redis write skipped: {e}")
        
        return response_payload
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Chat failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# =====================================================================
# TIER 1: DECOUPLED ASYNC INFERENCE PLANE (FOR LOAD TESTING)
# =====================================================================
@app.post("/api/v1/audit/async")
async def audit_async_entrypoint(request: ChatRequest, fastapi_req: Request):
    """Non-blocking ingestion route. Instantly queues heavy ML tasks to Celery."""
    cache_key = f"l1:{request.thread_id}:{request.question.strip().lower()}"
    
    # 1. Check Redis Cluster Cache L1 (< 1ms)
    if redis_client:
        try:
            if cached := redis_client.get(cache_key):
                logger.info("⚡ ASYNC L1 REDIS HIT! Serving instantly from shared memory.")
                return {"status": "completed", "cached": True, "result": json.loads(cached)}
        except Exception: pass

    # 2. Re-engage IP Token Shield
    client_ip = (fastapi_req.headers.get("x-forwarded-for") or fastapi_req.client.host or "127.0.0.1").split(",")[0].strip()
    enforce_security_shield(client_ip, request.question)

    # 3. Drop onto Celery ML Queue instantly
    task = execute_agent_audit.delay(request.question, request.thread_id)
    logger.info(f"📥 Dropped audit request onto Celery ML Broker. Task ID assigned: {task.id}")
    
    return {
        "status": "processing", 
        "task_id": task.id, 
        "thread_id": request.thread_id,
        "message": "Audit successfully queued to asynchronous background ML worker."
    }

@app.get("/api/v1/audit/status/{task_id}")
@app.get("/v1/audit/status/{task_id}")
def get_audit_status(task_id: str):
    """Polling endpoint for UI or Locust load-test metrics."""
    result = celery_engine.AsyncResult(task_id)
    response = {"task_id": task_id, "status": result.status}
    
    if result.status == "SUCCESS": 
        response["result"] = result.result
    elif result.status == "FAILURE": 
        response["error"] = str(result.info)
        
    return response
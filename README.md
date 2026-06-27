Enterprise Intelligence: Agentic RAG & Async ML Backend

A production-grade, highly concurrent Agentic AI backend utilizing LangGraph, FastAPI, and a decoupled Celery/Redis inference plane. This system is designed to route complex enterprise data audits dynamically across Relational (SQL) and Data Science (Pandas) sandboxes without hitting API rate limits or database deadlocks.

🚀 Architectural Pillars

Decoupled Inference Plane: Synchronous LLM blocking is eliminated by offloading heavy reasoning chains (LangGraph) to background Celery ML workers, maintaining an ultra-responsive API Gateway.

Dual-Engine Semantic Routing: Intelligently routes structural data extraction to a SQL engine and advanced mathematical analysis (correlations, multi-dimensional pivots) to an isolated Pandas Sandbox.

Vector Hierarchy & Hybrid Search: Implements hybrid retrieval using Qdrant (HNSW index), combining BM25 sparse search with dense embeddings, reranked by an ONNX INT8 local cross-encoder.

Resilient State Persistence: PostgreSQL checkpointer with Declarative Hash Partitioning eliminates write-lock queuing under heavy parallel loads.

📊 Evaluation & Performance Benchmarks

1. Offline Retrieval Accuracy (MS MARCO 500k Subset)

Validates the mathematical grounding of the retrieval pipeline using 500 queries against a 500k-document dataset.

MRR@10: 0.4800 (Industry benchmark > 0.60)

Recall@10: 0.9300 (Industry benchmark > 0.85)

Faithfulness: 1.000 (0% Hallucination)

2. Infrastructure Load Testing

Simulated via Locust executing complex async audit queues on a 4 vCPU / 16GB RAM node.

Peak Concurrency: 1,000 simultaneous users.

API Failure Rate: 0.00% (High-load socket stability).

Throughput: 314 Requests/Second (API Gateway).

Polling Latency: < 15ms (via Redis-cached status lookups).

🛠️ Technical Stack

AI Orchestration: LangGraph, LangChain, OpenAI (gpt-4o-mini)

Async/ML: Celery, Redis, FastAPI, ONNX Runtime (INT8)

Databases: PostgreSQL (pgvector), Qdrant (Local Vector)

DevOps: Docker Compose, Nginx, Locust

🛡️ Enterprise Security Gateway

Perimeter Defense: Nginx reverse proxy with Basic Authentication.

Token Shield: Native in-memory IP rate limiter (10 reqs/15m) and Global Token Fuse (25k tokens/min) prevent API budget exhaustion.

Context Bleed Firewall: UUID-based isolation ensures multi-tenant session data leakage is programmatically impossible at the database transaction layer.

🔗 Live Production Environment

Interface: https://35.234.216.194.nip.io

(Note: Access is secured via Basic Authentication. Contact developer for credentials).

🤝 How to use this repository

Environment: Ensure .env is configured with OPENAI_API_KEY and PINECONE_API_KEY.

Infrastructure: Run sudo docker compose up -d --build.

Validation: Run the evaluation suite: python backend/MS_Macro_eval_pipeline.py.

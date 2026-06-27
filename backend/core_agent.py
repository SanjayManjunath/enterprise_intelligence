from dotenv import load_dotenv
load_dotenv()
import os
import json
import logging
import difflib
from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver
import time
import re
from typing import Annotated, TypedDict, List
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_community.callbacks.manager import get_openai_callback

# --- IRON RULE: Absolute Imports for Enterprise Pathing ---
from services.factory import get_llm
from services.SQL_worker import SQLWorker
from services.Vector_worker import VectorWorker
from services.python_worker import PythonWorker
from tools import build_enterprise_tools

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =====================================================================
# AGENT STATE (V2: MESSAGE-BASED ARCHITECTURE)
# =====================================================================
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    question: str
    active_documents: List[str]
    final_output: str
    retries: int
    is_eval: bool

# =====================================================================
# BUDGET GUARDRAIL (PHASE 4 CIRCUIT BREAKER)
# =====================================================================
class EnterpriseBudgetManager:
    def __init__(self, pool: ConnectionPool, max_budget_usd=2.00):
        self.pool = pool
        self.max_budget_usd = max_budget_usd
        # Create table using the Postgres connection pool
        with self.pool.connection() as conn:
            conn.execute('''CREATE TABLE IF NOT EXISTS budget_log
                             (timestamp REAL, prompt_tokens INT, completion_tokens INT, cost_usd REAL)''')

    def check_budget(self):
        with self.pool.connection() as conn:
            cursor = conn.execute("SELECT SUM(cost_usd) FROM budget_log")
            total_cost = cursor.fetchone()[0]
            total_cost = total_cost if total_cost else 0.0
            
            if total_cost >= self.max_budget_usd:
                logger.error(f"🚨 HARD CIRCUIT BREAKER TRIPPED! Cost: ${total_cost:.4f} / ${self.max_budget_usd:.2f}")
                raise Exception(f"BUDGET LIMIT EXCEEDED (${total_cost:.4f} / ${self.max_budget_usd:.2f}). System halted to prevent API charges.")
            return total_cost

    def log_usage(self, cb):
        if cb.total_tokens > 0:
            try:
                with self.pool.connection() as conn:
                    # Note the change to Postgres %s placeholders
                    conn.execute("INSERT INTO budget_log VALUES (%s, %s, %s, %s)",
                                      (time.time(), cb.prompt_tokens, cb.completion_tokens, cb.total_cost))
                    
                    cursor = conn.execute("SELECT SUM(cost_usd) FROM budget_log")
                    total_cost = cursor.fetchone()[0] or 0.0
                    logger.info(f"💸 Budget Tracker: Run Cost ${cb.total_cost:.4f} | Total Spend ${total_cost:.4f} / ${self.max_budget_usd:.2f}")
            except Exception as e:
                logger.error(f"Budget logger failed: {e}")

class EnterpriseSemanticCache:
    def __init__(self, pool, embedding_model=None): 
        self.pool = pool
        self.embedding_model = embedding_model
        # Tracks schema health in memory so we don't hit Postgres DDL on every turn
        self._schema_ensured = False

    def _ensure_schema(self):
        """Self-healing schema verification executed prior to any vector I/O."""
        if self._schema_ensured:
            return

        try:
            with self.pool.connection() as conn:
                # 1. Enable pgvector extension (Separate commit required by PostgreSQL DDL rules)
                try:
                    conn.execute('CREATE EXTENSION IF NOT EXISTS vector;')
                    conn.commit()
                except Exception as ext_err:
                    logger.warning(f"pgvector extension note: {ext_err}. (Assuming pre-enabled by DB superuser).")
                    conn.rollback()

                # 2. Provision table and HNSW vector index with explicit transaction commit
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS semantic_cache_v2 (
                        dataset_context TEXT, 
                        intent TEXT, 
                        embedding vector(384), 
                        final_output TEXT
                    )
                ''')
                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_cache_embedding ON semantic_cache_v2 
                    USING hnsw (embedding vector_cosine_ops)
                ''')
                conn.execute('''
                    CREATE UNIQUE INDEX IF NOT EXISTS idx_tenant_intent_v2 
                    ON semantic_cache_v2(dataset_context, intent)
                ''')
                conn.commit()
                self._schema_ensured = True
                logger.info("✅ pgvector semantic_cache_v2 table and HNSW index successfully verified and committed.")
        except Exception as e:
            logger.error(f"Critical failure provisioning pgvector schema: {e}")

    def _is_conversational(self, text: str) -> bool:
        clean_text = text.strip().lower()
        if len(clean_text.split()) <= 3: return True
        if clean_text in ["proceed", "yes", "no", "go ahead", "do it", "sure"]: return True
        return False

    def _is_valid_analytical_report(self, text: str) -> bool:
        if not text: return False
        bad_signatures = ["Best Guess:", "System Interrupt", "Data unavailable", "The required data could not be found"]
        if any(sig in text for sig in bad_signatures): return False
        return True

    def check_cache(self, dataset_context: str, current_intent: str, threshold: float = 0.95):
        if self._is_conversational(current_intent) or not self.embedding_model: 
            return None
        
        # 🛡️ Self-heal schema before executing lookups
        self._ensure_schema()
        
        try:
            query_embedding = self.embedding_model.encode(current_intent).tolist()
            embedding_str = str(query_embedding)
            
            with self.pool.connection() as conn:
                cursor = conn.execute("""
                    SELECT final_output, 1 - (embedding <=> %s::vector) AS similarity 
                    FROM semantic_cache_v2 
                    WHERE dataset_context = %s 
                    ORDER BY similarity DESC 
                    LIMIT 1
                """, (embedding_str, dataset_context))
                
                row = cursor.fetchone()
                if row and row[1] >= threshold:
                    logger.info(f"🧠 L2 pgvector Semantic Hit! Cosine Similarity: {row[1]:.4f}")
                    return row[0]
        except Exception as e:
            logger.error(f"L2 pgvector lookup failed: {e}")
            
        return None

    def save_to_cache(self, dataset_context: str, intent: str, final_output: str):
        if self._is_conversational(intent) or not self._is_valid_analytical_report(final_output) or not self.embedding_model: 
            return 
            
        # 🛡️ Self-heal schema before committing records
        self._ensure_schema()
        
        try:
            intent_embedding = self.embedding_model.encode(intent).tolist()
            embedding_str = str(intent_embedding)
            
            with self.pool.connection() as conn:
                conn.execute("""
                    INSERT INTO semantic_cache_v2 (dataset_context, intent, embedding, final_output) 
                    VALUES (%s, %s, %s::vector, %s)
                    ON CONFLICT (dataset_context, intent) 
                    DO UPDATE SET final_output = EXCLUDED.final_output, embedding = EXCLUDED.embedding
                """, (dataset_context, intent, embedding_str, final_output))
                conn.commit()
                logger.info("💾 Successfully committed analytical audit to L2 pgvector cache.")
        except Exception as e:
            logger.error(f"L2 pgvector save failed: {e}")

# =====================================================================
# THE ENTERPRISE BRAIN (AGENTIC SUPERVISOR)
# =====================================================================
class EnterpriseBrain:

    def _discover_active_database(self, default_path):
        """Dynamically scans the storage volume for the active business database if the default is missing."""
        import os
        if os.path.exists(default_path):
            return default_path
            
        print(f"   ∟ [BRAIN] Default DB missing at {default_path}. Scanning /storage for alternative targets...")
        
        valid_dbs = []
        # Exclude system and memory databases from the business data scan
        ignore_list = ["brain_persistence.db", "hash_ledger.db", "chat_history.db", "budget_tracker.db", "semantic_cache_v2.db"]
        
        for root, _, files in os.walk("/storage"):
            for file in files:
                if file.endswith(".db") and file not in ignore_list:
                    valid_dbs.append(os.path.join(root, file))
                    
        if valid_dbs:
            # Sort by modified time and grab the most recent one
            latest_db = max(valid_dbs, key=os.path.getmtime)
            print(f"   ∟ [BRAIN] Dynamic Discovery Success -> Pivoting to: {latest_db}")
            return latest_db
            
        print("   ∟ [BRAIN] ⚠️ Critical: No viable business databases found during scan.")
        return default_path

    def __init__(self, 
                 db_path="/storage/databases/enterprise_data_large/enterprise_erp_large.db", 
                 logs_path="/storage/vector_indices/project_logs",
                 persistence_path="/storage/brain_persistence.db",
                 shared_vector_worker=None): 
        
        logger.info("Initializing EnterpriseBrain Controller: AGENTIC MODE ACTIVE.")
        
        # --- TRIGGER DYNAMIC LOOKUP ---
        self.default_db_path = self._discover_active_database(db_path)
        
        print("   ∟ [BRAIN] Loading LLM Factory Backend...")
        self.llm_fast = get_llm("openai")         
        self.llm_smart = get_llm("openai_smart")
        
        print("   ∟ [BRAIN] Booting Execution Workers...")
        self.sql_worker = SQLWorker(db_path=self.default_db_path)
        self.python_worker = PythonWorker(db_path=self.default_db_path)
        
        # FIX: Use the shared ML worker if provided to prevent HuggingFace deadlocks and RAM spikes
        if shared_vector_worker:
            self.vector_worker = shared_vector_worker
        else:
            self.vector_worker = VectorWorker(logs_path=logs_path)
        
        print("   ∟ [BRAIN] Equipping Agent Toolbox...")
        self.tools = build_enterprise_tools(self.sql_worker, self.vector_worker, self.python_worker, self.llm_fast)
        self.llm_with_tools = self.llm_fast.bind_tools(self.tools, parallel_tool_calls=False)
        
        # --- NEW: POSTGRESQL CONCURRENCY FIX ---
        print("   ∟ [BRAIN] Connecting PostgreSQL Persistence Pool...")
        
        # Connect using the Docker container hostname 'postgres-db'
        DB_URI = "postgresql://enterprise_user:enterprise_password@postgres-db:5432/langgraph_state"
        
        # Establish a connection pool to handle concurrent agent threads
        self.pool = ConnectionPool(
            conninfo=DB_URI,
            max_size=20, # Allows up to 20 simultaneous database locks
            kwargs={"autocommit": True}
        )
        
        # Open the connection pool
        self.pool.open()
        
        self.memory = PostgresSaver(self.pool)
        
        # Initialize the required LangGraph schema tables if they don't exist yet
        self.memory.setup()
        # -----------------------------------------
        
        print("   ∟ [BRAIN] Compiling Agentic State Graph...")
        self.budget_manager = EnterpriseBudgetManager(pool=self.pool, max_budget_usd=2.00) 
        self.graph = self._build_graph()
        self.cache = EnterpriseSemanticCache(pool=self.pool, embedding_model=self.vector_worker.model)

        self.tpm_limit = 5500
        self.current_tpm = 0
        self.last_token_reset = time.time()

    def update_context(self, db_path, logs_path, extra_folder=None):
        logger.info(f"Updating database context to: {os.path.basename(db_path)}")
        self.sql_worker.update_db_path(db_path)
        self.python_worker.db_path = db_path
        self.vector_worker.refresh_index(logs_path, secondary_path=extra_folder) 

    def supervisor_node(self, state: AgentState):
        self.budget_manager.check_budget() # GUARDRAIL CHECK
        logger.info("Supervisor Agent evaluating intent and invoking toolbelt...")
        docs = state.get("active_documents", [])
        docs_str = ", ".join(docs) if docs else "No session files. (Relying on Global Storage/Pinecone)"
        
        try:
            schema_context = self.sql_worker.get_full_schema()
            primary_table = list(schema_context.keys())[0] if schema_context else "DATA_TABLE"
        except Exception:
            primary_table = "DATA_TABLE"

        sys_msg = SystemMessage(content=f"""
        ACT AS: Principal Enterprise AI Auditor.
        SESSION DOCUMENTS & DATASETS: {docs_str}
        ACTIVE DATABASE TABLE: "{primary_table}"
        
        CRITICAL STATE OVERRIDE: 
        The user has loaded a valid dataset into your context. IF the previous chat history contains apologies about "unable to open database file", "data row count is 0", or "persistent connection issue", YOU MUST IGNORE THOSE FAILURES. The environment has been fixed. You MUST call your tools to answer the current question. Do NOT apologize.
        
        CRITICAL RULES:
        1. DATABASE DATA: If the user asks for numbers, trends, or data FROM THE DATABASE, ALWAYS use `query_relational_database`.
        2. DOCUMENT DATA: If the user asks about text or concepts, use `search_unstructured_documents`.
        3. STRICT ANTI-GUESSING: If a tool fails, rewrite your query/code and try again. If it fails 3 times, output the error. Do NOT attempt to answer the user's question without successful tool output.
        """)
        
        messages = [sys_msg] + state["messages"]
        
        with get_openai_callback() as cb:
            response = self.llm_with_tools.invoke(messages)
            self.budget_manager.log_usage(cb)
            
        return {"messages": [response]}

    def presentation_node(self, state: AgentState):
        self.budget_manager.check_budget()  # GUARDRAIL CHECK
        logger.info("Formatting final output via Dynamic Reasoning Escalator...")
        
        is_eval = state.get("is_eval", False)
        messages = state["messages"]
        last_ai_msg = messages[-1].content if messages[-1].content else ""
        tool_data = ""
        for msg in reversed(messages):
            if msg.type == "human": break
            if msg.type == "tool":
                tool_data += f"\n--- {msg.name} Result ---\n{msg.content[:2000]}\n"

        row_count = len([r for r in tool_data.split("\n") if r.strip() and "---" not in r])

        # DYNAMIC SCHEMA INJECTION: Protect token limits during RAGAS evaluation
        if is_eval:
            schema_instructions = """
            "executive_summary": "A concise, factual answer based ONLY on the context.",
            "data_table": null, // STRICT RULE: MUST BE null IN EVALUATION MODE TO SAVE TOKENS
            "chart_config": null, // STRICT RULE: MUST BE null IN EVALUATION MODE
            "context_used": ["List of exact source file names or table names used"],
            "confidence": 0.95
            """
        else:
            schema_instructions = """
            "executive_summary": "A concise, factual answer based ONLY on the context.",
            "data_table": "A Markdown formatted table of the exact retrieved data. Set to null if rows < 2.",
            "chart_config": { ... valid Plotly JSON object ... }, // Set to null if rows < 10
            "context_used": ["List of exact source file names or table names used"],
            "confidence": 0.95
            """

        prompt = f"""
        ACT AS: Principal Data Science Auditor.
        
        RAW SUPERVISOR ANALYSIS: {last_ai_msg}
        TOOL DATA CONTEXT (Evidence): {tool_data}
        USER_ORIGINAL_QUERY: "{state['question']}"
        DATA_ROW_COUNT: {row_count}
        
        TASK:
        Generate the final executive audit report based ONLY on the provided TOOL DATA CONTEXT.
        
        STRICT FAITHFULNESS CONTRACT (ANTI-HALLUCINATION):
        1. GROUNDING: You may ONLY use facts explicitly present in the TOOL DATA CONTEXT. 
        2. REFUSAL PROTOCOL: If the answer is not explicitly in the context, set the "executive_summary" to "The required data could not be found."
        
        CRITICAL INSTRUCTION - THE ZERO-OMISSION RULE:
        You must analyze the USER_ORIGINAL_QUERY and identify EVERY distinct question, sub-question, or analytical request. You must address every single one of them in your final executive_summary. 
        If the TOOL DATA CONTEXT does NOT contain the necessary information to answer a specific sub-question, you are strictly forbidden from silently ignoring it. Instead, you must explicitly state: "The retrieved context does not contain the data necessary to answer [State the specific missing part of the question]."
        
        OUTPUT FORMAT:
        You MUST output ONLY a valid JSON object with the following schema:
        {{
{schema_instructions}
        }}
        """
        
        # PASS 1: Attempt generation with Fast Model (gpt-4o-mini via self.llm_fast)
        try:
            with get_openai_callback() as cb:
                res = self.llm_fast.invoke(prompt)
                self.budget_manager.log_usage(cb)
                
            raw_content = res.content.strip()
            
            # ESCALATION CHECK: If JSON structure is missing or malformed, escalate to Smart Model
            if not re.search(r"\{.*\}", raw_content, re.DOTALL):
                logger.warning("Fast model failed JSON validation. Escalating to Smart Model...")
                with get_openai_callback() as cb:
                    res = self.llm_smart.invoke(prompt)
                    self.budget_manager.log_usage(cb)
                raw_content = res.content.strip()

            # Final JSON extraction
            json_match = re.search(r"\{.*\}", raw_content, re.DOTALL)
            final_output = json_match.group(0) if json_match else raw_content
            return {"final_output": final_output}
            
        except Exception as e:
            logger.error(f"Presentation Node Error: {str(e)}")
            error_json = json.dumps({
                "executive_summary": f"System error during report generation: {str(e)}",
                "data_table": None,
                "chart_config": None,
                "context_used": [],
                "confidence": 0.0
            })
            return {"final_output": error_json}

    def query_rewriter_node(self, state: AgentState):
        self.budget_manager.check_budget() # GUARDRAIL CHECK
        logger.info("⚠️ Vector Search Failed. Engaging Self-Healing Query Rewriter...")
        retries = state.get("retries", 0) + 1
        
        prompt = f"""Rewrite this query into a single, natural-sounding, highly academic or technical question using formal synonyms. Do not use bullet points. Original Query: "{state['question']}" """
        
        with get_openai_callback() as cb:
            res = self.llm.invoke(prompt)
            self.budget_manager.log_usage(cb)
            
        synonyms = res.content.strip()
        nudge = HumanMessage(content=f"SYSTEM INSTRUCTION: Try calling `search_unstructured_documents` again using this rewritten query: {synonyms}")
        return {"messages": [nudge], "retries": retries}   

    def should_continue(self, state: AgentState):
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            logger.info(f"Agent triggered {len(last_message.tool_calls)} tool calls. Routing to Workers...")
            return "tools"
        logger.info("Tool execution complete. Routing to Presentation Node.")
        return "presentation"

    def evaluate_tools(self, state: AgentState):
        messages = state["messages"]
        last_msg = messages[-1]
        retries = state.get("retries", 0)
        content_str = str(getattr(last_msg, 'content', last_msg))
        
        logger.info(f"   ∟ [ROUTER] Evaluating tool response: '{content_str[:100]}' | Current Retries: {retries}")
        if "semantic match" in content_str.lower() or "no qualitative data" in content_str.lower():
            if retries < 2:
                logger.info(f"   ∟ [ROUTER] Intercepted failed search. Routing to rewriter (Attempt {retries + 1}/2)")
                return "rewrite"
        return "supervisor"

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("supervisor", self.supervisor_node)
        workflow.add_node("tools", ToolNode(self.tools))
        workflow.add_node("presentation", self.presentation_node)
        workflow.add_node("query_rewriter", self.query_rewriter_node)
        
        workflow.set_entry_point("supervisor")
        workflow.add_conditional_edges("supervisor", self.should_continue, {"tools": "tools", "presentation": "presentation"})
        workflow.add_edge("tools", "supervisor")
        workflow.add_conditional_edges("tools", self.evaluate_tools, {"rewrite": "query_rewriter", "supervisor": "supervisor"})
        workflow.add_edge("query_rewriter", "supervisor")
        workflow.add_edge("presentation", END)
        
        return workflow.compile(checkpointer=self.memory)

    def run(self, question: str, thread_id: str):
        current_db = getattr(self.sql_worker, 'db_path', '')
        if current_db and current_db != ':memory:':
            import re
            uuid_match = re.search(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', current_db, re.I)
            if uuid_match and uuid_match.group(0) != thread_id:
                logger.warning(f"Context Bleed Prevented for thread {thread_id}")
                self.sql_worker.update_db_path(getattr(self, 'default_db_path', '/storage/enterprise_data_large/enterprise_erp_large.db'))

        session_folder = f"/storage/user_uploads/{thread_id}"
        active_docs = [f for f in os.listdir(session_folder) if not f.endswith('.db')] if os.path.exists(session_folder) else []

        active_dataset = os.path.basename(current_db)
        if active_dataset not in active_docs and active_dataset != "enterprise_erp_large.db":
            active_docs.append(active_dataset)

        current_time = time.time()
        if current_time - self.last_token_reset > 60:
            self.current_tpm = 0
            self.last_token_reset = current_time
        
        estimated_tokens = int(len(question) / 4) + 500
        self.current_tpm += estimated_tokens
        if self.current_tpm > self.tpm_limit:
            logger.warning("⚠️ GUARDRAIL: TPM limit approached. Engaging cooling backoff...")
            time.sleep(15)
            self.current_tpm = estimated_tokens
            self.last_token_reset = time.time()

        config = {"configurable": {"thread_id": thread_id}}
        
        cached_response = self.cache.check_cache(active_dataset, question)
        if cached_response:
            logger.info(f"⚡ Semantic Cache Hit! Bypassing engine for query: '{question}'")
            return cached_response

        input_state = {
            "messages": [HumanMessage(content=question)],
            "question": question,
            "active_documents": active_docs,
            "retries": 0,
            "is_eval": "validation_sim" in thread_id
        }

        try:
            logger.info("Engaging Agentic Graph Execution...")
            result = self.graph.invoke(input_state, config=config)
            raw_out = result.get("final_output", "The technical analysis could not be finalized.")
            
            # --- UI PRESENTATION PARSER ---
            try:
                # Strip Markdown formatting if the LLM wrapped the JSON
                clean_json = raw_out.replace("```json", "").replace("```", "").strip()
                parsed_data = json.loads(clean_json)
                
                # Reconstruct into clean Markdown for the Next.js frontend
                final_out = parsed_data.get("executive_summary", "Summary not available.")
                
                if parsed_data.get("data_table"):
                    final_out += f"\n\n### Analytical Data\n{parsed_data['data_table']}"
                    
            except json.JSONDecodeError:
                # Fallback: If the LLM didn't return JSON, just pass the raw text
                final_out = raw_out
            # ------------------------------
            
            self.cache.save_to_cache(active_dataset, question, final_out)
            return final_out
            
        except Exception as e:
            logger.error(f"Critical execution failure: {str(e)}")
            return "### ⚠️ System Interrupt\nA technical error occurred during the audit process. Please try refining your query."
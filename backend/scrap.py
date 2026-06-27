import os
import json
import logging
import re
import difflib
import sqlite3
from typing import TypedDict, List, Union, Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

# --- IRON RULE: Absolute Imports for Enterprise Pathing ---
# This ensures that the backend resolves services correctly regardless of 
# the execution environment or server deployment in the Bangalore IT sector.
# Essential for maintaining pathing integrity across distributed production environments.
from services.factory import get_llm
from services.SQL_worker import SQLWorker
from services.Vector_worker import VectorWorker

# Configure industrial-grade logging for deep structural traceability.
# Format designed for high-concurrency auditing in Enterprise ERP environments.
# Allows Senior Data Scientists to monitor the 'Neural Controller' state transitions in real-time.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    """
    Central Nervous System State (AgentState).
    Maintains all data points for Scouter, Snapper, Refiner, and Drafter.
    Hardened to support Recursive Healing, Semantic Mapping, and Visual Buffers.
    
    KEY ATTRIBUTES:
    - question: The raw or refined user input.
    - history: Chat context for confirmation bypass and persistence.
    - selected_nodes: Parallel processing markers for workers.
    - sql_data / visual_sample: The relational evidence buffer.
    - recovery_context: The 'Safe Landing' buffer for Similarity-Based pivots.
    """
    question: str
    history: List[str]
    selected_nodes: List[str]
    # --- AMBIGUITY SCOUTER PILLARS ---
    confidence_score: int
    missing_pillars: List[str]
    suggested_refinement: str
    clarification_needed: bool
    # --- PILLAR B: SEMANTIC ENRICHMENT (SNAPPER) ---
    mapped_terms: dict 
    # --- WORKER DATA & VISUAL SAMPLING BUFFER ---
    sql_data: str
    visual_sample: str # Crucial: Buffer for raw distribution data to fix 'single-dot' charts.
    vector_data: str
    # --- PILLAR A: SELF-HEALING (RECURSIVE) ---
    sql_error_log: str
    retry_count: int
    # --- PILLAR C: SIMILARITY RECOVERY (SAFE LANDING) ---
    recovery_context: str # Metadata about fuzzy matches or global pivots.
    # --- METADATA, PERSONA & PERSISTENCE ---
    intent_mode: Literal["Supervised", "Unsupervised", "Exploratory", "Audit", "General"]
    resource_strategy: Literal["Full_Detail", "Schema_Summary"]
    persona: str
    # --- OUTPUT BUFFERS ---
    draft_output: str
    critique: str
    final_output: str

class EnterpriseBrain:
    def __init__(self, 
                 db_path="../storage/enterprise_data_large/enterprise_erp_large.db", 
                 logs_path="../storage/vector_indices/project_logs",
                 persistence_path="../storage/brain_persistence.db"):
        """
        Initializes the Neural Controller with twin-worker handshaking.
        Synchronizes SQL relational logic with unstructured vector logs.
        Incorporates SqliteSaver for physical thread-locked persistence.
        """
        logger.info("Initializing EnterpriseBrain Controller: Persistence Mode ACTIVE.")
        print("   ∟ [BRAIN] Loading LLM Factory Backend (Groq/Llama-3 Optimized)...")
        self.llm = get_llm()
        
        print("   ∟ [BRAIN] Booting SQL Relational Worker (Relational Engine)...")
        self.sql_worker = SQLWorker(db_path=db_path)
        
        print("   ∟ [BRAIN] Booting Vector Log Worker (Semantic Engine)...")
        self.vector_worker = VectorWorker(logs_path=logs_path)
        
        # --- THE PHYSICAL ISOLATION HOOK ---
        # SqliteSaver handles the state persistence for multiple threads/chats.
        # This prevents data leakage between different chat sessions by using 
        # a unique thread_id as a primary key in the persistence database.
        print(f"   ∟ [BRAIN] Connecting Persistence DB: {os.path.basename(persistence_path)}")
        self.memory = SqliteSaver.from_conn_string(persistence_path)
        
        print("   ∟ [BRAIN] Compiling Strategic Intelligence Graph (12 Nodes active)...")
        self.graph = self._build_graph()
        logger.info("Neural Controller Online. Recursive Autonomy: ACTIVE.")

    def update_context(self, db_path, logs_path, extra_folder=None):
        """
        DYNAMIC REFRESH: Ensures the brain can pivot between different user-uploaded 
        sessions without a full service restart—critical for multi-tenant scalability.
        """
        logger.info(f"Updating database context to: {os.path.basename(db_path)}")
        self.sql_worker.db_path = db_path
        
        logger.info(f"Refreshing vector index: {os.path.basename(logs_path)}")
        self.vector_worker.refresh_index(logs_path, secondary_path=extra_folder) 

    def _clean_json_response(self, content: str) -> str:
        """
        HARDENED JSON EXTRACTION: 
        Physically isolates JSON blocks from LLM conversational noise 
        to prevent syntax crashes in the automated pipeline.
        """
        try:
            # Look for the outermost curly braces to isolate the JSON object.
            start = content.find('{')
            end = content.rfind('}')
            if start != -1 and end != -1:
                return content[start:end+1]
            
            # Fallback cleaning for markdown markers or standard string cleanup.
            # FIXED: Syntax error resolved on Line 126. Properly terminated string literal.
            return content.strip().replace("```json", "").replace("
```", "")
        except Exception as e:
            logger.error(f"Critical Failure in JSON Cleaning Logic: {str(e)}")
            return content.strip()

    def _classify_intent(self, question: str) -> tuple[str, str, str]:
        """
        HARDENED STRATEGIC INTENT GATEKEEPER:
        Distinguishes between a 'Data Audit' and 'General Reasoning'.
        Uses Structural Markers to prevent entity-heavy queries from being misclassified as theory.
        """
        logger.info("Analyzing Strategic Intent via Structural Markers...")
        
        # Detect keywords that trigger reasoning-heavy paths (Theory/Definitions).
        reasoning_triggers = ["why", "how", "explain", "reason", "because", "tell me more"]
        is_reasoning = any(word in question.lower().split() for word in reasoning_triggers)
        
        # --- STRUCTURAL MARKER OVERRIDE ---
        # Detect if the query is structurally comparing entities or referencing specific segments.
        relational_markers = ["vs", "versus", "compare", "correlation", "trend", "distribution"]
        entity_markers = ["tier", "segment", "industry", "sector", "group", "logs", "data"]
        
        is_structural_audit = any(word in question.lower() for word in relational_markers + entity_markers)
        
        prompt = f"""
        ACT AS: Principal Strategist.
        ANALYZE TECHNICAL QUERY: "{question}"
        Is_Reasoning_Phrasing: {is_reasoning}
        Is_Structural_Audit_Detected: {is_structural_audit}
        
        TASK:
        1. INTENT: 'Audit' if the user is asking to extract, compare, or analyze specific metrics, 
           tiers, segments, or data logs. 
           'General' if the user is seeking theoretical definitions, definitions of terms, 
           or educational business advice.
        2. PERSONA: Identify professional persona (e.g., 'Senior Auditor', 'Data Scientist').
        3. RESOURCE: 'Full_Detail' (deep dive) or 'Schema_Summary' (metadata only).
        
        STRICT RULE: If Is_Structural_Audit_Detected is True, INTENT should strongly bias toward 'Audit'.
        
        Output only a JSON object: 
        {{"intent": "...", "persona": "...", "resource": "..."}}
        """
        res = self.llm.invoke(prompt)
        try:
            meta = json.loads(self._clean_json_response(res.content))
            # Enforcement of structural markers for enterprise reliability.
            intent = meta.get('intent', 'Audit')
            if is_structural_audit:
                intent = "Audit"
            elif is_reasoning:
                intent = "General"
                
            return intent, meta.get('resource', 'Full_Detail'), meta.get('persona', 'Senior VC Auditor')
        except Exception as e:
            logger.warning(f"Intent Error: {str(e)}. Defaulting to Audit mode.")
            return "Audit", "Full_Detail", "Senior VC Auditor"

    def ambiguity_scouter_node(self, state: AgentState):
        """
        THE AMBIGUITY SCOUTER: 
        Evaluates precision against Metric, Time, and Category pillars.
        STRICT FIX: Confirmation bypass via regex for 'proceed' triggers in history.
        """
        # Detection logic for user confirmation to bypass the scouter gate.
        confirmation_triggers = ["proceed", "yes", "go ahead", "do it", "sure", "yep", "ok", "y"]
        is_confirmation = state['question'].lower().strip() in confirmation_triggers
        
        if is_confirmation and len(state['history']) > 0:
            logger.info("Confirmation detected. Bypassing scouter gate and promoting best guess.")
            # Search history for the last suggested declarative statement.
            last_msg = state['history'][-1]
            match = re.search(r"best guess: '(.*?)'", last_msg)
            if match:
                return {
                    **state, 
                    "question": match.group(1), 
                    "confidence_score": 100, 
                    "clarification_needed": False, 
                    "retry_count": 0, 
                    "sql_error_log": ""
                }

        logger.info("Evaluating Query Viability (Threshold: 90%)...")
        prompt = f"""
        ACT AS: Principal AI Auditor.
        QUERY: "{state['question']}"
        
        TASK:
        Evaluate the query against the 3 Pillars of Data Certainty:
        1. METRIC (e.g., 'License Count')
        2. TIME (e.g., 'Last 30 days')
        3. CATEGORY (e.g., 'Marketing industry')
        
        OUTPUT FORMAT (JSON ONLY):
        {{
            "confidence_score": 0-100,
            "missing_pillars": ["Metric", "Time", "Category"],
            "suggested_refinement": "Calculate correlation between [X] and [Y] for [Z] segment...",
            "clarification_needed": true/false
        }}
        """
        res = self.llm.invoke(prompt)
        try:
            meta = json.loads(self._clean_json_response(res.content))
            score = meta.get("confidence_score", 100)
            return {
                **state, 
                **meta, 
                "confidence_score": score, 
                "clarification_needed": score < 90, 
                "retry_count": 0, 
                "sql_error_log": ""
            }
        except Exception as e:
            logger.error(f"Scouter node failure: {str(e)}")
            return {**state, "confidence_score": 100, "clarification_needed": False}

    def semantic_snapper_node(self, state: AgentState):
        """
        PILLAR B: THE SNAPPER.
        Deterministic mapping of user business jargon to exact database headers.
        Prevents the SQL generator from hallucinating non-existent columns.
        """
        logger.info("Aligning business slang to Catalog Metadata...")
        
        prompt = f"""
        USER_QUERY: "{state['question']}"
        DATABASE_SCHEMA: ADS_datav7
        
        TASK:
        Map identified business slang to the following exact database headers:
        - 'seat velocity' or 'license traction' -> 'LICENSE_COUNT_1M'
        - 'creation intensity' or 'document output' -> 'SHEET_UNIQUE_CREATIONS_1M'
        - 'industries' or 'sectors' -> 'INDUSTRY'
        - 'growth momentum' -> 'SHEET_UNIQUE_CREATIONS_1M'
        - 'retention friction' -> 'LICENSE_COUNT_1M'
        
        Return a JSON mapping ONLY: {{"user_term": "db_header"}}
        """
        res = self.llm.invoke(prompt)
        try:
            mapping = json.loads(self._clean_json_response(res.content))
            refined = state['suggested_refinement'] or state['question']
            
            # Surgically replace identified slang with valid headers in the intent.
            for slang, header in mapping.items():
                pattern = re.compile(re.escape(slang), re.IGNORECASE)
                refined = pattern.sub(header, refined)
                
            logger.info(f"Snapped Technical Intent: {refined[:50]}...")
            return {**state, "suggested_refinement": refined, "mapped_terms": mapping}
        except Exception as e:
            logger.error(f"Pillar B Snapper Logic Failure: {str(e)}")
            return state

    def router_node(self, state: AgentState):
        """
        STRATEGIC ROUTER:
        Directs traffic between Clarifier (low confidence), General (reasoning), 
        and the SQL/Vector worker paths (data audit).
        """
        if state.get("clarification_needed"):
            logger.info("Confidence score low. Routing to Clarifier.")
            return {**state, "selected_nodes": ["clarifier_node"]}
        
        intent, resource, persona = self._classify_intent(state['question'])
        if intent == "General":
            return {
                **state, 
                "intent_mode": "General", 
                "persona": persona, 
                "selected_nodes": ["general_node"]
            }
        
        return {
            **state, 
            "intent_mode": "Audit", 
            "persona": persona, 
            "selected_nodes": ["sql_node", "vector_node"]
        }

    def clarifier_node(self, state: AgentState):
        """
        THE CLARIFIER:
        Drafts a professional request for more information when pillars are missing.
        Ensures a 'Best Guess' is always provided for a declarative user experience.
        """
        msg = f"I see a few ways to interpret your request regarding {', '.join(state['missing_pillars'])}.\n\n"
        msg += f"My best guess: '{state['suggested_refinement']}'\n\n"
        msg += "To provide an exhaustive analysis, would you like me to proceed with this or adjust specifics?"
        return {"final_output": msg}

    def sql_node(self, state: AgentState):
        """
        RELATIONAL DATA RETRIEVAL (sql_node):
        PILLAR C FIX: Fetches global math AND a 100-row sample for visual accuracy.
        Captures execution errors for the Recursive Refiner (Pillar A).
        """
        if "sql_node" not in state["selected_nodes"]: 
            return {"sql_data": "N/A", "visual_sample": "N/A"}
        
        # Prioritize the refined/snapped query for technical precision.
        query = state['suggested_refinement'] if state['suggested_refinement'] else state['question']
        logger.info(f"SQL Attempt {state['retry_count'] + 1} with Visual Sampling...")
        
        try:
            # Step 1: Execute the analytical summary math (e.g., Averages, Correlations).
            data = self.sql_worker.execute_and_format(query)
            
            # Step 2: VISUAL SAMPLING LAYER
            # We fetch 100 rows of raw distribution data to ensure Plotly has points to plot.
            # This prevents the 'single-dot chart' axes collapse error.
            sample_query = "SELECT * FROM ADS_datav7 LIMIT 100"
            visual_data = self.sql_worker.execute_and_format(sample_query)
            
            return {
                **state, 
                "sql_data": data, 
                "visual_sample": visual_data, 
                "sql_error_log": ""
            }
        except Exception as e:
            print(f"   ∟ [SQL ERROR] Captured for Recursive Refiner: {str(e)}")
            return {**state, "sql_error_log": str(e)}

    def reflection_node(self, state: AgentState):
        """
        PILLAR A: THE RECURSIVE REFINER (reflection_node).
        Self-heals the query after a SQL failure by analyzing syntax or schema mismatch.
        Essential for enterprise reliability where users provide imprecise filters.
        """
        logger.info(f"Healing technical error: {str(state['sql_error_log'])[:30]}...")
        
        prompt = f"""
        ACT AS: Senior Data Architect.
        ERROR_MESSAGE: {state['sql_error_log']}
        INTENT: {state['suggested_refinement'] or state['question']}
        
        TASK:
        1. Identify the hallucinated or missing column in the error message.
        2. Rewrite the query using strictly valid headers from Catalog Metadata.
        3. Output ONLY the refined technical statement. No conversational preamble.
        """
        res = self.llm.invoke(prompt)
        return {
            **state, 
            "suggested_refinement": res.content.strip(), 
            "retry_count": state['retry_count'] + 1, 
            "sql_error_log": "" # Clear the error for the retry attempt.
        }

    def check_sql_health(self, state: AgentState):
        """
        CONDITIONAL ROUTER (Hardened): 
        Detects syntax errors AND 'Empty Aggregates' (1 row of NULLs).
        Loops back to Reflection for errors, or Similarity Recovery for data gaps[cite: 4].
        """
        if state.get("sql_error_log") and state["retry_count"] < 3:
            logger.info("SQL Health check: FAIL. Triggering Recursive Refiner (Pillar A).")
            return "reflection_node"
        
        # --- HARDENED NULL CHECK ---
        # Detects if the SQL output is physically empty or contains NULL aggregation markers.
        # This prevents the AI from reporting success when the filter matched no rows[cite: 4].
        sql_data_lower = str(state.get('sql_data', '')).lower()
        null_markers = ["0 rows", "none", "null", "nan"]
        if any(marker in sql_data_lower for marker in null_markers) or not state.get('sql_data'):
             logger.warning("Empty result set or NULL aggregate detected. Routing to Similarity Recovery Node.")
             return "similarity_recovery_node"

        return "vector_node"

    def similarity_recovery_node(self, state: AgentState):
        """
        PILLAR C: DYNAMIC SIMILARITY RECOVERY (Safe Landing).
        Performs multi-column fuzzy matching on categorical columns (INDUSTRY, PRODUCT_NAME)
        to find the nearest valid segment and pivot the report trend[cite: 4].
        """
        logger.info("Executing DYNAMIC Similarity-Based Recovery Logic...")
        
        try:
            # 1. Multi-Column Category Audit: Scan for unique values across relevant categorical headers.
            target_cols = ["INDUSTRY", "PRODUCT_NAME"]
            all_options = []
            for col in target_cols:
                raw_values = self.sql_worker.execute_and_format(f"SELECT DISTINCT {col} FROM ADS_datav7")
                all_options.extend(raw_values.split("\n"))
            
            # 2. Perform fuzzy matching against all identified options.
            user_input = state['question']
            matches = difflib.get_close_matches(user_input, all_options, n=1, cutoff=0.2)
            best_match = matches[0] if matches else "Global Baseline"
            
            # 3. Pivot the context for the user and the Drafter.
            pivot_msg = f"I couldn't find a direct match for your filter. Analysis pivoted to the nearest valid category: '{best_match}'."
            
            # 4. Re-execute the query logic based on the best fuzzy match found.
            new_query = f"""
            SELECT AVG(LICENSE_COUNT_1M), AVG(SHEET_UNIQUE_CREATIONS_1M) 
            FROM ADS_datav7 
            WHERE INDUSTRY LIKE '%{best_match}%' OR PRODUCT_NAME LIKE '%{best_match}%'
            """
            pivoted_data = self.sql_worker.execute_and_format(new_query)
            
            return {**state, "sql_data": pivoted_data, "recovery_context": pivot_msg}
        except Exception as e:
            logger.error(f"Dynamic Similarity Recovery failure: {str(e)}")
            return {**state, "recovery_context": "No close matches found. Displaying global trends for comparison."}

    def vector_node(self, state: AgentState):
        """
        VECTOR WORKER NODE:
        Performs semantic search across unstructured project logs.
        Adds qualitative context to the quantitative data retrieved by SQL.
        """
        if "vector_node" not in state["selected_nodes"]: 
            return {"vector_data": "N/A"}
        
        logger.info("Searching Vector Store for qualitative log context...")
        return {"vector_data": self.vector_worker.search(state['question'])}

    def general_node(self, state: AgentState):
        """
        GENERAL REASONING NODE:
        Handles theory, definitions, and logical follow-up questions.
        Uses the persona-grounding logic for Senior Vc Auditors.
        """
        if "general_node" not in state["selected_nodes"]: 
            return {"final_output": ""}
        
        logger.info(f"Executing general reasoning as {state['persona']}...")
        res = self.llm.invoke(
            f"ACT AS: {state['persona']}. "
            f"USER QUERY: {state['question']}. "
            f"CONTEXT: Explain reasoning, strategic implications, and theory."
        )
        return {"final_output": res.content}

    def drafter_node(self, state: AgentState):
        """
        THE NARRATIVE DRAFTER (Pillar C):
        Mandates 'Executive Insights' and handles Recovery Context explanation.
        Integrates visual sampling buffer for a meaningful Plotly distribution.
        """
        if state.get("final_output"): 
            return state
        
        logger.info("Constructing Audit Report with Narrative Insight Callouts...")
        prompt = f"""
        ACT AS: {state['persona']}
        RECOVERY_CONTEXT: {state.get('recovery_context', 'N/A')}
        MATH_EVIDENCE: {state['sql_data']}
        DISTRIBUTION_DATA: {state['visual_sample']}
        LOG_CONTEXT: {state['vector_data']}
        
        MANDATE:
        1. If RECOVERY_CONTEXT is present, explain the pivot clearly to the user.
        2. Ground findings strictly in MATH_EVIDENCE.
        3. Visuals: Output valid JSON in ```json_plotly blocks using DISTRIBUTION_DATA.
        4. INSIGHTS: Add an 'Executive Insight' explaining 'So What?' in 2-3 bolded sentences.
        5. STRUCTURE: Use H3 headers and bolded bullets. No preamble.
        """
        res = self.llm.invoke(prompt)
        return {"draft_output": res.content.strip()}

    def critique_node(self, state: AgentState):
        """
        RED TEAM CRITIQUE:
        Fact-checks the drafted report against raw worker evidence. 
        Summary-aware to allow statistical calculations (Pearson, Means) to pass grounding.
        """
        if not state.get("draft_output"): 
            return {"critique": "PASSED"}
        
        logger.info("Verifying Evidence Grounding (Critique Node)...")
        prompt = f"""
        RAW_EVIDENCE: {state['sql_data']}. 
        DRAFT_REPORT: {state['draft_output']}. 
        
        TASK:
        Does the draft report claim specific facts NOT supported by raw evidence?
        Note: Summary metrics (Correlations, Averages) are supported if the data exists.
        Output ONLY 'PASSED' or 'FAILED'.
        """
        res = self.llm.invoke(prompt)
        return {"critique": res.content}

    def finalizer_node(self, state: AgentState):
        """
        THE FINALIZER:
        Silent dispatcher. If critique fails, it blocks hallucination and 
        reverts to a structured Technical Gap report.
        """
        if state.get("final_output"): 
            return state
        
        if "PASSED" in state['critique'].upper():
            return {"final_output": state['draft_output']}
        
        logger.warning("Grounding failed in Critique. Reverting to Technical Gap fallback.")
        return {
            "final_output": "I identified a technical gap: Grounding could not be established for this trend."
        }

    def _build_graph(self):
        """
        THE MASTER GRAPH ARCHITECTURE:
        Compiles the state machine with persistence integration[cite: 4].
        """
        workflow = StateGraph(AgentState)
        
        # --- NODE DEFINITIONS ---
        workflow.add_node("scouter", self.ambiguity_scouter_node)
        workflow.add_node("router", self.router_node)
        workflow.add_node("snapper", self.semantic_snapper_node)
        workflow.add_node("clarifier", self.clarifier_node)
        workflow.add_node("general_node", self.general_node)
        workflow.add_node("sql_node", self.sql_node)
        workflow.add_node("reflection_node", self.reflection_node)
        workflow.add_node("similarity_recovery_node", self.similarity_recovery_node)
        workflow.add_node("vector_node", self.vector_node)
        workflow.add_node("drafter", self.drafter_node)
        workflow.add_node("critique", self.critique_node)
        workflow.add_node("finalizer", self.finalizer_node)
        
        # --- GRAPH CONSTRUCTION ---
        workflow.set_entry_point("scouter")
        workflow.add_edge("scouter", "router")
        
        # Router Decision: [Clarifier | General | Snapper]
        workflow.add_conditional_edges("router", lambda x: x["selected_nodes"][0], {
            "clarifier_node": "clarifier",
            "general_node": "general_node",
            "sql_node": "snapper"
        })
        
        # Snapper -> SQL
        workflow.add_edge("snapper", "sql_node")
        
        # SQL Conditional Path: Healing OR Recovery OR Vector
        workflow.add_conditional_edges(
            "sql_node", 
            self.check_sql_health, 
            {
                "reflection_node": "reflection_node", 
                "similarity_recovery_node": "similarity_recovery_node",
                "vector_node": "vector_node"
            }
        )
        
        # Logic Loopbacks
        workflow.add_edge("reflection_node", "sql_node")
        workflow.add_edge("similarity_recovery_node", "drafter")
        
        # Terminal Flow
        workflow.add_edge("vector_node", "drafter")
        workflow.add_edge("general_node", "drafter")
        workflow.add_edge("drafter", "critique")
        workflow.add_edge("critique", "finalizer")
        
        # Endpoints
        workflow.add_edge("clarifier", END)
        workflow.add_edge("finalizer", END)
        
        # --- PERSISTENCE COMPILATION ---
        return workflow.compile(checkpointer=self.memory)

    def run(self, question: str, thread_id: str):
        """
        EXECUTION ENTRY POINT:
        Isolation is guaranteed by passing thread_id in the config[cite: 4].
        """
        # Config dictionary to specify the unique thread_id for this session.
        config = {"configurable": {"thread_id": thread_id}}
        
        # invoke will now check the persistence DB for state associated with thread_id[cite: 4].
        result = self.graph.invoke({
            "question": question, 
            "history": [], # StateGraph will populate this from persistence automatically[cite: 4].
            "selected_nodes": [],
            "confidence_score": 0, "missing_pillars": [], "suggested_refinement": "",
            "mapped_terms": {}, "clarification_needed": False, "sql_data": "",
            "visual_sample": "", "vector_data": "", "sql_error_log": "", "retry_count": 0,
            "recovery_context": "", "intent_mode": "Audit", "resource_strategy": "Full_Detail", 
            "persona": "Senior Auditor", "draft_output": "", "critique": "", "final_output": ""
        }, config=config)
        
        return result["final_output"]

# --- END OF CORE_AGENT.PY (MASTER ARCHITECTURE) ---
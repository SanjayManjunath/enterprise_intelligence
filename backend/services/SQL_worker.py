import sqlite3
import yaml
import re
import os
import json
import pandas as pd
from typing import TypedDict, Union, Literal
from langgraph.graph import StateGraph, END

# --- IRON RULE: Absolute Import Pathing for Enterprise Execution ---
from services.factory import get_llm

class AgentState(TypedDict):
    """
    Maintains the technical state of the SQL Worker.
    Includes retry counters and error tracking for self-correction loops.
    """
    question: str
    sql_query: str
    db_results: Union[dict, str]
    final_answer: str
    error_message: str
    retry_count: int 

class SQLWorker:
    def __init__(self, db_path):
        """
        Principal Data Scientist Core:
        Initializes the worker with absolute path hardening for catalog.yaml.
        """
        print(f"   ∟ [SQL WORKER] Initializing Engine with DB: {os.path.basename(db_path)}")
        self.llm = get_llm("openai_fast")
        self.db_path = db_path
        self.mem_conn = None  # Holds the in-memory SQLite connection for CSVs
        self.registered_tables = [] # Centralized Entity Registry
        
        # --- PATH HARDENING: Strategic resolution for catalog metadata ---
        base_dir = os.path.dirname(__file__)
        catalog_path = "/backend/catalog.yaml"
        
        try:
            with open(catalog_path, "r") as f:
                self.catalog = yaml.safe_load(f)
                print(f"   ∟ [SQL WORKER] Strategic Catalog successfully indexed at: {catalog_path}")
        except FileNotFoundError:
            print(f"⚠️ [SQL WORKER] Warning: catalog.yaml not found at {catalog_path}. Mappings disabled.")
            self.catalog = {}
        except Exception as e:
            print(f"⚠️ [SQL WORKER] Catalog Read Error: {str(e)}")
            self.catalog = {}

        self.graph = self._build_graph()

    def update_db_path(self, new_db_path: str):
        """
        DYNAMIC PIVOT & DATASET-AGNOSTIC LOADER: 
        Updates active DB path. If a CSV is detected, it loads it into an in-memory SQLite DB
        and registers the sanitized filename as the active table alias.
        """
        abs_path = os.path.abspath(new_db_path)
        if not os.path.exists(abs_path):
            print(f"⚠️ [SQL WORKER] Pivot failed: {abs_path} not found.")
            return

        # --- CSV to In-Memory SQL Bridge ---
        if abs_path.endswith('.csv'):
            print(f"   ∟ [SQL WORKER] CSV detected. Converting {os.path.basename(abs_path)} to in-memory SQL...")
            try:
                df = pd.read_csv(abs_path)
                self.mem_conn = sqlite3.connect(':memory:', check_same_thread=False)
                
                # Dynamically generate and register the table alias from the filename
                raw_name = os.path.basename(abs_path)
                safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', raw_name.replace('.csv', ''))
                
                df.to_sql(safe_name, self.mem_conn, index=False, if_exists='replace')
                self.db_path = ':memory:'
                self.registered_tables = [safe_name]
                print(f"   ∟ [SQL WORKER] Successfully loaded CSV into memory. Registered table: '{safe_name}'")
            except Exception as e:
                print(f"⚠️ [SQL WORKER] Failed to load CSV into memory: {str(e)}")
        else:
            # Standard SQLite DB pivot
            self.db_path = abs_path
            self.mem_conn = None
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
                self.registered_tables = [row[0] for row in cursor.fetchall()]
                conn.close()
                print(f"   ∟ [SQL WORKER] Successfully pivoted to DB. Registered tables: {self.registered_tables}")
            except Exception as e:
                print(f"⚠️ [SQL WORKER] Failed to read tables from DB: {str(e)}")
                self.registered_tables = []

    def _get_live_schema_and_sample(self):
        """
        DYNAMIC DISCOVERY ENGINE:
        Performs deep reflection of the database to ground the LLM's logic.
        """
        try:
            db_label = "In-Memory CSV" if self.db_path == ':memory:' else self.db_path
            print(f"   ∟ [SQL WORKER] Initiating Protected Schema Discovery on: {db_label}")
            
            # Use the in-memory connection if active, otherwise connect to the file
            conn = self.mem_conn if self.db_path == ':memory:' else sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
            tables = cursor.fetchall()
            
            discovery_report = []
            for table in tables:
                t_name = table[0]
                cursor.execute(f"PRAGMA table_info({t_name});")
                cols = cursor.fetchall()
                total_cols = len(cols)
                col_headers = [f"{c[1]} ({c[2]})" for c in cols]
                
                df_sample = pd.read_sql_query(f"SELECT * FROM {t_name} LIMIT 3;", conn)
                
                if total_cols > 10:
                    df_sample = df_sample.iloc[:, :10]
                    col_note = f"(Hard Pruned: Showing 10 of {total_cols} columns)"
                else:
                    col_note = f"(Total columns: {total_cols})"

                discovery_report.append(
                    f"ENTITY: {t_name} {col_note}\n"
                    f"COLUMNS: {', '.join(col_headers[:30])}{'...' if total_cols > 30 else ''}\n"
                    f"SAMPLES (3 Rows):\n{df_sample.to_string(index=False)}"
                )
            
            # Only close if it's a file-based connection; closing in-memory drops the DB!
            if self.db_path != ':memory:':
                conn.close()
                
            return "\n\n".join(discovery_report) if discovery_report else "No user tables found."
            
        except Exception as e:
            print(f"   ∟ [SQL WORKER] Critical Discovery Failure: {str(e)}")
            return f"CRITICAL_SCHEMA_ERROR: {str(e)}"
      
    def sql_generator_node(self, state: AgentState):
        """
        Universal Logic Node: Generates SQL using the Universal Hardening Policy.
        """
        attempt = state.get('retry_count', 0) + 1
        print(f"   ∟ [SQL GEN] Analyzing Intent with Catalog Metadata (Attempt {attempt})...")
        
        meta_triggers = ["schema", "list tables", "database structure", "column names"]
        if any(k in state['question'].lower() for k in meta_triggers):
             return {"sql_query": "--SCHEMA_AND_SAMPLE_REFLECT--", "retry_count": attempt}

        live_schema = self._get_live_schema_and_sample()
        
        feedback = f"\n⚠️ PREVIOUS ATTEMPT FAILED: {state['error_message']}\nCRITICAL: Fix the query." if state.get("error_message") else ""

        prompt = f"""
        Generate a strictly valid SQLite query for: "{state['question']}"
        
        AVAILABLE REGISTERED TABLES: {self.registered_tables}
        
        STRATEGIC CATALOG: {json.dumps(self.catalog, indent=2)}
        DATABASE CONTEXT: {live_schema}
        {feedback}
        
        STRICT RULES:
        1. Output ONLY the SQL query inside a ```sql code block.
        2. Wrap all column names in double quotes.
        3. NEVER invent or guess table names. You MUST use one of the AVAILABLE REGISTERED TABLES exactly as spelled.
        4. Use SQL 'CASE' statements if mapping categorical IDs.
        """
        response = self.llm.invoke(prompt)
        sql_match = re.search(r"```sql\n(.*?)\n```", response.content, re.DOTALL | re.IGNORECASE)
        sql = sql_match.group(1).strip() if sql_match else response.content.strip()
        
        return {"sql_query": sql, "retry_count": attempt}

    def execute_query_node(self, state: AgentState):
        """
        Execution Layer: Runs the generated SQL and handles numeric formatting.
        """
        if state["sql_query"] == "--SCHEMA_AND_SAMPLE_REFLECT--":
            return {"db_results": self._get_live_schema_and_sample(), "error_message": ""}

        if "ERROR" in state["sql_query"]:
            return {"db_results": "", "error_message": "Internal Logic Error in Generation."}
            
        print(f"   ∟ [SQL EXEC] Executing query...")
        try:
            # Use the in-memory connection if active, otherwise connect to the file
            conn = self.mem_conn if self.db_path == ':memory:' else sqlite3.connect(self.db_path)
            
            sql_clean = state["sql_query"].strip().rstrip(';') + ';'
            df = pd.read_sql_query(sql_clean, conn)
            
            # Only close if it's a file-based connection
            if self.db_path != ':memory:':
                conn.close()
            
            if df.empty:
                return {"db_results": "", "error_message": "Query execution returned 0 results."}
            
            display_df = df.head(15).copy()
            for col in display_df.select_dtypes(include=['number']).columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x:,.2f}" if pd.notnull(x) else "0.00")
            
            return {"db_results": display_df.to_string(index=False), "error_message": ""}
            
        except Exception as e:
            print(f"   ∟ [SQL EXEC] Critical Error: {str(e)}")
            return {"db_results": "", "error_message": str(e)}

    def should_retry(self, state: AgentState) -> Literal["generator", "formatter"]:
        if state.get("error_message") and state.get("retry_count", 0) < 3:
            return "generator"
        return "formatter"

    def answer_formatter_node(self, state: AgentState):
        return {"final_answer": state['db_results'] if state['db_results'] else "Search returned no records."}

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("generator", self.sql_generator_node)
        workflow.add_node("executor", self.execute_query_node)
        workflow.add_node("formatter", self.answer_formatter_node)
        
        workflow.set_entry_point("generator")
        workflow.add_edge("generator", "executor")
        workflow.add_conditional_edges("executor", self.should_retry, {"generator": "generator", "formatter": "formatter"})
        workflow.add_edge("formatter", END)
        
        return workflow.compile()

    def execute_and_format(self, question: str):
        final_state = self.graph.invoke({
            "question": question, "sql_query": "", "db_results": "", 
            "final_answer": "", "error_message": "", "retry_count": 0
        })
        
        # --- THE HARD FAIL ESCALATION ---
        # If the worker exhausted all retries, raise an exception to trigger the Pandas Sandbox.
        if final_state.get("error_message") and not final_state.get("db_results"):
            raise RuntimeError(f"SQL Relational Engine failed after 3 attempts. Last error: {final_state['error_message']}")
            
        return f"--- SQL_RESULT ---\n{final_state['final_answer']}\n\n--- SQL_QUERY_USED ---\n{final_state['sql_query']}"
import pandas as pd
import sqlite3
import traceback
import logging

logger = logging.getLogger(__name__)

class PythonWorker:
    """
    THE PANDAS SANDBOX (Dynamic Enterprise Edition)
    Hardened for memory safety, multi-format support, and schema-awareness.
    """
    def __init__(self, db_path: str):
        self.db_path = db_path

    def execute(self, code_string: str, table_name: str) -> tuple[bool, str]:
        logger.info(f"Initializing Pandas sandbox for dataset: {table_name}...")
        
        # 1. DYNAMIC ROUTING: Multi-Format Data Loading
        try:
            if self.db_path.lower().endswith('.csv'):
                logger.info("Detected CSV file. Loading directly into Pandas...")
                df_sample = pd.read_csv(self.db_path, nrows=10000)
                
                # Fallback hook for CSVs (SQL queries don't work natively on CSVs)
                def query_full_db(sql_query: str) -> pd.DataFrame:
                    return df_sample 
            else:
                logger.info("Detected SQLite DB. Loading sample into Pandas...")
                conn = sqlite3.connect(self.db_path)
                df_sample = pd.read_sql(f'SELECT * FROM "{table_name}" LIMIT 10000', conn)
                conn.close()
                
                # Aggregation hook for DBs
                def query_full_db(sql_query: str) -> pd.DataFrame:
                    with sqlite3.connect(self.db_path) as dynamic_conn:
                        return pd.read_sql(sql_query, dynamic_conn)
                        
        except Exception as e:
            logger.error(f"Failed to load data into Pandas: {str(e)}")
            return False, f"Failed to load dataset: {str(e)}"

        # --- SCHEMA VISION FIX ---
        # Capture the exact columns of the loaded dataframe to assist LLM self-correction
        actual_columns = list(df_sample.columns)

        # 2. Prepare the execution sandbox environment
        import numpy as np
        exec_globals = {
            "pd": pd, 
            "np": np, 
            "DataFrame": pd.DataFrame, 
            "Series": pd.Series,
            "df": df_sample, 
            "query_full_db": query_full_db, 
            "table_name": table_name
        }
        exec_locals = {}

        # 3. Clean the LLM code string
        clean_code = code_string.strip()
        if clean_code.lower().startswith("```python"): 
            clean_code = clean_code[9:]
        if clean_code.endswith("```"): 
            clean_code = clean_code[:-3]
        
        # 4. Execute the code block
        try:
            exec(clean_code.strip(), exec_globals, exec_locals)
            
            if "analyze" not in exec_locals:
                return False, f"Error: The generated code did not define an 'analyze(df)' function. Available columns are: {actual_columns}"
            
            result = exec_locals["analyze"](df_sample)
            
            # 5. Format and return
            if isinstance(result, pd.DataFrame):
                # THE GROUPBY FIX: Reset the index so grouped columns are not dropped by index=False
                if not isinstance(result.index, pd.RangeIndex): 
                    result = result.reset_index()
                return True, result.head(100).to_csv(index=False)
            
            return True, str(result)
            
        except KeyError as ke:
            logger.warning(f"Pandas Schema KeyError intercepted: {str(ke)}")
            # If LLM hallucinates a column, throw a guided error with the exact schema
            return False, f"Python Execution Error: KeyError {str(ke)}. You hallucinated a column name. The ACTUAL valid columns in this dataset are: {actual_columns}"
            
        except Exception as e:
            error_trace = traceback.format_exc()
            logger.error(f"Pandas sandbox crashed during execution:\n{error_trace}")
            # Append schema to general errors just in case it was a dataframe manipulation fault
            return False, f"Python Execution Error: {str(e)}. Reminder, the valid columns are: {actual_columns}"
# tools.py
import json
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

# =========================================================================
# INPUT SCHEMAS (STRICT LLM GUARDRAILS)
# =========================================================================
class SQLQueryInput(BaseModel):
    query: str = Field(description="The natural language question to answer using the relational database. Do NOT pass raw SQL code. Pass the plain English intent.")

class VectorSearchInput(BaseModel):
    query: str = Field(description="The specific search terms or concepts to look up in the uploaded documents, PDFs, and qualitative project logs.")

class PythonSandboxInput(BaseModel):
    intent: str = Field(description="The natural language description of the advanced mathematical or statistical analysis to perform.")
    table_name: str = Field(description="The exact name of the database table to analyze. You must retrieve this from the active schema.")

# =========================================================================
# ENTERPRISE TOOL FACTORY
# =========================================================================
def build_enterprise_tools(sql_worker, vector_worker, python_worker, llm):
    """
    Wraps the existing execution engines into LangChain-compatible tools.
    Injects the initialized workers directly so they share the exact same database context.
    """
    
    def query_database(query: str) -> str:
        """Searches the SQL database for quantitative metrics, rows, and structural data."""
        try:
            return sql_worker.execute_and_format(query)
        except Exception as e:
            return f"Database query failed. Error: {str(e)}"

    def search_documents(query: str) -> str:
        """Searches uploaded PDFs, Markdown, and qualitative logs for context."""
        try:
            # Slices to 2000 chars to maintain token protection limits
            results = vector_worker.search(query=query, top_k=5)
            if not results or "No semantic" in results:
                return "No qualitative data found for this query in the uploaded documents."
            return results 
        except Exception as e:
            return f"Document search failed. Error: {str(e)}"

    def execute_pandas_analysis(intent: str, table_name: str) -> str:
        """Executes advanced Python/Pandas math (e.g., standard deviation, variance, predictive modeling)."""
        prompt = f"""
        You are an expert Data Scientist. Write Python Pandas code to solve this: "{intent}"
        TARGET TABLE: "{table_name}"
        
        RULES:
        1. The dataset is already loaded as 'df'. Define `analyze(df)`.
        2. Output ONLY raw Python code. No markdown, no backticks.
        3. STRICT VISUALIZATION RULE: NEVER generate images, plots, or use libraries like matplotlib or seaborn. If the user asks for a plot or distribution, return the raw numerical data points (e.g., a dictionary of bin distributions or top outliers) so the downstream presentation layer can construct the Plotly chart natively.
        """
        try:
            py_res = llm.invoke(prompt)
            code = py_res.content.strip()
            if code.lower().startswith("```python"):
                code = code[9:]
            if code.endswith("```"):
                code = code[:-3]
                
            is_success, result = python_worker.execute(code.strip(), table_name)
            if is_success:
                return f"Pandas Execution Success:\n{result}"
            else:
                return f"Pandas Error (Consider trying a simpler SQL query or adjusting the Python logic): {result}"
        except Exception as e:
            return f"Pandas sandbox crashed: {str(e)}"

    # --- BIND TO STRUCTURED TOOLS ---
    sql_tool = StructuredTool.from_function(
        func=query_database,
        name="query_relational_database",
        description="Use this tool to find quantitative metrics, aggregate data, or filter rows from the database. Call this FIRST if the user asks for numbers.",
        args_schema=SQLQueryInput
    )

    vector_tool = StructuredTool.from_function(
        func=search_documents,
        name="search_unstructured_documents",
        description="Use this tool to read uploaded PDFs, research papers, policies, or project logs to find qualitative context. Call this if the user asks about documents.",
        args_schema=VectorSearchInput
    )

    pandas_tool = StructuredTool.from_function(
        func=execute_pandas_analysis,
        name="execute_advanced_math",
        description="Use this tool ONLY if advanced mathematical operations are required (e.g., regressions, standard deviations) that standard SQL cannot handle.",
        args_schema=PythonSandboxInput
    )

    return [sql_tool, vector_tool, pandas_tool]

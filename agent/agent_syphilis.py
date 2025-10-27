"""Syphilis Registry Query Agent

This agent provides natural language querying capabilities for a syphilis patient registry database.
"""

import os
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.os import AgentOS
from agno.os.interfaces.agui import AGUI
import requests
import pandas as pd
import numpy as np
from decimal import Decimal
import psycopg2

# Load environment variables
import dotenv
dotenv.load_dotenv()

# Define tools
def code_set_generator(domain: str, medical_term: str, term_from_text: str = None, sentence: str = None) -> str:
    """
    Generate medical code sets by querying the terminology search API.

    Args:
        domain: The medical domain (e.g., 'Condition', 'Medication')
        medical_term: The standardized medical term
        term_from_text: The term as it appears in the text (defaults to medical_term if not provided)
        sentence: The user's query containing the term (optional)

    Returns:
        A message confirming the codeset has been displayed
    """

    print(f"Generating code set for {domain} with medical term: {medical_term}")

    # Set default values if not provided
    if term_from_text is None:
        term_from_text = medical_term

    # Prepare request
    url = "http://localhost:4000/agent_terminology_search"
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": "db3cc6ce-0b73-40f0-8cc1-462692db5d17"
    }
    payload = {
        "term_from_text": term_from_text,
        "medical_term": medical_term,
        "domain": domain,
        "sentence": sentence or f"Patient has {medical_term}"
    }

    try:
        # Make API request
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # Raise exception for HTTP errors

        # Process response
        results = response.json()

        if not results:
            return f"No codes found for {medical_term} in domain {domain}."

        # Create DataFrame from results
        df = pd.DataFrame([{
            'Concept ID': item.get('concept_id', 'N/A'),
            'Concept Name': item.get('concept_name', 'N/A'),
            'Vocabulary': item.get('vocabulary_id', 'N/A'),
            'Code': item.get('concept_code', 'N/A'),
            'Records': item.get('RC', 'N/A'),
        } for item in results])

        # Sort data frame by RC descending
        df.sort_values(by='Records', ascending=False, inplace=True)
        # Remove Records column
        df.drop(columns=['Records'], inplace=True)

        # Return the JSON results so the agent can use the codes
        return str(results)

    except requests.exceptions.RequestException as e:
        error_message = f"❌ Error fetching code set: {str(e)}"
        return error_message

def get_sql_query(user_command: str, query_type: str, concept_results: dict, concept_ids: str) -> str:
    """
    Generate a SQL query by calling the query generation server.

    Args:
        user_command: The user's analytic question.
        query_type: The type of query (e.g., 'population').
        concept_results: Dictionary of concept sets from the elucidation process.
        concept_ids: Comma-separated string of concept IDs.

    Returns:
        The generated SQL query or an error message.
    """
    url = "http://localhost:4001/generate_query"

    if os.getenv("DATABASE_TYPE") == "POSTGRES":
        cdm_schema = os.getenv("CDM_SCHEMA")
        vocab_schema = os.getenv("VOCAB_SCHEMA")

        payload = {
            "user_command": user_command,
            "query_type": "population",
            "cdm_schema": cdm_schema,
            "vocab_schema": vocab_schema,
            "concept_results": concept_results,
            "concept_ids": concept_ids,
            "database_type": "POSTGRES"
        }
    elif os.getenv("DATABASE_TYPE") == "SNOWFLAKE":
        payload = {
            "user_command": user_command,
            "query_type": "population",
            "concept_results": concept_results,
            "concept_ids": concept_ids,
            "database_type": "SNOWFLAKE"
        }
    else:
        return "Unsupported database type. Please set DATABASE_TYPE to POSTGRES or SNOWFLAKE."

    print(payload)

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        print(response.json().get("query", "No SQL query returned from the server."))
        return response.json().get("query", "No SQL query returned from the server.")
    except requests.exceptions.RequestException as e:
        return f"Error calling query generation server: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

def execute_sql_query(sql_query: str) -> str:
    """
    Execute a SQL query directly against the PostgreSQL database.

    Args:
        sql_query: The SQL query to execute

    Returns:
        Query results in a string format or error message
    """
    # Database connection parameters
    db_params = {
        'host': 'localhost',
        'database': 'postgres',
        'user': 'postgres',
        'password': '',
        'port': '5432'
    }

    try:
        # Connect to PostgreSQL
        conn = psycopg2.connect(**db_params)
        cursor = conn.cursor()

        # Execute the query
        cursor.execute(sql_query)

        # Fetch results if it's a SELECT query
        if sql_query.strip().upper().startswith('SELECT'):
            # Get column names
            col_names = [desc[0] for desc in cursor.description] if cursor.description else []

            # Fetch all rows
            rows = cursor.fetchall()

            # Convert to list of dictionaries for better presentation
            results = []
            for row in rows:
                row_dict = dict(zip(col_names, row))
                results.append(row_dict)

            # Convert results to a readable string format
            if not results:
                return "Query executed successfully but returned no results."

            # Format results as a table-like string
            result_str = "Query Results:\n"
            result_str += "-" * 50 + "\n"

            # Add column headers
            if col_names:
                result_str += " | ".join(col_names) + "\n"
                result_str += "-" * (len(" | ".join(col_names))) + "\n"

            # Add data rows
            for row in results:
                row_str = " | ".join(str(value) for value in row.values())
                result_str += row_str + "\n"

            return result_str
        else:
            # For non-SELECT queries, return affected row count
            affected_rows = cursor.rowcount
            conn.commit()
            return f"Query executed successfully. {affected_rows} rows affected."

    except psycopg2.Error as e:
        error_msg = f"Database error: {str(e)}"
        return error_msg
    except Exception as e:
        error_msg = f"An unexpected error occurred: {str(e)}"
        return error_msg
    finally:
        # Clean up connections
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

def pandas_analysis(pandas_code: str = None, data: any = None, data_context: str = None, pandas_code_context: str = None) -> str:
    """
    Execute arbitrary pandas code safely for data analysis.

    Args:
        pandas_code: String containing pandas code to execute
        data: Data to analyze (DataFrame, dict, or list of dicts) - will be available as 'df' in the execution environment
        data_context: Optional context about the data being analyzed
        pandas_code_context: Alternative parameter name for pandas_code (for agent compatibility)

    Returns:
        Analysis results as formatted string or error message
    """
    # Handle the alternative parameter name
    if pandas_code_context is not None and pandas_code is None:
        pandas_code = pandas_code_context

    # Import required libraries at the top
    import pandas as pd
    import numpy as np
    from io import StringIO
    import traceback

    # Convert data parameter to DataFrame if provided
    if data is not None:
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            df = data
        else:
            return f"Error: Unsupported data type {type(data)}. Expected DataFrame, dict, or list."
    else:
        # If no data provided, create an empty DataFrame
        df = pd.DataFrame()

    # Create a restricted globals dictionary for safe execution
    safe_globals = {
        '__builtins__': {
            'len': len,
            'str': str,
            'int': int,
            'float': float,
            'bool': bool,
            'list': list,
            'dict': dict,
            'tuple': tuple,
            'set': set,
            'range': range,
            'enumerate': enumerate,
            'zip': zip,
            'sorted': sorted,
            'sum': sum,
            'min': min,
            'max': max,
            'abs': abs,
            'round': round,
            'print': print,
        },
        'pd': pd,
        'np': np,
        'StringIO': StringIO,
        'df': df,  # Make the DataFrame available as 'df' in the execution environment
    }

    # Try to add scipy if available
    try:
        import scipy.stats as stats
        safe_globals['stats'] = stats
        safe_globals['scipy'] = __import__('scipy')
    except ImportError:
        # scipy not available, continue without it
        pass

    # Create a local variables dictionary
    safe_locals = {}

    try:
        # Execute the pandas code
        exec(pandas_code, safe_globals, safe_locals)

        # Find the result variable (look for common result variable names)
        result = None
        result_names = ['result', 'df', 'data', 'analysis', 'output']

        for name in result_names:
            if name in safe_locals:
                result = safe_locals[name]
                break

        # If no standard result variable found, take the last assigned variable
        if result is None and safe_locals:
            last_var = list(safe_locals.keys())[-1]
            result = safe_locals[last_var]

        if result is None:
            return "No result variable found. Please assign your analysis result to a variable like 'result' or 'df'."

        # Handle different result types
        if isinstance(result, pd.DataFrame):
            return _format_dataframe_result(result)
        elif isinstance(result, pd.Series):
            return _format_series_result(result)
        elif isinstance(result, (list, tuple)):
            return _format_list_result(result)
        elif isinstance(result, dict):
            return _format_dict_result(result)
        else:
            return f"Analysis Result: {str(result)}"

    except Exception as e:
        error_msg = f"Error executing pandas code: {str(e)}\n"
        error_msg += f"Code executed: {pandas_code}\n"
        error_msg += f"Traceback: {traceback.format_exc()}"
        return error_msg

def _format_dataframe_result(df: pd.DataFrame) -> str:
    """Format DataFrame results for display"""
    # Return summary information
    summary = f"DataFrame Analysis Result:\n"
    summary += f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n"
    summary += f"Columns: {', '.join(df.columns.tolist())}\n\n"

    # Show first few rows
    summary += "First 5 rows:\n"
    summary += str(df.head())

    return summary

def _format_series_result(series: pd.Series) -> str:
    """Format Series results for display"""
    result = f"Series Analysis Result:\n"
    result += f"Name: {series.name}\n"
    result += f"Length: {len(series)}\n"
    result += f"Data type: {series.dtype}\n\n"

    # Show value counts for categorical data
    if series.dtype == 'object' or len(series.unique()) < 20:
        result += "Value counts:\n"
        result += str(series.value_counts())
    else:
        result += "Summary statistics:\n"
        result += str(series.describe())

    return result

def _format_list_result(data: any) -> str:
    """Format list/tuple results for display"""
    result_type = "List" if isinstance(data, list) else "Tuple"
    result = f"{result_type} Analysis Result:\n"
    result += f"Length: {len(data)}\n\n"

    # Show first 10 items
    if len(data) <= 10:
        result += "Contents:\n"
        for i, item in enumerate(data):
            result += f"[{i}]: {item}\n"
    else:
        result += "First 10 items:\n"
        for i in range(10):
            result += f"[{i}]: {data[i]}\n"
        result += f"... and {len(data) - 10} more items"

    return result

def _format_dict_result(data: dict) -> str:
    """Format dictionary results for display"""
    result = "Dictionary Analysis Result:\n"
    result += f"Keys: {len(data)}\n\n"

    for key, value in data.items():
        if isinstance(value, (pd.DataFrame, pd.Series)):
            result += f"{key}: {type(value).__name__} with shape {value.shape}\n"
        else:
            result += f"{key}: {value}\n"

    return result

def statistical_analysis(data: any, analysis_type: str, columns: list = None, **kwargs) -> str:
    """
    Perform statistical analysis on data.

    Args:
        data: Data to analyze (DataFrame, dict, or list of dicts)
        analysis_type: Type of analysis ('summary', 'correlation', 'chi_square', 'cramers_v', 'distribution', 'groupby')
        columns: Specific columns to analyze (optional)
        **kwargs: Additional parameters for specific analyses

    Returns:
        Statistical analysis results as formatted string
    """
    try:
        # Convert data to DataFrame if needed
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            df = data
        else:
            return f"Error: Unsupported data type {type(data)}. Expected DataFrame, dict, or list."

        analysis_type = analysis_type.lower()

        if analysis_type == 'summary':
            return _statistical_summary(df, columns)

        elif analysis_type == 'correlation':
            return _correlation_analysis(df, columns)

        elif analysis_type == 'chi_square':
            return _chi_square_test(df, columns, **kwargs)

        elif analysis_type == 'cramers_v':
            return _cramers_v_test(df, columns, **kwargs)

        elif analysis_type == 'distribution':
            return _distribution_analysis(df, columns)

        elif analysis_type == 'groupby':
            return _groupby_analysis(df, columns, **kwargs)

        else:
            return f"Unsupported analysis type: {analysis_type}. Supported types: summary, correlation, chi_square, cramers_v, distribution, groupby"

    except Exception as e:
        return f"Error in statistical analysis: {str(e)}"

def _statistical_summary(df: pd.DataFrame, columns: list = None) -> str:
    """Generate statistical summary of the data"""
    if columns:
        df_subset = df[columns]
    else:
        df_subset = df

    summary = "Statistical Summary:\n"
    summary += "=" * 50 + "\n"
    summary += str(df_subset.describe(include='all'))
    summary += "\n\nData Types:\n"
    summary += str(df_subset.dtypes)
    summary += f"\n\nShape: {df_subset.shape[0]} rows × {df_subset.shape[1]} columns"

    return summary

def _correlation_analysis(df: pd.DataFrame, columns: list = None) -> str:
    """Calculate correlation matrix"""
    if columns:
        numeric_cols = [col for col in columns if pd.api.types.is_numeric_dtype(df[col])]
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 2:
        return "Error: Need at least 2 numeric columns for correlation analysis"

    corr_matrix = df[numeric_cols].corr()

    result = "Correlation Analysis:\n"
    result += "=" * 50 + "\n"
    result += f"Columns analyzed: {', '.join(numeric_cols)}\n\n"
    result += "Correlation Matrix:\n"
    result += str(corr_matrix.round(3))

    # Add interpretation
    result += "\n\nInterpretation:\n"
    result += "- Values range from -1 (perfect negative correlation) to +1 (perfect positive correlation)\n"
    result += "- Values close to 0 indicate weak or no linear relationship\n"
    result += "- Strong correlations (|r| > 0.7) are highlighted for attention"

    return result

def _chi_square_test(df: pd.DataFrame, columns: list, **kwargs) -> str:
    """Perform chi-square test of independence"""
    if not columns or len(columns) != 2:
        return "Error: Chi-square test requires exactly 2 categorical columns"

    col1, col2 = columns

    try:
        # Create contingency table
        contingency_table = pd.crosstab(df[col1], df[col2])

        # Perform chi-square test
        from scipy.stats import chi2_contingency
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)

        result = "Chi-Square Test of Independence:\n"
        result += "=" * 50 + "\n"
        result += f"Testing relationship between: {col1} and {col2}\n\n"

        result += "Contingency Table:\n"
        result += str(contingency_table)
        result += "\n\n"

        result += "Expected Frequencies:\n"
        result += str(pd.DataFrame(expected, index=contingency_table.index, columns=contingency_table.columns).round(2))
        result += "\n\n"

        result += "Test Results:\n"
        result += f"Chi-Square Statistic: {chi2:.4f}\n"
        result += f"Degrees of Freedom: {dof}\n"
        result += f"P-value: {p_value:.6f}\n"
        result += "\n\nInterpretation:\n"
        if p_value < 0.05:
            result += f"❌ REJECT null hypothesis (p < 0.05)\n"
            result += f"There is a statistically significant association between {col1} and {col2}"
        else:
            result += f"✅ FAIL to reject null hypothesis (p ≥ 0.05)\n"
            result += f"No statistically significant association found between {col1} and {col2}"

        return result

    except ImportError:
        return "Error: scipy is required for chi-square test but is not available"
    except Exception as e:
        return f"Error performing chi-square test: {str(e)}"

def _cramers_v_test(df: pd.DataFrame, columns: list, **kwargs) -> str:
    """Calculate Cramér's V for categorical association strength"""
    if not columns or len(columns) != 2:
        return "Error: Cramér's V test requires exactly 2 categorical columns"

    col1, col2 = columns

    try:
        # Create contingency table
        contingency_table = pd.crosstab(df[col1], df[col2])

        # Perform chi-square test first
        from scipy.stats import chi2_contingency
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)

        # Calculate Cramér's V
        n = contingency_table.sum().sum()
        min_dim = min(contingency_table.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * min_dim))

        result = "Cramér's V Analysis:\n"
        result += "=" * 50 + "\n"
        result += f"Measuring association strength between: {col1} and {col2}\n\n"

        result += "Contingency Table:\n"
        result += str(contingency_table)
        result += "\n\n"

        result += "Results:\n"
        result += f"Chi-Square Statistic: {chi2:.4f}\n"
        result += f"P-value: {p_value:.4f}\n"
        result += f"Sample Size: {n}\n"
        result += f"Degrees of Freedom: {dof}\n\n"

        result += "Interpretation:\n"
        if cramers_v < 0.1:
            strength = "Very weak"
        elif cramers_v < 0.2:
            strength = "Weak"
        elif cramers_v < 0.3:
            strength = "Moderate"
        elif cramers_v < 0.4:
            strength = "Strong"
        else:
            strength = "Very strong"

        result += f"Cramér's V = {cramers_v:.4f} ({strength} association)\n"
        result += "- 0.0-0.1: Very weak association\n"
        result += "- 0.1-0.2: Weak association\n"
        result += "- 0.2-0.3: Moderate association\n"
        result += "- 0.3-0.4: Strong association\n"
        result += "- 0.4+: Very strong association"

        return result

    except ImportError:
        return "Error: scipy is required for Cramér's V calculation but is not available"
    except Exception as e:
        return f"Error calculating Cramér's V: {str(e)}"

def _distribution_analysis(df: pd.DataFrame, columns: list = None) -> str:
    """Analyze distribution of numeric columns"""
    if columns:
        numeric_cols = [col for col in columns if pd.api.types.is_numeric_dtype(df[col])]
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        return "Error: No numeric columns found for distribution analysis"

    result = "Distribution Analysis:\n"
    result += "=" * 50 + "\n"

    for col in numeric_cols:
        result += f"\n{col}:\n"
        result += "-" * 30 + "\n"
        result += str(df[col].describe())
        result += "\n\nSkewness: "
        result += f"{df[col].skew():.4f}\n"
        result += f"Kurtosis: {df[col].kurtosis():.4f}\n"
        # Check for normality
        try:
            from scipy.stats import shapiro
            stat, p = shapiro(df[col].dropna().sample(min(5000, len(df[col].dropna()))))
            result += f"Shapiro-Wilk test statistic: {stat:.4f}, p-value: {p:.4f}"
            if p > 0.05:
                result += " (appears normally distributed)"
            else:
                result += " (does not appear normally distributed)"
        except:
            result += " (normality test unavailable)"

        result += "\n"

    return result

def _groupby_analysis(df: pd.DataFrame, columns: list, **kwargs) -> str:
    """Perform groupby analysis"""
    if not columns or len(columns) < 2:
        return "Error: Groupby analysis requires at least 2 columns (grouping column + analysis column)"

    group_col = columns[0]
    analysis_cols = columns[1:]

    result = "Group Analysis:\n"
    result += "=" * 50 + "\n"
    result += f"Grouping by: {group_col}\n"
    result += f"Analyzing: {', '.join(analysis_cols)}\n\n"

    try:
        grouped = df.groupby(group_col)

        for col in analysis_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                result += f"\n{col} by {group_col}:\n"
                result += str(grouped[col].agg(['count', 'mean', 'std', 'min', 'max']).round(3))
            else:
                result += f"\n{col} distribution by {group_col}:\n"
                result += str(grouped[col].value_counts())

        result += "\n\nGroup Sizes:\n"
        result += str(grouped.size())

    except Exception as e:
        result += f"Error in groupby analysis: {str(e)}"

    return result

# Load instructions from prompt file
def load_prompt(filename: str) -> str:
    """Load prompt from file"""
    try:
        with open(filename, 'r') as f:
            return f.read()
    except FileNotFoundError:
        return "You are a helpful assistant for querying a syphilis patient registry database."

instructions = load_prompt("syphilis_query_elucidation_prompt.txt")

# Create the agent
agent = Agent(
    name="Syphilis Registry Query Agent",
    model=OpenAIChat(id="gpt-4o-mini"),
    instructions=instructions,
    tools=[
        code_set_generator,
        get_sql_query,
        execute_sql_query,
    ],
    markdown=True,
)

# Set up AgentOS
agent_os = AgentOS(agents=[agent], interfaces=[AGUI(agent=agent)])
app = agent_os.get_app()

if __name__ == "__main__":
    agent_os.serve(app="agent_syphilis:app", port=8001, reload=True)

import os
import re
import sys
import io

# These imports are stable and have worked in your previous logs
from langchain_ollama import ChatOllama
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

# --- CONFIGURATION ---
DEFAULT_SQL_DB = "sqlite:///mshauri_fedha_v6.db"
DEFAULT_VECTOR_DB = "mshauri_fedha_chroma_db"
DEFAULT_EMBED_MODEL = "nomic-embed-text"
DEFAULT_LLM_MODEL = "qwen3:32b"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:25000"

# --- 1. REPLACEMENT CLASS FOR 'Tool' ---
class SimpleTool:
    """A simple wrapper to replace langchain.tools.Tool"""
    def __init__(self, name, func, description):
        self.name = name
        self.func = func
        self.description = description
    
    def run(self, input_data):
        return self.func(input_data)

class PythonREPLTool(SimpleTool):
    def __init__(self):
        super().__init__(
            name="python_calculator",
            func=self.execute_python,
            description="Useful for ANY math, statistical analysis, or data transformation. Input should be valid Python code. PRINT the final result."
        )

    def execute_python(self, code):
        try:
            f = io.StringIO()
            with redirect_stdout(f):
                exec(code, {'__builtins__': __builtins__}, {}) 
            return f.getvalue()
        except Exception as e:
            return f"Error executing code: {e}"

# --- 2. REPLACEMENT CLASS FOR THE AGENT ---
class SimpleReActAgent:
    """A manual ReAct loop that doesn't rely on langchain.agents"""
    def __init__(self, llm, tools, verbose=True):
        self.llm = llm
        self.tools = {t.name: t for t in tools}
        self.verbose = verbose
        # Create the tool description string for the prompt
        self.tool_desc = "\n".join([f"{t.name}: {t.description}" for t in tools])
        self.tool_names = ", ".join([t.name for t in tools])
        
        self.prompt_template = """You are Mshauri Fedha, a senior financial advisor for Kenya. 
        Your goal is to provide accurate, data-backed advice.
        
        RULES:
        1. CITATIONS: You MUST cite your sources. 
           - SQL Data ->
           - Text Data ->
           - Code ->
        2. ADVICE: After presenting facts, add an "Advisory Opinion" section.
        3. CONFIDENCE: If data is old, state "Low Confidence".

{tool_desc}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question with citations

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

    def invoke(self, inputs):
        query = inputs["input"]
        scratchpad = ""
        
        print(f" Starting Agent Loop for: '{query}'")
        
        for step in range(10): # Max 10 steps
            # Fill the prompt
            prompt = self.prompt_template.format(
                tool_desc=self.tool_desc,
                tool_names=self.tool_names,
                input=query,
                agent_scratchpad=scratchpad
            )
            
            # Call LLM
            # stop=["\nObservation:"] prevents the LLM from hallucinating the tool output
            response = self.llm.invoke(prompt, stop=["\nObservation:"])
            response_text = response.content
            
            if self.verbose:
                print(f"\n🧠 Step {step+1}: {response_text.strip()}")

            scratchpad += response_text

            # Check for completion
            if "Final Answer:" in response_text:
                return {"output": response_text.split("Final Answer:")[-1].strip()}

            # Parse Action
            action_match = re.search(r"Action:\s*(.*?)\n", response_text)
            input_match = re.search(r"Action Input:\s*(.*)", response_text)
            
            if action_match and input_match:
                action_name = action_match.group(1).strip()
                action_input = input_match.group(1).strip()
                
                # Execute Tool
                if action_name in self.tools:
                    if self.verbose:
                        print(f"🛠️  Calling '{action_name}' with: {action_input}")
                    
                    try:
                        # Handle both SimpleTool (.run) and LangChain Tools (.invoke or .run)
                        tool = self.tools[action_name]
                        if hasattr(tool, 'invoke'):
                            tool_result = tool.invoke(action_input)
                        else:
                            tool_result = tool.run(action_input)
                            
                    except Exception as e:
                        tool_result = f"Error executing tool: {e}"
                        
                    observation = f"\nObservation: {tool_result}\n"
                else:
                    observation = f"\nObservation: Error: Tool '{action_name}' not found. Available: {self.tool_names}\n"
                
                scratchpad += observation
            else:
                # Fallback: if no action found but also no Final Answer
                if "Action:" in response_text:
                    scratchpad += "\nObservation: You provided an Action but no Action Input. Please provide the input.\n"
                else:
                    return {"output": response_text.strip()}
                    
        return {"output": "Agent timed out."}

# --- 3. MAIN SETUP FUNCTION ---

def create_mshauri_agent(
    sql_db_path=DEFAULT_SQL_DB,
    vector_db_path=DEFAULT_VECTOR_DB,
    llm_model=DEFAULT_LLM_MODEL,
    ollama_url=DEFAULT_OLLAMA_URL,
    temperature=0.1):
    print(f"  Initializing Mshauri Fedha (Model: {llm_model})...")
    
    # 1. Initialize LLM
    try:
        llm = ChatOllama(model=llm_model, base_url=ollama_url, temperature=0.1)
    except Exception as e:
        print(f" Error connecting to Ollama: {e}")
        return None

    # 2. LEFT BRAIN (SQL)
    if "sqlite" in sql_db_path:
        real_path = sql_db_path.replace("sqlite:///", "")
        if not os.path.exists(real_path):
             print(f"  Warning: SQL Database not found at {real_path}")

    db = SQLDatabase.from_uri(sql_db_path)
    # The Toolkit returns standard LangChain tools, which our SimpleReActAgent can handle
    sql_toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    sql_tools = sql_toolkit.get_tools()

    # 3. RIGHT BRAIN (Vector)
    # We define the Retriever function manually
    def search_docs(query):
        embeddings = OllamaEmbeddings(model=DEFAULT_EMBED_MODEL, base_url=ollama_url)
        vectorstore = Chroma(persist_directory=vector_db_path, embedding_function=embeddings)
        docs = vectorstore.similarity_search(query, k=4)
        return "\n\n".join([f"[Score: {score:.2f}] {d.page_content}" for d, score in docs])

    # Use our SimpleTool wrapper instead of importing from langchain
    retriever_tool = SimpleTool(
        name="search_financial_reports_and_news",
        func=search_docs,
        description="Searches CBK/KNBS reports and business news. Use this for qualitative questions (why, how, trends) or when SQL data is missing."
    )

    # 4. CREATE AGENT
    tools = sql_tools + [retriever_tool]
    agent = SimpleReActAgent(llm, tools)
    
    print(" Mshauri Agent Ready (Zero-Dependency Mode).")
    return agent

def ask_mshauri(agent, query):
    if not agent:
        print(" Agent not initialized.")
        return

    print(f"\n User: {query}")
    print("-" * 40)
    
    try:
        response = agent.invoke({"input": query})
        print("-" * 40)
        print(f" Mshauri: {response['output']}")
        return response['output']
    except Exception as e:
        print(f" Error during execution: {e}")
        return None

if __name__ == "__main__":
    # Quick Test
    agent = create_mshauri_agent()
    ask_mshauri(agent, "What is the inflation rate?")
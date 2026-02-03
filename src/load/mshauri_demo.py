import os
import math
import re
import sys
import io
import time
from contextlib import redirect_stdout
from typing import Any, List, Optional, Mapping

# Replaces HuggingFaceEndpoint with the robust Client
from huggingface_hub import InferenceClient 
from langchain_core.language_models.llms import LLM

from langchain_ollama import ChatOllama
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

# --- CONFIGURATION ---
DEFAULT_SQL_DB = "sqlite:///mshauri_fedha_v6.db"
DEFAULT_VECTOR_DB = "mshauri_fedha_chroma_db"
DEFAULT_EMBED_MODEL = "nomic-embed-text"
DEFAULT_LLM_MODEL = "qwen2.5:3b" 
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"

# --- CUSTOM WRAPPER 
class HuggingFaceChat(LLM):
    """
    Custom LangChain wrapper that hits the Chat API (v1/chat/completions).
    This fixes the 'Task Mismatch' error for Qwen, DeepSeek, and Llama.
    """
    repo_id: str
    hf_token: str
    temperature: float = 0.1
    max_new_tokens: int = 4096

    @property
    def _llm_type(self) -> str:
        return "hf_chat_api"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
            client = InferenceClient(model=self.repo_id, token=self.hf_token)
            messages = [{"role": "user", "content": prompt}]
            
            try:
                response = client.chat_completion(
                    messages=messages, 
                    max_tokens=self.max_new_tokens, 
                    temperature=self.temperature,
                    stream=False
                )
                content = response.choices[0].message.content
                
                # DeepSeek-R1 outputs thoughts in <think> tags. We must strip them 
                # so the Agent logic doesn't get confused.
                if "<think>" in content:
                    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
                
                return content
                
            except Exception as e:
                raise ValueError(f"API Error: {e}")

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"repo_id": self.repo_id}

# --- ROBUST MODEL LIST ---
CANDIDATE_MODELS = [
    "Qwen/Qwen2.5-72B-Instruct",                # Most Powerful
    "Qwen/Qwen2.5-32B-Instruct",                # Powerful & Balanced
    "Qwen/Qwen2.5-14B-Instruct",                # Faster Alternative
    "Qwen/Qwen2.5-7B-Instruct",                 # Lightweight
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", # Smartest Logic
    "meta-llama/Meta-Llama-3.1-8B-Instruct",    # High Availability
    "mistralai/Mistral-Nemo-Instruct-2407",     # Large Context
    "HuggingFaceH4/zephyr-7b-beta",             # Old Reliable
]

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
        self.tool_desc = "\n".join([f"{t.name}: {t.description}" for t in tools])
        self.tool_names = ", ".join([t.name for t in tools])
        
        # IMPROVED PROMPT: Explicitly tells agent to switch strategies if SQL fails
        self.prompt_template = """You are Mshauri Fedha, a senior financial advisor for Kenya. 
        Your goal is to provide accurate, data-backed advice.

        SCOPE (STRICT):
        1. You answer ONLY questions related to:
        - Economy (Inflation, GDP, Trade, Fiscal Policy, transport, health economics, taxes, and other economic matters).
        - Finance (Exchange Rates, Interest Rates, Banking, Investment).
        - Business Environment (Regulations, Taxes).
        2. If the user asks about politics, sports, entertainment, or personal advice, REFUSE nicely: "My scope is limited to economic and financial matters."

        RULES:
        1. CITATIONS & CONFIDENCE: You MUST cite your sources (,).
            - When using 'search_financial_reports_and_news', the result will have a [Confidence: X%] tag.
            - You MUST include citations or sources in your final answer. If there is no citations, state "Not available in database. Based on my knowledge as of 2023-11".
            - You MUST include this confidence score in your final answer.
            - Example: "According to CBK reports (Confidence: 85%), inflation is rising..."
            - SQL Data ->
            - Text Data -> 
            - Code -> PythonREPLTool
        2. STRATEGY: 
            - First, check SQL tables ('sql_db_list_tables').
            - IF the tables listed do NOT match the user's question, IMMEDIATELY switch to 'search_financial_reports_and_news'. 
            - Do NOT keep asking for tables if they are clearly not there.
        3. ADVICE PROTOCOL (CRITICAL) ***
            - IF the user asks for facts (e.g., "What is the rate?", "Show me the trend, what was the cause? How has it changed?"):
            -> Provide ONLY the data/facts. Do NOT give advice or recommendations. Be purely objective.
            
            - IF the user asks for help/guidance (e.g., "Is this good?", "What should I do?", "How does this affect me?, What can be done? How can we improve?"):
            -> First present the facts, THEN add a distinct section titled "Advisory Opinion" with your professional recommendation based on the facts.

        Tools Available:
        {tool_desc}

        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Thought: look at the tools and the question. Which tool is best?
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (repeat Thought/Action/Observation as needed)
        Thought: I have enough info.
        Final Answer: the final answer with citations.

        Begin!

        Question: {input}
        Thought:{agent_scratchpad}"""

    def invoke(self, inputs):
        query = inputs["input"]
        scratchpad = ""
        print(f"🚀 Starting Agent Loop for: '{query}'")
        
        for step in range(10):
            prompt = self.prompt_template.format(
                tool_desc=self.tool_desc,
                tool_names=self.tool_names,
                input=query,
                agent_scratchpad=scratchpad
            )
            
            # CRITICAL: Handle API Errors inside the loop to avoid crash
            try:
                response = self.llm.invoke(prompt, stop=["\nObservation:"])
                # Handle different return types (String vs Object)
                response_text = response if isinstance(response, str) else response.content
            except Exception as e:
                print(f"LLM Error: {e}")
                return {"output": "Error contacting AI service. Please try again."}
            
            if self.verbose:
                print(f"\nStep {step+1}: {response_text.strip()}")

            scratchpad += response_text

            # Check for explicit Final Answer
            if "Final Answer:" in response_text:
                return {"output": response_text.split("Final Answer:")[-1].strip()}

            # Check for Action (Tool Use)
            action_match = re.search(r"Action:\s*(.*?)\n", response_text)
            
            #  FALLBACK: If there is NO Action and NO Final Answer, assume the whole text is the answer.
            if not action_match and "Action:" not in response_text:
                return {"output": response_text.strip()}
            input_match = re.search(r"Action Input:\s*(.*)", response_text)
            
            if action_match:
                action_name = action_match.group(1).strip()
                action_input = input_match.group(1).strip() if input_match else ""
                
                if action_name in self.tools:
                    if self.verbose:
                        print(f"🛠️ Calling '{action_name}' with: '{action_input}'")
                    try:
                        tool = self.tools[action_name]
                        if hasattr(tool, 'invoke'):
                            tool_result = tool.invoke(action_input)
                        else:
                            tool_result = tool.run(action_input)
                    except Exception as e:
                        tool_result = f"Error: {e}"
                    
                    if self.verbose:
                        print(f"Observation: {str(tool_result)[:200]}...")
                    observation = f"\nObservation: {tool_result}\n"
                else:
                    observation = f"\nObservation: Error: Tool '{action_name}' not found.\n"
                scratchpad += observation
            else:
                 if "Action:" in response_text:
                    scratchpad += "\nObservation: Missing Action Input.\n"
                 else:
                    return {"output": response_text.strip()}
                    
        return {"output": "Agent timed out."}

# --- MAIN SETUP FUNCTION ---

def create_mshauri_agent(
    sql_db_path=DEFAULT_SQL_DB,
    vector_db_path=DEFAULT_VECTOR_DB,
    llm_model=DEFAULT_LLM_MODEL,
    ollama_url=DEFAULT_OLLAMA_URL):
    print(f"Initializing Mshauri Fedha...")
    
    hf_token = os.getenv("HF_TOKEN")
    llm = None

    # 1. ROBUST SERVERLESS LOADING LOOP
    if hf_token:
        print("⚡ HF Token found. Testing models...")
        
        for model_id in CANDIDATE_MODELS:
            print(f"Trying model: {model_id}...")
            try:
                # USE CUSTOM WRAPPER
                candidate_llm = HuggingFaceChat(
                    repo_id=model_id, 
                    hf_token=hf_token,
                    temperature=0.1
                )
                # TEST CALL
                candidate_llm.invoke("Ping")
                
                print(f"SUCCESS: Connected to {model_id}")
                llm = candidate_llm
                break
            except Exception as e:
                print(f"Failed: {str(e)[:100]}...")
                time.sleep(1)

    # FALLBACK
    if not llm:
        print("\nFalling back to Local CPU Ollama...")
        try:
            llm = ChatOllama(model="qwen2.5:3b", base_url=ollama_url, temperature=0.1)
        except Exception as e:
            print(f"Error connecting to Ollama: {e}")
            return None

    # TOOLS
    if "sqlite" in sql_db_path:
        real_path = sql_db_path.replace("sqlite:///", "")
        if not os.path.exists(real_path):
             print(f"Warning: SQL Database not found at {real_path}")

    try:
        db = SQLDatabase.from_uri(sql_db_path)
        sql_toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        sql_tools = sql_toolkit.get_tools()
    except Exception as e:
        print(f"SQL Setup Failed: {e}")
        sql_tools = []

    def search_docs(query):
        embeddings = OllamaEmbeddings(model=DEFAULT_EMBED_MODEL, base_url=ollama_url)
        vectorstore = Chroma(persist_directory=vector_db_path, embedding_function=embeddings)
        
        # Get L2 Distance (Lower is better)
        results = vectorstore.similarity_search_with_score(query, k=5)
        
        formatted_results = []
        
        # SIGMA: This controls how strict you are.
        sigma = 350   # pivot point for scaling
        
        for doc, score in results:
            # Gaussian RBF Kernel
            # Formula: exp( - (score^2) / (2 * sigma^2) )
            confidence_float = math.exp(- (score**2) / (2 * sigma**2))
            
            # Convert to Percentage
            confidence_pct = int(confidence_float * 100)
            
            # Optional: Filter out low confidence noise (< 20%)
            if confidence_pct < 20:
                continue
                
            formatted_results.append(f"[Confidence: {confidence_pct}%] {doc.page_content}")

        if not formatted_results:
             return "No relevant financial documents found."

        return "\n\n".join(formatted_results)

    retriever_tool = SimpleTool(
        name="search_financial_reports_and_news",
        func=search_docs,
        description="Searches CBK reports/news."
    )
    
    repl_tool = PythonREPLTool()
    tools = sql_tools + [retriever_tool, repl_tool]
    agent = SimpleReActAgent(llm, tools)
    
    print("Agent Ready.")
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
import os
import math
import re
import sys
import io
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import redirect_stdout
from typing import Any, List, Optional, Mapping

# --- IMPORT FOR TRANSLATION ---
from deep_translator import GoogleTranslator

# Replaces HuggingFaceEndpoint with the robust Client
from huggingface_hub import InferenceClient
from langchain_core.language_models.llms import LLM

from langchain_ollama import ChatOllama
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

# --- CONFIGURATION ---
DEFAULT_SQL_DB = "sqlite:///mshauri_fedha_v6.db"
DEFAULT_VECTOR_DB = "mshauri_fedha_chroma_db"
DEFAULT_EMBED_MODEL = "nomic-embed-text"
DEFAULT_LLM_MODEL = "qwen2.5:7b" 
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"

# --- CUSTOM WRAPPER ---
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
                stream=False,
                stop=stop,
            )
            content = response.choices[0].message.content

            # Strip <think> tags if present
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
    "mistralai/Mistral-Nemo-Instruct-2407",      # Large Context
    "HuggingFaceH4/zephyr-7b-beta",             # Old Reliable
]


# --- VOTING ENSEMBLE ---
class VotingLLM:
    """
    Calls all available LLMs in parallel and selects the response with
    the highest peer-consensus score (Jaccard word similarity).

    Models that fail or time out are silently skipped, providing
    built-in fallback behavior without a separate fallback chain.
    If ALL models fail, raises ValueError so the agent can handle it.
    """
    def __init__(self, llms: list, timeout: int = 45):
        self.llms = llms
        self.timeout = timeout

    def invoke(self, prompt, stop=None):
        def call_one(llm):
            result = llm.invoke(prompt, stop=stop) if stop else llm.invoke(prompt)
            text = result if isinstance(result, str) else result.content
            return text.strip() if text else None

        responses = []
        with ThreadPoolExecutor(max_workers=len(self.llms)) as executor:
            futures = {executor.submit(call_one, llm): llm for llm in self.llms}
            try:
                for future in as_completed(futures, timeout=self.timeout):
                    llm_name = futures[future].__class__.__name__
                    try:
                        result = future.result()
                        if result:
                            responses.append(result)
                            print(f"Vote received: {llm_name}", flush=True)
                    except Exception as e:
                        print(f"Voter {llm_name} failed: {str(e)[:80]}", flush=True)
            except TimeoutError:
                print("Voting timed out. Using responses collected so far.", flush=True)

        if not responses:
            raise ValueError("All LLMs failed to respond during voting.")

        if len(responses) == 1:
            return responses[0]

        return self._pick_consensus(responses)

    def _pick_consensus(self, responses: list) -> str:
        """Returns the response with the highest average Jaccard similarity to all others.
        This is the 'centroid' of the group — the most broadly agreed-upon answer.
        """
        best_score, best_response = -1.0, responses[0]
        for i, r1 in enumerate(responses):
            words1 = set(r1.lower().split())
            scores = []
            for j, r2 in enumerate(responses):
                if i == j:
                    continue
                words2 = set(r2.lower().split())
                union = len(words1 | words2)
                scores.append(len(words1 & words2) / union if union else 0.0)
            avg = sum(scores) / len(scores) if scores else 0.0
            if avg > best_score:
                best_score, best_response = avg, r1
        print(f"Consensus winner: score={best_score:.2f}, voters={len(responses)}", flush=True)
        return best_response

def get_robust_llm():
    """Builds a voting ensemble LLM from all available providers.

    All available models vote simultaneously on every query. The response
    with the highest peer-consensus score (Jaccard word similarity) wins.
    Models that fail during a vote are silently skipped — providing
    built-in fallback behavior.

    Priority / collection order:
        1. Hugging Face  (Qwen 72B)     - requires HF_TOKEN
        2. Groq          (Llama 70B)    - requires GROQ_API_KEY
        3. Gemini        (1.5 Flash)    - requires GEMINI_API_KEY
        4. Local Ollama  (Qwen 7B)      - always included

    Returns:
        (robust_llm, base_llm)
        robust_llm: VotingLLM for the agent brain (or single model if only 1 available)
        base_llm:   Plain highest-priority model for SQLDatabaseToolkit
    """
    available_llms = []  # All working models, in priority order

    # 1. HuggingFace — test candidate models until one responds
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        print("HF Token found. Testing candidate models...", flush=True)
        for model_id in CANDIDATE_MODELS:
            print(f"  Trying: {model_id}...", flush=True)
            try:
                candidate = HuggingFaceChat(repo_id=model_id, hf_token=hf_token, temperature=0.1)
                candidate.invoke("Ping")
                available_llms.append(candidate)
                print(f"HF voter ready: {model_id}", flush=True)
                break
            except Exception as e:
                print(f"  Failed {model_id}: {str(e)[:100]}...", flush=True)
                time.sleep(0.5)

    # 2. Groq (Llama-3.3-70B)
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        groq_llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            api_key=groq_key,
        )
        available_llms.append(groq_llm)
        print("Groq voter ready.", flush=True)

    # 3. Gemini (1.5 Flash)
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        gemini_llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.1,
            google_api_key=gemini_key,
        )
        available_llms.append(gemini_llm)
        print("Gemini voter ready.", flush=True)

    # 4. Local Ollama — always included as the guaranteed baseline voter
    local_llm = ChatOllama(model="qwen2.5:7b", temperature=0)
    available_llms.append(local_llm)
    print("Ollama voter ready.", flush=True)

    if not available_llms:
        return None, None

    # base_llm: highest-priority plain model for SQLDatabaseToolkit
    # (VotingLLM is not a LangChain BaseLanguageModel and cannot be passed to the Toolkit)
    base_llm = available_llms[0]

    if len(available_llms) == 1:
        print("Single voter mode (only Ollama available).", flush=True)
        return base_llm, base_llm

    print(f"Voting ensemble active: {len(available_llms)} models will collaborate.", flush=True)
    return VotingLLM(available_llms), base_llm

# --- CLASS FOR 'Tool' ---
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

# --- CLASS FOR THE AGENT ---
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
        - Economy (Inflation, GDP, Trade, Fiscal Policy, education, transport, health economics, taxes, and other economic matters).
        - Politics (Elections, Governance, Policies impacting economy).
        - Programs (Government initiatives, social programs affecting economy).
        - Projects (Infrastructure, public works with economic impact).
        - Protests (Strikes, demonstrations with economic/political implications).
        - Finance (Exchange Rates, Interest Rates, Banking, Investment).
        - Business Environment (Regulations, Taxes).
        - Any other topics directly impacting Kenya's economic and financial landscape.
        2. If the user asks about sports, Geography, entertainment, or personal advice, REFUSE nicely: "My scope is limited to economic and financial matters."

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
        4. If you are unsure or the data is inconclusive, say "The data is inconclusive on this matter. Based on my knowledge as of 2023-11, I would recommend further monitoring and cautious decision-making."
        5. ALWAYS use the tools when relevant data is not in your immediate knowledge. Do NOT make up data or advice without tool support.
        6. If the user asks for information that is outside your scope, politely decline and remind them of your focus on economic and financial matters in Kenya.
        7. If a user asks you to predict the future, say "I cannot predict the future, but based on current trends and data (cite sources), here are some possible scenarios..."
        8. If a user asks you to compare two things, provide a clear comparison based on data, and cite your sources for each point of comparison.
        9. If a user asks you to analyze a policy or program, break down the analysis into clear sections (e.g., "Economic Impact", "Social Impact", "Political Implications") and provide data-backed insights in each section with citations.
        10. If a user asks you to explain a concept, provide a clear and concise explanation, and if possible, relate it to the Kenyan context with examples and citations.
        11. If a user asks you to summarize a report or news article, provide a brief summary of the key points, and include the confidence score from your 'search_financial_reports_and_news' tool in your summary.
        12. If a user asks you to answer in Swahili, provide your answer in Swahili, but still follow all the rules above (citations, strategy, advice protocol). You can use English sources but translate your final answer to Swahili.
        13. If a user asks you to provide data visualizations, use the PythonREPLTool to generate the visualization code, execute it, and then describe the insights from the visualization in your final answer with citations.
        14. If a user asks you to provide historical trends, use the SQL tools to extract the relevant data, and then use the PythonREPLTool to analyze and describe the trends with citations.
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
        # Flush=True forces logs to appear instantly
        print(f"Starting Agent Loop for: '{query}'", flush=True)
        
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
                print(f"\nStep {step+1}: {response_text.strip()}", flush=True)

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
                        print(f"Calling '{action_name}' with: '{action_input}'", flush=True)
                    try:
                        tool = self.tools[action_name]
                        if hasattr(tool, 'invoke'):
                            tool_result = tool.invoke(action_input)
                        else:
                            tool_result = tool.run(action_input)
                    except Exception as e:
                        tool_result = f"Error: {e}"
                    
                    if self.verbose:
                        print(f"Observation: {str(tool_result)[:200]}...", flush=True)
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

# --- 3. MULTILINGUAL AGENT WRAPPER ---
class MultilingualAgent:
    """
    Wraps the core agent to handle English/Swahili translation transparently.
    1. Detects Swahili input.
    2. Translates SW -> EN.
    3. Runs Agent in English (better reasoning/tools).
    4. Translates EN Output -> SW.
    """
    def __init__(self, agent):
        self.agent = agent
        # Initialize translators
        self.en_to_sw = GoogleTranslator(source='en', target='sw')
        self.sw_to_en = GoogleTranslator(source='sw', target='en')

    def detect_and_translate_input(self, query):
        """
        Heuristic check: If query contains common Swahili words, treat as Swahili.
        """
        # Keywords common in Kenyan financial/economic context
        swahili_keywords = [
            'habari', 'pesa', 'shilingi', 'bei', 'uchumi', 'mkopo', 'faida', 
            'hasara', 'benki', 'riba', 'soko', 'mfumuko', 'kodi', 'ushuru',
            'serikali', 'biashara', 'uwekezaji', 'bajeti', 'deni', 'kipato'
        ]
        
        # Check if any keyword exists
        is_swahili = any(word in query.lower() for word in swahili_keywords)
        
        # Also check if user explicitly requested Swahili
        if "swahili" in query.lower() or "kiswahili" in query.lower():
            is_swahili = True

        if is_swahili:
            print(f"Swahili context detected. Translating input...", flush=True)
            try:
                translated_query = self.sw_to_en.translate(query)
                print(f"   Original: '{query}' -> English: '{translated_query}'", flush=True)
                return translated_query, True
            except Exception as e:
                print(f"  Translation failed: {e}. Using original.", flush=True)
                return query, False
        
        return query, False

    def invoke(self, inputs):
        query = inputs["input"]
        
        # 1. PRE-PROCESS: Translate Input
        processed_query, is_swahili_mode = self.detect_and_translate_input(query)
        
        # 2. CORE PROCESS: Run Agent in English
        # We pass the ENGLISH query to the agent so it can use SQL/Vector tools correctly.
        result = self.agent.invoke({"input": processed_query})
        english_output = result.get("output", "Error")
        
        # 3. POST-PROCESS: Translate Output if needed
        if is_swahili_mode:
            print(f"Translating response to Swahili...", flush=True)
            try:
                # Translating the final answer
                swahili_output = self.en_to_sw.translate(english_output)
                return {"output": swahili_output}
            except Exception as e:
                print(f" Response translation failed: {e}", flush=True)
                return {"output": f"{english_output} (Translation Error)"}
        
        return {"output": english_output}

# --- MAIN SETUP FUNCTION ---

def create_mshauri_agent(
    sql_db_path=DEFAULT_SQL_DB,
    vector_db_path=DEFAULT_VECTOR_DB,
    ollama_url=DEFAULT_OLLAMA_URL):
    print("Initializing Mshauri Fedha...", flush=True)

    # Build a resilient LLM with automatic fallback cascade
    # UNPACK BOTH LLMS HERE
    robust_llm, base_llm = get_robust_llm()
    if robust_llm is None or base_llm is None:
        print("Error: No LLM could be initialised. Check your API keys and Ollama service.")
        return None

    # TOOLS
    if "sqlite" in sql_db_path:
        real_path = sql_db_path.replace("sqlite:///", "")
        if not os.path.exists(real_path):
             print(f"Warning: SQL Database not found at {real_path}")

    try:
        db = SQLDatabase.from_uri(sql_db_path)
        # PASS 'base_llm' to the Toolkit to bypass the Pydantic type-check error
        sql_toolkit = SQLDatabaseToolkit(db=db, llm=base_llm)
        sql_tools = sql_toolkit.get_tools()
    except Exception as e:
        print(f"SQL Setup Failed: {e}")
        sql_tools = []

    def search_docs(query):
        embeddings = OllamaEmbeddings(model=DEFAULT_EMBED_MODEL, base_url=ollama_url)

        # The Chroma collection was created with default L2 distance (no cosine metadata).
        # nomic-embed-text produces unit-normalised vectors, so we can recover exact cosine
        # similarity from L2 distance using: cosine_sim = 1 - (l2_distance^2 / 2)
        # We pass this as a custom relevance_score_fn so LangChain uses it instead of
        # its default L2 normalisation formula (1 - l2/2), giving true cosine percentages.
        def cosine_from_l2(l2_distance: float) -> float:
            cosine_sim = 1.0 - (l2_distance ** 2) / 2.0
            # Clamp to [0, 1] to guard against floating-point edge cases
            return max(0.0, min(1.0, cosine_sim))

        vectorstore = Chroma(
            persist_directory=vector_db_path,
            embedding_function=embeddings,
            relevance_score_fn=cosine_from_l2,
        )

        # Returns (doc, cosine_similarity) pairs in [0, 1]
        results = vectorstore.similarity_search_with_relevance_scores(query, k=5)

        formatted_results = []

        for doc, cosine_sim in results:
            # Map cosine similarity [0, 1] directly to a percentage
            confidence_pct = int(cosine_sim * 100)

            # Filter out low-confidence noise (< 20%)
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
    
    # PASS 'robust_llm' to the Agent so its brain has fallback protection
    agent = SimpleReActAgent(robust_llm, tools)
    
    # WRAP THE AGENT IN TRANSLATION LAYER
    print("Agent Ready (Multilingual Mode).", flush=True)
    return MultilingualAgent(agent)

def ask_mshauri(agent, query):
    if not agent:
        print(" Agent not initialized.")
        return

    print(f"\nUser: {query}")
    print("-" * 40)
    
    try:
        response = agent.invoke({"input": query})
        print("-" * 40)
        print(f"Mshauri: {response['output']}")
        return response['output']
    except Exception as e:
        print(f"Error during execution: {e}")
        return None

if __name__ == "__main__":
    # Quick Test
    agent = create_mshauri_agent()
    # Test English
    ask_mshauri(agent, "What is the inflation rate?")
    # Test Swahili
    ask_mshauri(agent, "Hali ya uchumi ni vipi?")
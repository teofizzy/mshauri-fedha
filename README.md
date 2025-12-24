# Mshauri Fedha 🇰🇪

**An AI-Powered Financial Research Assistant for the Kenyan Economy.**

Mshauri Fedha ("Financial Advisor") is a Proof of Concept (PoC) AI agent designed to provide accurate, data-driven insights into the Kenyan financial landscape. Unlike standard chatbots that hallucinate numbers, Mshauri Fedha uses a **Dual-Brain Architecture** to separate exact statistical retrieval from qualitative analysis.

## 🧠 The Architecture

This project solves the "Broken Table Problem" and the "Hallucination Problem" by splitting the AI's cognition into two distinct systems managed by a Supervisor Agent:

### 1. The Left Brain (Structured / SQL) 📉
* **Role:** The Mathematician.
* **Source:** A SQLite database (`mshauri_fedha_v6.db`) containing rigorous data from **KNBS (Kenya National Bureau of Statistics)** and **CBK (Central Bank of Kenya)**.
* **Capability:** Executes precise SQL queries to answer questions like *"What was the exact inflation rate in 2023?"* or *"Calculate the average tea export value for the last 5 years."*
* **Tech:** LangChain SQL Toolkit.

### 2. The Right Brain (Unstructured / Vector RAG) 📖
* **Role:** The Analyst.
* **Source:** A Vector Database (`ChromaDB`) containing:
    * **Official Reports:** Full Markdown conversions of KNBS Economic Surveys and CBK reports (tables preserved).
    * **Market News:** Cleaned and deduplicated business news articles (stripped of URLs and noise).
* **Capability:** Performs Semantic Search to answer questions like *"Why did the shilling depreciate?"* or *"What is the sentiment regarding the new Finance Bill?"*
* **Tech:** `nomic-embed-text` embeddings, ChromaDB.

### 3. The Supervisor (ReAct Agent) 🕵️
* **Role:** The Manager.
* **Logic:** A custom-built **Zero-Dependency ReAct Agent** (Python-based) that analyzes user intent and routes the query to the correct "Brain"—or both—to synthesize a comprehensive answer.

---

## 🛠️ Technology Stack

* **LLM Inference:** [Ollama](https://ollama.com/) (Local)
* **Model:** `qwen2.5:14b` (Chosen for high reasoning capability)
* **Embeddings:** `nomic-embed-text` (High performance for long documents)
* **Orchestration:** Python (Custom ReAct Implementation) & LangChain Community
* **Vector Store:** ChromaDB
* **Database:** SQLite

---

## 🚀 Setup & Installation

### 1. Prerequisites
* Python 3.10+
* [Ollama](https://ollama.com/) installed and running.

### 2. Install Dependencies
```bash
pip install pandas langchain-ollama langchain-community langchain-chroma chromadb duckduckgo-search tqdm
```

### 3. Pull Required Models
Ensure your local Ollama instance has the "Brain" and "Eyes" installed:
```bash
ollama pull qwen2.5:14b
ollama pull nomic-embed-text
```

### 4. Configuration
Ensure your Ollama server is running.

## 📂 Data Ingestion (Building the Brains)
Before the agent can work, you must populate its knowledge base

### Step 1: Ingest News (Right Brain)
Parses CSVs and chunks text.

```bash
python src/load/ingest_news.py
```

### Step 2: Ingest Reports (Right Brain)
Loads Markdown files (converted from PDFs via Marker), preserving table structures for the AI to read.
```bash
python src/load/ingest_md.py
```

### Step 3: Verify SQL (Left Brain)
Ensure `mshauri_fedha_v6.db exists`

## Usage
You can interact with Mshauri Fedha directly via the modular agent script or within a Jupyter Notebook.

**Running via Python Script**
```python
from mshauri_demo import create_mshauri_agent, ask_mshauri

# Initialize the Supervisor
agent = create_mshauri_agent()

# Ask a Hybrid Question (Uses both SQL and Vector)
ask_mshauri(agent, "What is the inflation rate in 2023 and why is it rising?")
```
**Sample Output**
```plaintext
❓ User: What is the inflation rate in 2023 and why is it rising?
----------------------------------------
🚀 Starting Agent Loop...

🧠 Step 1: Thought: The user is asking for a specific number (inflation rate) and a reason (why). 
I should first check the SQL database for the exact rate, then check the reports for the reasons.
Action: sql_db_query
Action Input: SELECT rate FROM inflation WHERE year = 2023

🧠 Step 2: Observation: [(7.7,)]
Thought: I have the rate (7.7%). Now I need the reasons.
Action: search_financial_reports_and_news
Action Input: reasons for high inflation 2023 Kenya

🧠 Step 3: Observation: ...Reports mention high fuel prices, depreciation of the shilling...
...
💡 Mshauri: The inflation rate in 2023 was 7.7%. This rise was primarily driven by increased fuel costs and the depreciation of the Kenyan Shilling against major currencies, which raised the cost of imports [Source: KNBS/News].
```
## Future Work
Chainlit UI: Deploy a chat interface for easier interaction.

GraphRAG: Implement knowledge graphs to better link entities (e.g., specific politicians to policies).

Live Scraping: Automate the news ingestion to run daily.

-----
**Status**: Prototype (Proof of concept)
**Author** Teofilo Ligawa

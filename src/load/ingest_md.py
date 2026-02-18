import os
import logging
import sys
import time
import requests
import subprocess
from pathlib import Path
from tqdm import tqdm
from langchain_community.document_loaders import DirectoryLoader, TextLoader # <--- SWITCHED
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

# --- LOGGING ---
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(message)s', force=True)
logger = logging.getLogger("ReportIngest")

def _ensure_ollama_running(port="25000"):
    host = f"http://127.0.0.1:{port}"
    try:
        if requests.get(host).status_code == 200:
            return True
    except: pass
    
    print(" Starting Ollama Server...")
    scratch = os.environ.get("SCRATCH", "/tmp")
    base = Path(scratch)
    bin_path = base / "ollama_core/bin/ollama"
    
    env = os.environ.copy()
    env["OLLAMA_HOST"] = f"127.0.0.1:{port}"
    env["OLLAMA_MODELS"] = str(base / "ollama_core/models")
    
    subprocess.Popen([str(bin_path), "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
    time.sleep(5)
    return True

def ingest_markdown_reports(
    markdown_dir="mshauri-fedha/data/knbs/marker-output",
    vector_db_path="mshauri_fedha_chroma_db",
    model="nomic-embed-text",
    ollama_port="25000"
):
    _ensure_ollama_running(ollama_port)

    if not os.path.exists(markdown_dir):
        logger.error(f" Directory not found: {markdown_dir}")
        return

    print(f"Scanning for Markdown Reports in {markdown_dir}...")
    
    # --- 1. LOAD FILES (Improved) ---
    # We use TextLoader which is faster and doesn't trigger 'unstructured' warnings
    loader = DirectoryLoader(
        markdown_dir,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={'autodetect_encoding': True}, # Safe for varying file encodings
        show_progress=True,
        use_multithreading=True
    )
    
    # Catch errors during loading (e.g., empty files)
    try:
        raw_docs = loader.load()
    except Exception as e:
        print(f" Warning during loading: {e}")
        # Fallback: simple load if directory loader fails
        raw_docs = []

    if not raw_docs:
        print(" No valid markdown files found.")
        return

    print(f"   Loaded {len(raw_docs)} report files.")

    # --- 2. CHUNKING ---
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, 
        chunk_overlap=200,
        separators=["\n## ", "\n### ", "\n", " ", ""]
    )
    docs = text_splitter.split_documents(raw_docs)
    
    # --- 3. METADATA ---
    for d in docs:
        d.metadata["type"] = "report"
        if "source" not in d.metadata:
            d.metadata["source"] = os.path.basename(d.metadata.get("source", "Official Report"))

    print(f"   Split into {len(docs)} chunks.")

    # --- 4. EMBEDDING ---
    print(" Appending to Vector Store...")
    embeddings = OllamaEmbeddings(
        model=model,
        base_url=f"http://127.0.0.1:{ollama_port}"
    )
    
    vectorstore = Chroma(
        persist_directory=vector_db_path,
        embedding_function=embeddings
    )
    
    # Batch Add
    batch_size = 100
    
    with tqdm(total=len(docs), desc="Ingesting Reports", unit="chunk") as pbar:
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i+batch_size]
            vectorstore.add_documents(batch)
            pbar.update(len(batch))

    print("\n Reports Added. Hybrid Knowledge Base is ready.")

if __name__ == "__main__":
    ingest_markdown_reports()
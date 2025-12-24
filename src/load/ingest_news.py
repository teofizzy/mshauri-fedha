import pandas as pd
import os
import re
import ast
import logging
import glob
import time
import requests
import subprocess
import sys
from pathlib import Path
from tqdm import tqdm # <--- New Import
from dateutil import parser
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

# --- CONFIG ---
MIN_CONTENT_LENGTH = 100 

# --- LOGGING ---
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(message)s', force=True)
logger = logging.getLogger("NewsIngest")

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

# --- CLEANING HELPERS ---
def clean_text(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'(?:https?|ftp)://\S+|www\.\S+', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'^[\W_]+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def parse_standard_date(date_str):
    try:
        if pd.isna(date_str): return "Unknown Date"
        dt = parser.parse(str(date_str))
        return dt.strftime("%Y-%m-%d")
    except: return "Unknown Date"

def extract_publisher_from_dict(pub_str):
    try:
        if isinstance(pub_str, str) and "{" in pub_str:
            data = ast.literal_eval(pub_str)
            return data.get('title', 'Google News')
        return str(pub_str)
    except: return "Google News"

def normalize_news_df(df, filename):
    cols = df.columns.tolist()
    normalized = []

    def create_entry(row, title_col, content_col, date_col, source_val):
        title = clean_text(row.get(title_col, ''))
        content = clean_text(row.get(content_col, ''))
        if len(content) < MIN_CONTENT_LENGTH: return None
        return {
            'title': title,
            'content': content,
            'date': parse_standard_date(row.get(date_col, '')),
            'source': source_val,
            'url': row.get('url', ''),
            'file_origin': filename
        }

    if 'publisher' in cols and 'description' in cols:
        for _, row in df.iterrows():
            source = extract_publisher_from_dict(row.get('publisher', ''))
            entry = create_entry(row, 'title', 'description', 'published date', source)
            if entry: normalized.append(entry)
    elif 'full_content' in cols:
        for _, row in df.iterrows():
            c_col = 'full_content' if isinstance(row.get('full_content'), str) and len(str(row.get('full_content'))) > 50 else 'summary'
            entry = create_entry(row, 'title', c_col, 'date', str(row.get('source', 'Unknown')))
            if entry: normalized.append(entry)
    elif 'content' in cols and 'source' in cols:
        for _, row in df.iterrows():
            entry = create_entry(row, 'title', 'content', 'date', str(row.get('source', 'Unknown')))
            if entry: normalized.append(entry)
    return pd.DataFrame(normalized)

def ingest_news_data(news_dir, vector_db_path="mshauri_fedha_chroma_db", model="nomic-embed-text"):
    _ensure_ollama_running()

    csv_files = glob.glob(os.path.join(news_dir, "*.csv"))
    if not csv_files:
        print("No files found.")
        return

    print(f" Found {len(csv_files)} news files. Processing...")
    
    all_articles = []
    
    # Progress bar for loading files
    for f in tqdm(csv_files, desc="Reading CSVs", unit="file"):
        try:
            df = pd.read_csv(f)
            clean_df = normalize_news_df(df, os.path.basename(f))
            if not clean_df.empty:
                all_articles.extend(clean_df.to_dict('records'))
        except Exception as e:
            pass

    # Deduplication
    unique_docs = {}
    for art in all_articles:
        key = f"{art['title']}_{art['date']}"
        if art['title'] in art['content']:
            page_content = f"Date: {art['date']}\nSource: {art['source']}\n\n{art['content']}"
        else:
            page_content = f"Title: {art['title']}\nDate: {art['date']}\nSource: {art['source']}\n\n{art['content']}"
        
        if key in unique_docs:
            if len(page_content) > len(unique_docs[key].page_content):
                unique_docs[key] = Document(page_content=page_content, metadata={"source": art['source'], "date": art['date'], "type": "news"})
        else:
            unique_docs[key] = Document(page_content=page_content, metadata={"source": art['source'], "date": art['date'], "type": "news"})
            
    raw_docs = list(unique_docs.values())
    print(f"   Condensed into {len(raw_docs)} unique articles.")

    # Chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200, separators=["\n\n", "\n", ". ", " "])
    final_docs = text_splitter.split_documents(raw_docs)
    
    if final_docs:
        print(f" Embedding {len(final_docs)} chunks into Vector DB...")
        embeddings = OllamaEmbeddings(model=model, base_url="http://127.0.0.1:25000")
        vectorstore = Chroma(persist_directory=vector_db_path, embedding_function=embeddings)
        
        batch_size = 100
        # Progress bar for embedding
        with tqdm(total=len(final_docs), desc="Embedding News", unit="chunk") as pbar:
            for i in range(0, len(final_docs), batch_size):
                batch = final_docs[i:i+batch_size]
                vectorstore.add_documents(batch)
                pbar.update(len(batch))
            
        print("\n News Ingestion Complete.")
    else:
        print("No valid articles extracted.")
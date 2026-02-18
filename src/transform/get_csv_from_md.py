import os
import re
import json
import io
import time
import logging
import requests
import subprocess
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine
from ollama import Client

# --- LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("KNBS_Ingest")

# --- 1. INFRASTRUCTURE ---

def _manage_ollama_server(ollama_host, ollama_port, ollama_bin, model):
    """Ensures Ollama is running (reuses existing logic)."""
    try:
        if requests.get(ollama_host).status_code == 200:
            logger.info(" Ollama connected.")
            return True
    except: pass

    logger.info(f"Starting Ollama ({model})...")
    scratch_env = os.environ.get("SCRATCH", "/tmp")
    models_dir = Path(scratch_env) / "ollama_core/models"
    
    server_env = os.environ.copy()
    server_env["OLLAMA_HOST"] = f"127.0.0.1:{ollama_port}"
    server_env["OLLAMA_MODELS"] = str(models_dir)
    
    try:
        subprocess.Popen([str(ollama_bin), "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=server_env)
        time.sleep(5)
        subprocess.run([str(ollama_bin), "pull", model], env=server_env, check=True)
        return True
    except Exception as e:
        logger.error(f" Server Error: {e}")
        return False

# --- 2. MARKDOWN PARSING ENGINE ---

def extract_tables_from_markdown(md_content: str) -> list[pd.DataFrame]:
    """
    Scans markdown text for pipe-delimited tables (| col | col |) 
    and converts them to Pandas DataFrames.
    """
    tables = []
    lines = md_content.split('\n')
    buffer = []
    inside_table = False
    
    for line in lines:
        stripped = line.strip()
        # Detect table lines (must start and end with |)
        if stripped.startswith('|') and stripped.endswith('|'):
            inside_table = True
            buffer.append(stripped)
        else:
            if inside_table:
                # Table block ended, process buffer
                if buffer:
                    try:
                        table_str = '\n'.join(buffer)
                        # Read using pandas, handling markdown separators
                        df = pd.read_csv(
                            io.StringIO(table_str), 
                            sep="|", 
                            skipinitialspace=True, 
                            engine='python'
                        )
                        
                        # CLEANUP PANDAS ARTIFACTS
                        # 1. Drop empty columns (pandas creates empty cols for leading/trailing pipes)
                        df = df.dropna(axis=1, how='all')
                        
                        # 2. Filter out the markdown divider row (e.g. ---|---|---)
                        if not df.empty:
                            df = df[~df.iloc[:,0].astype(str).str.contains('---', regex=False)]
                            
                        if not df.empty and len(df.columns) > 1:
                            tables.append(df)
                            
                    except Exception as e:
                        logger.warning(f"Failed to parse a table block: {e}")
                
                buffer = []
                inside_table = False
                
    return tables

# --- 3. LLM HEADER CLEANER (KNBS SPECIFIC) ---

def clean_knbs_headers(df: pd.DataFrame, filename: str, table_index: int, client: Client, model: str) -> pd.DataFrame:
    """
    Uses LLM to sanitize headers, handling split headers common in PDF-to-Markdown.
    """
    raw_headers = [str(c).strip() for c in df.columns]
    
    # Context: Provide first 2 rows to help identify if headers are split across rows
    data_preview = df.head(2).astype(str).values.tolist()
    
    prompt = f"""
    You are a Data Engineer cleaning Kenya National Bureau of Statistics (KNBS) data.
    
    Source File: "{filename}"
    Table Index: {table_index}
    
    Current Headers: {raw_headers}
    Data Preview (First 2 Rows): {data_preview}
    
    Task: Return a list of {len(raw_headers)} clean, snake_case SQL column names.
    
    RULES:
    1. INFER MEANING: If header is "Gross" and Row 1 is "Domestic Product", the column name is "gdp".
    2. HANDLE YEARS: If headers are "2019", "2020", keep as "year_2019".
    3. HANDLE GARBAGE: If header is "Unnamed: 1" look at Data Preview. If it contains items like "Agriculture", name it "sector".
    4. KNBS reports often have a "Total" column. Ensure it is named "total".
    
    Respond ONLY with a JSON list of strings.
    """
    
    try:
        res = client.chat(model=model, messages=[{'role': 'user', 'content': prompt}], format='json')
        new_headers = json.loads(res['message']['content'])
        
        # Handle dictionary wrapper if LLM returns {"headers": [...]}
        if isinstance(new_headers, dict):
            for val in new_headers.values():
                if isinstance(val, list): 
                    new_headers = val
                    break
        
        # Validation: Length must match
        if isinstance(new_headers, list) and len(new_headers) == len(df.columns):
            df.columns = new_headers
        else:
            # Fallback: keep originals but snake_case them
            df.columns = [re.sub(r'[^a-zA-Z0-9]', '_', str(c).strip()).lower() for c in df.columns]
            
    except Exception as e:
        logger.warning(f"LLM Header clean failed (Table {table_index}): {e}")
        
    return df

# --- 4. MAIN PIPELINE EXPORT ---

def ingest_knbs_data(input_dir: str, db_name: str, model: str = "qwen2.5:14b"):
    """
    Main entry point to run the KNBS ingestion pipeline.
    Recursively scans input_dir for all .md files.
    """
    # Paths
    SCRATCH = os.environ.get("SCRATCH", "/tmp")
    BASE_DIR = Path(SCRATCH)
    
    INPUT_PATH = Path(input_dir)
    if not INPUT_PATH.exists():
        INPUT_PATH = BASE_DIR / input_dir
    
    if not INPUT_PATH.exists():
        logger.error(f" Input directory not found: {INPUT_PATH}")
        return

    OLLAMA_BIN = BASE_DIR / "ollama_core/bin/ollama"
    CUSTOM_PORT = "25000"
    OLLAMA_HOST = f"http://127.0.0.1:{CUSTOM_PORT}"
    
    # Infrastructure
    if not _manage_ollama_server(OLLAMA_HOST, CUSTOM_PORT, OLLAMA_BIN, model): return
    
    engine = create_engine(f"sqlite:///{db_name}")
    client = Client(host=OLLAMA_HOST)
    
    # Process Files (RECURSIVE SEARCH using rglob)
    files = sorted(list(INPUT_PATH.rglob("*.md")))
    logger.info(f"Found {len(files)} KNBS markdown files (Recursive Scan). Starting ingestion...")
    
    for f in files:
        logger.info(f"Processing {f.name}...")
        try:
            with open(f, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
            
            # A. Extract Tables
            dfs = extract_tables_from_markdown(content)
            
            if not dfs:
                continue
                
            logger.info(f"   found {len(dfs)} tables.")
            
            # B. Clean & Load Tables
            for i, df in enumerate(dfs):
                # Basic cleanup
                df = df.dropna(how='all', axis=1).dropna(how='all', axis=0)
                if df.empty or len(df) < 2: continue # Skip empty/tiny tables
                
                # LLM Semantic Cleaning
                df = clean_knbs_headers(df, f.name, i, client, model)
                
                # Sanitize numeric data
                for c in df.columns:
                    if any(x in str(c).lower() for x in ['rate', 'value', 'amount', 'total', 'year', 'price']):
                        df[c] = df[c].apply(lambda x: pd.to_numeric(str(x).replace(',', '').replace('%',''), errors='ignore'))

                # Naming: knbs_{filename_slug}_tab{index}
                slug = re.sub(r'[^a-zA-Z0-9]', '_', f.stem).lower()[:40].lstrip('_')
                table_name = f"{slug}_tab{i}"
                
                df['source_file'] = f.name
                
                df.to_sql(table_name, engine, if_exists='replace', index=False)
                logger.info(f"     -> Saved table: {table_name} ({len(df)} rows)")
                
        except Exception as e:
            logger.error(f"   Failed {f.name}: {e}")

    logger.info(" KNBS Ingestion Complete.")
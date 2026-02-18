import os
import re
import json
import time
import logging
import requests
import subprocess
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from pathlib import Path
from sqlalchemy import create_engine, text
from ollama import Client

# --- LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("SmartIngestV6")

# --- 0. KNOWLEDGE BASE (THE RULES) ---

# Files to ignore completely 
SKIP_PATTERNS = [
    "december__2019_tap",
    "lcr_return",
    "lcr_sheet",
    "quarterly_gdp",
    "remittances",
    "depository_corporation_survey_(expanded)"
]

# Exact column expectations based on prior analysis [cite: 1, 2, 5, 6, 8, 14, 15, 16, 17]
SCHEMA_DEFINITIONS = {
    "annual_gdp": ["year", "month", "nominal_gdp_prices", "real_gdp_growth", "real_gdp_prices"],
    "bop_annual": ["bpm6_concept", "year_2019", "year_2020", "year_2021", "year_2022", "year_2023", "year_2024"],
    "indicative_rates": ["date", "currency", "mean_rate", "buy_rate", "sell_rate"],
    "exchange_rates": ["date", "currency", "mean_rate", "buy_rate", "sell_rate"], # Catch-all for historical/indicative
    "central_bank_rates": ["year", "month", "reverse_repo", "interbank_rate", "tbill_91_day", "tbill_182_day", "tbill_364_day", "reserve_requirement", "cbr"],
    "commercial_bank_rates": ["year", "month", "deposit_rate", "savings_rate", "lending_rate", "overdraft_rate"],
    "domestic_debt": ["fiscal_year", "treasury_bills", "treasury_bonds", "govt_stocks", "overdraft_cbk", "advances_commercial", "other_debt", "total_debt"],
    "forex_bureau": ["bureau_name", "usd_buy", "usd_sell", "usd_margin", "gbp_buy", "gbp_sell", "gbp_margin", "euro_buy", "euro_sell", "euro_margin"],
    "treasury_bills": ["issue_date", "amount_offered", "tenure", "amount_received", "amount_accepted", "yield_rate", "alloted", "rejected", "redeemed", "outstanding"],
    "treasury_bonds": ["issue_date", "bond_code", "amount_offered", "amount_received", "amount_accepted", "coupon_rate", "alloted", "rejected", "redeemed", "outstanding"],
    "exports": ["year", "month", "commodity", "value_millions", "total"],
    "imports": ["year", "month", "commodity", "value_millions", "total"],
    "revenue": ["year", "month", "tax_revenue", "non_tax_revenue", "total_revenue", "recurrent_expenditure", "development_expenditure"],
    "depository_corporation_survey": ["category", "data_values"] # Wide table handling triggered later
}

# --- 1. INFRASTRUCTURE ---

def _manage_ollama_server(ollama_host, ollama_port, ollama_bin, model):
    try:
        if requests.get(ollama_host).status_code == 200:
            logger.info(" Ollama connected.")
            return True
    except: pass

    logger.info(f" Starting Ollama ({model})...")
    scratch_env = os.environ.get("SCRATCH", "/tmp")
    models_dir = Path(scratch_env) / "ollama_core/models"
    
    server_env = os.environ.copy()
    server_env["OLLAMA_HOST"] = f"127.0.0.1:{ollama_port}"
    server_env["OLLAMA_MODELS"] = str(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        subprocess.Popen([str(ollama_bin), "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=server_env)
        time.sleep(5)
        subprocess.run([str(ollama_bin), "pull", model], env=server_env, check=True)
        return True
    except Exception as e:
        logger.error(f" Server Error: {e}")
        return False

# --- 2. HEADER HUNTER (Geometric Scanner) ---

def read_csv_robust(file_path: Path) -> pd.DataFrame:
    encodings = ['utf-8', 'latin1', 'cp1252', 'ISO-8859-1']
    for enc in encodings:
        try:
            return pd.read_csv(file_path, header=None, dtype=str, encoding=enc).fillna("")
        except UnicodeDecodeError:
            continue
    return pd.DataFrame()

def find_best_header_row(df_raw: pd.DataFrame, expected_keywords: List[str]) -> Tuple[int, int]:
    """Scores rows based on expected keywords for this specific file type."""
    scores = {}
    scan_depth = min(30, len(df_raw))
    
    # If we have no expectations, use generic keywords
    if not expected_keywords:
        expected_keywords = ['year', 'month', 'date', 'rate', 'bank', 'shilling', 'total']

    for i in range(scan_depth):
        row_str = " ".join(df_raw.iloc[i].astype(str)).lower()
        score = 0
        
        # Reward: Matches expected schema
        for kw in expected_keywords:
            if kw.lower() in row_str:
                score += 3
        
        # Penalty: Looks like Data (Dense numbers)
        num_cells = sum(1 for c in df_raw.iloc[i].astype(str) if c.replace(',','').replace('.','').isdigit())
        if num_cells > len(df_raw.columns) * 0.5:
            score -= 10
            
        scores[i] = score

    best_header = max(scores, key=scores.get)
    if scores[best_header] <= 0:
        return _geometric_scan(df_raw)
    
    return best_header, best_header + 1

def _geometric_scan(df_raw):
    """Fallback: Find first dense block of numbers."""
    def is_data(x):
        try: 
            float(str(x).replace(',', ''))
            return 1
        except: return 0
    scores = df_raw.map(is_data).sum(axis=1)
    if scores.empty or scores.max() <= 1: return 0, 1
    data_rows = scores[scores >= scores.max() * 0.5].index.tolist()
    if not data_rows: return 0, 1
    data_start = data_rows[0]
    header_idx = max(0, data_start - 1)
    # Search up for content
    while header_idx > 0:
        if df_raw.iloc[header_idx].str.join("").str.strip().any(): break
        header_idx -= 1
    return header_idx, data_start

# --- 3. HYBRID PROMPT STRATEGY ---

def get_clean_headers(raw_headers: List[str], first_row: List[str], filename: str, client: Client, model: str) -> List[str]:
    # 1. Identify File Type & Expectations
    expected_cols = []
    file_type = "generic"
    for key, cols in SCHEMA_DEFINITIONS.items():
        if key in filename.lower():
            file_type = key
            expected_cols = cols
            break
            
    # 2. Build Prompt
    valid_raw = [str(h).strip() for h in raw_headers]
    valid_data = [str(d).strip()[:15] for d in first_row]
    
    prompt = f"""
    You are a Financial Data Engineer.
    
    File: "{filename}"
    Detected Type: "{file_type}"
    Expected Schema: {expected_cols}
    
    Current Headers (Row N): {valid_raw}
    First Data Row (Row N+1): {valid_data}
    
    Task: Return a list of {len(raw_headers)} clean snake_case column names.
    
    CRITICAL RULES:
    1. PRIORITIZE THE EXPECTED SCHEMA. If the data looks like it matches the expectation, use those names.
    2. If Expected Schema has 5 cols but file has 7, keep the 5 and name the others based on context (e.g., 'total').
    3. If header is a Year ("1999"), keep it as "year_1999".
    4. If header is empty/garbage, use the Data Row to guess (e.g. "Kenya Commercial Bank" -> "bank_name").
    
    Respond ONLY with a JSON list of strings.
    """
    
    try:
        res = client.chat(model=model, messages=[{'role': 'user', 'content': prompt}], format='json')
        content = json.loads(res['message']['content'])
        
        if isinstance(content, dict):
            for val in content.values():
                if isinstance(val, list): return val
        return content if isinstance(content, list) else [f"col_{i}" for i in range(len(raw_headers))]
    except:
        # FALLBACK: If LLM fails, return the Expected Schema (padded if needed)
        if expected_cols:
            if len(expected_cols) < len(raw_headers):
                return expected_cols + [f"extra_{i}" for i in range(len(raw_headers)-len(expected_cols))]
            return expected_cols[:len(raw_headers)]
        return [f"col_{i}" for i in range(len(raw_headers))]

# --- 4. SPECIFIC TRANSFORMS ---

def apply_specific_transforms(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    fname = filename.lower()
    
    # Rule 20: Revenue & Expenditure - Remove top 3 rows
    if "revenue" in fname:
        if len(df) > 3: df = df.iloc[3:].reset_index(drop=True)
            
    # Rule 9: Depository Survey - Wide Table Logic
    if "depository_corporation" in fname:
        # This is a massive wide table. We usually want to melt it.
        # Assuming col 0 is Category and rest are dates
        try:
            id_vars = [df.columns[0]]
            value_vars = [c for c in df.columns if c != df.columns[0]]
            df = df.melt(id_vars=id_vars, value_vars=value_vars, var_name="date", value_name="amount_millions")
        except: pass

    # Rule 1/19/21/22: Year + Month merging
    # Check if we have 'year' and 'month' columns
    cols = [str(c).lower() for c in df.columns]
    if 'year' in cols and 'month' in cols:
        try:
            # Simple merge
            y_idx = cols.index('year')
            m_idx = cols.index('month')
            df['period'] = df.iloc[:, y_idx].astype(str) + '-' + df.iloc[:, m_idx].astype(str)
        except: pass
        
    return df

# --- 5. PROCESSING CORE ---

def process_file_v6(file_path: Path, engine, client, model):
    # 1. Skip Check
    if any(p in file_path.name.lower() for p in SKIP_PATTERNS):
        logger.warning(f" Skipping {file_path.name} (Blacklisted)")
        return

    logger.info(f"Processing {file_path.name}...")
    
    # 2. Read
    df_raw = read_csv_robust(file_path)
    if df_raw.empty: return

    # 3. Identify Expectations for Header Scanning
    expected_keys = []
    for key, cols in SCHEMA_DEFINITIONS.items():
        if key in file_path.name.lower():
            expected_keys = cols
            break

    # 4. Find Header
    header_idx, data_start = find_best_header_row(df_raw, expected_keys)
    
    # 5. Extract Headers
    raw_headers = df_raw.iloc[header_idx].tolist()
    
    # Double Header Check
    if header_idx > 0:
        row_above = df_raw.iloc[header_idx-1].fillna("").astype(str).tolist()
        if sum(len(x) for x in row_above) > 10:
            raw_headers = [f"{p} {c}".strip() for p, c in zip(row_above, raw_headers)]

    if len(raw_headers) != len(df_raw.columns):
        raw_headers = [f"col_{i}" for i in range(len(df_raw.columns))]

    # 6. LLM / Hybrid Map
    first_row = df_raw.iloc[data_start].tolist() if data_start < len(df_raw) else [""]*len(raw_headers)
    clean_headers = get_clean_headers(raw_headers, first_row, file_path.name, client, model)
    
    # Align Lengths
    if len(clean_headers) < len(df_raw.columns):
        clean_headers += [f"extra_{i}" for i in range(len(df_raw.columns) - len(clean_headers))]
    clean_headers = clean_headers[:len(df_raw.columns)]

    # 7. Build DF
    df = df_raw.iloc[data_start:].copy()
    df.columns = clean_headers
    
    # 8. Transforms
    df = apply_specific_transforms(df, file_path.name)
    
    # 9. Clean & Save
    df = df.loc[:, ~df.columns.str.contains('^unnamed', case=False)]
    df.dropna(thresh=1, inplace=True)
    
    for c in df.columns:
        if any(x in str(c).lower() for x in ['rate', 'value', 'amount', 'mean', 'buy', 'sell']):
            df[c] = df[c].apply(lambda x: pd.to_numeric(str(x).replace(',', '').replace('(', '-').replace(')', ''), errors='ignore'))

    table_name = re.sub(r'cbk_batch_\d+_\d+_', '', file_path.stem)
    table_name = re.sub(r'[^a-zA-Z0-9]', '_', table_name).lower()[:60].lstrip('_')
    df['source_file'] = file_path.name
    
    try:
        df.to_sql(table_name, engine, if_exists='replace', index=False)
        logger.info(f"    Saved {len(df)} rows to '{table_name}'")
    except Exception as e:
        logger.error(f"    SQL Error: {e}")

# --- MAIN ---

def process_cbk_files(input_dir: str, db_name="mshauri_fedha_v6.db", model="qwen2.5:14b"):
    SCRATCH = os.environ.get("SCRATCH", "/tmp")
    BASE_DIR = Path(SCRATCH)
    INPUT_PATH = Path(input_dir) if Path(input_dir).exists() else BASE_DIR / input_dir
    
    if not INPUT_PATH.exists(): return

    OLLAMA_BIN = BASE_DIR / "ollama_core/bin/ollama"
    CUSTOM_PORT = "25000"
    OLLAMA_HOST = f"http://127.0.0.1:{CUSTOM_PORT}"
    
    if not _manage_ollama_server(OLLAMA_HOST, CUSTOM_PORT, OLLAMA_BIN, model): return
    
    engine = create_engine(f"sqlite:///{db_name}")
    client = Client(host=OLLAMA_HOST)
    
    files = sorted(list(INPUT_PATH.glob("*.csv")))
    print(f"Processing {len(files)} files...")
    
    for f in files:
        process_file_v6(f, engine, client, model)
        
    print("\n Done.")
    with engine.connect() as conn:
        tables = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'")).fetchall()
        print(f"Created {len(tables)} tables.")

if __name__ == "__main__":
    pass
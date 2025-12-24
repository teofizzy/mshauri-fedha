import os
import json
import zipfile
import time
import gc
import threading
import shutil
import requests
import subprocess
import pymupdf
import pdfplumber
import pandas as pd
import urllib3
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Import shared config
from config import ProcessingConfig

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class UniversalProcessor:
    def __init__(self, config: ProcessingConfig):
        self.config = config
        retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 504])
        adapter = HTTPAdapter(max_retries=retry)
        self.session = requests.Session()
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def download(self, url, safe_title) -> Path:
        try:
            response = self.session.get(url, timeout=30, stream=True, verify=False)
            response.raise_for_status()
            ext = Path(url).suffix.lower() or '.pdf'
            safe_name = safe_title[:50].replace(' ', '_').replace('/', '_')
            filepath = self.config.local_dirs['pdfs'] / f"{safe_name}{ext}"
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return filepath
        except: return None

    def process(self, filepath: Path, url: str, safe_title: str):
        # Only does basic validation/extraction for the initial pass
        try:
            res = subprocess.run(["python", "pdf_canary.py", str(filepath)], capture_output=True, timeout=15)
            if res.returncode != 0: return None
            
            # Basic PyMuPDF extraction for quick preview
            doc = pymupdf.open(filepath)
            text = "".join([page.get_text() for page in doc])
            doc.close()
            
            return {
                'text': text,
                'tables': [],
                'images': [],
                'metadata': {'pages': len(doc)}
            }
        except: return None

class BatchPipeline:
    def __init__(self, config: ProcessingConfig, processor: UniversalProcessor):
        self.config = config
        self.processor = processor
        self.lock = threading.Lock()
        self.config.setup()

    def _append_log(self, log_key, record):
        with self.lock:
            with open(self.config.logs[log_key], 'a', encoding='utf-8') as f:
                f.write(json.dumps(record) + '\n')

    def _worker(self, item):
        row = item['row']
        title = str(row.get('text', 'untitled'))
        url = row['file_url']
        
        path = self.processor.download(url, title)
        if not path or path.stat().st_size < 500: return None
        
        data = self.processor.process(path, url, title)
        if not data: return None
        
        # Save Preview Text
        with open(self.config.local_dirs['texts'] / f"{path.stem}.txt", 'w') as f:
            f.write(data['text'])
            
        self._append_log('docs', {'url': url, 'file': path.name, 'status': 'downloaded'})
        return True

    def _zip_and_ship(self, batch_id):
        ts = datetime.now().strftime("%H%M%S")
        zname = f"{self.config.source_name}_{batch_id}_{ts}.zip"
        local_z = Path(f"/tmp/{zname}")
        drive_z = Path(self.config.drive_zip_dir) / zname
        
        with zipfile.ZipFile(local_z, 'w', zipfile.ZIP_DEFLATED) as z:
            for root, _, files in os.walk(self.config.local_work_dir):
                for f in files:
                    fp = os.path.join(root, f)
                    z.write(fp, os.path.relpath(fp, self.config.local_work_dir))
        
        shutil.copy(local_z, drive_z)
        self.config.setup() # Wipe local
        os.remove(local_z)

    def run(self, df, ignore_history=False):
        done = set()
        if not ignore_history and os.path.exists(self.config.logs['docs']):
            with open(self.config.logs['docs']) as f:
                for l in f: 
                    try: done.add(json.loads(l)['url'])
                    except: continue
        
        queue = [r for _, r in df.iterrows() if r['file_url'] not in done]
        print(f" Queued: {len(queue)} files.")
        
        bs = self.config.batch_size
        for i in range(0, len(queue), bs):
            batch = queue[i:i+bs]
            bid = f"batch_{i//bs + 1}"
            print(f"Processing {bid}...")
            
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as ex:
                futures = [ex.submit(self._worker, {'row': item}) for item in batch]
                for _ in tqdm(as_completed(futures), total=len(batch)): pass
            
            self._zip_and_ship(bid)
            gc.collect()

if __name__ == "__main__":
    # Example Usage
    ROOT_DIR = "/scratch/user/mshauri_data"
    conf = ProcessingConfig(root_dir=ROOT_DIR, source_name='cbk')
    pipe = BatchPipeline(conf, UniversalProcessor(conf))
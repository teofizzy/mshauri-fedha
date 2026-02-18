import random

import os
import sys
import time
import queue
import logging
import gc
import multiprocessing as mp
import argparse
from pathlib import Path
import torch

def configure_parallelism(workers_per_gpu=5):
    """
    Tuned specifically for 96GB VRAM GPUs.
    We cap workers to prevent 'Thundering Herd' and use VRAM for 'Batch Power'.
    """
    if not torch.cuda.is_available():
        return max(1, mp.cpu_count() // 2), 1, 0

    num_gpus = torch.cuda.device_count()
    gpu_properties = torch.cuda.get_device_properties(0)
    total_vram_gb = gpu_properties.total_memory / (1024**3)
    
    # --- THE STABILITY STRATEGY ---
    # On 96GB, 8-10 workers is the "Sweet Spot". 
    # More workers than this creates too much 'context switching' overhead on the GPU. 
    
    total_slots = num_gpus * workers_per_gpu
    
    print(f"GH200/A100 Detected: {num_gpus} GPUs | {total_vram_gb:.1f} GB VRAM")
    print(f"Stability Config: {workers_per_gpu} workers/GPU | {total_slots} Total Slots")
    
    # --- SYSTEM TUNING ---
    os.environ["OLLAMA_NUM_PARALLEL"] = str(total_slots)
    os.environ["OLLAMA_MAX_QUEUE"] = "2048" # Large buffer for requests
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    return total_slots, workers_per_gpu, num_gpus



# --- CRITICAL: MULTIPROCESSING SETUP ---
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass 

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from marker.config.parser import ConfigParser

# Configure Logger
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - [GPU-%(processName)s] - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def worker_routine(worker_id, gpu_id, batch_queue, output_dir, ollama_config, marker_config):
    """
    Optimized Worker for GH200:
    1. Receives a BATCH of files (List) to reduce queue overhead.
    2. Uses torch.compile for architectural optimization.
    3. Skips image extraction for speed.
    4. Fails fast on tables (Triage) to prevent LLM stalls.
    """
    
    time.sleep(worker_id * 1.5) 
    
    mp.current_process().name = f"{worker_id}:Dev{gpu_id}"
    logger.info(f"Initializing Worker {worker_id}...")
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["TORCH_DEVICE"] = f"cuda:{gpu_id}"
    
    # 2. Model Initialization
    try:
        # Load Surya/OCR weights
        artifact_dict = create_model_dict(
            device=f"cuda:{gpu_id}", 
            dtype=torch.bfloat16, 
            attention_implementation="flash_attention_2"
        )

        # --- OPTIMIZATION 1: torch.compile ---
        logger.info("Compiling models with torch.compile... (One-time setup)")
        for key, model in artifact_dict.items():
            if hasattr(model, 'forward'):
                artifact_dict[key] = torch.compile(model, mode="max-autotune")

        # --- OPTIMIZATION 2: Config Tuning ---
        full_config = {
            "output_format": "markdown",
            "disable_multiprocessing": True,
            "extract_images": False,         # Speed up: Skip image extraction
            "ocr_all_pages": False,
            "use_llm": True,
            "llm_service": "marker.services.ollama.OllamaService",
            
            # --- TRIAGE STRATEGY ---
            "max_table_retries": 0,          # Fail fast if table extraction stalls
            "llm_service_timeout": 150, # Don't let a table hold a worker for more than x minutes
            
            **ollama_config,
            **marker_config
        }
        
        config_parser = ConfigParser(full_config)
        
        converter = PdfConverter(
            config=config_parser.generate_config_dict(),
            artifact_dict=artifact_dict,
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
            llm_service=config_parser.get_llm_service()
        )
        logger.info(f"Worker Ready. Waiting for batches...")
        
    except Exception as e:
        logger.error(f"Initialization Failed: {e}")
        return

    # 3. Batch Work Loop
    batches_processed = 0
    while True:
        try:
            # Get a list of files (Batch)
            batch_files = batch_queue.get(timeout=5)
        except queue.Empty:
            logger.info(f"Queue empty. Worker shutting down. Processed {batches_processed} batches.")
            break
            
        # Process the batch locally
        for pdf_path_str in batch_files:
            try:
                pdf_path = Path(pdf_path_str)
                doc_out_dir = Path(output_dir) / pdf_path.stem
                md_file = doc_out_dir / f"{pdf_path.stem}.md"

                if md_file.exists():
                    continue

                # Heavy Compute (OCR + LLM)
                rendered = converter(str(pdf_path))
                
                if rendered is None:
                    logger.warning(f"Skipping {pdf_path.name}: Converter returned None")
                    continue

                text, meta, images = text_from_rendered(rendered)

                # Write Output
                doc_out_dir.mkdir(parents=True, exist_ok=True)
                with open(md_file, "w", encoding="utf-8") as f:
                    f.write(text)
            
            except Exception as e:
                logger.error(f"Failed on {pdf_path.name}: {e}")
        
        # Cleanup after batch to keep VRAM healthy
        batches_processed += 1
        
        # Aggressive GC after every batch prevents "memory creep" on long runs
        gc.collect()
        torch.cuda.empty_cache()

class MarkerFolderProcessor:
    def __init__(self, output_dir, ollama_url, ollama_model, batch_multiplier, workers_per_gpu, num_gpus):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_gpus = num_gpus
        
        # We now accept the dynamic num_gpus passed from __main__
        if self.num_gpus > 0:
            print(f" Detected {self.num_gpus} GPUs (Dynamic Mode)")
        else:
            print(" No GPUs detected. Running in CPU mode.")

        self.workers_per_gpu = workers_per_gpu
        
        # Configs passed to workers
        self.ollama_config = {
            "ollama_base_url": ollama_url,
            "ollama_model": ollama_model,
            "ollama_timeout": 600, # 10 mins max per request
            "ollama_options": {
                "num_ctx": 32768,
                "num_predict": 2048,
                "temperature": 0.0
            }
        }
        self.marker_config = {"batch_multiplier": batch_multiplier}

    def process_folder(self, source_folder, batch_size=10, subset=None):
        if subset is not None:
            # Use the partitioned list provided by run_transform.py
            pdfs = [Path(p) for p in subset]
        else:
            source_path = Path(source_folder)
            pdfs = sorted(list(source_path.glob("*.pdf")))
        
        if not pdfs:
            print("No PDFs to process.")
            return

        manager = mp.Manager()
        batch_queue = manager.Queue()
        
        # --- BATCHING STRATEGY ---
        # Chunk the list of PDFs into batches
        chunks = [pdfs[i:i + batch_size] for i in range(0, len(pdfs), batch_size)]
        print(f" Created {len(chunks)} batches of {batch_size} files each.")
        
        for chunk in chunks:
            batch_queue.put([str(p) for p in chunk])

        total_workers = (self.num_gpus * self.workers_per_gpu) if self.num_gpus > 0 else 1
        print(f" Launching {total_workers} workers on {self.num_gpus} GPUs...")

        processes = []
        for i in range(total_workers):
            gpu_id = i % self.num_gpus if self.num_gpus > 0 else 0
            p = mp.Process(
                target=worker_routine,
                args=(i, gpu_id, batch_queue, self.output_dir, self.ollama_config, self.marker_config)
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
            
        print(" Extraction Complete.")

# This block only runs if you execute 'python extract.py' directly.
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input folder of PDFs")
    parser.add_argument("--output", required=True, help="Output folder")
    parser.add_argument("--url", default="http://localhost:11434", help="Ollama URL")
    parser.add_argument("--model", default="llama3", help="Ollama Model Name")
    args = parser.parse_args()

    # --- DYNAMIC HARDWARE DETECTION ---
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        gpu_properties = torch.cuda.get_device_properties(0)
        total_vram_gb = gpu_properties.total_memory / (1024**3)
        
        # Calculate optimal workers: (VRAM - 2GB overhead) / 5GB per worker
        workers_per_gpu = int((total_vram_gb - 2) // 5)
        workers_per_gpu = max(1, workers_per_gpu) # Minimum 1
        
        total_slots = num_gpus * workers_per_gpu
        print(f"Dynamic Config: {num_gpus} GPUs | {workers_per_gpu} workers/GPU | {total_slots} Total Slots")
        
        # Set Env vars for external tools (optional, but good practice)
        os.environ["OLLAMA_NUM_PARALLEL"] = str(total_slots)
        
    else:
        num_gpus = 0
        workers_per_gpu = 1
        print("  No GPU detected. Defaulting to 1 worker.")

    # --- PROCESSOR INIT ---
    processor = MarkerFolderProcessor(
        output_dir=args.output,
        ollama_url=args.url,
        ollama_model=args.model,
        batch_multiplier=2,      
        workers_per_gpu=workers_per_gpu, # Passed dynamically
        num_gpus=num_gpus                # Passed dynamically
    )
    
    processor.process_folder(args.input, batch_size=10)
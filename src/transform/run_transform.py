import sys
import os
import subprocess
import time
import logging
import requests
import torch
from pathlib import Path
from datetime import timedelta

# --- 1. LOGGING SETUP ---
# Identify Node Rank for logging clarity
NODE_ID = os.environ.get("SLURM_PROCID", "0")

logging.basicConfig(
    level=logging.INFO,
    format=f'%(asctime)s - [Node {NODE_ID}] - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"logs/node_{NODE_ID}_transform.log")
    ]
)
logger = logging.getLogger(__name__)

def main():
    t_start = time.perf_counter()
    logger.info(f"Starting Transformation Pipeline on Node {NODE_ID}")

    # --- 2. ENVIRONMENT & PATHS ---
    SCRATCH = Path(os.environ.get("SCRATCH", "/tmp"))
    INPUT_PDFS_DIR = SCRATCH / "mshauri-fedha/data/knbs/pdfs"
    OUTPUT_DIR = SCRATCH / "mshauri-fedha/data/knbs/marker-output"
    
    OLLAMA_HOME = SCRATCH / "ollama_core"
    OLLAMA_BIN = OLLAMA_HOME / "bin/ollama"
    OLLAMA_HOST = "http://localhost:11434"

    # Important: Ensure the current directory is in sys.path for 'extract' import
    if os.getcwd() not in sys.path:
        sys.path.append(os.getcwd())

    try:
        from extract import MarkerFolderProcessor, configure_parallelism
    except ImportError as e:
        logger.error(f"Could not import extract.py from {os.getcwd()}")
        raise e

    # --- 3. DYNAMIC PARALLELISM & OLLAMA CONFIG ---
    # Calculates workers based on node hardware (GH200 96GB)
    total_slots, workers_per_gpu, num_gpus = configure_parallelism()
    
    # Clean up any zombie servers on this node
    subprocess.run(["pkill", "-f", "ollama serve"], stderr=subprocess.DEVNULL)
    time.sleep(5)

    # Set server environment variables
    server_env = os.environ.copy()
    server_env["OLLAMA_NUM_PARALLEL"] = str(total_slots)
    server_env["OLLAMA_MAX_LOADED_MODELS"] = "1"
    server_env["OLLAMA_MAX_QUEUE"] = "2048"
    server_env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    logger.info(f"Launching Ollama Server with {total_slots} slots...")
    subprocess.Popen(
        [str(OLLAMA_BIN), "serve"], 
        stdout=subprocess.DEVNULL, 
        stderr=subprocess.DEVNULL,
        env=server_env
    )

    # Heartbeat check
    for i in range(60):
        try:
            if requests.get(OLLAMA_HOST).status_code == 200:
                logger.info(" Ollama Server is online.")
                break
        except:
            time.sleep(1)
    else:
        raise RuntimeError(" Ollama server heartbeat failed.")

    # --- 4. MODEL SETUP ---
    BASE_MODEL = "qwen2.5:7b" 
    CUSTOM_MODEL_NAME = "qwen2.5-7b-16k"

    logger.info(f" Pulling {BASE_MODEL}...")
    subprocess.run([str(OLLAMA_BIN), "pull", BASE_MODEL], check=True, capture_output=True)

    logger.info(f" Creating custom 16k context model...")
    modelfile_path = Path(f"Modelfile_node_{NODE_ID}")
    modelfile_path.write_text(f"FROM {BASE_MODEL}\nPARAMETER num_ctx 16384")
    subprocess.run([str(OLLAMA_BIN), "create", CUSTOM_MODEL_NAME, "-f", str(modelfile_path)], check=True, capture_output=True)

    # --- 5. AUTOMATED DATA PARTITIONING ---
    # Get all PDFs and sort them for deterministic behavior
    all_pdfs = sorted(list(INPUT_PDFS_DIR.glob("*.pdf")))
    total_nodes = int(os.environ.get("SLURM_NTASKS", 1))
    node_rank = int(NODE_ID)

    # Each node takes every Nth file (Node 0 takes index 0, 2, 4... Node 1 takes 1, 3, 5...)
    my_pdfs = all_pdfs[node_rank::total_nodes]
    my_pdf_strs = [str(p) for p in my_pdfs]

    logger.info(f" Data Partitioning: Node {node_rank}/{total_nodes} handling {len(my_pdfs)} files.")

    # --- 6. EXECUTION ---
    os.chdir(SCRATCH)
    
    processor = MarkerFolderProcessor(
        output_dir=OUTPUT_DIR,
        ollama_url=OLLAMA_HOST,
        ollama_model=CUSTOM_MODEL_NAME,
        batch_multiplier=4, 
        workers_per_gpu=workers_per_gpu,
        num_gpus=num_gpus                   
    )

    logger.info(f"Processing PDFs...")
    # Using the 'subset' parameter in process_folder (ensure extract.py supports this)
    processor.process_folder(INPUT_PDFS_DIR, batch_size=5, subset=my_pdf_strs)

    # --- 7. CLEANUP & TIMING ---
    t_end = time.perf_counter()
    duration = timedelta(seconds=t_end - t_start)
    logger.info(" Transformation process finished.")
    logger.info(f"Total Duration for Node {NODE_ID}: {duration}")
    
    # Shutdown server
    subprocess.run(["pkill", "-f", "ollama serve"], stderr=subprocess.DEVNULL)
    if modelfile_path.exists(): modelfile_path.unlink()

if __name__ == "__main__":
    main()
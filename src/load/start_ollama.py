import os
import subprocess
import time
import requests
import json
import sys
from pathlib import Path

def start_ollama_server():
    """Checks if Ollama is running on port 25000, if not, starts it."""
    OLLAMA_PORT = "25000"
    OLLAMA_HOST = f"http://127.0.0.1:{OLLAMA_PORT}"
    
    # 1. Check if already running
    try:
        if requests.get(OLLAMA_HOST).status_code == 200:
            print(" Ollama is already running.")
            return True
    except:
        pass

    print(" Starting Ollama Server...")
    
    # 2. Define Paths (CSCS Environment)
    SCRATCH = os.environ.get("SCRATCH", "/tmp")
    BASE_DIR = Path(SCRATCH)
    OLLAMA_BIN = BASE_DIR / "ollama_core/bin/ollama"
    MODELS_DIR = BASE_DIR / "ollama_core/models"
    
    # 3. Setup Environment
    server_env = os.environ.copy()
    server_env["OLLAMA_HOST"] = f"127.0.0.1:{OLLAMA_PORT}"
    server_env["OLLAMA_MODELS"] = str(MODELS_DIR)
    
    # 4. Start Background Process
    try:
        subprocess.Popen(
            [str(OLLAMA_BIN), "serve"], 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL, 
            env=server_env
        )
        print(" Waiting for server to boot...")
        time.sleep(10) # Give it time to initialize
        
        # 5. Verify
        if requests.get(OLLAMA_HOST).status_code == 200:
            print(" Server started successfully.")
            return True
    except Exception as e:
        print(f" Failed to start server: {e}")
        return False



def pull_embedding_model(model_name="nomic-embed-text"):
    url = "http://127.0.0.1:25000/api/pull"
    print(f"  Requesting pull for '{model_name}'...")
    
    try:
        # Send the pull request to the running server
        response = requests.post(url, json={"name": model_name}, stream=True)
        response.raise_for_status()
        
        # Stream the progress so you know it's working
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                status = data.get('status', '')
                completed = data.get('completed', 0)
                total = data.get('total', 1)
                
                # Print progress bar or status
                if total > 1 and completed > 0:
                    percent = int((completed / total) * 100)
                    sys.stdout.write(f"\r   {status}: {percent}%")
                else:
                    sys.stdout.write(f"\r   {status}")
                sys.stdout.flush()
                
        print(f"\n Model '{model_name}' installed successfully!")
        
    except Exception as e:
        print(f"\n Failed to pull model: {e}")

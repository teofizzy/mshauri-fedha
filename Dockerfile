# Use the official Python minimal image
FROM python:3.10-slim

# 1. Install system tools
#    CRITICAL: Added 'git-lfs' so we download the REAL database files, not pointers.
#    Added 'zstd' for Ollama.
RUN apt-get update && apt-get install -y \
    curl \
    git \
    git-lfs \
    build-essential \
    zstd \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install  # <--- Initialize LFS

# 2. Install Ollama (Root)
RUN curl -fsSL https://ollama.com/install.sh | sh

# 3. Setup User
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# 4. Workdir
WORKDIR $HOME/app

# 5. Requirements
COPY --chown=user requirements.txt $HOME/app/
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy Code
COPY --chown=user . $HOME/app

# 7. Startup
#    We clone the dataset. git-lfs ensures we get the big files.
#    CHANGE: Pulling 'qwen2.5:3b' instead of '7b' for a faster fallback.
CMD git clone https://huggingface.co/datasets/teofizzy/mshauri-data data_download && \
    mv data_download/mshauri_fedha_v6.db . && \
    mv data_download/mshauri_fedha_chroma_db . && \
    rm -rf data_download && \
    echo "Starting Ollama..." && \
    ollama serve & \
    sleep 10 && \
    echo "Pulling Fallback Model (3B)..." && \
    ollama pull qwen2.5:3b && \
    ollama pull nomic-embed-text && \
    echo "Models Ready. Launching App..." && \
    streamlit run src/app.py --server.port 7860 --server.address 0.0.0.0
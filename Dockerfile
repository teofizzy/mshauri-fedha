# Use the official Python minimal image (Best for CPU Basic tier)
FROM python:3.10-slim

# 1. Install system tools
#    Added 'zstd' which is now REQUIRED by the Ollama installer
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    zstd \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Ollama (Must be done as ROOT)
RUN curl -fsSL https://ollama.com/install.sh | sh

# 3. Setup User (Hugging Face Requirement)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# 4. Set Working Directory
WORKDIR $HOME/app

# 5. Install Python Requirements
COPY --chown=user requirements.txt $HOME/app/
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy Application Code
COPY --chown=user . $HOME/app

# 7. STARTUP COMMAND
CMD git clone https://huggingface.co/datasets/teofizzy/mshauri-data data_download && \
    mv data_download/mshauri_fedha_v6.db . && \
    mv data_download/mshauri_fedha_chroma_db . && \
    rm -rf data_download && \
    echo "⬇️ Starting Ollama..." && \
    ollama serve & \
    sleep 10 && \
    echo "⬇️ Pulling Models..." && \
    ollama pull qwen2.5:7b && \
    ollama pull nomic-embed-text && \
    echo "✅ Models Ready. Launching App..." && \
    streamlit run src/app.py --server.port 7860 --server.address 0.0.0.0
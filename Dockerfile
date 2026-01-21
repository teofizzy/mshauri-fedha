# Use the official Python minimal image (Best for CPU Basic tier)
FROM python:3.10-slim

# 1. Install system tools (curl for Ollama, git for data cloning)
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. Setup User (Hugging Face Requirement)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

# 3. Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# 4. Install Python Requirements
COPY --chown=user requirements.txt $HOME/app/
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy Application Code
COPY --chown=user . $HOME/app

# 6. STARTUP COMMAND
# A. Clone Data from Dataset Repo
# B. Move Data to root (so src/app.py can find it easily)
# C. Start Ollama Server
# D. Pull Models (7b fits in 16GB RAM)
# E. Run Streamlit pointing to src/app.py
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
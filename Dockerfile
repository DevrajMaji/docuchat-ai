# ── Hugging Face Spaces Dockerfile ──────────────────────────────────────────
# HF Spaces requires the app to listen on port 7860

FROM python:3.10-slim

# Metadata
LABEL maintainer="Devraj Maji"
LABEL description="DocuChat AI — End-to-End RAG Document Chatbot"

# Set working directory
WORKDIR /app

# System dependencies for faiss-cpu and sentence-transformers
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download the sentence-transformer model so cold starts are fast
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy the rest of the project
COPY . .

# Expose port 7860 (required by Hugging Face Spaces)
EXPOSE 7860

# HF Spaces injects secrets as env variables (GROQ_API_KEY)
ENV GROQ_API_KEY=""

# Run Streamlit on port 7860
CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.fileWatcherType=none", \
     "--browser.gatherUsageStats=false"]

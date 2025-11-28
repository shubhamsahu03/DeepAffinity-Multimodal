# Use a lightweight Python base
FROM python:3.9-slim

# 1. Setup User (Hugging Face Requirement)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# 2. Setup Working Directory
WORKDIR /app

# 3. Setup Cache Permissions
ENV TRANSFORMERS_CACHE=/app/cache/transformers
ENV TORCH_HOME=/app/cache/torch
RUN mkdir -p /app/cache/transformers && mkdir -p /app/cache/torch

# 4. Install System Dependencies (Root needed temporarily)
USER root
RUN apt-get update && apt-get install -y \
    libxrender1 \
    libxext6 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Switch back to user
USER user

# 5. Install Python Libraries
# CRITICAL FIX: We pin "numpy<2.0" to prevent the PyTorch crash
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir \
    "numpy<2.0" \
    torch==2.1.0 \
    torch-geometric \
    rdkit \
    transformers \
    streamlit \
    pandas \
    matplotlib \
    seaborn

# 6. Copy Project Files
COPY --chown=user . /app

# 7. Expose HF Port
EXPOSE 7860

# 8. Launch App
CMD ["streamlit", "run", "app/app.py", "--server.port=7860", "--server.address=0.0.0.0"]
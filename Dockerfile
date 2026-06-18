FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y git build-essential && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
# CPU-only for Codespaces (no FAISS-GPU)
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install faiss-cpu==1.8.0  # override GPU version for CPU environment
COPY . .
RUN pip install -e .
EXPOSE 7860
CMD ["python", "demo/app.py"]

# Hugging Face Spaces runs this to build and start the app.
FROM python:3.12-slim

# A writable home for the model cache; Spaces runs as a non-root user.
ENV HOME=/home/user \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/user/.cache/huggingface \
    ANONYMIZED_TELEMETRY=False \
    HF_HUB_DISABLE_TELEMETRY=1 \
    ASK_PROVIDERS=google,groq

RUN useradd -m -u 1000 user
USER user
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

# Download the embedding and reranking models at build time, so the first
# visitor does not wait for them.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
    SentenceTransformer('all-MiniLM-L6-v2'); \
    CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Build the search index at image build time rather than on first request.
RUN python src/extract_chunks.py && python src/build_index.py

EXPOSE 7860
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "7860"]

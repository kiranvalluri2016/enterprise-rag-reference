FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir .
ENV RAG_HOST=0.0.0.0 RAG_PORT=8000
EXPOSE 8000
USER 65534:65534
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8000/live', timeout=2)"
CMD ["python", "-m", "rag_reference.api"]

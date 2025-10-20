FROM python:3.11-slim

# Create user with ID 1000 (required for HF Spaces)
RUN useradd -m -u 1000 user

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    huggingface-hub==0.20.1 \
    python-multipart==0.0.6 \
    jinja2==3.1.2 \
    httpx==0.25.2

# Create data directory with proper permissions
RUN mkdir -p /data && chown -R user:user /data

# Copy application files
COPY --chown=user:user app.py /app/

# Switch to user
USER user

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]


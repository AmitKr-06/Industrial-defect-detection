FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_SOURCE=local \
    TORCH_HOME=/opt/torch

WORKDIR /app

# 1) PyTorch CPU wheels: heavy layer that rarely changes, so it is cached
RUN pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 \
        --index-url https://download.pytorch.org/whl/cpu

# 2) Remaining Python dependencies
COPY docker/docker-requirements.txt .
RUN pip install --no-cache-dir -r docker-requirements.txt

# 3) Download the backbone weights into the image (no internet needed at runtime)
RUN python -c "from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights; wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)"

# 4) Application code 
COPY app/ ./app/
COPY src/ ./src/
COPY params.yaml .

# 4b) ONLY the two inference artifacts — NOT the 2.1 GB of training dumps
COPY models/memory_bank.pt ./models/
COPY models/threshold.json ./models/

# 5) Non-root user with writable logs and uploads folders
RUN useradd --create-home appuser \
    && mkdir -p /app/logs /app/uploads \
    && chown -R appuser:appuser /app /opt/torch
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status == 200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
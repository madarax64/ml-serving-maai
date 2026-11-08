FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN pip install torch==2.15.0+cu132 torchvision==0.29.2+cu132 \
        --extra-index-url https://download.pytorch.org/whl/cu132 \
    && pip install pillow==13.0.0 litserve==0.2.19
RUN useradd --create-home --uid 1000 app
USER app
WORKDIR /app

COPY --chown=app:app fasterrcnn_resnet50_fpn_v2_coco.pth .
COPY --chown=app:app server.py .

EXPOSE 24242

# LitServe serves /health. The long start period covers model loading.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:24242/health', timeout=3)"

CMD ["python", "server.py"]

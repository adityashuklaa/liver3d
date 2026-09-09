# CPU image. For GPU, start from pytorch/pytorch:*-cuda*-runtime and drop the
# CPU index-url below.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1     PIP_NO_CACHE_DIR=1     LIVER3D_WORK_DIR=/app/outputs/api

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends         libgl1 libglib2.0-0     && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch     && pip install -r requirements.txt

COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY docs ./docs

RUN useradd -m -u 10001 liver3d && mkdir -p /app/outputs && chown -R liver3d /app
USER liver3d

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s     CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/healthz')"

# LIVER3D_CHECKPOINT and LIVER3D_API_KEYS must be supplied at run time.
CMD ["uvicorn", "src.api.service:app", "--host", "0.0.0.0", "--port", "8000"]

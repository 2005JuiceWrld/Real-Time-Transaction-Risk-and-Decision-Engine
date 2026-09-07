# Shared image for every plain-Python service: the API, the Kafka producer,
# the stream scorer, and the one-off data-gen/train jobs. They differ only
# in the command docker-compose runs, not in what's installed — one image,
# several roles, rather than near-identical Dockerfiles copy-pasted four times.
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY scripts ./scripts
COPY configs ./configs

# models/ and data/ are bind-mounted at runtime (see docker-compose.yml) —
# the model artifact and datasets are produced by scripts/train_model.py and
# shared with every service that reads them, not baked into the image.

CMD ["uvicorn", "src.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]

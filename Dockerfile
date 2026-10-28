FROM python:3.11-slim@sha256:e88e9763f943ec1834f992a4b51e0f24500486803e8bc534e5767af9ea65f6ce
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 II_PROJECT_ROOT=/app II_ARTIFACT_ROOT=/app/reports/runtime
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir --timeout 120 pip==26.2.1
COPY pyproject.toml setup.py MANIFEST.in README.md requirements-core-linux-py311.txt ./
COPY src/industrial_inspection ./src/industrial_inspection
COPY configs ./configs
COPY scripts ./scripts
COPY web ./web
RUN python -m pip install --no-cache-dir --timeout 60 --resume-retries 3 -c requirements-core-linux-py311.txt .
EXPOSE 8000
CMD ["uvicorn", "industrial_inspection.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

# --- stage 1: build the frontend -------------------------------------------
FROM node:20-slim AS frontend
WORKDIR /app/web
COPY web/package.json web/package-lock.json* ./
RUN npm ci
COPY web/ ./
RUN npm run build

# --- stage 2: backend runtime ------------------------------------------------
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
WORKDIR /app

# git: ingest_job.py shells out to it to clone repos. ca-certificates: needed
# for both git-over-https and outbound calls to the OpenAI API.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project

COPY api/ ./api/
COPY pipeline/ ./pipeline/
COPY index_repo.py query.py ask.py ./
COPY --from=frontend /app/web/dist ./web/dist

# Overridden by fly.toml to point at the mounted volume; this default only
# matters for `docker run` without an explicit volume.
ENV APP_DATA_DIR=/data
EXPOSE 8000

CMD ["uv", "run", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

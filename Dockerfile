# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:0.13.1@sha256:4a1f3666816dafadd20ceb71eac9a7d56d7ef30279f7d64048519ab901ada336 AS uv

FROM python:3.15-slim@sha256:65a65fc7b30b06cb6df4930fd4650697cb23f4c2698ab81fc87e4b031bc36928 AS build
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-dev --no-install-project
COPY alembic.ini ./
COPY migrations ./migrations
COPY app ./app
# One-off jobs, as the database owner: `python -m scripts.seed_demo` and `scripts.grant_access`.
COPY scripts ./scripts

FROM python:3.15-slim@sha256:65a65fc7b30b06cb6df4930fd4650697cb23f4c2698ab81fc87e4b031bc36928 AS runtime
RUN useradd --system --uid 10001 --no-create-home app
WORKDIR /app
COPY --from=build --chown=app:app /app /app
# PYTHONSAFEPATH keeps the working directory off sys.path, and the project isn't installed into
# the venv, so PYTHONPATH names /app explicitly: `python -m app.workers.connector` (the worker
# deployment's command) and alembic then import `app`; uvicorn already adds its app dir.
ENV PATH="/app/.venv/bin:$PATH" PYTHONSAFEPATH=1 PYTHONPATH=/app
USER app
EXPOSE 8000
# Keep-alive above the ALB's 60 s idle timeout; graceful shutdown inside the pod's grace period.
# Set FORWARDED_ALLOW_IPS to the load balancer's subnet range in the deployment.
CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", \
     "--proxy-headers", "--timeout-keep-alive", "75", "--timeout-graceful-shutdown", "20"]

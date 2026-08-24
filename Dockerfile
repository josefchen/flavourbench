FROM python:3.12.11-slim-bookworm@sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN useradd --create-home --uid 10001 flavourbench

COPY --chown=flavourbench:flavourbench pyproject.toml requirements.lock README.md alembic.ini Dockerfile ./
COPY --chown=flavourbench:flavourbench alembic ./alembic
COPY --chown=flavourbench:flavourbench contracts ./contracts
COPY --chown=flavourbench:flavourbench src ./src

RUN pip install --require-hashes --no-deps -r requirements.lock \
    && pip install --no-deps --no-build-isolation .

USER flavourbench

EXPOSE 8090
CMD ["uvicorn", "flavourbench.main:app", "--host", "0.0.0.0", "--port", "8090"]

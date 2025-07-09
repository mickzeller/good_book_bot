FROM python:3.12-slim
LABEL authors="mickzeller"

WORKDIR /app

# Install Poetry with cache mount 🔥
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install poetry==1.8.3

RUN poetry config virtualenvs.create false

# dependencies are only re-installed if these files change.
COPY pyproject.toml poetry.lock* ./

RUN --mount=type=cache,target=/root/.cache/pypoetry \
    --mount=type=cache,target=/root/.cache/pip \
    poetry install --no-interaction --no-ansi --no-root

COPY . .

EXPOSE 8888

CMD ["poetry", "run", "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
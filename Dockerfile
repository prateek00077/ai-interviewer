# Two stages: build wheels with a compiler present, run without one.
#
# WHY THE RUNTIME STAGE IS NOT `-slim` WITHOUT EXTRAS: WeasyPrint does not
# bundle a text stack. It links against Pango, Cairo and HarfBuzz at *runtime*
# via ctypes, so a missing library is not a build error -- it is an ImportError
# the first time a report renders, in a worker, in production. The apt list
# below is exactly that stack plus libmagic.
#
# The image serves both roles. The API and the worker share every dependency
# except the entrypoint, and two images would double the build time and let
# their dependency sets drift apart silently.

FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# build-essential for anything without a manylinux wheel; libpq-dev is not
# needed (asyncpg is pure-python plus its own wheel) but curl is, for uv-style
# installs and for the healthcheck in compose.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY pyproject.toml README.md ./
COPY app ./app

# The voice extra is NOT optional for this image, despite being an extra in
# pyproject. `app.main` imports the API router, which reaches session_manager,
# which imports pipecat -- so a plain `pip install .` produces a container that
# builds cleanly and dies with ModuleNotFoundError the moment uvicorn starts.
# The extra exists so that CI and auth-only work install in seconds, not because
# the deployed API can do without it.
#
# `nim` too: the Riva client is what the offline transcript pass streams
# through, and that runs in the worker.
#
# Wheels into a venv, which the runtime stage copies whole -- it never sees a
# compiler.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip && pip install '.[voice,nim]'


FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

# Runtime shared libraries for two ctypes-loading dependencies. See the note at
# the top of this file: omitting any of these produces a green build that dies
# on first use.
#
#   libpango/cairo/harfbuzz/gdk-pixbuf  WeasyPrint -- first report render
#   libgl1, libglib2.0-0                OpenCV, pulled in by pipecat -- app import
#
# libGL in particular is needed by a *headless* server that will never draw
# anything: opencv-python links it unconditionally, and the ImportError it
# raises names libGL.so.1 rather than opencv, which makes it a confusing
# half-hour if you meet it in production instead of here.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpango-1.0-0 \
        libpangoft2-1.0-0 \
        libharfbuzz0b \
        libcairo2 \
        libgdk-pixbuf-2.0-0 \
        libffi8 \
        libgl1 \
        libglib2.0-0 \
        shared-mime-info \
        fonts-dejavu-core \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

WORKDIR /srv
COPY app ./app
COPY alembic ./alembic
COPY alembic.ini ./
COPY config ./config
COPY scripts ./scripts
# The manual test console. Not mounted when ENVIRONMENT=production (see
# api/dev.py), but the image is what people run locally to try the product, and
# without this file /dev answers 500 -- which reads as a broken build rather
# than a missing asset.
COPY static ./static

# Non-root. Nothing here writes to the filesystem -- uploads go to S3 and logs
# go to stdout -- so there is no reason for the process to own its own code.
RUN useradd --system --create-home --uid 10001 aii \
    && chown -R aii:aii /srv
USER aii

EXPOSE 8000

# Liveness only. Readiness is /ready and belongs to the orchestrator, which can
# pull an instance out of rotation without killing it; a Docker HEALTHCHECK can
# only restart, and restarting on a database blip is the wrong response.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

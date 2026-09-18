FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HUNCH_HOST=0.0.0.0 HUNCH_PORT=8791 HUNCH_LOG=warning
WORKDIR /app
COPY requirements.lock .
RUN apt-get update && apt-get install -y --no-install-recommends tini && rm -rf /var/lib/apt/lists/* \
 && pip install -r requirements.lock
COPY hunch ./hunch
COPY hunch.toml.example ./
RUN useradd --system --uid 10001 hunch && chown -R hunch /app
USER hunch
EXPOSE 8791
HEALTHCHECK --interval=30s --timeout=6s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8791/health', timeout=5).status == 200 else 1)"
# Listening on 0.0.0.0 requires HUNCH_API_KEYS (or HUNCH_ALLOW_NOAUTH=1 behind an authenticating proxy).
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-m", "hunch"]

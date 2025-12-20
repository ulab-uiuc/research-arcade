FROM python:3.11-slim

LABEL maintainer="ResearchArcade"
LABEL description="Continuous arXiv paper crawler"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    cron \
    tzdata \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app
COPY app/ /app/
COPY research_arcade/ /app/research_arcade/

# Copy requirements first for better caching
COPY docker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ /app/

# Copy application code
COPY . .

# Create directories for output
RUN mkdir -p /app/download /app/arxiv_ids /app/logs

# Copy and set up entrypoint
COPY docker/docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# Default environment variables for crawling
ENV CRAWL_INTERVAL="daily"
ENV CRAWL_HOUR="6"
ENV CRAWL_MINUTE="0"
ENV ARXIV_FIELD=""
ENV DEST_DIR="/app/download"
ENV ARXIV_ID_DEST="/app/arxiv_ids"
ENV LOOKBACK_DAYS="1"

# Expose volume mount points
VOLUME ["/app/download", "/app/arxiv_ids", "/app/logs"]

ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["continuous"]

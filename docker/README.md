# arXiv Continuous Crawler

A Docker-based continuous arXiv paper crawler for ResearchArcade.

## Quick Start

### Build the image

```bash
docker build -t arxiv-crawler .
```

### Run with Docker Compose (recommended)

```bash
# Start the crawler
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the crawler
docker-compose down
```

### Run with Docker directly

```bash
# Continuous mode (default) - runs immediately then loops
docker run -d \
  -e ARXIV_FIELD=cs.AI \
  -e CRAWL_INTERVAL=daily \
  -e LOOKBACK_DAYS=1 \
  -v $(pwd)/data/download:/app/download \
  -v $(pwd)/data/arxiv_ids:/app/arxiv_ids \
  -v $(pwd)/data/logs:/app/logs \
  --name arxiv-crawler \
  arxiv-crawler continuous

# One-time run
docker run --rm \
  -e ARXIV_FIELD=cs.AI \
  -v $(pwd)/data:/app/download \
  arxiv-crawler once 2024-01-01 2024-01-31

# Cron-based scheduling
docker run -d \
  -e CRAWL_INTERVAL=daily \
  -e CRAWL_HOUR=6 \
  -e CRAWL_MINUTE=0 \
  -v $(pwd)/data:/app/download \
  --name arxiv-crawler \
  arxiv-crawler cron
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CRAWL_INTERVAL` | `daily` | `hourly`, `daily`, `weekly`, or number of seconds |
| `CRAWL_HOUR` | `6` | Hour to run (0-23, UTC) |
| `CRAWL_MINUTE` | `0` | Minute to run (0-59) |
| `ARXIV_FIELD` | (empty) | arXiv category (e.g., `cs.AI`, `cs.CV`, `stat.ML`) |
| `LOOKBACK_DAYS` | `1` | Number of days to look back |
| `DEST_DIR` | `/app/download` | Directory for downloaded papers |
| `ARXIV_ID_DEST` | `/app/arxiv_ids` | Directory for arXiv ID lists |
| `TZ` | `UTC` | Timezone |

## Modes

### Continuous (default)
Runs the crawler immediately, then sleeps for the specified interval before running again. Simple and reliable.

```bash
docker run arxiv-crawler continuous
```

### Cron
Uses cron for precise scheduling. Better for specific time-based requirements.

```bash
docker run arxiv-crawler cron
```

### Once
Runs a single crawl and exits. Useful for testing or one-off crawls.

```bash
docker run arxiv-crawler once 2024-01-01 2024-01-31
```

## Multi-Field Crawling

To crawl multiple arXiv categories, use the multi-field profile:

```bash
docker-compose --profile multi-field up -d
```

This starts separate crawlers for cs.AI, stat.ML, and cs.CV with staggered schedules.

## Volumes

| Container Path | Description |
|----------------|-------------|
| `/app/download` | Downloaded papers and metadata |
| `/app/arxiv_ids` | Lists of processed arXiv IDs |
| `/app/logs` | Crawler logs (daily rotation) |

## Directory Structure

```
arxiv-crawler/
├── Dockerfile
├── docker-compose.yml
├── docker-entrypoint.sh
├── requirements.txt
├── README.md
├── scripts/
│   └── crawl_arxiv.py
├── paper_crawler/
│   └── crawler_job.py
└── data/                    # Created at runtime (mounted volumes)
    ├── download/
    ├── arxiv_ids/
    └── logs/
```

## Monitoring

### View logs
```bash
# Docker Compose
docker-compose logs -f arxiv-crawler

# Docker
docker logs -f arxiv-crawler

# Inside container
tail -f /app/logs/crawler_*.log
```

### Check status
```bash
docker-compose ps
```

### Interactive shell
```bash
docker-compose exec arxiv-crawler bash
# or
docker run -it arxiv-crawler shell
```

## Customization

### Custom cron schedule
Set `CRAWL_INTERVAL` to a cron expression:

```bash
docker run -e CRAWL_INTERVAL="0 */6 * * *" arxiv-crawler cron
```

### Add dependencies
Edit `requirements.txt` and rebuild:

```bash
docker-compose build --no-cache
docker-compose up -d
```

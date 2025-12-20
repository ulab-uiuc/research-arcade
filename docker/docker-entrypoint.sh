#!/bin/bash
set -e

# Colors for logging
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Function to run the crawler
run_crawler() {
    local start_date=$1
    local end_date=$2
    
    log_info "Starting crawl from $start_date to $end_date"
    
    FIELD_ARG=""
    if [ -n "$ARXIV_FIELD" ]; then
        FIELD_ARG="--field $ARXIV_FIELD"
    fi
    
    python -u scripts/crawl_arxiv.py \
        --start-date "$start_date" \
        --end-date "$end_date" \
        --dest "$DEST_DIR" \
        --arxiv_id_dest "$ARXIV_ID_DEST" \
        $FIELD_ARG \
        2>&1 | tee -a /app/logs/crawler_$(date +%Y%m%d).log
    
    local exit_code=${PIPESTATUS[0]}
    
    if [ $exit_code -eq 0 ]; then
        log_info "Crawl completed successfully"
    else
        log_error "Crawl failed with exit code $exit_code"
    fi
    
    return $exit_code
}

# Calculate start date based on lookback days
get_start_date() {
    date -d "$LOOKBACK_DAYS days ago" +%Y-%m-%d
}

# Get today's date
get_today() {
    date +%Y-%m-%d
}

# Setup cron job for scheduled crawling
setup_cron() {
    log_info "Setting up cron job..."
    
    # Create the cron script
    cat > /app/run_crawler.sh << 'CRONSCRIPT'
#!/bin/bash
cd /app
source /etc/environment

FIELD_ARG=""
if [ -n "$ARXIV_FIELD" ]; then
    FIELD_ARG="--field $ARXIV_FIELD"
fi

START_DATE=$(date -d "$LOOKBACK_DAYS days ago" +%Y-%m-%d)
END_DATE=$(date +%Y-%m-%d)

python -u scripts/crawl_arxiv.py \
    --start-date "$START_DATE" \
    --end-date "$END_DATE" \
    --dest "$DEST_DIR" \
    --arxiv_id_dest "$ARXIV_ID_DEST" \
    $FIELD_ARG \
    >> /app/logs/crawler_$(date +%Y%m%d).log 2>&1
CRONSCRIPT
    chmod +x /app/run_crawler.sh
    
    # Export environment variables for cron
    printenv | grep -E "^(ARXIV_|DEST_|LOOKBACK_|PATH|PYTHON)" >> /etc/environment
    
    # Create cron schedule based on interval
    case "$CRAWL_INTERVAL" in
        "hourly")
            CRON_SCHEDULE="$CRAWL_MINUTE * * * *"
            ;;
        "daily")
            CRON_SCHEDULE="$CRAWL_MINUTE $CRAWL_HOUR * * *"
            ;;
        "weekly")
            CRON_SCHEDULE="$CRAWL_MINUTE $CRAWL_HOUR * * 0"
            ;;
        *)
            # Custom cron expression
            CRON_SCHEDULE="$CRAWL_INTERVAL"
            ;;
    esac
    
    log_info "Cron schedule: $CRON_SCHEDULE"
    
    # Write crontab
    echo "$CRON_SCHEDULE /app/run_crawler.sh" | crontab -
    
    # Start cron daemon
    cron
    
    log_info "Cron daemon started"
}

# Continuous loop-based crawling (alternative to cron)
run_loop() {
    log_info "Starting continuous crawl loop..."
    
    # Calculate sleep interval in seconds
    case "$CRAWL_INTERVAL" in
        "hourly")
            SLEEP_SECONDS=3600
            ;;
        "daily")
            SLEEP_SECONDS=86400
            ;;
        "weekly")
            SLEEP_SECONDS=604800
            ;;
        *)
            # Assume it's a number of seconds
            SLEEP_SECONDS=${CRAWL_INTERVAL:-86400}
            ;;
    esac
    
    log_info "Crawl interval: $SLEEP_SECONDS seconds"
    
    while true; do
        START_DATE=$(get_start_date)
        END_DATE=$(get_today)
        
        run_crawler "$START_DATE" "$END_DATE" || true
        
        log_info "Sleeping for $SLEEP_SECONDS seconds until next crawl..."
        sleep $SLEEP_SECONDS
    done
}

# Main entrypoint logic
case "$1" in
    "continuous"|"loop")
        log_info "Mode: Continuous loop"
        log_info "Field: ${ARXIV_FIELD:-all}"
        log_info "Lookback days: $LOOKBACK_DAYS"
        log_info "Destination: $DEST_DIR"
        
        # Run initial crawl immediately
        START_DATE=$(get_start_date)
        END_DATE=$(get_today)
        run_crawler "$START_DATE" "$END_DATE" || true
        
        # Then continue with loop
        run_loop
        ;;
        
    "cron")
        log_info "Mode: Cron-based scheduling"
        log_info "Field: ${ARXIV_FIELD:-all}"
        log_info "Lookback days: $LOOKBACK_DAYS"
        log_info "Destination: $DEST_DIR"
        
        setup_cron
        
        # Run initial crawl immediately
        START_DATE=$(get_start_date)
        END_DATE=$(get_today)
        run_crawler "$START_DATE" "$END_DATE" || true
        
        # Keep container running and tail logs
        log_info "Monitoring logs..."
        tail -f /app/logs/crawler_*.log 2>/dev/null || tail -f /dev/null
        ;;
        
    "once")
        log_info "Mode: Single run"
        START_DATE=${2:-$(get_start_date)}
        END_DATE=${3:-$(get_today)}
        
        run_crawler "$START_DATE" "$END_DATE"
        ;;
        
    "shell"|"bash")
        exec /bin/bash
        ;;
        
    *)
        # If arguments look like dates, treat as one-time run
        if [[ "$1" =~ ^[0-9]{4}-[0-9]{2} ]]; then
            run_crawler "$1" "${2:-$(get_today)}"
        else
            echo "Usage: docker run <image> [mode] [options]"
            echo ""
            echo "Modes:"
            echo "  continuous  - Run crawler in a continuous loop (default)"
            echo "  cron        - Use cron for scheduling"
            echo "  once        - Run once and exit"
            echo "  shell       - Open interactive shell"
            echo ""
            echo "Environment variables:"
            echo "  CRAWL_INTERVAL   - hourly, daily, weekly, or seconds (default: daily)"
            echo "  CRAWL_HOUR       - Hour to run (0-23, for daily/weekly, default: 6)"
            echo "  CRAWL_MINUTE     - Minute to run (0-59, default: 0)"
            echo "  ARXIV_FIELD      - arXiv category (e.g., cs.AI, cs.CV)"
            echo "  LOOKBACK_DAYS    - Days to look back (default: 1)"
            echo "  DEST_DIR         - Download directory (default: /app/download)"
            echo "  ARXIV_ID_DEST    - arXiv IDs output dir (default: /app/arxiv_ids)"
            exit 1
        fi
        ;;
esac

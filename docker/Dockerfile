FROM python:3.11-slim

WORKDIR /app

# Install system dependencies if needed
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire project
COPY . .

# Install the package and dependencies
RUN pip install --no-cache-dir -e .

# Create directories for data and logs
RUN mkdir -p /app/download /app/logs

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the scheduler service
CMD ["python", "-m", "research_arcade.services.scheduler"]
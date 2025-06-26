# Dockerfile
FROM python:3.8.20-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Expose the port your app listens on
EXPOSE 5000

# Run the app; adapt if you use a different framework/entrypoint
CMD ["python", "app.py"]

# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    bash \
    coreutils \
    && rm -rf /var/lib/apt/lists/*

# Create the shared directory for named pipes
RUN mkdir -p /shared

# Copy requirements file and install Python dependencies
COPY agent/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the runners directory (for GeminiClient)
COPY runners/ /app/runners/

# Copy the Datadog integration module
COPY datadog_integration/ /app/datadog_integration/

# Copy the Python scripts
COPY agent/command_client.py /app/agent/command_client.py
COPY agent/run_loop.py /app/run_loop.py

# Copy scenario configuration files
COPY agent/PROMPT.json /app/PROMPT.json
COPY agent/DESCRIPTION.md /app/DESCRIPTION.md

# Make scripts executable
RUN chmod +x /app/agent/command_client.py /app/run_loop.py

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default command runs the test loop
CMD ["python3", "/app/run_loop.py"]

# Alternative: you can also run the command client directly by overriding the command
# docker run <image> python3 /app/agent/command_client.py "ls -la"

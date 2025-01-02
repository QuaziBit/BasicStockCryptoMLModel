# Use a base image with Python 3.10.5
FROM python:3.10-slim

# Set environment variables to prevent Python from writing .pyc files and buffering
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into the container
COPY . .

# Make the runner.sh script executable
RUN chmod +x /app/runner.sh

# Set the default output directory inside the container
ENV OUTPUT_DIR="/app/output"

# Create the default output directory
RUN mkdir -p $OUTPUT_DIR

# Set the entrypoint to the runner.sh script
ENTRYPOINT ["/bin/bash", "/app/runner.sh"]

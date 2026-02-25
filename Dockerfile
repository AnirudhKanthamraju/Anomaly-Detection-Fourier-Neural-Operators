# ============================================================
# Avionic Anomaly Detection Pipeline - Dockerfile
# Base: Python 3.11 slim for a lightweight image
# ============================================================

FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy dependency list first (leverages Docker layer caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files
COPY data_loaders.py .
COPY models.py .

# Copy pre-generated cache (pickle files) so no CSV parsing is needed
COPY cache/ ./cache/

# Default command: run the data loader
CMD ["python", "data_loaders.py"]

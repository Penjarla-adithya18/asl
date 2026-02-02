# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV and MediaPipe (updated for newer Debian)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements_hf.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY flask_api_new.py .
COPY asl_mobilenet_best_keras3.h5 .

# Expose port 7860 (HuggingFace Spaces default)
EXPOSE 7860

# Run the Flask app with gunicorn
CMD ["gunicorn", "flask_api_new:app", "--bind", "0.0.0.0:7860", "--timeout", "120", "--workers", "1", "--threads", "1"]

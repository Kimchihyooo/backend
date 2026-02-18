# Use Python 3.10 slim for a smaller image size
FROM python:3.10-slim

# Set environment variables
# PYTHONDONTWRITEBYTECODE: Prevents Python from writing .pyc files to disk
# PYTHONUNBUFFERED: Ensures that Python output is sent straight to terminal without buffering
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Install system dependencies required by OpenCV, FFmpeg, and other libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set up a new user named "user" with user ID 1000 to avoid running as root
RUN useradd -m -u 1000 user

# Set the working directory in the container
WORKDIR $HOME/app

# Copy requirements.txt first to leverage Docker cache
COPY --chown=user requirements.txt .

# Switch to the non-root user
USER user

# Install Python dependencies
# We upgrade pip and then install the requirements
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
# This is done after installing requirements to keep the build fast when only code changes
COPY --chown=user . .

# Expose the port the app runs on
EXPOSE 8000

# Start the application using uvicorn
# Using app:app as defined in backend/app.py
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

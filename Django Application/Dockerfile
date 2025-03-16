# Use the NVIDIA CUDA base image with Python 3.6.8
# Or your desired CUDA version
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu18.04

# Install Python 3.6.8 (if not already present)
RUN apt-get update && apt-get install -y \
    python3.6 \
    python3.6-dev \
    python3-pip \
    cmake \
    && ln -s /usr/bin/python3.6 /usr/bin/python \
    && ln -s /usr/bin/pip3 /usr/bin/pip

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create a directory for static files
RUN mkdir -p /app/staticfiles/app/uploaded_videos/
WORKDIR /app

# Copy and install requirements
COPY ./requirements.txt /app/requirements.txt
RUN pip install --upgrade pip
RUN pip install opencv-python==4.2.0.32
RUN pip install -r requirements.txt

# Copy application files
COPY . /app

# Collect static files
RUN python manage.py collectstatic --noinput

# Install Gunicorn
RUN pip install gunicorn

# Create upload directory
RUN mkdir -p /app/uploaded_videos/app/uploaded_videos/

# Define volume and entrypoint
VOLUME /app/run/
ENTRYPOINT ["/app/bin/gunicorn_start.sh"]
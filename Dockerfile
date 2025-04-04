# Use an official Python image
FROM python:3.12

# Install Poetry
RUN pip install --no-cache-dir poetry
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libpulse0 \
    libasound2 \
    ffmpeg \
    x11-utils \
    x11-xserver-utils \
    # qt5-default \
    qtbase5-dev \
    qtbase5-dev-tools \
    pulseaudio
# This installs: 
# Qt libs (qt5-default, etc.), 
# Audio libs (libpulse0, libasound2), 
# Video/audio tools like ffmpeg, 
# X11 client tools

# Set the working directory inside the container
WORKDIR /app

# Copy project files (adjust if needed)
COPY pyproject.toml poetry.lock ./

# Install dependencies (without root)
RUN poetry install --no-root

# Copy the rest of your code
# COPY . .
# Explicitly copy the main script, classifiers, audio & image assets
COPY run.py ./                      
COPY models/ ./models/              
COPY media/ ./media/                

# Default command to run the app
CMD ["poetry", "run", "python", "run.py"]
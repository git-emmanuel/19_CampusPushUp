# Use an official Python image
FROM python:3.12

# Install Poetry
RUN pip install --no-cache-dir poetry
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev

# Set the working directory inside the container
WORKDIR /app

# Copy project files (adjust if needed)
COPY pyproject.toml poetry.lock ./

# Install dependencies (without root, no dev dependencies)
RUN poetry install --no-root

# Copy the rest of your code
COPY . .

# Default command to run the app
CMD ["poetry", "run", "python", "run.py"]
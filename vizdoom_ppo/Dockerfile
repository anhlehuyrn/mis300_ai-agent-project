
# Use a lightweight Python base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Install ViZDoom dependencies
RUN apt-get update && apt-get install -y \
    libsdl2-dev \
    libboost-all-dev \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Command to run the training script
CMD ["python", "-u", "vizdoom_ppo/train.py"]

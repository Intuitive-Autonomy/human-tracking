FROM python:3.8

# Install system dependencies for OpenCV and computer vision
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    python3-opencv \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy dependency file
COPY requirements.txt* ./

# Install Python dependencies
RUN if [ -f "requirements.txt" ]; then pip install --no-cache-dir -r requirements.txt; fi

# Copy source code
COPY . .

CMD ["bash"]

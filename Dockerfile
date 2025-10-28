# ==========================================
# Base image: NVIDIA Triton Inference Server
# ==========================================
FROM nvcr.io/nvidia/tritonserver:24.03-py3

# ------------------------------------------
# Set working directory
# ------------------------------------------
WORKDIR /app

# ------------------------------------------
# Copy requirements and install dependencies
# ------------------------------------------
COPY requirements.txt .

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-dev && \
    pip install --upgrade pip && \
    pip install -r requirements.txt && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ------------------------------------------
# Copy model repository and source code
# ------------------------------------------
COPY ./models /models
COPY ./src /src

# ------------------------------------------
# Expose Triton ports
# ------------------------------------------
EXPOSE 8000 8001 8002

# ------------------------------------------
# Default command: run the Triton server
# ------------------------------------------
CMD ["tritonserver", "--model-repository=/models", "--strict-model-config=false", "--log-verbose=1"]

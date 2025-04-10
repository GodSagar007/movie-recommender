FROM python:3.10

# Set working directory
WORKDIR /app

# Install system dependencies required for NumPy and scikit-surprise
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    python3-dev \
    gcc \
    g++ \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Upgrade pip
RUN pip install --upgrade pip

# Clean install of a specific compatible version of NumPy
RUN pip install --no-cache-dir numpy==1.24.4

# Now install scikit-surprise
RUN pip install scikit-surprise==1.1.3

# Install remaining requirements
RUN pip install -r requirements.txt

# Expose the FastAPI port
EXPOSE 8000

# Run the app
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

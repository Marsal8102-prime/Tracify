FROM python:3.12-slim

WORKDIR /app/ml-engine

# Install system dependencies required by OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY ml-engine/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY ml-engine ./

EXPOSE 8001

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8001"]

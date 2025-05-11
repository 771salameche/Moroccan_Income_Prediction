# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the API code and model
COPY api/ /app/api/
COPY models/ /app/models/
COPY src/ /app/src/

# Set environment variables
ENV PYTHONPATH=/app
ENV PORT=8000

# Expose the port
EXPOSE 8000

# Command to run the API
CMD ["uvicorn", "api.api:app", "--host", "0.0.0.0", "--port", "8000"]
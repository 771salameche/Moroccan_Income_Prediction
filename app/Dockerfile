# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Streamlit app code and related files
COPY app/ /app/app/
COPY src/ /app/src/

# Set environment variables
ENV PYTHONPATH=/app
ENV API_URL=http://api:8000

# Expose the port
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
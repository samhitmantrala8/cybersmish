# Use an official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy only requirements first (for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the Hugging Face model to reduce runtime memory
RUN python -c "from transformers import pipeline; pipeline('text-classification', model='samhitmantrala/smish_fin')"

# Copy all app files
COPY . .

# Expose the port Render will use
EXPOSE 10000

# Start command
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "app:app", "--workers", "1", "--threads", "4"]

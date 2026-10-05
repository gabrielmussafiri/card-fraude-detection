FROM python:3.13-slim

WORKDIR /app

# Copy requirements FIRST for layer caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the application code and the model artifact
COPY app/app.py .
COPY model.pkl .

# Document the port
EXPOSE 8000

# Run Uvicorn, binding to 0.0.0.0 so it's accessible from outside the container
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
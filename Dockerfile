FROM python:3.11-slim
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    playwright install chromium --with-deps

# Copy all project files
COPY . .

# Hugging Face Spaces requires apps to listen on port 7860
ENV PORT=7860
EXPOSE 7860

# Run the API
CMD ["python", "api.py"]

# Use official Python image
FROM python:3.11.9

# Set working directory
WORKDIR /app

COPY . .


# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Flask API port
EXPOSE 7860

# Run Flask app with Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]
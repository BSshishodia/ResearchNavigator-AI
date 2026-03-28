# Use Python 3.11 to satisfy NetworkX 3.5 constraints
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install system dependencies for building C-extensions (needed for Spacy/Numpy)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .

# Install dependencies (ensure NumPy is 2.0+ to avoid the Thinc conflict)
RUN pip install --no-cache-dir -r requirements.txt

RUN python -m nltk.downloader punkt punkt_tab averaged_perceptron_tagger_eng stopwords

# Copy the rest of the project files
COPY . .

# Create the uploads directory for PDF/DOCX processing
RUN mkdir -p uploads

# EXPOSE tells Docker which port the container is listening on internally
EXPOSE 5000

# Start the application using the 0.0.0.0 host binding
CMD ["python", "app.py"]
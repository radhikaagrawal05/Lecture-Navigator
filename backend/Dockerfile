# Use the official Python image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Create a user for Hugging Face Spaces (Spaces run as user 1000)
RUN useradd -m -u 1000 user
USER user

# Copy the requirements file and install dependencies
# We use the user's local bin directory for pip installs
ENV PATH="/home/user/.local/bin:$PATH"
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Need to install uvicorn as well if it's not in requirements.txt
RUN pip install --no-cache-dir uvicorn

# Copy the rest of the backend files
COPY --chown=user . /app

# Download NLTK data locally to the expected directory
RUN python setup_nltk.py || true

# Expose port 7860 (default for Hugging Face Spaces)
EXPOSE 7860

# Run the FastAPI app with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]

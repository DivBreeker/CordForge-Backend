# Use Python 3.9
FROM python:3.9

# Set working directory
WORKDIR /code

# Copy requirements and install them
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Create a folder for the model cache (permissions fix for HF)
RUN mkdir -p /code/cache && chmod 777 /code/cache
ENV TRANSFORMERS_CACHE=/code/cache
ENV HF_HOME=/code/cache

# Copy the rest of the code
COPY . .

# Run the app on port 7860 (Required for Hugging Face)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
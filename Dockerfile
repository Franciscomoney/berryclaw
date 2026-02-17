FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY berryclaw.py .
COPY integrations/ integrations/

# Config and secrets are mounted as volumes — copy examples as fallback
COPY config.json.example config.json.example
COPY secrets.json.example secrets.json.example

CMD ["python3", "-u", "berryclaw.py"]

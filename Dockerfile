FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir .

COPY bot.py .

EXPOSE 7860

CMD ["python", "bot.py", "--host", "0.0.0.0", "--port", "7860", "-t", "telnyx"]

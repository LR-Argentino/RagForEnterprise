FROM python:3.11-slim

RUN apt-get update && apt-get install -y libgomp1 curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt


EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]

FROM python:3.13-slim
WORKDIR /app

COPY ingester/requirements.txt .
RUN apt-get update && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir -r requirements.txt

COPY utils/ ./utils/
COPY ingester/ ./ingester/

CMD ["python", "-m", "ingester.markdown_ingester"]
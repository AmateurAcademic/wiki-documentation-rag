FROM python:3.13-slim
WORKDIR /app

COPY egester/requirements.txt .
RUN pip install --no-cache-dir -r egester/requirements.txt

COPY utils/ ./utils/
COPY egester/ ./egester/

EXPOSE 8001
CMD ["uvicorn", "egester.note_writer_api:app", "--host", "0.0.0.0", "--port", "8001"]
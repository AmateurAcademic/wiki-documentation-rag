FROM python:3.13-slim
WORKDIR /app

COPY wiki_tools/egester/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY utils/ ./utils/
COPY wiki_tools/egester/ ./egester/

EXPOSE 8001
CMD ["uvicorn", "egester.note_writer_api:app", "--host", "0.0.0.0", "--port", "8001"]

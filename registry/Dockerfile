FROM python:3.10-slim
WORKDIR /app

COPY registry/requirements.txt .
RUN pip install -r requirements.txt

COPY shared/ ./shared/
COPY registry/ ./

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]

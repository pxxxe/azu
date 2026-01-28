FROM python:3.10-slim
WORKDIR /app

# Copy Requirements
COPY scheduler/requirements.txt .
RUN pip install -r requirements.txt

# Copy Shared Lib
COPY shared/ ./shared/
# Copy App
COPY api/ ./

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]

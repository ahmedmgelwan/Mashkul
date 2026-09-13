FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements_prod.txt .
RUN pip install --no-cache-dir -r requirements_prod.txt

COPY src/ ./src/
COPY api/ ./api/
COPY assets/ ./assets/

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/main.py .
COPY src/datasets.py .
COPY src/model.py .
COPY src/utils.py .
CMD ["python", "src/main.py"]
# Gunakan image resmi Python versi 3.10.12 (slim agar sizenya lebih kecil)
FROM python:3.10.12-slim

# Set environment variable agar output Python langsung tampil di log terminal
ENV PYTHONUNBUFFERED=1

# Buat folder kerja di dalam container
WORKDIR /app

# Install system dependencies (Biasanya dibutuhkan untuk library ML/NLP/Database)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt dulu agar Docker bisa nge-cache layer ini
COPY requirements.txt .

# Install library Python
RUN pip install --no-cache-dir -r requirements.txt

# Copy seluruh source code project kamu ke dalam container
COPY . .

# Ekspose port yang digunakan oleh chatbot kamu (Contoh: 8000 untuk FastAPI/Flask)
EXPOSE 8000

# Perintah untuk menjalankan aplikasi (SESUAIKAN DENGAN FRAMEWORK KAMU)
# Contoh untuk FastAPI: CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
# Contoh untuk Flask: CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]
# Contoh script biasa: CMD ["python", "app.py"]
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
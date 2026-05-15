FROM python:3.11-slim

WORKDIR /app

# Системные зависимости для аудио + ffmpeg для kairos-asr
RUN apt-get update && apt-get install -y --no-install-recommends \
    libportaudio2 \
    libsndfile1 \
    libsdl2-mixer-2.0-0 \
    libmpg123-0 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Python-зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Код приложения
COPY . .

# Порт FastAPI
EXPOSE 8000

# Запуск
CMD ["python", "app.py"]

# Голосовой сборщик корзины ВкусВилл

Ассистент для IVR-системы: принимает диктовку товаров, распознаёт речь, ищет в каталоге через MCP, собирает корзину и передаёт оператору ссылку.

## Архитектура

- Распознавание: GigaAM RNNT (kairos-asr v0.7.0, 885 МБ, лицензия MIT)
- Синтез речи: edge-tts
- Детектор речи: RMS-based (src/audio/rms_detector.py) с подавлением эха
- Поиск: MCP-сервер ВкусВилла
- Fuzzy-поиск: rapidfuzz
- Веб-интерфейс: FastAPI + WebSockets
- Оркестрация: K3s + Podman

## Быстрый старт (без Kubernetes)

```bash
git clone https://github.com/KiriruSokoru/mcp_voice_app.git
cd mcp_voice_app
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

Открыть http://localhost:8000

### Системные зависимости

ffmpeg (kairos-asr использует для загрузки аудио):
```bash
# Fedora
sudo dnf install -y ffmpeg-free
sudo ln -s /usr/bin/ffmpeg-free /usr/local/bin/ffmpeg

# Ubuntu/Debian
sudo apt install -y ffmpeg
```

## Production (K3s)

Требования: Fedora 44+/RHEL 9+, SELinux Enforcing, 8+ CPU, 16GB RAM, 10GB диск.

k8s/ — манифесты для развёртывания:
- namespace.yaml
- registry.yaml (локальный Docker Registry)
- model-pv.yaml (PersistentVolume + Job загрузки модели)
- deployment.yaml (ассистент + Service)

## История изменений

### 15.05.2026 — STT миграция Whisper на GigaAM, эхоподавление

- STT заменён с faster-whisper large-v3-turbo-ct2 (1.6 ГБ, 11.7с) на GigaAM RNNT (885 МБ, 0.3-0.6с)
  - Точность на русском телефонном разговоре: WER 9.5% (Whisper: 23.9%)
  - Архитектура CTC/RNNT не подвержена галлюцинациям
- Добавлен RMS-детектор речи с функцией mute микрофона на время TTS
- Двойной слив аудио-очереди перед каждой транскрипцией
- Дедупликация TTS-аудио по хешу
- Короткие подсказки вместо повторяющихся инструкций
- Индикатор состояния микрофона в UI
- Ложные срабатывания детектора сохраняются в datasets/silence_samples/

## Известные ограничения

- Словарь опечаток ограничен
- Нет контекстной памяти между фразами
- Каталог для fuzzy загружается из hardcoded списка
- Качество распознавания зависит от микрофона (рекомендуется гарнитура)

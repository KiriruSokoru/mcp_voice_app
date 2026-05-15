# Голосовой сборщик корзины ВкусВилл

Ассистент для IVR-системы: принимает диктовку товаров, распознаёт через Whisper, ищет в каталоге через MCP, собирает корзину и передаёт оператору ссылку.

## Архитектура

- Распознавание: faster-whisper-large-v3-turbo-ct2 (int8)
- Синтез речи: edge-tts
- Поиск: MCP-сервер ВкусВилла
- Fuzzy-поиск: rapidfuzz
- Веб-интерфейс: FastAPI + WebSockets
- Оркестрация: K3s + Podman

## Быстрый старт (без Kubernetes)

git clone https://github.com/KiriruSokoru/mcp_voice_app.git
cd mcp_voice_app
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py

Открыть http://localhost:8000

## Production (K3s)

Требования: Fedora 44+/RHEL 9+, SELinux Enforcing, 8+ CPU, 16GB RAM, 10GB диск.

k8s/ — манифесты для развёртывания:
- namespace.yaml
- registry.yaml (локальный Docker Registry)
- model-pv.yaml (PersistentVolume + Job загрузки модели)
- deployment.yaml (ассистент + Service)

## Известные ограничения

- Словарь опечаток ограничен
- Нет контекстной памяти между фразами
- Каталог для fuzzy загружается из hardcoded списка
- Эхоподавление при работе через динамики

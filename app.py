"""
app.py MVP v11 — Голосовой сборщик корзины для ВкусВилл
"""
import asyncio, base64, hashlib, json, os, queue, re, tempfile, threading, time
import soundfile as sf
from typing import Dict, List, Optional

import numpy as np
import sounddevice as sd
import edge_tts
from kairos_asr import KairosASR
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn
from rapidfuzz import fuzz, process

from mcp_client import VkusVillMCP
from src.audio.rms_detector import RMSDetector, RMSConfig, AudioState

# ── КОНФИГ ───────────────────────────────────────
SAMPLE_RATE    = 16000
POST_TTS_PAUSE = 0.5
MAX_ITEMS      = 5
MAX_ERRORS     = 2

TTS_VOICE = "ru-RU-SvetlanaNeural"

DATASET_DIR = os.path.join(os.path.dirname(__file__), "datasets", "silence_samples")
detector = RMSDetector(
    RMSConfig(
        speech_threshold=0.045,
        silence_threshold=0.015,
        silence_duration=1.2,
        max_speech_duration=15.0,
        min_speech_duration=0.3,
        sample_rate=SAMPLE_RATE,
    ),
    dataset_dir=DATASET_DIR
)

POPULAR_ITEMS = [
    "молоко", "кефир", "творог", "сметана", "йогурт", "ряженка",
    "сыр", "масло сливочное", "масло оливковое", "сёмга", "форель",
    "курица", "говядина", "картофель", "помидоры", "огурцы",
    "хлеб", "батон", "пельмени", "вареники", "колбаса", "сосиски",
    "яйца", "гречка", "рис", "макароны", "печенье", "конфеты",
    "чай", "кофе", "сок", "вода", "лимонад", "квас", "пиво",
    "чипсы", "сухарики", "орешки", "бананы", "яблоки", "груши",
    "апельсины", "мандарины", "виноград", "клубника", "малина"
]

# Топ покупаемых товаров (Росстат, продовольственная корзина 2026)
TOP_ITEMS = [
    "говядина", "свинина", "курица", "индейка", "баранина", "печень",
    "колбаса", "сосиски", "сардельки",
    "рыба", "форель", "семга", "горбуша", "сельдь",
    "молоко", "творог", "кефир", "сметана", "ряженка", "йогурт",
    "сыр", "масло сливочное", "масло растительное",
    "яйца",
    "хлеб", "батон", "хлеб ржаной",
    "гречка", "рис", "пшено", "макароны", "спагетти", "мука",
    "картофель", "лук", "морковь", "капуста", "свекла",
    "помидоры", "огурцы",
    "яблоки", "бананы", "апельсины", "лимоны",
    "сахар", "соль", "чай", "кофе", "печенье",
    "вода", "сок", "лимонад", "квас",
]

def ts():   return time.strftime("%H:%M:%S")
def log(m): print(f"[{ts()}] {m}", flush=True)

# ── TTS ──────────────────────────────────────────
try:
    import pygame
    pygame.mixer.init()
    TTS_AVAILABLE = True
except Exception:
    TTS_AVAILABLE = False
    log("⚠️ Звук недоступен, TTS будет работать только текстом")
    class MockPygame:
        class _music:
            @staticmethod
            def load(path): pass
            @staticmethod
            def play(): pass
            @staticmethod
            def unload(): pass
            @staticmethod
            def get_busy(): return False
        mixer = type('obj', (object,), {
            'init': lambda: None,
            'music': _music
        })()
    pygame = MockPygame()

_tts_done    = threading.Event()
_tts_done.set()
_tts_q: queue.Queue = queue.Queue()
_last_audio_hash = ""

def _tts_worker():
    global _last_audio_hash
    loop = asyncio.new_event_loop()

    async def _speak(text: str):
        global _last_audio_hash
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tmp = f.name
        try:
            # Mute микрофон ПЕРЕД синтезом
            detector.mute_for_tts()
            
            await edge_tts.Communicate(text, TTS_VOICE).save(tmp)
            with open(tmp, 'rb') as af:
                audio_bytes = af.read()
            audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
            audio_hash = hashlib.md5(audio_bytes).hexdigest()
            if audio_hash != _last_audio_hash:
                _last_audio_hash = audio_hash
                try:
                    _bcast_q.put_nowait(json.dumps({"audio": audio_b64}, ensure_ascii=False))
                except asyncio.QueueFull:
                    pass
            if TTS_AVAILABLE:
                pygame.mixer.music.load(tmp)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    await asyncio.sleep(0.05)
        finally:
            try:
                if TTS_AVAILABLE:
                    pygame.mixer.music.unload()
                os.unlink(tmp)
            except Exception:
                pass
            # Unmute ПОСЛЕ проигрывания + пауза
            await asyncio.sleep(POST_TTS_PAUSE)
            detector.unmute_after_tts()
            # Слить аудио-мусор, накопившийся за время TTS
            while not _audio_q.empty():
                try:
                    _audio_q.get_nowait()
                except queue.Empty:
                    break

    while True:
        text = _tts_q.get()
        if text is None:
            break
        _tts_done.clear()
        log(f"🔊 TTS → «{text}»")
        loop.run_until_complete(_speak(text))
        _tts_done.set()

threading.Thread(target=_tts_worker, daemon=True).start()
log(f"🗣 TTS голос: {TTS_VOICE} (звук: {'да' if TTS_AVAILABLE else 'нет'})")

# ── СОСТОЯНИЕ ────────────────────────────────────
class Session:
    def reset(self):
        self.active     = False
        self.cart: List[Dict] = []
        self.history: List[str] = []
        self.errors     = 0
        self.start_ts   = 0.0
        self.status     = "⏸ Ожидание"
        self.basket_url = ""
        self.pending: Optional[Dict] = None
        self.pending_items: Optional[List[Dict]] = None
        self.pending_query: Optional[str] = None
        self.catalog: List[str] = []
        self.hotwords: List[str] = []
        self.mic_level  = 0
        self.mic_bar    = ""
    def __init__(self): self.reset()

S   = Session()
mcp = VkusVillMCP()
_audio_q:    queue.Queue     = queue.Queue()
_ws_clients: List[WebSocket] = []
_bcast_q:    asyncio.Queue   = asyncio.Queue()

def push():
    data = json.dumps(
        {
            "log": S.history,
            "cart": S.cart,
            "status": S.status,
            "url": S.basket_url,
            "mic_level": S.mic_level,
            "mic_bar": S.mic_bar,
            "active": S.active
        },
        ensure_ascii=False
    )
    try: _bcast_q.put_nowait(data)
    except asyncio.QueueFull: pass

def say(text: str, wait: bool = False):
    S.history.append(f"🤖 {text}")
    _tts_q.put(normalize_for_tts(text))
    push()
    if wait:
        _tts_done.wait()

# ── ИНИЦИАЛИЗАЦИЯ КАТАЛОГА ───────────────────────
async def init_catalog():
    log("📚 Загружаю каталог из MCP по топовым товарам...")
    catalog_set = set(POPULAR_ITEMS)
    
    for query in TOP_ITEMS:
        try:
            results = await mcp.search(query)
            for item in results:
                catalog_set.add(item["name"].lower())
        except Exception as e:
            log(f"   '{query}': ошибка ({e})")
    
    S.catalog = list(catalog_set)
    S.hotwords = [item[:20] for item in S.catalog[:100]]
    log(f"✅ Каталог: {len(S.catalog)} товаров (из {len(TOP_ITEMS)} топ-запросов)")

# ── NLU ──────────────────────────────────────────
_CONFIRM = re.compile(r'\b(да|верно|правильно|ок|окей|хорошо|подтверждаю|конечно|именно|точно|угу|ага)\b', re.I)
_DONE    = re.compile(r'\b(всё|все|хватит|достаточно|готово|оформляй|оформите|заканчивай|больше ничего)\b', re.I)
_OP      = re.compile(r'\b(оператор|человек|живой|менеджер|помогите|соедини|переключи|стоп)\b', re.I)
_DENY    = re.compile(r'\b(нет|не верно|неверно|не то|другой|другое|не тот|не та|не надо|не нужно|не хочу)\b', re.I)
_SELECT  = re.compile(r'\b(первый|второй|третий|1-й|2-й|3-й|1|2|3)\b', re.I)

_QTY_WORDS = {
    "один": 1, "одну": 1, "одна": 1,
    "два": 2, "две": 2,
    "три": 3, "четыре": 4, "пять": 5,
    "шесть": 6, "семь": 7, "восемь": 8,
    "девять": 9, "десять": 10,
}

_STOP_WORDS = re.compile(
    r'\b(мне|пожалуйста|ещё|еще|можно|просто|купи|возьми|закажи|'
    r'положи|добавь|добавьте|добавить|хочу|дай|нужно|нужен|нужна|нужны)\b',
    re.I
)

def fuzzy_fix(query: str) -> str:
    if not query or len(query) < 3 or not S.catalog:
        return query
    result = process.extractOne(query, S.catalog, scorer=fuzz.ratio, score_cutoff=70)
    if result:
        fixed, score = result
        if score > 80:
            log(f"   🔧 Fuzzy: '{query}' → '{fixed}' ({score}%)")
            return fixed
    return query

def _extract_qty_and_clean(text: str):
    qty = 1
    result = text
    for word, n in _QTY_WORDS.items():
        if re.search(rf'\b{word}\b', result, re.I):
            result = re.sub(rf'\b{word}\b', '', result, flags=re.I)
            qty = n
            break
    else:
        m = re.search(r'(?<![,.\d])([2-9])(?![,.\d%])', result)
        if m:
            qty = int(m.group(1))
            result = result[:m.start()] + result[m.end():]
    return qty, result.strip()

def _build_search_query(text: str) -> str:
    q = _STOP_WORDS.sub(' ', text)
    q = re.sub(r'\s+', ' ', q).strip()
    return q

def normalize_query(query: str) -> str:
    query = re.sub(r'(\d+),(\d+%)', r'\1.\2', query)
    # Не удаляем "жирности" — это важное слово для поиска
    query = re.sub(r'\s+', ' ', query).strip()
    # Исправляем S1 → С1 (яйца)
    query = re.sub(r'\bS1\b', 'С1', query, flags=re.I)
    query = re.sub(r'\bS2\b', 'С2', query, flags=re.I)
    query = re.sub(r'\bS0\b', 'С0', query, flags=re.I)
    return query

def extract_select_index(text: str) -> Optional[int]:
    m = _SELECT.search(text)
    if m:
        word = m.group(1).lower()
        if word in ["первый", "1-й", "1"]: return 0
        if word in ["второй", "2-й", "2"]: return 1
        if word in ["третий", "3-й", "3"]: return 2
    return None

def nlu(text: str) -> dict:
    t = text.strip()
    if _OP.search(t):      return {"intent": "operator"}
    if _DONE.search(t):    return {"intent": "done"}
    if _CONFIRM.search(t): return {"intent": "confirm"}
    if _DENY.search(t):    return {"intent": "deny"}

    if S.pending_items:
        idx = extract_select_index(t)
        if idx is not None and idx < len(S.pending_items):
            return {"intent": "select_item", "index": idx}

    t = fuzzy_fix(t)
    qty, cleaned = _extract_qty_and_clean(t)
    query = _build_search_query(cleaned)

    if len(query) >= 2:
        is_single_word = len(query.split()) == 1
        return {"intent": "add_item", "product": query, "qty": qty, "single_word": is_single_word}
    return {"intent": "unknown"}

# ── Нормализация для TTS ─────────────────────────
_TTS_NORMALIZE = [
    (re.compile(r'\b(\d+)\s*г(?:\b|рамм)'), r'\1 грамм'),  # не трогаем "грамм" если уже есть
    (re.compile(r'\b(\d+)\s*кг(?:\b|илограмм)'), r'\1 килограмм'),
    (re.compile(r'\b(\d+)\s*мл(?:\b|иллилитр)'), r'\1 миллилитров'),
    (re.compile(r'\b(\d+)\s*л(?:\b|итр)'), r'\1 литр'),
    (re.compile(r'\b(\d+)\s*см(?:\b|антиметр)'), r'\1 сантиметр'),
    (re.compile(r'\b(\d+)\s*шт(?:\b|тук)'), r'\1 штук'),
    (re.compile(r'(\d+),(\d+)\s*%'), r'\1.\2 процента'),
    (re.compile(r'(\d+)\s*%'), r'\1 процент'),
    (re.compile(r'ул\.?\s*'), 'улица '),
]

def normalize_for_tts(text: str) -> str:
    for pattern, replacement in _TTS_NORMALIZE:
        text = pattern.sub(replacement, text)
    return text

# ── АУДИО ────────────────────────────────────────
def _find_mic():
    try:
        devices = sd.query_devices()
        log(f"🎤 Найдено {len(devices)} устройств")
        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                try:
                    sd.check_input_settings(device=i, samplerate=SAMPLE_RATE, channels=1)
                    log(f"   ✅ Микрофон [{i}]: {dev['name']}")
                    return i
                except sd.PortAudioError:
                    continue
        log("⚠️ Подходящий микрофон не найден")
    except Exception as e:
        log(f"⚠️ Ошибка сканирования: {e}")
    return None

MIC_DEVICE = _find_mic()

def audio_capture(stop_ev: threading.Event):
    """Захват аудио через RMS-детектор."""
    
    def cb(indata, frames, t, status):
        if status:
            log(f"⚠️ audio status: {status}")
        
        # Обновляем уровень микрофона для UI
        vol = int(np.abs(indata.flatten()).mean())
        S.mic_level = vol
        S.mic_bar = "█" * min(vol // 30, 30)
        
        # Пропускаем через RMS-детектор
        result = detector.process_chunk(indata)
        
        if result is not None:
            dur = len(result) / SAMPLE_RATE
            S.history.append(f"🎤 запись {dur:.1f}с")
            _audio_q.put(result)
            log(f"✂️  фраза {dur:.2f}с → очередь")
        
        push()
    
    # Привязываем колбэк для ложных срабатываний
    detector.on_false_trigger = lambda arr: log(
        f"📊 Ложное срабатывание сохранено ({len(arr)/SAMPLE_RATE:.2f}с)"
    )
    
    log(f"🎙 Микрофон запущен (device={MIC_DEVICE}, RMS-детектор)")
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                            dtype="int16", blocksize=512, callback=cb,
                            device=MIC_DEVICE):
            while not stop_ev.is_set():
                time.sleep(0.05)
    except Exception as e:
        log(f"⚠️ Микрофон недоступен: {e}")

# ── STT ──────────────────────────────────────────
_HALLUCINATIONS = re.compile(
    r'^(спасибо|до свидания|пробки?|субтитры|подписывайтесь|продолжение следует'
    r'|в этом видео|редактирование|субтитры создавал|играет музыка|музыка|'
    r'\.+|-+|\s*)$',
    re.I
)

print("⏳ Загружаю GigaAM (kairos-asr)...")
_asr = KairosASR()
print("✅ GigaAM готов")

def transcribe(audio: np.ndarray) -> str:
    """Распознавание через GigaAM — без галлюцинаций, оптимизирован под русский."""
    t0 = time.time()
    
    # Сохраняем аудио во временный WAV-файл (kairos-asr ожидает путь к файлу)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name
    try:
        sf.write(tmp_path, audio, SAMPLE_RATE)
        result = _asr.transcribe(tmp_path)
        # kairos-asr возвращает объект TranscriptionResult с полем .text
        text = result.full_text.strip().rstrip('.!?,;:')
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
    
    elapsed = time.time() - t0
    log(f"🎯 GigaAM {elapsed:.1f}с: «{text}»")

    if not text or len(text) < 2:
        log(f"   STT: пустой результат")
        return ""

    if _HALLUCINATIONS.match(text):
        log(f"   STT: галлюцинация: {text!r}")
        return ""
    if re.search(r'субтитры|продолжение следует|играет музыка', text, re.I):
        log(f"   STT: расширенная галлюцинация: {text!r}")
        return ""
    return text

# ── HANDOFF ──────────────────────────────────────
async def do_handoff(reason: str):
    S.active = False
    S.status = "📞 Передача оператору..."
    push()
    if S.cart:
        try:
            url = await mcp.cart_link(
                [{"xml_id": i["xml_id"], "q": i["qty"]} for i in S.cart]
            )
            S.basket_url = url
            S.history.append(f"🔗 {url}")
            log(f"🔗 {url}")
        except Exception as e:
            log(f"⚠️ cart_link: {e}")
    items_str = ", ".join(i["name"] for i in S.cart) if S.cart else "корзина пуста"
    say(f"{reason} В корзине: {items_str}. Оператор всё видит.", wait=True)
    S.status = "✅ Оператор подключается"
    push()

async def show_top_choices(query: str, qty: int, results: List[Dict]):
    top_items = results[:3]
    S.pending_items = top_items
    S.pending_query = query

    message = "Я нашёл несколько вариантов:\n"
    for i, item in enumerate(top_items, 1):
        price = item.get("price", {}).get("current", "?")
        message += f"{i}. {item['name']} — {price} руб.\n"
    message += "Какой добавить? Скажите «первый», «второй» или «третий»."

    say(message, wait=True)
    S.status = "⏳ Жду выбора товара"
    push()

# ── ГОЛОСОВОЙ ЦИКЛ ───────────────────────────────
HINT_SHORT = "Назовите товар"
INTRO_FIRST = (
    "Здравствуйте! Я помогу собрать корзину. "
    "Называйте товары, например: «молоко», «хлеб», «творог». "
    "Я буду уточнять детали. Когда закончите — скажите «всё»."
)

async def voice_loop():
    log("🔄 voice_loop запущен")
    while True:
        try:
            audio = _audio_q.get_nowait()
        except queue.Empty:
            await asyncio.sleep(0.05)
            continue

        if not S.active:
            while not _audio_q.empty():
                try:
                    _audio_q.get_nowait()
                except queue.Empty:
                    break
            continue

        # Слить всё накопившееся, оставить только последнюю фразу
        while not _audio_q.empty():
            try:
                audio = _audio_q.get_nowait()
            except queue.Empty:
                break

        await asyncio.to_thread(_tts_done.wait)

        # Повторно слить — за время TTS микрофон насобирал мусор
        while not _audio_q.empty():
            try:
                audio = _audio_q.get_nowait()
            except queue.Empty:
                break

        S.status = "⚙️ Распознаю..."
        push()
        t0   = time.time()
        text = await asyncio.to_thread(transcribe, audio)
        elapsed = time.time() - t0
        log(f"📝 GigaAM {elapsed:.1f}с: «{text}»")

        if not text:
            S.status = "🎤 Слушаю..."
            push()
            continue

        # Контекстная коррекция команд подтверждения/отмены
        if S.pending or S.pending_items:
            text_lower = text.lower().strip()
            # "перно", "верна", "верн" → "верно"
            if len(text_lower) <= 6 and text_lower not in ("да", "нет", "всё", "все",
                "первый", "второй", "третий", "верно", "хорошо", "готово", "оператор"):
                if re.match(r'^[вп][еи]?р[нм]', text_lower):
                    log(f"   🔧 Контекст: '{text}' → 'верно'")
                    text = "верно"
                elif re.match(r'^н[еиа]', text_lower) and len(text_lower) <= 4:
                    log(f"   🔧 Контекст: '{text}' → 'нет'")
                    text = "нет"

        S.history.append(f"👤 {text}")
        intent = nlu(text)
        log(f"🧠 {intent}")
        action = intent["intent"]

        if S.pending_items and action == "select_item":
            idx = intent["index"]
            item = S.pending_items[idx]
            price = item.get("price", {}).get("current", "?")
            qty = 1

            S.cart.append({"xml_id": int(item["xml_id"]), "name": item["name"],
                           "price": price, "qty": qty})
            S.pending_items = None
            S.pending_query = None
            S.errors = 0

            if len(S.cart) >= MAX_ITEMS:
                await do_handoff(f"Добавил {item['name']}. Набрали {MAX_ITEMS} позиций.")
            else:
                say(f"Добавил {item['name']}. {HINT_SHORT}", wait=True)
                S.status = "🎤 Слушаю..."
                push()
            continue

        if S.pending:
            if action == "confirm":
                item = S.pending
                S.cart.append(item)
                S.pending = None
                S.pending_items = None
                S.errors = 0
                if len(S.cart) >= MAX_ITEMS:
                    await do_handoff(f"Добавил {item['name']}. Набрали {MAX_ITEMS} позиций.")
                else:
                    say(f"Добавил {item['name']}. {HINT_SHORT}", wait=True)
                    S.status = "🎤 Слушаю..."; push()
                continue
            if action == "deny":
                S.pending = None
                S.pending_items = None
                S.errors = 0
                say(f"Хорошо, отменил. {HINT_SHORT}", wait=True)
                S.status = "🎤 Слушаю..."; push()
                continue
            if action not in ("operator", "done"):
                log("↩️ pending сброшен")
                S.pending = None
                S.pending_items = None

        if action == "done":
            if not S.cart:
                say("Корзина пустая. Назовите хотя бы один товар.", wait=True)
                S.status = "🎤 Слушаю..."; push()
            else:
                await do_handoff("Передаю оператору.")
            continue

        if action == "operator":
            await do_handoff("Соединяю с оператором.")
            continue

        if action == "add_item":
            q, qty, is_single = intent["product"], intent["qty"], intent.get("single_word", False)

            q_normalized = normalize_query(q)
            if q_normalized != q:
                log(f"   🔧 Нормализация: '{q}' → '{q_normalized}'")
                q = q_normalized

            S.status = f"🔍 Ищу «{q}»..."; push()
            log(f"🔍 MCP search: {q!r} qty={qty}")
            try:
                results = await mcp.search(q)
            except Exception as e:
                log(f"   MCP error: {e}"); results = []

            log(f"   MCP нашёл {len(results)} товаров")

            if not results:
                S.errors += 1
                if S.errors >= MAX_ERRORS:
                    await do_handoff("Не могу найти товары, передаю оператору.")
                else:
                    say(f"Не нашёл «{q}». Попробуйте назвать проще.", wait=True)
                    S.status = "🎤 Слушаю..."; push()
                continue

            if is_single and len(results) > 1:
                await show_top_choices(q, qty, results)
                continue

            item = results[0]
            price = item.get("price", {}).get("current", "?")
            S.pending = {"xml_id": int(item["xml_id"]), "name": item["name"],
                         "price": price, "qty": qty}
            S.errors = 0
            S.status = "⏳ Жду подтверждения"
            say(f"Нашёл: {item['name']}, {price} рублей, {qty} штука. Верно?", wait=True)
            push()
            continue

        S.errors += 1
        if S.errors >= MAX_ERRORS:
            await do_handoff("Не понимаю запрос, передаю оператору.")
        else:
            say(f"Не понял. {HINT_SHORT}.", wait=True)
            S.status = "🎤 Слушаю..."; push()

# ── FASTAPI ──────────────────────────────────────
app = FastAPI()

HTML = """<!DOCTYPE html><html><head><meta charset="utf-8"><title>Голосовой ассистент ВкусВилл</title>
<style>
*{box-sizing:border-box}
body{font-family:system-ui;max-width:960px;margin:0 auto;padding:1rem;background:#f0f2f5}
h2{margin:0 0 1rem;color:#333}
.status{padding:10px 16px;border-radius:8px;background:#1565C0;color:#fff;font-weight:600;margin-bottom:0.5rem}
.mic-bar{font-family:monospace;font-size:12px;padding:4px 16px;background:#263238;color:#4caf50;border-radius:4px;margin-bottom:1rem;min-height:24px}
.btns{display:flex;gap:10px;margin-bottom:1rem;flex-wrap:wrap}
button{padding:10px 22px;font-size:14px;font-weight:600;border:none;border-radius:7px;cursor:pointer}
.go{background:#2e7d32;color:#fff}.op{background:#c62828;color:#fff}.rs{background:#546e7a;color:#fff}
button:disabled{opacity:0.5;cursor:not-allowed}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:1rem}
.box{background:#fff;border-radius:10px;padding:14px;border:1px solid #ddd}
.box h3{font-size:12px;text-transform:uppercase;color:#888;margin:0 0 10px}
.log{height:320px;overflow-y:auto;font-size:13px;line-height:1.8}
.bot{color:#1565C0;font-weight:500}.usr{color:#2e7d32}.lnk{color:#999;font-size:12px}
.ci{display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #f0f0f0;font-size:14px}
.url{margin-top:1rem;padding:12px 16px;background:#e8f5e9;border-radius:8px;font-size:13px;display:none}
.url a{color:#1b5e20;font-weight:600;word-break:break-all}
.empty{color:#bbb;font-style:italic;font-size:13px}
.hint{background:#fff8e1;border-left:3px solid #f9a825;padding:10px 14px;border-radius:4px;font-size:13px;margin-bottom:1rem;line-height:1.8}
</style></head><body>
<h2>🎙 Голосовой сборщик корзины</h2>
<div class="hint">
  <b>«да»</b> — подтвердить &nbsp;·&nbsp;
  <b>«нет»</b> — отменить &nbsp;·&nbsp;
  <b>«первый/второй/третий»</b> — выбрать &nbsp;·&nbsp;
  <b>«всё»</b> / <b>«готово»</b> — передать оператору &nbsp;·&nbsp;
  <b>«оператор»</b> — оператор немедленно
</div>
<div class="status" id="st">⏸ Ожидание</div>
<div class="mic-bar" id="mic">
  <span id="mic-icon">⏸️</span>
  <span id="mic-text">Микрофон не активен</span>
  <span id="mic-level"></span>
</div>
<div class="btns">
  <button class="go" id="btnStart" onclick="cmd('start')">▶ Начать звонок</button>
  <button class="op" onclick="cmd('handoff')">📞 Передать оператору</button>
  <button class="rs" onclick="cmd('reset')">↺ Сброс</button>
</div>
<div class="grid">
  <div class="box"><h3>Диалог</h3>
    <div class="log" id="log"><p class="empty">Нажмите «Начать звонок»</p></div>
  </div>
  <div class="box"><h3>Корзина</h3>
    <div id="cart"><p class="empty">Пусто</p></div>
    <div id="total" style="text-align:right;font-size:13px;color:#555;margin-top:8px"></div>
  </div>
</div>
<div class="url" id="url"></div>
<script>
const ws=new WebSocket(`ws://${location.host}/ws`);
ws.onmessage=({data})=>{
  const d=JSON.parse(data);
  // Воспроизводим аудио, если пришло
  if(d.audio){
    const audio=new Audio('data:audio/mp3;base64,'+d.audio);
    audio.play().catch(e=>console.log('Audio play failed:',e));
  }
  document.getElementById('st').textContent=d.status;
  const micIcon = document.getElementById('mic-icon');
  const micText = document.getElementById('mic-text');
  const micLevel = document.getElementById('mic-level');

  if (d.active) {
    if (d.status.includes('Слушаю')) {
      micIcon.textContent = '🎤';
      micText.textContent = 'Слушаю...';
    } else if (d.status.includes('Распознаю') || d.status.includes('Ищу')) {
      micIcon.textContent = '🧠';
      micText.textContent = 'Обрабатываю...';
    } else if (d.status.includes('Жду')) {
      micIcon.textContent = '⏳';
      micText.textContent = d.status;
    } else {
      micIcon.textContent = '🔊';
      micText.textContent = 'Отвечаю...';
    }
    micLevel.textContent = d.mic_bar || '';
  } else {
    micIcon.textContent = '⏸️';
    micText.textContent = 'Микрофон не активен';
    micLevel.textContent = '';
  }
  const btn=document.getElementById('btnStart');
  if(d.active){btn.textContent='🛑 Завершить';btn.className='op';btn.onclick=()=>cmd('handoff')}
  else{btn.textContent='▶ Начать звонок';btn.className='go';btn.onclick=()=>cmd('start')}
  const lg=document.getElementById('log');
  lg.innerHTML=d.log.length
    ?d.log.map(l=>`<div class="${l.startsWith('🤖')?'bot':l.startsWith('👤')?'usr':l.startsWith('🎤')?'lnk':'lnk'}">${l}</div>`).join('')
    :'<p class="empty">Ожидание...</p>';
  lg.scrollTop=lg.scrollHeight;
  let total=0;
  document.getElementById('cart').innerHTML=d.cart.length
    ?d.cart.map(i=>{total+=(+i.price||0)*i.qty;
      return`<div class="ci"><span>${i.name}</span><b>${i.qty}×${i.price}₽</b></div>`}).join('')
    :'<p class="empty">Пусто</p>';
  document.getElementById('total').textContent=total>0?`Итого ≈ ${total} ₽`:'';
  const ub=document.getElementById('url');
  if(d.url){ub.style.display='block';ub.innerHTML=`🔗 <a href="${d.url}" target="_blank">${d.url}</a>`;}
  else ub.style.display='none';
};
function cmd(a){ws.send(JSON.stringify({action:a}));}
</script></body></html>"""

@app.get("/", response_class=HTMLResponse)
async def root(): return HTML

@app.websocket("/ws")
async def ws_ep(ws: WebSocket):
    await ws.accept()
    _ws_clients.append(ws)
    push()
    try:
        while True:
            msg = await ws.receive_json()
            a = msg.get("action")
            if a == "start":
                S.reset()
                detector.force_reset()   # <-- ДОБАВИТЬ эту строку
                S.active = True; S.start_ts = time.time()
                S.status = "🎤 Слушаю клиента..."; push()
                say(INTRO_FIRST)
            elif a == "handoff":
                await do_handoff("Оператор подключается.")
            elif a == "reset":
                S.reset(); push()
    except WebSocketDisconnect:
        if ws in _ws_clients: _ws_clients.remove(ws)

async def _ws_sender():
    while True:
        data = await _bcast_q.get()
        dead = []
        for ws in _ws_clients:
            try: await ws.send_text(data)
            except: dead.append(ws)
        for ws in dead:
            if ws in _ws_clients: _ws_clients.remove(ws)

def _run_uvicorn():
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")

async def main():
    await init_catalog()

    threading.Thread(target=_run_uvicorn, daemon=True).start()
    stop_ev = threading.Event()
    threading.Thread(target=audio_capture, args=(stop_ev,), daemon=True).start()
    asyncio.create_task(_ws_sender())
    log(f"🌐 http://localhost:8000  RMS-threshold={detector.config.speech_threshold}  silence={detector.config.silence_duration}s")
    await voice_loop()

if __name__ == "__main__":
    asyncio.run(main())

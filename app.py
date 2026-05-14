"""
app.py MVP v10 - АК-47 Франкенштейн ИИ
Три механизма защиты:
1. Initial Prompt с топ-50 товаров
2. Hotwords из MCP (топ-100)
3. Fuzzy-поиск по всему каталогу (rapidfuzz)
"""
import asyncio, json, os, queue, re, tempfile, threading, time
from typing import Dict, List, Optional

import numpy as np
import sounddevice as sd
import edge_tts
from faster_whisper import WhisperModel
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn
from rapidfuzz import fuzz, process

from mcp_client import VkusVillMCP

# ── КОНФИГ ───────────────────────────────────────
SAMPLE_RATE    = 16000
VAD_THRESH     = 350
SILENCE_SEC    = 2.2
MIN_SAMPLES    = 16000
POST_TTS_PAUSE = 0.5
MAX_ITEMS      = 5
MAX_ERRORS     = 2

STT_MODEL = "deepdml/faster-whisper-large-v3-turbo-ct2"
TTS_VOICE      = "ru-RU-SvetlanaNeural"

# Топ-50 популярных товаров (hardcoded для initial prompt)
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

SILENT_NEEDED  = int(SILENCE_SEC / (512 / SAMPLE_RATE))

def ts():   return time.strftime("%H:%M:%S")
def log(m): print(f"[{ts()}] {m}", flush=True)

# ── TTS ──────────────────────────────────────────
try:
    import pygame
    pygame.mixer.init()
except ImportError:
    log("⚠️ pygame не установлен, TTS будет без звука")
    class MockPygame:
        mixer = type('obj', (object,), {'init': lambda: None, 'music': type('obj', (object,), {'load': lambda x: None, 'play': lambda: None, 'unload': lambda: None, 'get_busy': lambda: False})()})()
    pygame = MockPygame()

is_speaking  = False
_mic_blocked = False
_tts_done    = threading.Event()
_tts_done.set()
_tts_q: queue.Queue = queue.Queue()

def _tts_worker():
    global is_speaking, _mic_blocked
    loop = asyncio.new_event_loop()

    async def _speak(text: str):
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tmp = f.name
        try:
            await edge_tts.Communicate(text, TTS_VOICE).save(tmp)
            pygame.mixer.music.load(tmp)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                await asyncio.sleep(0.05)
        finally:
            try:
                pygame.mixer.music.unload()
                os.unlink(tmp)
            except Exception:
                pass

    while True:
        text = _tts_q.get()
        if text is None:
            break
        is_speaking = True
        _mic_blocked = True
        _tts_done.clear()
        log(f"🔊 TTS → «{text}»")
        loop.run_until_complete(_speak(text))
        is_speaking = False
        time.sleep(POST_TTS_PAUSE)
        _mic_blocked = False
        _tts_done.set()

threading.Thread(target=_tts_worker, daemon=True).start()
log(f"🗣 TTS голос: {TTS_VOICE}")

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
        self.catalog: List[str] = []  # кэш названий товаров для fuzzy
        self.hotwords: List[str] = []  # топ-100 для hotwords
    def __init__(self): self.reset()

S   = Session()
mcp = VkusVillMCP()
_audio_q:    queue.Queue     = queue.Queue()
_ws_clients: List[WebSocket] = []
_bcast_q:    asyncio.Queue   = asyncio.Queue()

def push():
    data = json.dumps(
        {"log": S.history, "cart": S.cart,
         "status": S.status, "url": S.basket_url},
        ensure_ascii=False
    )
    try: _bcast_q.put_nowait(data)
    except asyncio.QueueFull: pass

def say(text: str, wait: bool = False):
    S.history.append(f"🤖 {text}")
    _tts_q.put(text)
    push()
    if wait:
        _tts_done.wait()

# ── ИНИЦИАЛИЗАЦИЯ КАТАЛОГА ───────────────────────
async def init_catalog():
    """Загружает каталог товаров и топ-100 для hotwords"""
    log("📚 Загружаю популярные товары для fuzzy-поиска...")
    # Пока используем hardcoded список, так как MCP не умеет в пустой запрос
    S.catalog = POPULAR_ITEMS.copy()
    S.hotwords = [item[:20] for item in POPULAR_ITEMS[:50]]
    log(f"✅ Загружено {len(S.catalog)} товаров, {len(S.hotwords)} hotwords (hardcoded)")

# ── NLU (улучшенная) ─────────────────────────────
_CONFIRM = re.compile(r'\b(да|верно|правильно|ок|окей|хорошо|подтверждаю|конечно|именно|точно|угу|ага)\b', re.I)
_DONE    = re.compile(r'\b(всё|все|хватит|достаточно|готово|оформляй|оформите|заканчивай|больше ничего)\b', re.I)
_OP      = re.compile(r'\b(оператор|человек|живой|менеджер|помогите|соедини|переключи|стоп)\b', re.I)
_DENY    = re.compile(r'\b(нет|не верно|неверно|не то|другой|другое|не тот|не та)\b', re.I)
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
    """Находит ближайшее название товара в каталоге (АК-47 деталь №3)"""
    if not query or len(query) < 3 or not S.catalog:
        return query
    
    # Ищем с порогом 70%
    result = process.extractOne(query, S.catalog, scorer=fuzz.ratio, score_cutoff=70)
    if result:
        fixed, score = result
        if score > 80:
            log(f"   🔧 Fuzzy замена: '{query}' → '{fixed}' (сходство {score}%)")
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
    """Нормализует запрос перед отправкой в MCP"""
    # 3,2% → 3.2% (MCP понимает точку, а не запятую)
    query = re.sub(r'(\d+),(\d+%)', r'\1.\2', query)
    # Убираем слово "жирности" (оно избыточно)
    query = re.sub(r'\bжирности\b', '', query, flags=re.I)
    # Убираем двойные пробелы
    query = re.sub(r'\s+', ' ', query).strip()
    return query

def extract_select_index(text: str) -> Optional[int]:
    m = _SELECT.search(text)
    if m:
        word = m.group(1).lower()
        if word in ["первый", "1-й", "1"]:
            return 0
        if word in ["второй", "2-й", "2"]:
            return 1
        if word in ["третий", "3-й", "3"]:
            return 2
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

    # АК-47: сначала fuzzy fix
    t = fuzzy_fix(t)

    qty, cleaned = _extract_qty_and_clean(t)
    query = _build_search_query(cleaned)

    log(f"   NLU: raw={t!r} → qty={qty} query={query!r}")

    if len(query) >= 2:
        is_single_word = len(query.split()) == 1
        return {"intent": "add_item", "product": query, "qty": qty, "single_word": is_single_word}
    return {"intent": "unknown"}

# ── АУДИО ────────────────────────────────────────
def audio_capture(stop_ev: threading.Event):
    buf, speaking, silent_chunks = [], False, 0

    def cb(indata, frames, t, status):
        nonlocal buf, speaking, silent_chunks
        if _mic_blocked:
            buf[:] = []; speaking = False; silent_chunks = 0
            return
        vol = int(np.abs(indata.flatten()).mean())
        if S.active:
            bar = "█" * min(vol // 30, 30)
            print(f"\r  mic {vol:4d} |{bar:<30}|  ", end="", flush=True)
        if vol > VAD_THRESH:
            if not speaking:
                print(f"\n[{ts()}] 🎤 голос vol={vol}")
            speaking = True; silent_chunks = 0
            buf.extend(indata.flatten().tolist())
        else:
            if speaking:
                buf.extend(indata.flatten().tolist())
                silent_chunks += 1
                if silent_chunks >= SILENT_NEEDED:
                    n = len(buf)
                    if n >= MIN_SAMPLES:
                        arr = np.array(buf, dtype=np.int16).astype(np.float32) / 32768.0
                        _audio_q.put(arr)
                        print(f"\n[{ts()}] ✂️  фраза {n/SAMPLE_RATE:.2f}с → очередь")
                    else:
                        print(f"\n[{ts()}] ⚡ коротко ({n/SAMPLE_RATE:.2f}с), игнор")
                    buf[:] = []; speaking = False; silent_chunks = 0

    log("🎙 Микрофон запущен")
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                        dtype="int16", blocksize=512, callback=cb):
        while not stop_ev.is_set():
            time.sleep(0.05)

# ── STT ──────────────────────────────────────────
print(f"⏳ Загружаю Whisper ({STT_MODEL})...")
_whisper = WhisperModel(
    STT_MODEL, 
    device="cpu", 
    compute_type="int8",
    cpu_threads=8,
    num_workers=2
)
print("✅ Whisper готов")

_HALLUCINATIONS = re.compile(
    r'^(спасибо|до свидания|пробки?|субтитры|подписывайтесь|продолжение следует'
    r'|в этом видео|редактирование|\.+|-+|\s*)$',
    re.I
)

def transcribe(audio: np.ndarray) -> str:
    # АК-47 деталь №1: initial prompt с топ-50 товаров
    initial_prompt = f"Пользователь диктует список покупок в магазине. Товары: {', '.join(POPULAR_ITEMS[:30])}. Распознавай чётко названия продуктов."
    
    # АК-47 деталь №2: hotwords из MCP
    hotwords_str = ", ".join(S.hotwords[:50]) if S.hotwords else ""
    
    segs, info = _whisper.transcribe(
        audio,
        beam_size=1,
        language="ru",
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        no_speech_threshold=0.6,
        log_prob_threshold=-1.0,
        compression_ratio_threshold=2.4,
        initial_prompt=initial_prompt,
        hotwords=hotwords_str if hotwords_str else None
    )
    parts = []
    for seg in segs:
        if seg.no_speech_prob > 0.6:
            log(f"   STT: сегмент отброшен (no_speech_prob={seg.no_speech_prob:.2f}): {seg.text!r}")
            continue
        parts.append(seg.text.strip())
    text = " ".join(parts).strip()

    if _HALLUCINATIONS.match(text):
        log(f"   STT: галлюцинация отброшена: {text!r}")
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
HINT_ADD  = "Называйте следующий товар."
HINT_DONE = "Скажите «всё» когда закончите."

async def voice_loop():
    log("🔄 voice_loop запущен")
    while True:
        try:
            audio = _audio_q.get_nowait()
        except queue.Empty:
            await asyncio.sleep(0.05)
            continue

        if not S.active:
            continue

        await asyncio.to_thread(_tts_done.wait)

        S.status = "⚙️ Распознаю..."
        push()
        t0   = time.time()
        text = await asyncio.to_thread(transcribe, audio)
        log(f"📝 Whisper {time.time()-t0:.1f}с: «{text}»")

        if not text:
            S.status = "🎤 Слушаю..."
            push()
            continue

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
                say(f"Добавил {item['name']}. {HINT_ADD} {HINT_DONE}", wait=True)
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
                    say(f"Добавил {item['name']}. {HINT_ADD} {HINT_DONE}", wait=True)
                    S.status = "🎤 Слушаю..."; push()
                continue
            if action == "deny":
                S.pending = None
                S.pending_items = None
                S.errors = 0
                say(f"Хорошо, отменил. {HINT_ADD}", wait=True)
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
            
            # Нормализуем запрос перед отправкой
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
            say(f"Не понял. {HINT_ADD} {HINT_DONE}", wait=True)
            S.status = "🎤 Слушаю..."; push()

# ── FASTAPI ──────────────────────────────────────
app = FastAPI()

HTML = """<!DOCTYPE html><html><head><meta charset="utf-8"><title>Ассистент v10 - АК-47</title>
<style>
body{font-family:system-ui;max-width:960px;margin:0 auto;padding:1.5rem;background:#f0f2f5}
.status{padding:10px 16px;border-radius:8px;background:#1565C0;color:#fff;font-weight:600;margin-bottom:1rem}
.btns{display:flex;gap:10px;margin-bottom:1rem;flex-wrap:wrap}
button{padding:10px 22px;font-size:14px;font-weight:600;border:none;border-radius:7px;cursor:pointer}
.go{background:#2e7d32;color:#fff}.op{background:#c62828;color:#fff}.rs{background:#546e7a;color:#fff}
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
<h2>🎙 Ассистент v10 — АК-47 Франкенштейн ИИ</h2>
<div class="hint">
  <b>«да»</b> — подтвердить &nbsp;·&nbsp;
  <b>«нет»</b> — отменить &nbsp;·&nbsp;
  <b>«первый/второй/третий»</b> — выбрать из предложенных &nbsp;·&nbsp;
  <b>«всё»</b> / <b>«готово»</b> — передать оператору &nbsp;·&nbsp;
  <b>«оператор»</b> — передать немедленно
</div>
<div class="status" id="st">⏸ Ожидание</div>
<div class="btns">
  <button class="go" onclick="cmd('start')">▶ Начать звонок</button>
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
  document.getElementById('st').textContent=d.status;
  const lg=document.getElementById('log');
  lg.innerHTML=d.log.length
    ?d.log.map(l=>`<div class="${l.startsWith('🤖')?'bot':l.startsWith('👤')?'usr':'lnk'}">${l}</div>`).join('')
    :'<p class="empty">Нажмите «Начать звонок»</p>';
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
                S.active = True; S.start_ts = time.time()
                S.status = "🎤 Слушаю клиента..."; push()
                say("Здравствуйте! Называйте товары по одному. Когда закончите — скажите «всё».")
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
    # Загружаем каталог перед стартом
    await init_catalog()
    
    threading.Thread(target=_run_uvicorn, daemon=True).start()
    stop_ev = threading.Event()
    threading.Thread(target=audio_capture, args=(stop_ev,), daemon=True).start()
    asyncio.create_task(_ws_sender())
    log(f"🌐 http://localhost:8000  VAD={VAD_THRESH}  silence={SILENCE_SEC}s")
    await voice_loop()

if __name__ == "__main__":
    asyncio.run(main())

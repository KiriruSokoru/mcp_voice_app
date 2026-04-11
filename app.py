"""
app.py MVP v4 — с выводом в терминал (логгирование действий)
python app.py → http://localhost:8000 → нажми "Начать звонок"
"""
import asyncio, json, threading, queue, time, re
from typing import List, Dict, Optional
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
import pyttsx3
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn
from mcp_client import VkusVillMCP

# ── КОНФИГ ───────────────────────────────────────
SAMPLE_RATE = 16000
VAD_THRESH  = 400
SILENCE_SEC = 1.5
MIN_SAMPLES = 6000    # 0.375 сек минимум
STT_MODEL   = "small"
MAX_ITEMS   = 5
MAX_ERRORS  = 2

def ts(): return time.strftime("%H:%M:%S")
def log(msg): print(f"[{ts()}] {msg}", flush=True)

# ── TTS ──────────────────────────────────────────
_tts_q: queue.Queue = queue.Queue()

def _tts_worker():
    engine = pyttsx3.init()
    engine.setProperty("rate", 145)
    for v in engine.getProperty("voices"):
        name = v.name.lower()
        if any(x in name for x in ["irina","pavel","русский","russian"]):
            engine.setProperty("voice", v.id)
            break
    while True:
        text = _tts_q.get()
        if text is None: break
        log(f"🔊 TTS: {text}")
        engine.say(text)
        engine.runAndWait()

threading.Thread(target=_tts_worker, daemon=True).start()

# ── СОСТОЯНИЕ ────────────────────────────────────
class Session:
    def reset(self):
        self.active   = False
        self.cart: List[Dict] = []
        self.history: List[str]  = []
        self.errors   = 0
        self.start_ts = 0.0
        self.status   = "⏸ Ожидание"
        self.basket_url = ""
        self.pending: Optional[Dict] = None
    def __init__(self): self.reset()

S = Session()
_audio_q: queue.Queue = queue.Queue()
_ws_clients: List[WebSocket] = []
_bcast_q: asyncio.Queue = asyncio.Queue()
mcp = VkusVillMCP()

def say(text: str):
    S.history.append(f"🤖 {text}")
    _tts_q.put(text)
    push()

def push():
    data = json.dumps(
        {"log": S.history, "cart": S.cart, "status": S.status, "url": S.basket_url},
        ensure_ascii=False
    )
    try: _bcast_q.put_nowait(data)
    except asyncio.QueueFull: pass

# ── NLU ──────────────────────────────────────────
_ADD     = re.compile(r'\b(добавь|добавьте|положи|хочу|дай|нужно|нужен|нужна|нужны|купи|возьми|закажи)\b', re.I)
_CONFIRM = re.compile(r'\b(да|верно|правильно|ок|окей|хорошо|подтверждаю|конечно|именно|точно|угу|ага)\b', re.I)
_DONE    = re.compile(r'\b(всё|хватит|достаточно|готово|оформляй|оформите|заканчивай|всё готово)\b', re.I)
_OP      = re.compile(r'\b(оператор|человек|живой|менеджер|помогите|соедини|переключи|стоп)\b', re.I)
_QTY_W   = {"один":1,"одну":1,"одна":1,"два":2,"две":2,"три":3,"четыре":4,"пять":5}
_JUNK    = re.compile(r'\b(мне|пожалуйста|ещё|еще|можно|просто|литр|литра|грамм|кг|упаковку|штуку|пачку|бутылку)\b', re.I)

def _qty(text):
    for w,n in _QTY_W.items():
        if re.search(rf'\b{w}\b', text, re.I):
            return n, re.sub(rf'\b{w}\b','',text,flags=re.I).strip()
    m = re.search(r'\b(\d+)\b', text)
    if m: return min(int(m.group(1)),10), (text[:m.start()]+text[m.end():]).strip()
    return 1, text

def nlu(text: str) -> dict:
    t = text.strip()
    if _OP.search(t):      return {"intent":"operator"}
    if _CONFIRM.search(t): return {"intent":"confirm"}
    if _DONE.search(t):    return {"intent":"done"}
    clean = _ADD.sub("", t).strip() or t
    qty, clean = _qty(clean)
    clean = _JUNK.sub(" ", clean)
    clean = re.sub(r'[^\w\s\-]',' ',clean)
    clean = re.sub(r'\s+',' ',clean).strip()
    if len(clean) >= 2:
        return {"intent":"add_item","product":clean,"qty":qty}
    return {"intent":"unknown"}

# ── АУДИО ────────────────────────────────────────
def audio_capture(stop: threading.Event):
    buf, last_voice, speaking = [], time.time(), False

    def cb(indata, frames, t, status):
        nonlocal buf, last_voice, speaking
        vol = int(np.abs(indata.flatten()).mean())
        # показываем уровень только при активной сессии
        if S.active:
            bar = "█" * min(vol//40, 30)
            print(f"\r  mic {vol:4d} |{bar:<30}|  ", end="", flush=True)

        if vol > VAD_THRESH:
            if not speaking:
                print(f"\n[{ts()}] 🎤 ГОЛОС vol={vol}")
            speaking = True
            last_voice = time.time()
            buf.extend(indata.flatten().tolist())
        elif speaking and time.time()-last_voice > SILENCE_SEC:
            n = len(buf)
            if n >= MIN_SAMPLES:
                arr = np.array(buf, dtype=np.int16).astype(np.float32) / 32768.0
                _audio_q.put(arr)
                print(f"\n[{ts()}] ✂️  фраза {n/SAMPLE_RATE:.2f}с → очередь (размер: {_audio_q.qsize()})")
            else:
                print(f"\n[{ts()}] ⚡ слишком коротко ({n} сэмплов), игнор")
            buf, speaking = [], False

    log("🎙 Запускаю микрофон...")
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                        dtype="int16", blocksize=512, callback=cb):
        while not stop.is_set():
            time.sleep(0.05)

# ── STT ──────────────────────────────────────────
print("⏳ Загружаю Whisper...")
_whisper = WhisperModel(STT_MODEL, device="cpu", compute_type="int8")
print("✅ Whisper готов")

def transcribe(audio: np.ndarray) -> str:
    segs, _ = _whisper.transcribe(audio, beam_size=1, language="ru", vad_filter=True)
    return " ".join(s.text for s in segs).strip()

# ── HANDOFF ──────────────────────────────────────
async def do_handoff(reason: str):
    S.active = False
    S.status = "📞 Передача оператору..."
    say(reason)
    push()
    if S.cart:
        try:
            url = await mcp.cart_link([{"xml_id": i["xml_id"], "q": i["qty"]} for i in S.cart])
            S.basket_url = url
            S.history.append(f"🔗 {url}")
            log(f"🔗 Корзина: {url}")
        except Exception as e:
            log(f"⚠️ cart_link ошибка: {e}")
    S.status = "✅ Готово — у оператора"
    push()

# ── ГОЛОСОВОЙ ЦИКЛ ───────────────────────────────
async def voice_loop():
    log("🔄 voice_loop запущен")
    while True:
        try:
            audio = _audio_q.get_nowait()
        except queue.Empty:
            await asyncio.sleep(0.05)
            continue

        if not S.active:
            log("⏭ сессия не активна — пропускаю аудио")
            continue

        S.status = "⚙️ Распознаю..."
        push()
        log("⚙️ Транскрипция...")

        t0 = time.time()
        text = await asyncio.to_thread(transcribe, audio)
        log(f"📝 Whisper ({time.time()-t0:.1f}с): «{text}»")

        if not text:
            log("   (пусто — игнор)")
            S.status = "🎤 Слушаю..."
            push()
            continue

        S.history.append(f"👤 {text}")
        intent = nlu(text)
        log(f"🧠 intent={intent}")
        action = intent["intent"]

        # ПОДТВЕРЖДЕНИЕ
        if S.pending and action == "confirm":
            item = S.pending
            S.cart.append(item)
            S.pending = None
            S.errors = 0
            log(f"✅ Добавлен: {item['name']}")
            say(f"Добавил {item['name']}. Что ещё?")
            S.status = "🎤 Слушаю..."
            push()
            if len(S.cart) >= MAX_ITEMS:
                await do_handoff(f"Набрали {len(S.cart)} позиций. Передаю оператору.")
            continue

        # если pending но не confirm — сбрасываем pending и обрабатываем как новый
        if S.pending and action not in ("operator","done"):
            log(f"↩️ pending сброшен (получен {action})")
            S.pending = None

        if action == "add_item":
            q, qty = intent["product"], intent["qty"]
            S.status = f"🔍 Ищу «{q}»..."
            push()
            log(f"🔍 MCP search: {q!r}")
            try:
                results = await mcp.search(q)
                log(f"   → {len(results)} результатов")
            except Exception as e:
                log(f"   → MCP error: {e}")
                results = []

            if not results:
                S.errors += 1
                log(f"❌ не найдено (ошибок: {S.errors}/{MAX_ERRORS})")
                if S.errors >= MAX_ERRORS:
                    await do_handoff("Не могу найти товары. Передаю оператору.")
                else:
                    say(f"Не нашёл «{q}». Попробуйте назвать иначе.")
                    S.status = "🎤 Слушаю..."
                    push()
                continue

            item = results[0]
            price = item.get("price", {}).get("current", "?")
            S.pending = {"xml_id": int(item["xml_id"]), "name": item["name"], "price": price, "qty": qty}
            S.errors = 0
            S.status = "⏳ Жду подтверждения"
            say(f"{item['name']}, {price} рублей, {qty} штуки. Верно?")
            push()
            continue

        if action == "done":
            await do_handoff("Передаю оператору с корзиной.")
            continue

        if action == "operator":
            await do_handoff("Соединяю с оператором.")
            continue

        # UNKNOWN
        S.errors += 1
        log(f"❓ unknown (ошибок: {S.errors}/{MAX_ERRORS})")
        if S.errors >= MAX_ERRORS:
            await do_handoff("Не понимаю. Передаю оператору.")
        else:
            say("Повторите, пожалуйста. Назовите товар.")
            S.status = "🎤 Слушаю..."
            push()

# ── FASTAPI ──────────────────────────────────────
app = FastAPI()

HTML = """<!DOCTYPE html><html><head><meta charset="utf-8"><title>Ассистент</title>
<style>
body{font-family:system-ui;max-width:900px;margin:0 auto;padding:1.5rem;background:#f5f5f5}
.status{padding:10px 16px;border-radius:8px;background:#1565C0;color:#fff;font-weight:600;margin-bottom:1rem}
.btns{display:flex;gap:10px;margin-bottom:1rem}
button{padding:10px 22px;font-size:14px;font-weight:600;border:none;border-radius:7px;cursor:pointer}
.go{background:#2e7d32;color:#fff} .op{background:#c62828;color:#fff} .rs{background:#546e7a;color:#fff}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:1rem}
.box{background:#fff;border-radius:10px;padding:14px;border:1px solid #ddd}
.box h3{font-size:13px;color:#666;margin-bottom:8px}
.log{height:300px;overflow-y:auto;font-size:13px;line-height:1.7}
.ci{display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #eee;font-size:14px}
.url{margin-top:1rem;padding:12px;background:#e8f5e9;border-radius:8px;font-size:13px;display:none}
.url a{color:#1b5e20;font-weight:600;word-break:break-all}
.empty{color:#999;font-style:italic}
</style></head><body>
<h2>🎙 Ассистент предзаполнения заказа</h2>
<div class="status" id="st">⏸ Ожидание</div>
<div class="btns">
  <button class="go" onclick="cmd('start')">▶ Начать звонок</button>
  <button class="op" onclick="cmd('handoff')">📞 Забрать (оператор)</button>
  <button class="rs" onclick="cmd('reset')">↺ Сброс</button>
</div>
<div class="grid">
  <div class="box"><h3>ДИАЛОГ</h3><div class="log" id="log"><p class="empty">Нажмите "Начать звонок"</p></div></div>
  <div class="box"><h3>КОРЗИНА</h3><div id="cart"><p class="empty">Пусто</p></div></div>
</div>
<div class="url" id="url"></div>
<script>
const ws=new WebSocket(`ws://${location.host}/ws`);
ws.onmessage=({data})=>{
  const d=JSON.parse(data);
  document.getElementById('st').textContent=d.status;
  const lg=document.getElementById('log');
  lg.innerHTML=d.log.length?d.log.map(l=>`<div>${l}</div>`).join(''):'<p class="empty">Нажмите "Начать звонок"</p>';
  lg.scrollTop=lg.scrollHeight;
  document.getElementById('cart').innerHTML=d.cart.length
    ?d.cart.map(i=>`<div class="ci"><span>${i.name}</span><b>${i.qty}шт·${i.price}₽</b></div>`).join('')
    :'<p class="empty">Пусто</p>';
  const ub=document.getElementById('url');
  if(d.url){ub.style.display='block';ub.innerHTML=`🔗 <a href="${d.url}" target="_blank">${d.url}</a>`;}
  else{ub.style.display='none';}
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
    log(f"🔌 WebSocket подключён (всего: {len(_ws_clients)})")
    try:
        while True:
            msg = await ws.receive_json()
            a = msg.get("action")
            log(f"📨 WS action: {a}")
            if a == "start":
                S.reset()
                S.active, S.start_ts = True, time.time()
                S.status = "🎤 Слушаю клиента..."
                push()
                say("Здравствуйте! Назовите товар.")
            elif a == "handoff":
                await do_handoff("Оператор подключается.")
            elif a == "reset":
                S.reset(); push()
    except WebSocketDisconnect:
        if ws in _ws_clients: _ws_clients.remove(ws)
        log(f"🔌 WebSocket отключён")

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
    threading.Thread(target=_run_uvicorn, daemon=True).start()
    stop = threading.Event()
    threading.Thread(target=audio_capture, args=(stop,), daemon=True).start()
    asyncio.create_task(_ws_sender())
    log(f"🌐 http://localhost:8000  |  VAD={VAD_THRESH}  |  silence={SILENCE_SEC}s")
    await voice_loop()

if __name__ == "__main__":
    asyncio.run(main())

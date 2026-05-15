"""
RMS-детектор речи с подавлением эха.
Заменяет пороговый VAD в audio_capture().
"""
import time
import wave
import uuid
from pathlib import Path
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Callable, List
import numpy as np
import structlog

logger = structlog.get_logger()

# ── Конфигурация ─────────────────────────────────
@dataclass
class RMSConfig:
    speech_threshold: float = 0.02       # RMS > этого → речь
    silence_threshold: float = 0.008     # RMS < этого → тишина
    silence_duration: float = 1.2        # секунд тишины для конца фразы
    max_speech_duration: float = 15.0    # макс. длительность фразы
    min_speech_duration: float = 0.3     # мин. длительность (короче = шум)
    sample_rate: int = 16000
    window_size: int = 512               # ~32ms при 16kHz


class AudioState(Enum):
    SILENCE = "silence"
    SPEECH = "speech"
    TTS_PLAYING = "tts"


class RMSDetector:
    """
    Детектор речи на основе RMS-энергии сигнала.
    
    Заменяет:
    - VAD_THRESH = 350
    - Логику speaking / silent_chunks в audio_capture()
    
    Добавляет:
    - mute_for_tts() / unmute_after_tts() для эхоподавления
    - Сохранение ложных срабатываний в datasets/silence_samples/
    """
    
    def __init__(self, config: RMSConfig = None, dataset_dir: str = None):
        self.config = config or RMSConfig()
        self.state = AudioState.SILENCE
        self._buffer: List[int] = []        # int16 семплы
        self._silence_start: Optional[float] = None
        self._speech_start: Optional[float] = None
        self._is_muted = False
        
        # Колбэки (вызываются из audio_capture)
        self.on_speech_start: Optional[Callable[[], None]] = None
        self.on_speech_end: Optional[Callable[[np.ndarray], None]] = None  # float32 массив
        self.on_false_trigger: Optional[Callable[[np.ndarray], None]] = None
        
        # Директория для ложных срабатываний
        self._dataset_dir = Path(dataset_dir) if dataset_dir else None
        if self._dataset_dir:
            self._dataset_dir.mkdir(parents=True, exist_ok=True)
    
    def process_chunk(self, indata: np.ndarray) -> Optional[np.ndarray]:
        """
        Вызывается из callback audio_capture для каждого блока.
        indata: int16 массив (512 семплов) от sounddevice.
        Возвращает: собранную фразу (float32 массив) или None.
        """
        if self._is_muted or self.state == AudioState.TTS_PLAYING:
            return None
        
        # Конвертируем int16 → float32 для RMS
        float_data = indata.astype(np.float32) / 32768.0
        rms = np.sqrt(np.mean(float_data ** 2))
        
        now = time.time()
        
        # ── Машина состояний ──
        if self.state == AudioState.SILENCE:
            if rms > self.config.speech_threshold:
                # Начало речи
                self.state = AudioState.SPEECH
                self._speech_start = now
                self._buffer = indata.flatten().tolist()
                self._silence_start = None
                logger.debug("speech_start", rms=round(rms, 4))
                if self.on_speech_start:
                    self.on_speech_start()
        
        elif self.state == AudioState.SPEECH:
            self._buffer.extend(indata.flatten().tolist())
            
            # Защита от бесконечной записи
            if now - self._speech_start > self.config.max_speech_duration:
                return self._finalize_speech()
            
            if rms < self.config.silence_threshold:
                if self._silence_start is None:
                    self._silence_start = now
                elif now - self._silence_start >= self.config.silence_duration:
                    return self._finalize_speech()
            else:
                self._silence_start = None
        
        return None
    
    def mute_for_tts(self):
        """Заглушить детектор на время озвучки (подавление эха)."""
        self._is_muted = True
        self.state = AudioState.TTS_PLAYING
        logger.debug("mic_muted_for_tts")
    
    def unmute_after_tts(self):
        """Вернуть детектор в рабочий режим после озвучки."""
        self._is_muted = False
        self.state = AudioState.SILENCE
        self._buffer = []
        self._silence_start = None
        logger.debug("mic_unmuted")
    
    def force_reset(self):
        """Сброс состояния (при старте нового звонка)."""
        self._buffer = []
        self._silence_start = None
        self._speech_start = None
        if not self._is_muted:
            self.state = AudioState.SILENCE
    
    def _finalize_speech(self) -> Optional[np.ndarray]:
        """Завершает сбор фразы, проверяет на ложное срабатывание."""
        self.state = AudioState.SILENCE
        self._silence_start = None
        
        if not self._buffer:
            return None
        
        duration = time.time() - self._speech_start
        audio_array = np.array(self._buffer, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Проверка на ложное срабатывание (слишком короткий фрагмент)
        if duration < self.config.min_speech_duration:
            logger.info("false_trigger", duration=round(duration, 3),
                       samples=len(self._buffer))
            if self.on_false_trigger:
                self.on_false_trigger(audio_array)
            self._save_false_trigger(self._buffer)
            return None
        
        logger.info("speech_end", duration=round(duration, 2),
                   samples=len(self._buffer))
        return audio_array
    
    def _save_false_trigger(self, buffer: List[int]):
        """Сохраняет ложное срабатывание как WAV для датасета."""
        if not self._dataset_dir:
            return
        try:
            filename = self._dataset_dir / f"false_trigger_{uuid.uuid4().hex[:8]}.wav"
            int16_array = np.array(buffer, dtype=np.int16)
            with wave.open(str(filename), 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(self.config.sample_rate)
                wf.writeframes(int16_array.tobytes())
            logger.debug("false_trigger_saved", filename=str(filename))
        except Exception as e:
            logger.error("save_false_trigger_error", error=str(e))

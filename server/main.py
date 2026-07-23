"""word-agent — FastAPI-прокси над `claude -p --model haiku` для сайта-словаря.

Сайт (GitHub Pages) шлёт английское слово/выражение, сервер прогоняет его
через шаблон Альбины и возвращает markdown-разбор.

Устройство как у langme-llm:
- только haiku (дешёвая модель, экономим квоту подписки);
- один запрос одновременно (asyncio lock) -> 429 когда занято;
- дисковый кэш по нормализованному слову (повторные запросы бесплатны);
- простой per-IP rate limit;
- статичный токен в X-Auth-Token (лежит в JS сайта — защита от сканеров портов,
  не от целенаправленного злоумышленника).
"""

import asyncio
import hashlib
import os
import subprocess
import time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

TOKEN = os.environ.get("WORD_AGENT_TOKEN", "")
MODEL = os.environ.get("WORD_AGENT_MODEL", "haiku")
CLAUDE_BIN = os.environ.get("WORD_AGENT_CLAUDE_BIN", "claude")
TIMEOUT_S = int(os.environ.get("WORD_AGENT_TIMEOUT", "90"))
CACHE_DIR = Path(os.environ.get("WORD_AGENT_CACHE", "~/.cache/word_agent")).expanduser()
CACHE_DIR.mkdir(parents=True, exist_ok=True)

RATE_WINDOW_S = 60
RATE_MAX = 10  # запросов с одного IP в минуту (кэш-хиты не считаются)

app = FastAPI(title="word-agent")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://fosttt.github.io",
        "http://localhost:8080",
    ],
    allow_methods=["POST", "GET"],
    allow_headers=["X-Auth-Token", "Content-Type"],
)

lock = asyncio.Lock()
_hits: dict[str, deque] = defaultdict(deque)

TEMPLATE = """Обрабатывай английское слово или выражение по данному шаблону. \
Отвечай в Markdown.

**СЛОВО ИЛИ ВЫРАЖЕНИЕ** (всегда КАПСОМ и жирным) - русский эквивалент
На следующей строке уровень сложности с цветом и обозначением:
A1 🟢 A2 🟡 B1 🔵 B2 🟣 C1 🟠 C2 🔴 (укажи только один подходящий, например: B1 🔵)

Частотность: прогресс-бар из 10 сегментов (например ▓▓▓▓▓▓▓░░░ 7/10)

Стиль:
Formal | Informal | Neutral (укажи один)

Актуальность:
Obsolete | Limited Use | Current (укажи один)

Словосочетания:
Три умеренно сложных и часто употребляемых в речи словосочетания (*курсивом*), \
где само слово или выражение всегда **жирное**. Перед каждым словосочетанием \
ставь 2 подходящих эмодзи. Сразу после каждого словосочетания приводи пример \
предложения с этим словосочетанием — в предложении слово или выражение должно \
быть **жирным**.

Семейство слов:
Просто перечисли родственные слова (без пояснений).

Исследование:
Кратко объясни: от какого слова произошло, почему используется именно в таком \
значении. (1-2 коротких предложения, без воды)

Устойчивость словосочетания:
Указывай ТОЛЬКО если запрос — словосочетание! (слабая / средняя / сильная)

Требования:
Естественный, повседневный английский. Без перевода и русского текста (кроме \
первого эквивалента). Разные ситуации и значения. Никаких дополнительных \
комментариев до или после разбора.

Опечатки:
Если есть очевидная опечатка — исправь автоматически до наиболее вероятной \
формы и работай с ней, без пояснений. Если исправление невозможно, обрабатывай \
как редкое слово уровня C2 с частотностью 0/10.

Слово или выражение для разбора:
"""


def check_token(token: str | None):
    if not TOKEN:
        raise HTTPException(500, "WORD_AGENT_TOKEN is not configured on the server")
    if token != TOKEN:
        raise HTTPException(401, "bad token")


def check_rate(ip: str):
    now = time.monotonic()
    q = _hits[ip]
    while q and now - q[0] > RATE_WINDOW_S:
        q.popleft()
    if len(q) >= RATE_MAX:
        raise HTTPException(429, "too many requests, wait a minute")
    q.append(now)


def cache_path(key: str) -> Path:
    return CACHE_DIR / (hashlib.sha256(key.encode()).hexdigest() + ".md")


def run_claude(prompt: str) -> str:
    # промпт через stdin + запрет инструментов: prompt-injection в запросе не
    # сможет заставить claude читать файлы или выполнять команды на сервере
    try:
        proc = subprocess.run(
            [CLAUDE_BIN, "-p", "--model", MODEL, "--disallowedTools", "*"],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(504, "LLM timeout")
    except FileNotFoundError:
        raise HTTPException(500, f"claude CLI not found: {CLAUDE_BIN}")
    if proc.returncode != 0:
        raise HTTPException(502, f"claude failed: {proc.stderr.strip()[:300]}")
    text = proc.stdout.strip()
    if not text:
        raise HTTPException(502, "empty LLM response")
    return text


class WordReq(BaseModel):
    text: str


@app.get("/health")
async def health():
    return {"ok": True, "model": MODEL}


@app.post("/word")
async def word(req: WordReq, request: Request,
               x_auth_token: str | None = Header(default=None)):
    check_token(x_auth_token)
    text = " ".join(req.text.split()).strip()
    if not text:
        raise HTTPException(400, "empty request")
    if len(text) > 80:
        raise HTTPException(400, "too long — one word or phrase, please")

    key = text.lower()
    p = cache_path(key)
    if p.exists():
        return {"text": p.read_text(), "cached": True}

    ip = request.client.host if request.client else "?"
    check_rate(ip)
    if lock.locked():
        raise HTTPException(429, "busy, retry in a few seconds")
    async with lock:
        answer = await asyncio.to_thread(run_claude, TEMPLATE + text)
    p.write_text(answer)
    return {"text": answer, "cached": False}

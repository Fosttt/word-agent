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
CHAPTER_TIMEOUT_S = int(os.environ.get("WORD_AGENT_CHAPTER_TIMEOUT", "420"))
CHAPTER_MAX_CHARS = int(os.environ.get("WORD_AGENT_CHAPTER_MAX", "20000"))
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
Obsolete ☠️ | Limited Use ⚠️ | Current ✅ (укажи один, обязательно вместе с эмодзи)

Словосочетания:
Три умеренно сложных и часто употребляемых в речи словосочетания (*курсивом*), \
где само слово или выражение всегда **жирное**. Перед каждым словосочетанием \
ставь 2 подходящих эмодзи, а сразу после словосочетания — его русский перевод \
в двойных фигурных скобках: {{перевод}}. Пример предложения с этим \
словосочетанием приводи С НОВОЙ СТРОКИ (не в той же строке!) — в предложении \
слово или выражение должно быть **жирным**. Сразу после предложения, ещё с \
новой строки, — русский перевод всего предложения в двойных фигурных скобках: \
{{перевод предложения}}.

Семейство слов:
Просто перечисли родственные слова (без пояснений), после каждого слова сразу \
ставь его русский перевод в двойных фигурных скобках: {{перевод}}.

Исследование:
Кратко объясни НА АНГЛИЙСКОМ: от какого слова произошло, почему используется \
именно в таком значении. (1-2 коротких предложения, без воды) Сразу после \
объяснения, с новой строки, — русский перевод всего объяснения в двойных \
фигурных скобках: {{перевод объяснения}}.

Устойчивость словосочетания:
Указывай ТОЛЬКО если запрос — словосочетание! (слабая / средняя / сильная)

Требования:
Естественный, повседневный английский. Без перевода и русского текста, кроме \
первого эквивалента и переводов в двойных фигурных скобках {{...}}. Разные \
ситуации и значения. Никаких дополнительных комментариев до или после разбора.

Опечатки:
Если есть очевидная опечатка — исправь автоматически до наиболее вероятной \
формы и работай с ней, без пояснений. Если исправление невозможно, обрабатывай \
как редкое слово уровня C2 с частотностью 0/10.

Названия разделов ВСЕГДА пиши по-русски, ровно так: «Частотность:», «Стиль:», \
«Актуальность:», «Словосочетания:», «Семейство слов:», «Исследование:».

Слово или выражение для разбора:
"""

CHAPTER_TEMPLATE = """Составь учебный словарь по тексту главы книги на английском.

Выпиши ВСЕ уникальные значимые слова: существительные, глаголы, прилагательные, \
наречия и полезные устойчивые выражения. Исключи артикли, предлоги, союзы, \
местоимения, вспомогательные глаголы, числительные и имена собственные \
(имена людей, названия мест).

Каждое слово приведи в начальной форме (лемме): существительные в единственном \
числе, глаголы в инфинитиве без to. Каждое слово — только один раз. \
Отсортируй по алфавиту.

Ответ — ТОЛЬКО строки такого формата, по одной на слово, без заголовков, \
нумерации, markdown и любых пояснений:
слово | транскрипция IPA | часть речи | перевод | уровень

Часть речи по-английски: noun, verb, adjective, adverb, phrase, idiom и т.п.
Перевод — краткий, по-русски, в контексте употребления в этой главе.
Уровень — строго один из: A1, A2, B1, B2, C1, C2.

Пример строк:
abandon | əˈbændən | verb | покидать, бросать | B2
brave | breɪv | adjective | смелый | A2

Текст главы:
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


def run_claude(prompt: str, timeout_s: int = TIMEOUT_S) -> str:
    # промпт через stdin + запрет инструментов: prompt-injection в запросе не
    # сможет заставить claude читать файлы или выполнять команды на сервере
    try:
        proc = subprocess.run(
            [CLAUDE_BIN, "-p", "--model", MODEL, "--disallowedTools", "*"],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout_s,
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


@app.post("/chapter")
async def chapter(req: WordReq, request: Request,
                  x_auth_token: str | None = Header(default=None)):
    check_token(x_auth_token)
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "empty request")
    if len(text) > CHAPTER_MAX_CHARS:
        raise HTTPException(
            400,
            f"chapter too long ({len(text)} chars, max {CHAPTER_MAX_CHARS}) — "
            "разбей главу на части",
        )

    key = "chapter:" + text.lower()
    p = cache_path(key)
    if p.exists():
        return {"text": p.read_text(), "cached": True}

    ip = request.client.host if request.client else "?"
    check_rate(ip)
    if lock.locked():
        raise HTTPException(429, "busy, retry in a few seconds")
    async with lock:
        answer = await asyncio.to_thread(
            run_claude, CHAPTER_TEMPLATE + text, CHAPTER_TIMEOUT_S
        )
    p.write_text(answer)
    return {"text": answer, "cached": False}

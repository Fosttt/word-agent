"""word-agent — FastAPI-прокси над `claude -p --model haiku` для сайта-словаря.

Сайт (GitHub Pages) шлёт английское слово/выражение, сервер прогоняет его
через шаблон Альбины и возвращает markdown-разбор.

Устройство как у langme-llm:
- только haiku (дешёвая модель, экономим квоту подписки);
- глобальный семафор на 2 процесса claude (больше 2ГБ VPS не потянет);
- дисковый кэш по нормализованному слову (повторные запросы бесплатны);
- простой per-IP rate limit;
- статичный токен в X-Auth-Token (лежит в JS сайта — защита от сканеров портов,
  не от целенаправленного злоумышленника).

Режим «Глава» (переделан 26.07 — «прорыв» по скорости):
- слова из главы выделяет САМ СЕРВЕР (токенизация + стоп-слова + отсев имён
  собственных + лемматизация simplemma), LLM главу больше не читает и ничего
  не решает про состав словаря;
- вечный пословный кэш dict.jsonl: однажды переведённое слово никогда не
  генерится заново — следующие главы той же книги почти мгновенны;
- неизвестные слова уходят в claude мелкими батчами параллельно (семафор 2);
- /chapter2 стримит NDJSON: кэшированные слова прилетают сразу, остальные —
  по мере готовности батчей; при обрыве связи задача досчитывает и кладёт
  всё в кэши, повторный запрос собирается из кэша;
- отдельный вызов по полному тексту достаёт устойчивые выражения (output
  маленький, поэтому быстрый);
- /chapter оставлен как раньше (один JSON-ответ) для старых вкладок.
"""

import asyncio
import hashlib
import json
import os
import re
import subprocess
import time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import offline_dict

TOKEN = os.environ.get("WORD_AGENT_TOKEN", "")
MODEL = os.environ.get("WORD_AGENT_MODEL", "haiku")
CLAUDE_BIN = os.environ.get("WORD_AGENT_CLAUDE_BIN", "claude")
TIMEOUT_S = int(os.environ.get("WORD_AGENT_TIMEOUT", "90"))
BATCH_TIMEOUT_S = int(os.environ.get("WORD_AGENT_BATCH_TIMEOUT", "240"))
BATCH_SIZE = int(os.environ.get("WORD_AGENT_BATCH", "60"))
CHAPTER_MAX_CHARS = int(os.environ.get("WORD_AGENT_CHAPTER_MAX", "20000"))
CACHE_DIR = Path(os.environ.get("WORD_AGENT_CACHE", "~/.cache/word_agent")).expanduser()
CACHE_DIR.mkdir(parents=True, exist_ok=True)
DICT_PATH = CACHE_DIR / "dict.jsonl"

RATE_WINDOW_S = 60
RATE_MAX = 10  # запросов с одного IP в минуту (кэш-хиты не считаются)

app = FastAPI(title="word-agent")


class BodyTimeoutASGI:
    """Обрывает запрос, если тело не дошло за timeout секунд.

    Мобильные операторы иногда теряют пакеты с телом POST (заголовки дошли,
    JSON — нет): без таймаута соединение висит в ESTAB вечно и копится.
    Таймаут действует только до полного получения тела; дальше receive()
    используется стримингом для слежения за разрывом — там ждать можно сколько
    угодно.
    """

    def __init__(self, app, timeout: float = 15):
        self.app = app
        self.timeout = timeout

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        body_done = False

        async def recv():
            nonlocal body_done
            if body_done:
                return await receive()
            try:
                msg = await asyncio.wait_for(receive(), timeout=self.timeout)
            except asyncio.TimeoutError:
                body_done = True
                return {"type": "http.disconnect"}
            if msg["type"] == "http.disconnect" or (
                    msg["type"] == "http.request" and not msg.get("more_body")):
                body_done = True
            return msg

        await self.app(scope, recv, send)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://fosttt.github.io",
        "http://localhost:8080",
    ],
    allow_methods=["POST", "GET"],
    allow_headers=["X-Auth-Token", "Content-Type"],
)
app.add_middleware(BodyTimeoutASGI, timeout=15)

# Глобальный лимит: максимум 2 процесса claude одновременно (2ГБ VPS).
# word_lock дополнительно держит очередь слов честной (по одному),
# chapter_lock — одна глава за раз.
claude_sem = asyncio.Semaphore(2)
word_lock = asyncio.Lock()
chapter_lock = asyncio.Lock()
_word_waiters = 0
WORD_QUEUE_MAX = 3  # 1 активный + 2 ждут; дальше честный 429
_hits: dict[str, deque] = defaultdict(deque)

# Разбор слова разбит на два независимых промпта, которые идут в claude
# ПАРАЛЛЕЛЬНО (семафор 2): основная карточка (уровень, частотность,
# словосочетания) и хвост (семейство слов, исследование). Полное время ≈
# времени длинной половины вместо суммы всех разделов.
TEMPLATE_MAIN = """Обрабатывай английское слово или выражение по данному шаблону. \
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

Требования:
Естественный, повседневный английский. Без перевода и русского текста, кроме \
первого эквивалента и переводов в двойных фигурных скобках {{...}}. Разные \
ситуации и значения. Никаких дополнительных комментариев до или после разбора. \
НЕ добавляй никаких других разделов (семейство слов и происхождение делает \
другой процесс).

Опечатки:
Если есть очевидная опечатка — исправь автоматически до наиболее вероятной \
формы и работай с ней, без пояснений. Если исправление невозможно, обрабатывай \
как редкое слово уровня C2 с частотностью 0/10.

Названия разделов ВСЕГДА пиши по-русски, ровно так: «Частотность:», «Стиль:», \
«Актуальность:», «Словосочетания:».

Слово или выражение для разбора:
"""

TEMPLATE_EXTRA = """Для английского слова или выражения ниже выдай в Markdown \
ТОЛЬКО перечисленные разделы, начиная ответ сразу со строки «Семейство слов:».

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
Названия разделов пиши по-русски, ровно так: «Семейство слов:», \
«Исследование:», «Устойчивость словосочетания:». Никаких других разделов, \
заголовков и комментариев до или после.

Опечатки:
Если есть очевидная опечатка — исправь автоматически до наиболее вероятной \
формы и работай с ней, без пояснений.

Слово или выражение:
"""

BATCH_TEMPLATE = """Для каждого английского слова из списка ниже выдай ровно \
одну строку строго такого формата:
слово | транскрипция IPA | часть речи | краткий русский перевод | уровень

Правила:
- слово в первой колонке — ровно как в списке, без изменений и без пропусков;
- часть речи по-английски: noun, verb, adjective, adverb и т.п.;
- перевод краткий: 1-3 самых употребимых значения через запятую;
- уровень — строго один из: A1, A2, B1, B2, C1, C2;
- если слово — имя собственное (имя, фамилия, название места, бренда), \
вместо разбора выдай ровно: слово | - | name | - | -

Никаких заголовков, нумерации, markdown, пояснений и пустых строк — \
только строки формата, по одной на каждое слово списка.

Список слов:
"""

PHRASES_TEMPLATE = """Прочитай текст главы книги на английском и выпиши из \
него 5-15 полезных для изучения устойчивых выражений, фразовых глаголов и \
идиом — только те, что реально встречаются в этом тексте.

Ответ — ТОЛЬКО строки такого формата, по одной на выражение, без заголовков, \
нумерации, markdown и пояснений:
выражение | транскрипция IPA | phrase | краткий русский перевод | уровень

Уровень — строго один из: A1, A2, B1, B2, C1, C2. \
Если подходящих выражений нет — верни пустой ответ.

Текст главы:
"""

# ---- локальное выделение слов из главы -------------------------------------

# Только служебные слова (артикли, предлоги, союзы, местоимения,
# вспомогательные/модальные глаголы, числительные) — значимые наречия
# вроде very/just остаются в словаре, как и раньше делала модель.
STOPWORDS = set("""
a an the
and or but nor so yet if then than because although though while whereas
unless until once since as when whenever where wherever whether either neither
both
about above across after against along amid among around at before behind
below beneath beside besides between beyond by despite down during except for
from in inside into like near of off on onto out outside over past per through
throughout till to toward towards under underneath up upon via with within
without
i you he she it we they me him her us them my your his its our their mine
yours hers ours theirs myself yourself himself herself itself ourselves
yourselves themselves this that these those who whom whose which what someone
anyone everyone none nothing something anything everything somebody anybody
everybody nobody one another each other any some all
am is are was were be been being do does did doing done have has had having
will would shall should can could may might must ought not
don't doesn't didn't can't cannot won't wouldn't couldn't shouldn't isn't
aren't wasn't weren't haven't hasn't hadn't mustn't needn't ain't let's i'm
i've i'll i'd you're you've you'll you'd he's she's it's we're we've we'll
we'd they're they've they'll they'd there's here's that's what's who's
where's how's
there here too yes no oh ah mr mrs ms dr
two three four five six seven eight nine ten eleven twelve thirteen fourteen
fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty
seventy eighty ninety hundred thousand million billion first second third
""".split())

WORD_RE = re.compile(r"[A-Za-z][A-Za-z'’\-]*")

# формы, которые simplemma не сводит к базовой (went трактует как wend);
# неоднозначные (felt/left/found…) не трогаем — модель даёт оба значения
LEMMA_FIX = {"went": "go", "gone": "go"}


def extract_words(text: str) -> list[str]:
    """Уникальные значимые леммы главы, по алфавиту.

    Имена собственные отсеиваются эвристикой: слово, которое встречается
    с большой буквы НЕ в начале предложения и ни разу — со строчной.
    """
    import simplemma  # ленивый импорт: словарь en грузится при первой главе

    tokens: list[tuple[str, bool]] = []  # (слово, стоит в начале предложения)
    seen_lower: set[str] = set()
    seen_cap_mid: set[str] = set()
    for m in WORD_RE.finditer(text):
        w = m.group(0).strip("'’-")
        if len(w) < 2:
            continue
        before = text[: m.start()].rstrip(" \t")
        at_start = not before or before[-1] in "\n.!?\"'“”‘’…"
        tokens.append((w, at_start))
        wl = w.lower()
        if w[0].islower():
            seen_lower.add(wl)
        elif not at_start:
            seen_cap_mid.add(wl)

    out: set[str] = set()
    for w, _ in tokens:
        if w.isupper() and len(w) > 1:  # акронимы (OK, TV, USA)
            continue
        wl = w.lower()
        if wl.endswith("'s") or wl.endswith("’s"):
            wl = wl[:-2]
        if "'" in wl or "’" in wl:  # прочие сокращения (o'clock и т.п.)
            continue
        if len(wl) < 2 or wl in STOPWORDS:
            continue
        if wl in seen_cap_mid and wl not in seen_lower:  # имя собственное
            continue
        lemma = LEMMA_FIX.get(wl) or simplemma.lemmatize(wl, lang="en")
        if len(lemma) < 2 or lemma in STOPWORDS:
            continue
        out.add(lemma)
    return sorted(out)


# ---- вечный пословный кэш ---------------------------------------------------

DICT: dict[str, str] = {}
if DICT_PATH.exists():
    for _line in DICT_PATH.read_text().splitlines():
        try:
            _rec = json.loads(_line)
            DICT[_rec["w"]] = _rec["line"]
        except (json.JSONDecodeError, KeyError):
            continue


def is_name(line: str) -> bool:
    """Строка-маркер имени собственного (pos == name) — в словарь не идёт."""
    parts = line.split("|")
    return len(parts) > 2 and parts[2].strip() == "name"


def dict_put(w: str, line: str):
    if w in DICT:
        return
    DICT[w] = line
    with DICT_PATH.open("a") as f:
        f.write(json.dumps({"w": w, "line": line}, ensure_ascii=False) + "\n")


def parse_dict_lines(raw: str, restrict: set[str] | None) -> dict[str, str]:
    """Строки 'слово | ipa | pos | перевод | уровень' → {слово: строка}."""
    out: dict[str, str] = {}
    for line in raw.splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 4 or not parts[0]:
            continue
        w = parts[0].strip("*#` ").lower()
        if restrict is not None and w not in restrict:
            continue
        ipa = parts[1].strip("/") if len(parts) > 1 else ""
        pos = parts[2] if len(parts) > 2 else ""
        ru = parts[3] if len(parts) > 3 else ""
        lvl = ""
        if len(parts) > 4:
            m = re.search(r"[ABC][12]", parts[4].upper())
            lvl = m.group(0) if m else ""
        out[w] = f"{w} | {ipa} | {pos} | {ru} | {lvl}"
    return out


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


# без автообновлений и телеметрии CLI стартует на пару секунд быстрее;
# thinking выключен — шаблоны механические, размышления только тянут время
CLAUDE_ENV = {**os.environ,
              "DISABLE_AUTOUPDATER": "1",
              "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
              "MAX_THINKING_TOKENS": "0"}


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
            env=CLAUDE_ENV,
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


async def run_claude_async(prompt: str, timeout_s: int = TIMEOUT_S) -> str:
    async with claude_sem:
        return await asyncio.to_thread(run_claude, prompt, timeout_s)


# ---- тёплый пул процессов claude -------------------------------------------
# Старт node на этом VPS стоит ~2-2,5с — заметная часть задержки каждого слова.
# Фронт при первом нажатии клавиши шлёт /prewarm, сервер заранее поднимает
# процессы; к моменту Enter node уже загружен, остаётся только время модели.

WARM_MAX = 2          # по числу параллельных половин разбора
WARM_TTL_S = 120      # неиспользованный тёплый процесс убиваем (память 2ГБ VPS)
_warm: list[asyncio.subprocess.Process] = []


async def _spawn_stream_proc() -> asyncio.subprocess.Process:
    proc = await asyncio.create_subprocess_exec(
        CLAUDE_BIN, "-p", "--model", MODEL, "--disallowedTools", "*",
        "--output-format", "stream-json", "--include-partial-messages",
        "--verbose",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
        env=CLAUDE_ENV,
    )
    # CLI умирает, если stdin пуст 3 секунды — кормим таймер пробелом
    # (промпт читается до EOF, лишний пробел в начале безвреден)
    proc.stdin.write(b" ")
    await proc.stdin.drain()
    return proc


async def _reap_warm(proc: asyncio.subprocess.Process):
    await asyncio.sleep(WARM_TTL_S)
    if proc in _warm:
        _warm.remove(proc)
        if proc.returncode is None:
            proc.kill()
        await proc.wait()


async def prewarm_pool():
    while len(_warm) < WARM_MAX:
        proc = await _spawn_stream_proc()
        _warm.append(proc)
        asyncio.get_running_loop().create_task(_reap_warm(proc))


def _take_warm() -> asyncio.subprocess.Process | None:
    while _warm:
        proc = _warm.pop(0)
        if proc.returncode is None:
            return proc
    return None


async def stream_claude(prompt: str, timeout_s: int = TIMEOUT_S):
    """Генератор ('delta', кусок) … ('result', полный текст) из stream-json."""
    proc = _take_warm()
    if proc is None:
        proc = await _spawn_stream_proc()
    try:
        try:
            proc.stdin.write(prompt.encode())
            await proc.stdin.drain()
            proc.stdin.close()
        except (BrokenPipeError, ConnectionResetError):
            # тёплый процесс успел умереть — один холодный ретрай
            proc.kill()
            await proc.wait()
            proc = await _spawn_stream_proc()
            proc.stdin.write(prompt.encode())
            await proc.stdin.drain()
            proc.stdin.close()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        while True:
            left = deadline - loop.time()
            if left <= 0:
                raise HTTPException(504, "LLM timeout")
            line = await asyncio.wait_for(proc.stdout.readline(), timeout=left)
            if not line:
                break
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("type") == "stream_event":
                ev = obj.get("event") or {}
                if ev.get("type") == "content_block_delta":
                    d = ev.get("delta") or {}
                    if d.get("type") == "text_delta" and d.get("text"):
                        yield "delta", d["text"]
            elif obj.get("type") == "result":
                res = obj.get("result")
                if isinstance(res, str) and res.strip():
                    yield "result", res.strip()
        await proc.wait()
    finally:
        if proc.returncode is None:
            proc.kill()


async def collect_stream(prompt: str, timeout_s: int = TIMEOUT_S) -> str:
    """Полный текст ответа claude (через тёплый пул, под семафором)."""
    async with claude_sem:
        full, result = [], None
        async for kind, chunk in stream_claude(prompt, timeout_s):
            if kind == "delta":
                full.append(chunk)
            else:
                result = chunk
        return (result or "".join(full)).strip()


class WordReq(BaseModel):
    text: str


@app.get("/health")
async def health():
    return {"ok": True, "model": MODEL, "dict_words": len(DICT)}


@app.post("/prewarm")
async def prewarm(x_auth_token: str | None = Header(default=None)):
    """Фронт дёргает при первом нажатии клавиши — греем процессы заранее."""
    check_token(x_auth_token)
    await prewarm_pool()
    return {"warm": len(_warm)}


@app.get("/prewarm")
async def prewarm_get(token: str | None = None):
    """GET-вариант без preflight — см. word2_get."""
    check_token(token)
    await prewarm_pool()
    return {"warm": len(_warm)}


@app.get("/health/offline")
async def health_offline():
    return await asyncio.to_thread(offline_dict.stats)


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
    # ждём своей очереди вместо мгновенного 429 (фронт и так крутит спиннер)
    global _word_waiters
    if _word_waiters >= WORD_QUEUE_MAX:
        raise HTTPException(429, "busy, retry in a few seconds")
    _word_waiters += 1
    try:
        try:
            await asyncio.wait_for(word_lock.acquire(), timeout=120)
        except asyncio.TimeoutError:
            raise HTTPException(429, "busy, retry in a few seconds")
        try:
            main, extra = await asyncio.gather(
                collect_stream(TEMPLATE_MAIN + text),
                collect_stream(TEMPLATE_EXTRA + text))
        finally:
            word_lock.release()
    finally:
        _word_waiters -= 1
    if not main:
        raise HTTPException(502, "empty LLM response")
    answer = main + ("\n\n" + extra if extra else "")
    if extra:  # без хвоста не кэшируем — пусть следующий запрос доберёт всё
        p.write_text(answer)
    return {"text": answer, "cached": False}


def _word2_response(raw_text: str, request: Request):
    """Стриминг разбора слова NDJSON: delta… → done (или error)."""
    text = " ".join(raw_text.split()).strip()
    if not text:
        raise HTTPException(400, "empty request")
    if len(text) > 80:
        raise HTTPException(400, "too long — one word or phrase, please")

    def j(**obj):
        return json.dumps(obj, ensure_ascii=False) + "\n"

    key = text.lower()
    p = cache_path(key)
    if p.exists():
        return StreamingResponse(
            iter([j(type="delta", text=p.read_text()) + j(type="done")]),
            media_type="application/x-ndjson")

    ip = request.client.host if request.client else "?"
    check_rate(ip)
    if _word_waiters >= WORD_QUEUE_MAX:
        raise HTTPException(429, "busy, retry in a few seconds")

    async def gen():
        global _word_waiters
        _word_waiters += 1
        try:
            try:
                await asyncio.wait_for(word_lock.acquire(), timeout=120)
            except asyncio.TimeoutError:
                yield j(type="error", detail="сервер занят — попробуй ещё раз через минуту")
                return
            try:
                # хвост (семейство/исследование) генерится параллельно
                # с основной карточкой — на втором слоте семафора
                extra_task = asyncio.create_task(
                    collect_stream(TEMPLATE_EXTRA + text))
                async with claude_sem:
                    full, result = [], None
                    async for kind, chunk in stream_claude(TEMPLATE_MAIN + text):
                        if kind == "delta":
                            full.append(chunk)
                            yield j(type="delta", text=chunk)
                        else:
                            result = chunk
                main = (result or "".join(full)).strip()
                if not main:
                    extra_task.cancel()
                    yield j(type="error", detail="пустой ответ — попробуй ещё раз")
                    return
                try:
                    extra = await extra_task
                except Exception:
                    extra = ""
                if extra:
                    yield j(type="delta", text="\n\n" + extra)
                    # без хвоста не кэшируем — следующий запрос доберёт всё
                    p.write_text(main + "\n\n" + extra)
                yield j(type="done")
            finally:
                word_lock.release()
        except HTTPException as e:
            yield j(type="error", detail=str(e.detail))
        except Exception as e:
            yield j(type="error", detail=f"внутренняя ошибка: {e}")
        finally:
            _word_waiters -= 1

    return StreamingResponse(gen(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-store"})


@app.post("/word2")
async def word2(req: WordReq, request: Request,
                x_auth_token: str | None = Header(default=None)):
    check_token(x_auth_token)
    return _word2_response(req.text, request)


@app.get("/word2")
async def word2_get(request: Request, text: str = "", token: str | None = None):
    """GET-вариант: без CORS-preflight и без тела запроса.

    Мобильные сети/DPI иногда съедают тело POST (заголовки доходят, JSON нет) —
    GET умещается в один маленький пакет и проходит там, где POST виснет.
    Токен и так лежит открыто в JS сайта, в query он ничем не хуже заголовка.
    """
    check_token(token)
    return _word2_response(text, request)


# ---- глава ------------------------------------------------------------------

async def chapter_job(text: str, cache_file: Path,
                      q: asyncio.Queue | None) -> str:
    """Собирает словарь главы; события прогресса кладёт в q (если дана).

    Живёт как самостоятельная задача: если клиент отвалился, всё равно
    досчитывает и пишет кэши — повторный запрос соберётся мгновенно.
    """

    async def emit(**obj):
        if q is not None:
            await q.put(obj)

    try:
        async with chapter_lock:
            words = await asyncio.to_thread(extract_words, text)
            if not words:
                raise HTTPException(400, "не нашла английских слов в тексте")
            known = [w for w in words if w in DICT and not is_name(DICT[w])]
            cached_lines = [DICT[w] for w in known]
            unknown = [w for w in words if w not in DICT]

            # офлайн-резолв (MUSE+IPA+CEFR): большинство новых слов — без LLM
            def offline_pass():
                inst, rest = [], []
                for w in unknown:
                    line = offline_dict.resolve(w)
                    if line:
                        inst.append(line)
                    else:
                        rest.append(w)
                return inst, rest

            offline_lines, llm_words = await asyncio.to_thread(offline_pass)
            for line in offline_lines:
                dict_put(line.split("|")[0].strip(), line)

            instant = cached_lines + offline_lines
            await emit(type="meta", total=len(known) + len(unknown),
                       cached=len(instant))
            lines = list(instant)
            if instant:
                await emit(type="lines", text="\n".join(instant))

            async def word_batch(batch: list[str]):
                raw = await run_claude_async(
                    BATCH_TEMPLATE + "\n".join(batch), BATCH_TIMEOUT_S)
                return "word", parse_dict_lines(raw, restrict=set(batch))

            async def phrases():
                raw = await run_claude_async(
                    PHRASES_TEMPLATE + text, BATCH_TIMEOUT_S)
                res = parse_dict_lines(raw, restrict=None)
                # модель путает колонку pos — принудительно ставим phrase
                for w, line in res.items():
                    parts = [p.strip() for p in line.split("|")]
                    res[w] = f"{parts[0]} | {parts[1]} | phrase | {parts[3]} | {parts[4]}"
                return "phrase", res

            batches = [llm_words[i:i + BATCH_SIZE]
                       for i in range(0, len(llm_words), BATCH_SIZE)]
            tasks = [asyncio.create_task(word_batch(b)) for b in batches]
            # выражения — самый долгий вызов; стартует следом за батчами,
            # на паре предложений искать нечего — не жжём квоту
            ph_task = asyncio.create_task(phrases()) if len(text) >= 600 else None

            got: dict[str, str] = {}

            async def collect(pending):
                for fut in asyncio.as_completed(pending):
                    try:
                        kind, res = await fut
                    except Exception:
                        continue  # упавший батч добьём ретраем
                    new = []
                    for w, line in res.items():
                        if kind == "word":
                            if w in got:
                                continue
                            got[w] = line
                            dict_put(w, line)
                        if not is_name(line):
                            new.append(line)
                    if new:
                        lines.extend(new)
                        await emit(type="lines", text="\n".join(new))

            await collect(tasks)
            missing = [w for w in llm_words if w not in got]
            if missing:  # модель пропустила / батч упал — один ретрай
                retry = [asyncio.create_task(word_batch(missing[i:i + BATCH_SIZE]))
                         for i in range(0, len(missing), BATCH_SIZE)]
                await collect(retry)

            if ph_task is not None:
                await emit(type="status", text="слова готовы — ищу устойчивые выражения…")
                await collect([ph_task])

            final = "\n".join(sorted(set(lines), key=str.lower))
            cache_file.write_text(final)
            await emit(type="done", words=len(set(lines)))
            return final
    except HTTPException as e:
        await emit(type="error", detail=str(e.detail))
        raise
    except Exception as e:
        await emit(type="error", detail=f"внутренняя ошибка: {e}")
        raise


def _chapter_validate(req: WordReq, request: Request,
                      x_auth_token: str | None) -> tuple[str, Path]:
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
    return text, cache_path("chapter:" + text.lower())


@app.post("/chapter")
async def chapter(req: WordReq, request: Request,
                  x_auth_token: str | None = Header(default=None)):
    """Старый интерфейс: один JSON-ответ целиком (для закэшированных вкладок)."""
    text, p = _chapter_validate(req, request, x_auth_token)
    if p.exists():
        return {"text": p.read_text(), "cached": True}
    ip = request.client.host if request.client else "?"
    check_rate(ip)
    if chapter_lock.locked():
        raise HTTPException(429, "busy, retry in a few seconds")
    answer = await chapter_job(text, p, None)
    return {"text": answer, "cached": False}


@app.post("/chapter2")
async def chapter2(req: WordReq, request: Request,
                   x_auth_token: str | None = Header(default=None)):
    """Стриминг NDJSON: meta → lines… → done (или error)."""
    text, p = _chapter_validate(req, request, x_auth_token)

    def ndjson(*objs):
        return "".join(json.dumps(o, ensure_ascii=False) + "\n" for o in objs)

    if p.exists():
        cached = p.read_text()
        n = len(cached.splitlines())
        return StreamingResponse(
            iter([ndjson({"type": "meta", "total": n, "cached": n},
                         {"type": "lines", "text": cached},
                         {"type": "done", "words": n})]),
            media_type="application/x-ndjson")

    ip = request.client.host if request.client else "?"
    check_rate(ip)
    if chapter_lock.locked():
        raise HTTPException(429, "busy, retry in a few seconds")

    q: asyncio.Queue = asyncio.Queue()
    task = asyncio.create_task(chapter_job(text, p, q))
    task.add_done_callback(lambda t: t.exception())  # гасим "never retrieved"

    async def gen():
        while True:
            item = await q.get()
            yield json.dumps(item, ensure_ascii=False) + "\n"
            if item["type"] in ("done", "error"):
                return

    return StreamingResponse(gen(), media_type="application/x-ndjson")

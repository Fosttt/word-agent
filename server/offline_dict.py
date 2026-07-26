"""Офлайн-резолвер словарных строк — слово в строку словаря БЕЗ LLM.

Источники (server/data/, скачаны 26.07.2026):
- muse_en_ru.txt — переводы EN→RU (facebook MUSE bilingual dictionaries);
- en_US_ipa.txt — транскрипции IPA (open-dict-data/ipa-dict, en_US);
- cefrj.csv + cefr_c1c2.csv — уровни CEFR и части речи
  (openlanguageprofiles/olp-en-cefrj: CEFR-J A1-B2 + Octanove C1-C2);
- фолбэк уровня — частота wordfreq (zipf → CEFR-подобная шкала).

Слово без перевода в MUSE резолвится в None и уходит в LLM-батч.
"""

import csv
from pathlib import Path

DATA = Path(__file__).parent / "data"

_ipa: dict[str, str] = {}
_ru: dict[str, list[str]] = {}
_cefr: dict[str, tuple[str, str]] = {}  # слово -> (pos, уровень)
_loaded = False

LEVELS = {"A1", "A2", "B1", "B2", "C1", "C2"}


def _has_cyr(s: str) -> bool:
    return any("а" <= c <= "я" or c == "ё" for c in s.lower())


def _load():
    global _loaded
    if _loaded:
        return
    for line in (DATA / "en_US_ipa.txt").read_text().splitlines():
        if "\t" in line:
            w, ipa = line.split("\t", 1)
            _ipa.setdefault(w.lower(), ipa.split(",")[0].strip().strip("/"))
    for line in (DATA / "muse_en_ru.txt").read_text().splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        en, ru = parts
        if not _has_cyr(ru):
            continue
        lst = _ru.setdefault(en.lower(), [])
        if ru not in lst and len(lst) < 3:
            lst.append(ru)
    for fn in ("cefrj.csv", "cefr_c1c2.csv"):
        with (DATA / fn).open() as f:
            for row in csv.DictReader(f):
                hw = (row.get("headword") or "").strip().lower()
                lvl = (row.get("CEFR") or "").strip().upper()
                pos = (row.get("pos") or "").strip().lower()
                if not hw or lvl not in LEVELS:
                    continue
                # заголовки вида "a.m./A.M./am" — раскрываем варианты
                for w in hw.replace("/", " ").split():
                    _cefr.setdefault(w.lower(), (pos, lvl))
    _loaded = True


def _freq_level(w: str) -> str:
    from wordfreq import zipf_frequency
    z = zipf_frequency(w, "en")
    for thr, lvl in ((5.3, "A1"), (4.9, "A2"), (4.3, "B1"),
                     (3.7, "B2"), (3.1, "C1")):
        if z >= thr:
            return lvl
    return "C2"


def resolve(w: str) -> str | None:
    """'слово | ipa | pos | перевод | уровень' или None, если перевода нет."""
    _load()
    rus = _ru.get(w)
    if not rus:
        return None
    pos, lvl = _cefr.get(w, ("", ""))
    if not lvl:
        lvl = _freq_level(w)
    return f"{w} | {_ipa.get(w, '')} | {pos} | {', '.join(rus)} | {lvl}"


def stats() -> dict:
    _load()
    return {"ru": len(_ru), "ipa": len(_ipa), "cefr": len(_cefr)}

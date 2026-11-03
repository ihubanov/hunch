"""Typed values from extracted text: code, not the model, turns "$450 a month" into {amount: 450, currency: USD}.

The model only copies a span out of the source. These parsers read the span; when one can't, the
field says so (status "unparsed") and keeps the text, rather than guess.

  number    "1,250.5", "12k", "3 million", "seven"       -> 1250.5, 12000, 3000000, 7
  money     "$450", "EUR 1.2m", "450 dollars"            -> {"amount": 450, "currency": "USD"}
  date      "2026-03-03", "March 3, 2026", "tomorrow"    -> "2026-03-03" (relative dates need `now`)
  time      "5pm", "5:30 p.m.", "17:30", "noon"          -> "17:00", "17:30", "12:00"
  duration  "2 hours", "1h30m", "half an hour", "90 min" -> seconds
  text      anything                                      -> the span itself
"""
from __future__ import annotations

import datetime as dt
import re
from typing import Any

TYPES = ("text", "number", "money", "date", "time", "duration")

_WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
          "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30,
          "forty": 40, "fifty": 50, "sixty": 60, "ninety": 90, "hundred": 100, "a": 1, "an": 1}
_SCALE = {"k": 1e3, "thousand": 1e3, "m": 1e6, "mm": 1e6, "mn": 1e6, "million": 1e6,
          "b": 1e9, "bn": 1e9, "billion": 1e9}
_NUM = re.compile(r"(?<![\w.])([-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|[-+]?\.\d+)\s*"
                  r"(k|thousand|mm|mn|m|million|bn|b|billion)?(?![a-z])", re.I)


def _tidy(x: float) -> int | float:
    return int(x) if float(x).is_integer() else round(x, 10)


def parse_number(s: str) -> int | float | None:
    m = _NUM.search(s)
    if m:
        x = float(m.group(1).replace(",", ""))
        if m.group(2):
            x *= _SCALE[m.group(2).lower()]
        return _tidy(x)
    words = re.findall(r"[a-z]+", s.lower())
    for i, w in enumerate(words):
        if w in _WORDS and w not in ("a", "an"):
            x = float(_WORDS[w])
            if i + 1 < len(words) and words[i + 1] in _SCALE:
                x *= _SCALE[words[i + 1]]
            return _tidy(x)
    return None


_SYMBOLS = [("US$", "USD"), ("C$", "CAD"), ("A$", "AUD"), ("NZ$", "NZD"), ("HK$", "HKD"), ("R$", "BRL"),
            ("$", "USD"), ("€", "EUR"), ("£", "GBP"), ("¥", "JPY"), ("₹", "INR"), ("₩", "KRW"), ("₽", "RUB"),
            ("₺", "TRY"), ("₪", "ILS"), ("₴", "UAH"), ("zł", "PLN"), ("лв", "BGN"), ("kr", "SEK")]
_CODES = {"USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD", "CNY", "HKD", "SGD", "INR", "KRW", "BRL",
          "MXN", "ZAR", "SEK", "NOK", "DKK", "PLN", "CZK", "HUF", "RON", "BGN", "TRY", "ILS", "AED", "SAR",
          "RUB", "UAH", "THB", "IDR", "MYR", "PHP", "VND", "NGN", "EGP", "KES"}
_CURRENCY_WORDS = {"dollar": "USD", "dollars": "USD", "bucks": "USD", "euro": "EUR", "euros": "EUR",
                   "pound": "GBP", "pounds": "GBP", "sterling": "GBP", "yen": "JPY", "rupee": "INR",
                   "rupees": "INR", "franc": "CHF", "francs": "CHF", "lev": "BGN", "leva": "BGN"}


def parse_money(s: str) -> dict | None:
    amount = parse_number(s)
    if amount is None:
        return None
    currency = None
    for code in re.findall(r"\b[A-Z]{3}\b", s):
        if code in _CODES:
            currency = code
            break
    if currency is None:
        for sym, code in _SYMBOLS:
            if sym in s:
                currency = code
                break
    if currency is None:
        for w in re.findall(r"[a-z]+", s.lower()):
            if w in _CURRENCY_WORDS:
                currency = _CURRENCY_WORDS[w]
                break
    return {"amount": amount, "currency": currency}


_MONTHS = {m: i for i, ms in enumerate(
    [("jan", "january"), ("feb", "february"), ("mar", "march"), ("apr", "april"), ("may",), ("jun", "june"),
     ("jul", "july"), ("aug", "august"), ("sep", "sept", "september"), ("oct", "october"), ("nov", "november"),
     ("dec", "december")], 1) for m in ms}
_WEEKDAYS = {d: i for i, ds in enumerate(
    [("mon", "monday"), ("tue", "tues", "tuesday"), ("wed", "wednesday"), ("thu", "thur", "thurs", "thursday"),
     ("fri", "friday"), ("sat", "saturday"), ("sun", "sunday")]) for d in ds}
_MONTH_RE = "|".join(sorted(_MONTHS, key=len, reverse=True))


def _safe_date(y: int, m: int, d: int) -> str | None:
    try:
        return dt.date(y, m, d).isoformat()
    except ValueError:
        return None


def parse_date(s: str, now: dt.date | None = None) -> str | None:
    t = s.lower()
    m = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", t)
    if m:
        return _safe_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = re.search(r"\b(\d{1,2})[/.](\d{1,2})[/.](\d{2,4})\b", t)
    if m:
        a, b, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        y += 2000 if y < 100 else 0
        if a > 12 >= b:
            return _safe_date(y, b, a)          # day first
        if b > 12 >= a:
            return _safe_date(y, a, b)          # month first
        return None                             # 03/04/2026: ambiguous, so don't guess
    m = re.search(rf"\b({_MONTH_RE})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s+(\d{{4}}))?\b", t)
    if m:
        y = int(m.group(3)) if m.group(3) else (now.year if now else None)
        return _safe_date(y, _MONTHS[m.group(1)], int(m.group(2))) if y else None
    m = re.search(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({_MONTH_RE})\.?(?:,?\s+(\d{{4}}))?\b", t)
    if m:
        y = int(m.group(3)) if m.group(3) else (now.year if now else None)
        return _safe_date(y, _MONTHS[m.group(2)], int(m.group(1))) if y else None
    if now is None:
        return None
    if re.search(r"\bday after tomorrow\b", t):
        return (now + dt.timedelta(days=2)).isoformat()
    for word, delta in (("today", 0), ("tonight", 0), ("tomorrow", 1), ("yesterday", -1)):
        if re.search(rf"\b{word}\b", t):
            return (now + dt.timedelta(days=delta)).isoformat()
    m = re.search(r"\bin\s+(\d+|a|an|one|two|three|four|five|six|seven|ten)\s+(day|week)s?\b", t)
    if m:
        n = int(m.group(1)) if m.group(1).isdigit() else _WORDS[m.group(1)]
        return (now + dt.timedelta(days=n * (7 if m.group(2) == "week" else 1))).isoformat()
    m = re.search(r"\b(next|this|on|coming)?\s*(" + "|".join(sorted(_WEEKDAYS, key=len, reverse=True)) + r")\b", t)
    if m:
        ahead = (_WEEKDAYS[m.group(2)] - now.weekday()) % 7
        if ahead == 0 and m.group(1) == "next":
            ahead = 7
        return (now + dt.timedelta(days=ahead)).isoformat()
    return None


def parse_time(s: str) -> str | None:
    t = s.lower()
    if re.search(r"\b(noon|midday)\b", t):
        return "12:00"
    if re.search(r"\bmidnight\b", t):
        return "00:00"
    m = re.search(r"\b(\d{1,2})(?:[:.](\d{2}))?\s*([ap])\.?\s*m\b\.?", t)
    if m:
        h, mi = int(m.group(1)), int(m.group(2) or 0)
        if not 1 <= h <= 12 or mi > 59:
            return None
        h = h % 12 + (12 if m.group(3) == "p" else 0)
        return f"{h:02d}:{mi:02d}"
    m = re.search(r"\b([01]?\d|2[0-3])[:h](\d{2})\b", t)
    if m:
        return f"{int(m.group(1)):02d}:{m.group(2)}"
    return None


_UNITS = {"s": 1, "sec": 1, "secs": 1, "second": 1, "seconds": 1,
          "m": 60, "min": 60, "mins": 60, "minute": 60, "minutes": 60,
          "h": 3600, "hr": 3600, "hrs": 3600, "hour": 3600, "hours": 3600,
          "d": 86400, "day": 86400, "days": 86400, "w": 604800, "wk": 604800, "wks": 604800,
          "week": 604800, "weeks": 604800}
_UNIT_RE = "|".join(sorted(_UNITS, key=len, reverse=True))
# a digit may touch its unit ("1h30m"); a word needs a space ("an hour"), so "and" is never "an" + "d"
_DUR = re.compile(r"(?:(?<![\d.])(\d+(?:\.\d+)?)\s*-?\s*|(?<![a-z])(half an?|an?|one|two|three|four|five|six|seven|eight|nine|"
                  r"ten|eleven|twelve|fifteen|twenty|thirty|forty|fifty|sixty|ninety)\s+)(" + _UNIT_RE + r")(?![a-z])", re.I)


def parse_duration(s: str) -> int | float | None:
    t = s.lower()
    total, found, last_unit = 0.0, False, None
    for m in _DUR.finditer(t):
        q = m.group(1) or m.group(2)
        n = 0.5 if q.startswith("half") else float(q) if q[0].isdigit() else float(_WORDS[q])
        last_unit = _UNITS[m.group(3)]
        total += n * last_unit
        found = True
    if found and last_unit and re.search(r"\band a half\b", t):
        total += 0.5 * last_unit
    return _tidy(total) if found else None


def parse(kind: str, text: str, now: dt.date | None = None) -> Any:
    """The typed value of `text`, or None when the parser can't read one."""
    if kind == "text":
        return text
    if kind == "number":
        return parse_number(text)
    if kind == "money":
        return parse_money(text)
    if kind == "date":
        return parse_date(text, now)
    if kind == "time":
        return parse_time(text)
    if kind == "duration":
        return parse_duration(text)
    raise ValueError(f"unknown type {kind!r}")

from __future__ import annotations

import json
import re


def char_count(text: str) -> int:
    return len(re.sub(r"\s+", "", text or ""))


def clip(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n【以上内容已截断】"


def extract_json(text: str) -> dict:
    raw = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", raw, re.S | re.I)
    if fence:
        raw = fence.group(1).strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("模型没有返回 JSON 大纲")
    blob = raw[start : end + 1]
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        blob = re.sub(r",\s*([}\]])", r"\1", blob)
        data = json.loads(blob)
    if not isinstance(data, dict):
        raise ValueError("大纲 JSON 必须是对象")
    return data


def strip_leading_heading(content: str, title: str) -> str:
    lines = (content or "").strip().splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    if not lines:
        return ""
    first = lines[0].strip()
    if first.startswith("#"):
        heading = re.sub(r"^#+\s*", "", first).strip()
        heading = re.sub(r"^[\d.]+\s*", "", heading).strip()
        plain_title = re.sub(r"^[\d.]+\s*", "", title).strip()
        if plain_title and (plain_title in heading or heading in plain_title):
            lines = lines[1:]
            while lines and not lines[0].strip():
                lines.pop(0)
    return "\n".join(lines).strip()


def safe_filename(name: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*]', "_", (name or "").strip())
    cleaned = cleaned.rstrip(". ")
    return cleaned or "标书"


def infer_level(number: str, fallback: int = 1) -> int:
    if re.fullmatch(r"\d+(\.\d+)*", number or ""):
        return min(3, number.count(".") + 1)
    try:
        level = int(fallback)
    except (TypeError, ValueError):
        level = 1
    return min(3, max(1, level))


def number_sort_key(number: str) -> tuple:
    if re.fullmatch(r"\d+(\.\d+)*", number or ""):
        return (0, tuple(int(part) for part in number.split(".")))
    return (1, (number or "",))

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from pathlib import Path

import jieba
import joblib

from tender_agent.parsers import read_document

INDEX_NAME = "index.joblib"
SUPPORTED_SUFFIXES = {".txt", ".md", ".markdown", ".docx", ".pdf"}
_JIEBA_LOCK = threading.Lock()
_JIEBA_READY = False

STOPWORDS = frozenset(
    "的 了 和 与 或 及 在 是 为 对 将 并 等 中 上 下 也 就 都 而 被 把 从 以 到 由 及其 以及 一个 我们 可以 进行 通过 相关 包括 具有 能够 需要 如果 因此 同时 以及 不是 没有 这个 那个 以及 其中 对于 关于 以及 自己 什么 怎么 因为 所以".split()
)


@dataclass
class ChunkHit:
    source: str
    text: str
    score: float


@dataclass
class BuildReport:
    files: int
    chunks: int
    warnings: list[str]


def tokenize(text: str) -> list[str]:
    global _JIEBA_READY
    if not _JIEBA_READY:
        with _JIEBA_LOCK:
            if not _JIEBA_READY:
                jieba.initialize()
                _JIEBA_READY = True
    tokens: list[str] = []
    for token in jieba.lcut(text or ""):
        word = token.strip().lower()
        if len(word) < 2 or word in STOPWORDS:
            continue
        if re.fullmatch(r"[\W_]+", word):
            continue
        tokens.append(word)
    return tokens


def chunk_text(text: str, size: int = 700, overlap: int = 120) -> list[str]:
    normalized = re.sub(r"\r\n?", "\n", text or "")
    normalized = re.sub(r"[ \t]+\n", "\n", normalized)
    paragraphs = [part.strip() for part in re.split(r"\n+", normalized) if part.strip()]
    chunks: list[str] = []
    buffer = ""

    def push_long(paragraph: str) -> None:
        step = max(size - overlap, 200)
        for start in range(0, len(paragraph), step):
            piece = paragraph[start : start + size].strip()
            if len(piece) >= 80:
                chunks.append(piece)

    for paragraph in paragraphs:
        if len(paragraph) > size * 2:
            if buffer.strip():
                chunks.append(buffer.strip())
                buffer = ""
            push_long(paragraph)
            continue
        candidate = f"{buffer}\n{paragraph}".strip() if buffer else paragraph
        if buffer and len(candidate) > size:
            chunks.append(buffer.strip())
            tail = buffer.strip()[-overlap:] if overlap else ""
            buffer = f"{tail}\n{paragraph}".strip()
        else:
            buffer = candidate
    if len(buffer.strip()) >= 40:
        chunks.append(buffer.strip())
    return chunks


def format_hits(hits: list[ChunkHit], limit: int = 3000) -> str:
    parts: list[str] = []
    used = 0
    for hit in hits:
        block = f"【来源：{hit.source}】\n{hit.text.strip()}\n"
        if used + len(block) > limit:
            remain = limit - used
            if remain > 120:
                parts.append(block[:remain])
            break
        parts.append(block)
        used += len(block)
    return "\n".join(parts).strip()


class KnowledgeBase:
    def __init__(self, root: Path):
        self.root = root
        self.raw_dir = root / "raw"
        self.index_dir = root / "index"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._cache = None
        self._cache_mtime: float | None = None

    @property
    def index_path(self) -> Path:
        return self.index_dir / INDEX_NAME

    def list_sources(self) -> list[Path]:
        files = [
            path
            for path in self.raw_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
        ]
        return sorted(files, key=lambda item: str(item).lower())

    def build(self) -> BuildReport:
        from sklearn.feature_extraction.text import TfidfVectorizer

        warnings: list[str] = []
        chunks: list[dict] = []
        files = 0
        for path in self.list_sources():
            files += 1
            try:
                text = read_document(path)
            except Exception as exc:
                warnings.append(f"{path.name} 读取失败：{exc}")
                continue
            if len(re.sub(r"\s+", "", text)) < 40:
                warnings.append(f"{path.name} 几乎没有可提取的文字，已跳过。扫描版 PDF 需要先做文字识别。")
                continue
            relative = path.relative_to(self.raw_dir).as_posix()
            for index, piece in enumerate(chunk_text(text), start=1):
                chunks.append({"source": relative, "text": piece, "index": index})
        if not chunks:
            if self.index_path.exists():
                self.index_path.unlink()
            self._cache = None
            self._cache_mtime = None
            warnings.append("没有可用文本，索引已清空。")
            return BuildReport(files=files, chunks=0, warnings=warnings)

        vectorizer = TfidfVectorizer(
            analyzer=tokenize,
            lowercase=False,
            min_df=1,
            max_df=1.0,
        )
        matrix = vectorizer.fit_transform([item["text"] for item in chunks])
        payload = {"version": 1, "vectorizer": vectorizer, "matrix": matrix, "chunks": chunks}
        self.index_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(payload, self.index_path)
        self._cache = payload
        self._cache_mtime = self.index_path.stat().st_mtime
        return BuildReport(files=files, chunks=len(chunks), warnings=warnings)

    def ready(self) -> bool:
        return self.index_path.exists()

    def _load(self) -> dict | None:
        if not self.index_path.exists():
            self._cache = None
            self._cache_mtime = None
            return None
        mtime = self.index_path.stat().st_mtime
        if self._cache is None or self._cache_mtime != mtime:
            self._cache = joblib.load(self.index_path)
            self._cache_mtime = mtime
        return self._cache

    def search(self, query: str, k: int = 4) -> list[ChunkHit]:
        from sklearn.metrics.pairwise import cosine_similarity

        data = self._load()
        if not data or not (query or "").strip():
            return []
        vector = data["vectorizer"].transform([query])
        scores = cosine_similarity(vector, data["matrix"]).ravel()
        order = scores.argsort()[::-1]
        hits: list[ChunkHit] = []
        for position in order[: max(k, 1)]:
            score = float(scores[position])
            if score <= 0:
                continue
            chunk = data["chunks"][int(position)]
            hits.append(ChunkHit(source=chunk["source"], text=chunk["text"], score=score))
        return hits

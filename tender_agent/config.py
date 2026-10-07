from __future__ import annotations

import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"
MODELS_PATH = CONFIG_DIR / "models.yaml"
SECRETS_PATH = CONFIG_DIR / "secrets.yaml"
KNOWLEDGE_DIR = ROOT / "knowledge"
OUTPUT_DIR = ROOT / "output"

load_dotenv(ROOT / ".env")

PROVIDERS = {
    "deepseek": "DeepSeek",
    "openai_compat": "OpenAI 兼容接口",
    "cursor": "Cursor",
}

DEFAULT_PROFILES = [
    {
        "id": "deepseek-chat",
        "label": "DeepSeek Chat（正文）",
        "provider": "deepseek",
        "model": "deepseek-chat",
        "base_url": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY",
        "temperature": 0.5,
        "max_tokens": 8192,
    },
    {
        "id": "deepseek-reasoner",
        "label": "DeepSeek Reasoner（大纲）",
        "provider": "deepseek",
        "model": "deepseek-reasoner",
        "base_url": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY",
        "temperature": 0.0,
        "max_tokens": 8192,
    },
    {
        "id": "cursor-composer",
        "label": "Cursor Composer",
        "provider": "cursor",
        "model": "composer-2.5",
        "base_url": "",
        "api_key_env": "CURSOR_API_KEY",
        "temperature": 0.3,
        "max_tokens": 4096,
    },
]


@dataclass
class ModelProfile:
    id: str
    label: str
    provider: str
    model: str
    base_url: str = ""
    api_key_env: str = ""
    temperature: float = 0.5
    max_tokens: int = 4096

    def resolved_base_url(self) -> str:
        url = (self.base_url or "").strip().rstrip("/")
        if self.provider == "deepseek":
            return url or "https://api.deepseek.com"
        return url

    def to_dict(self) -> dict:
        data = asdict(self)
        data["temperature"] = float(self.temperature)
        data["max_tokens"] = int(self.max_tokens)
        return data


def ensure_dirs() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    (KNOWLEDGE_DIR / "raw").mkdir(parents=True, exist_ok=True)
    (KNOWLEDGE_DIR / "index").mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not MODELS_PATH.exists():
        _dump_yaml(
            MODELS_PATH,
            {"default_profile": DEFAULT_PROFILES[0]["id"], "profiles": DEFAULT_PROFILES},
        )


def _dump_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path.name} 格式不正确")
    return data


def _parse_profile(raw: dict) -> ModelProfile:
    provider = str(raw.get("provider") or "deepseek").strip()
    if provider not in PROVIDERS:
        raise ValueError(f"未知模型提供商: {provider}")
    model = str(raw.get("model") or "").strip()
    if not model:
        raise ValueError("模型名称不能为空")
    profile_id = str(raw.get("id") or "").strip() or slugify(str(raw.get("label") or model))
    return ModelProfile(
        id=profile_id,
        label=str(raw.get("label") or model).strip(),
        provider=provider,
        model=model,
        base_url=str(raw.get("base_url") or "").strip(),
        api_key_env=str(raw.get("api_key_env") or "").strip(),
        temperature=float(raw.get("temperature", 0.5)),
        max_tokens=int(raw.get("max_tokens", 4096)),
    )


def load_settings() -> tuple[list[ModelProfile], str]:
    ensure_dirs()
    data = _load_yaml(MODELS_PATH)
    raw_profiles = data.get("profiles") or []
    profiles = [_parse_profile(item) for item in raw_profiles if isinstance(item, dict)]
    if not profiles:
        profiles = [_parse_profile(item) for item in DEFAULT_PROFILES]
    default_id = str(data.get("default_profile") or profiles[0].id)
    if default_id not in {item.id for item in profiles}:
        default_id = profiles[0].id
    return profiles, default_id


def load_secrets() -> dict[str, str]:
    data = _load_yaml(SECRETS_PATH)
    return {str(key): str(value) for key, value in data.items() if value}


def save_secrets(secrets: dict[str, str]) -> None:
    _dump_yaml(SECRETS_PATH, secrets)


def save_settings(profiles: list[ModelProfile], default_id: str) -> None:
    ensure_dirs()
    if default_id not in {item.id for item in profiles} and profiles:
        default_id = profiles[0].id
    _dump_yaml(
        MODELS_PATH,
        {
            "default_profile": default_id,
            "profiles": [item.to_dict() for item in profiles],
        },
    )


def resolve_api_key(profile: ModelProfile) -> str:
    secrets = load_secrets()
    saved = (secrets.get(profile.id) or "").strip()
    if saved:
        return saved
    if profile.api_key_env:
        env_value = (os.environ.get(profile.api_key_env) or "").strip()
        if env_value:
            return env_value
    if profile.provider in {"deepseek", "openai_compat"}:
        return (os.environ.get("DEEPSEEK_API_KEY") or "").strip()
    if profile.provider == "cursor":
        return (os.environ.get("CURSOR_API_KEY") or "").strip()
    return ""


def has_api_key(profile: ModelProfile) -> bool:
    return bool(resolve_api_key(profile))


def slugify(label: str) -> str:
    text = re.sub(r"\s+", "-", (label or "").strip().lower())
    text = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff_-]", "", text)
    return text or "model"


def unique_id(label: str, existing: set[str]) -> str:
    base = slugify(label)
    candidate = base
    index = 2
    while candidate in existing:
        candidate = f"{base}-{index}"
        index += 1
    return candidate

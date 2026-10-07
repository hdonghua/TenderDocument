from __future__ import annotations

import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from tender_agent.config import ModelProfile, resolve_api_key


@dataclass
class LLMResult:
    text: str
    truncated: bool = False


class OpenAICompatClient:
    """DeepSeek 与其他 OpenAI Chat Completions 兼容接口。"""

    def __init__(self, profile: ModelProfile, api_key: str):
        from openai import OpenAI

        base_url = profile.resolved_base_url()
        if not base_url:
            raise RuntimeError(f"模型「{profile.label}」缺少 Base URL")
        self.profile = profile
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=180)

    def complete(self, system: str, user: str, *, json_mode: bool = False) -> LLMResult:
        use_temperature = "reasoner" not in self.profile.model.lower()
        payload = {
            "model": self.profile.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": int(self.profile.max_tokens),
        }
        if use_temperature:
            payload["temperature"] = float(self.profile.temperature)
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        last_error: Exception | None = None
        for attempt in range(2):
            try:
                try:
                    response = self._client.chat.completions.create(**payload)
                except Exception as exc:
                    if json_mode and "response_format" in payload:
                        payload.pop("response_format", None)
                        response = self._client.chat.completions.create(**payload)
                    else:
                        raise exc
                choice = response.choices[0]
                message = choice.message
                text = (message.content or "").strip()
                reasoning = getattr(message, "reasoning_content", None)
                if not text and reasoning:
                    raise RuntimeError(
                        "模型只返回了推理过程，正文为空。请增大 max_tokens，或换用 Chat 模型写正文。"
                    )
                return LLMResult(text=text, truncated=choice.finish_reason == "length")
            except Exception as exc:
                last_error = exc
                message = str(exc).lower()
                if attempt == 0 and ("429" in message or "rate limit" in message):
                    time.sleep(2)
                    continue
                break
        raise RuntimeError(f"模型调用失败（{self.profile.label}）: {last_error}") from last_error


class CursorClient:
    """通过 Cursor SDK 的一次性 Agent 生成文本。"""

    def __init__(self, profile: ModelProfile, api_key: str):
        self.profile = profile
        self.api_key = api_key

    def complete(self, system: str, user: str, *, json_mode: bool = False) -> LLMResult:
        try:
            from cursor_sdk import Agent, AgentOptions, LocalAgentOptions
        except ImportError as exc:
            raise RuntimeError(
                "未安装 Cursor SDK。请执行 pip install -r requirements-cursor.txt 后再使用该模型。"
            ) from exc

        instruction = (
            "你是标书撰写助手。只把最终正文写在回复里。"
            "不要创建、修改或删除任何文件，不要运行命令，不要调用工具。"
        )
        if json_mode:
            instruction += "最终回复只能是一个 JSON 对象，不要加 Markdown 代码块。"
        prompt = f"{instruction}\n\n{system}\n\n{user}"
        workdir = Path(tempfile.mkdtemp(prefix="tender-cursor-"))
        try:
            result = Agent.prompt(
                prompt,
                AgentOptions(
                    api_key=self.api_key,
                    model=self.profile.model,
                    local=LocalAgentOptions(cwd=str(workdir)),
                ),
            )
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

        status = getattr(result, "status", "")
        text = (getattr(result, "result", None) or "").strip()
        if status == "error" or not text:
            raise RuntimeError(f"Cursor 运行失败: {status or 'empty'} {text}")
        return LLMResult(text=text, truncated=False)


def build_client(profile: ModelProfile):
    api_key = resolve_api_key(profile)
    if not api_key:
        env_hint = profile.api_key_env or (
            "CURSOR_API_KEY" if profile.provider == "cursor" else "DEEPSEEK_API_KEY"
        )
        raise RuntimeError(
            f"模型「{profile.label}」没有可用的 API Key。请在「模型配置」中填写，或设置环境变量 {env_hint}。"
        )
    if profile.provider == "cursor":
        return CursorClient(profile, api_key)
    return OpenAICompatClient(profile, api_key)

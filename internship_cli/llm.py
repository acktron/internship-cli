"""LLM backends for optional classification (Gemini or local Ollama).

Both backends expose the same chat + tool-calling surface so callers do not
branch on provider. Select with LLM_BACKEND=gemini|ollama.

Project-root `.env` is loaded once via python-dotenv (does not override
variables already exported in the shell).
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

BACKEND_ENV = "LLM_BACKEND"
OLLAMA_MODEL_ENV = "OLLAMA_MODEL"
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"

DEFAULT_BACKEND = "gemini"
DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"
DEFAULT_OLLAMA_MODEL = "qwen2.5:3b"
OLLAMA_BASE_URL = "http://localhost:11434/v1"
# Local models are slow under load (esp. after a long scrape); give them room.
OLLAMA_HTTP_TIMEOUT = 120
OLLAMA_KEEP_ALIVE = "10m"

GEMINI_URL_TEMPLATE = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
)

_ENV_LOADED = False


def load_project_env() -> Optional[Path]:
    """Load `.env` from cwd or the package project root into os.environ.

    Existing environment variables are left alone (shell export wins).
    Safe to call repeatedly; only the first call reads the file.
    """
    global _ENV_LOADED
    if _ENV_LOADED:
        return None
    _ENV_LOADED = True

    try:
        from dotenv import load_dotenv
    except ImportError:
        return None

    here = Path(__file__).resolve().parent.parent
    for candidate in (Path.cwd() / ".env", here / ".env"):
        if candidate.is_file():
            load_dotenv(candidate, override=False)
            return candidate
    return None


class LLMError(RuntimeError):
    pass


@dataclass(frozen=True)
class Message:
    role: str  # system | user | assistant | tool
    content: str = ""
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[list["ToolCall"]] = None


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str  # JSON object as a string


@dataclass(frozen=True)
class Completion:
    content: Optional[str] = None
    tool_calls: list[ToolCall] = field(default_factory=list)


def get_backend() -> str:
    load_project_env()
    value = os.environ.get(BACKEND_ENV, DEFAULT_BACKEND).strip().lower()
    return value or DEFAULT_BACKEND


def get_llm_client(model: Optional[str] = None) -> "LLMClient":
    """Factory: gemini (default) or ollama from LLM_BACKEND."""
    load_project_env()
    backend = get_backend()
    if backend == "ollama":
        chosen = (
            (model or "").strip()
            or os.environ.get(OLLAMA_MODEL_ENV, "").strip()
            or DEFAULT_OLLAMA_MODEL
        )
        return OllamaClient(model=chosen)
    if backend == "gemini":
        key = os.environ.get(GEMINI_API_KEY_ENV, "").strip()
        if not key:
            raise LLMError(
                f"{GEMINI_API_KEY_ENV} is not set. Export an API key, set "
                f"{BACKEND_ENV}=ollama for a local model, or drop --classify-llm "
                "and use --allowlist instead."
            )
        chosen = (model or "").strip() or DEFAULT_GEMINI_MODEL
        return GeminiClient(model=chosen, api_key=key)
    raise LLMError(f"Unknown {BACKEND_ENV}={backend!r}. Use 'gemini' or 'ollama'.")


class LLMClient:
    """Shared chat / tool-calling interface for all backends."""

    backend: str
    model: str

    def complete(
        self,
        messages: list[Message],
        tools: Optional[list[dict[str, Any]]] = None,
        *,
        temperature: float = 0.0,
        json_mode: bool = False,
        timeout: int = 60,
    ) -> Completion:
        raise NotImplementedError


def _post_json(url: str, payload: dict, headers: dict[str, str], timeout: int) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


class GeminiClient(LLMClient):
    backend = "gemini"

    def __init__(self, model: str, api_key: str) -> None:
        self.model = model
        self._api_key = api_key

    def complete(
        self,
        messages: list[Message],
        tools: Optional[list[dict[str, Any]]] = None,
        *,
        temperature: float = 0.0,
        json_mode: bool = False,
        timeout: int = 60,
    ) -> Completion:
        payload: dict[str, Any] = {
            "contents": _messages_to_gemini(messages),
            "generationConfig": {"temperature": temperature},
        }
        if json_mode:
            payload["generationConfig"]["response_mime_type"] = "application/json"
        system = _system_text(messages)
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        gemini_tools = _tools_to_gemini(tools)
        if gemini_tools:
            payload["tools"] = gemini_tools

        url = GEMINI_URL_TEMPLATE.format(model=self.model, key=self._api_key)
        try:
            response = _post_json(
                url,
                payload,
                {"Content-Type": "application/json"},
                timeout,
            )
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:400]
            raise LLMError(f"Gemini API error {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise LLMError(f"Could not reach the Gemini API: {exc.reason}") from exc
        except json.JSONDecodeError as exc:
            raise LLMError("Gemini returned a non-JSON body.") from exc

        return _completion_from_gemini(response)


class OllamaClient(LLMClient):
    backend = "ollama"

    def __init__(self, model: str, base_url: str = OLLAMA_BASE_URL) -> None:
        self.model = model
        self._base_url = base_url.rstrip("/")

    def complete(
        self,
        messages: list[Message],
        tools: Optional[list[dict[str, Any]]] = None,
        *,
        temperature: float = 0.0,
        json_mode: bool = False,
        timeout: int = OLLAMA_HTTP_TIMEOUT,
    ) -> Completion:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": _messages_to_openai(messages),
            "temperature": temperature,
            # Keep the model resident between tool-selection and summary turns.
            "keep_alive": OLLAMA_KEEP_ALIVE,
            # Newer OpenAI-compat builds also read keep_alive from options.
            "options": {"keep_alive": OLLAMA_KEEP_ALIVE},
        }
        if tools:
            payload["tools"] = tools
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        url = f"{self._base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        last_exc: BaseException | None = None

        # One retry: cold starts / brief stalls after scraping often recover.
        for attempt in range(2):
            try:
                response = _post_json(url, payload, headers, timeout)
                return _completion_from_openai(response)
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:400]
                raise LLMError(f"Ollama API error {exc.code}: {detail}") from exc
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                last_exc = exc
                if attempt == 0:
                    continue
                detail = f"{type(exc).__name__}: {exc}"
                print("Ollama not reachable — is it running?", file=sys.stderr)
                print(f"  underlying: {detail}", file=sys.stderr)
                raise LLMError(
                    f"Ollama not reachable — is it running? ({detail})"
                ) from exc
            except json.JSONDecodeError as exc:
                raise LLMError("Ollama returned a non-JSON body.") from exc

        # Unreachable, but keeps type-checkers happy.
        detail = f"{type(last_exc).__name__}: {last_exc}" if last_exc else "unknown error"
        raise LLMError(f"Ollama not reachable — is it running? ({detail})") from last_exc


def _system_text(messages: list[Message]) -> str:
    return "\n\n".join(m.content for m in messages if m.role == "system" and m.content)


def _messages_to_openai(messages: list[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for msg in messages:
        row: dict[str, Any] = {"role": msg.role, "content": msg.content or ""}
        if msg.name:
            row["name"] = msg.name
        if msg.tool_call_id:
            row["tool_call_id"] = msg.tool_call_id
        if msg.tool_calls:
            row["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments},
                }
                for tc in msg.tool_calls
            ]
        out.append(row)
    return out


def _messages_to_gemini(messages: list[Message]) -> list[dict[str, Any]]:
    """Map OpenAI-style roles onto Gemini user/model turns + function replies."""
    out: list[dict[str, Any]] = []
    for msg in messages:
        if msg.role == "system":
            continue
        if msg.role == "tool":
            out.append(
                {
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": msg.name or "tool",
                                "response": _safe_json_object(msg.content),
                            }
                        }
                    ],
                }
            )
            continue
        if msg.role == "assistant":
            parts: list[dict[str, Any]] = []
            if msg.content:
                parts.append({"text": msg.content})
            for tc in msg.tool_calls or []:
                parts.append(
                    {
                        "functionCall": {
                            "name": tc.name,
                            "args": _safe_json_object(tc.arguments),
                        }
                    }
                )
            if parts:
                out.append({"role": "model", "parts": parts})
            continue
        # user (and anything else)
        out.append({"role": "user", "parts": [{"text": msg.content or ""}]})
    return out


def _tools_to_gemini(tools: Optional[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    if not tools:
        return []
    declarations = []
    for tool in tools:
        if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
            fn = tool["function"]
        else:
            fn = tool
        declarations.append(
            {
                "name": fn["name"],
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
            }
        )
    return [{"function_declarations": declarations}] if declarations else []


def _completion_from_openai(response: dict) -> Completion:
    try:
        message = response["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError(f"Unexpected Ollama response shape: {str(response)[:300]}") from exc

    content = message.get("content")
    tool_calls: list[ToolCall] = []
    for raw in message.get("tool_calls") or []:
        if not isinstance(raw, dict):
            continue
        fn = raw.get("function") or {}
        tool_calls.append(
            ToolCall(
                id=str(raw.get("id") or fn.get("name") or "tool"),
                name=str(fn.get("name") or ""),
                arguments=fn.get("arguments")
                if isinstance(fn.get("arguments"), str)
                else json.dumps(fn.get("arguments") or {}),
            )
        )
    return Completion(content=content, tool_calls=tool_calls)


def _completion_from_gemini(response: dict) -> Completion:
    try:
        parts = response["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError(f"Unexpected Gemini response shape: {str(response)[:300]}") from exc

    texts: list[str] = []
    tool_calls: list[ToolCall] = []
    for index, part in enumerate(parts if isinstance(parts, list) else []):
        if not isinstance(part, dict):
            continue
        if "text" in part and part["text"] is not None:
            texts.append(str(part["text"]))
        fc = part.get("functionCall") or part.get("function_call")
        if isinstance(fc, dict) and fc.get("name"):
            args = fc.get("args") if "args" in fc else fc.get("arguments")
            tool_calls.append(
                ToolCall(
                    id=f"call_{index}_{fc['name']}",
                    name=str(fc["name"]),
                    arguments=args if isinstance(args, str) else json.dumps(args or {}),
                )
            )
    content = "".join(texts) if texts else None
    return Completion(content=content, tool_calls=tool_calls)


def _safe_json_object(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"result": raw}
    return parsed if isinstance(parsed, dict) else {"result": parsed}

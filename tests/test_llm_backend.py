"""LLM backend selection and shared completion interface."""

from __future__ import annotations

import json
import os
import urllib.error
from unittest.mock import patch

import pytest

from internship_cli.llm import (
    OLLAMA_HTTP_TIMEOUT,
    OLLAMA_KEEP_ALIVE,
    Completion,
    GeminiClient,
    LLMError,
    Message,
    OllamaClient,
    ToolCall,
    get_backend,
    get_llm_client,
)


@pytest.fixture(autouse=True)
def _clear_backend_env(monkeypatch):
    # Skip project .env so unit tests control the environment explicitly.
    monkeypatch.setattr("internship_cli.llm._ENV_LOADED", True)
    monkeypatch.delenv("LLM_BACKEND", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)


def test_default_backend_is_gemini():
    assert get_backend() == "gemini"


def test_ollama_client_uses_default_model(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "ollama")
    client = get_llm_client()
    assert isinstance(client, OllamaClient)
    assert client.model == "qwen2.5:3b"


def test_ollama_model_env(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.2")
    client = get_llm_client()
    assert client.model == "llama3.2"


def test_gemini_requires_api_key(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "gemini")
    with pytest.raises(LLMError, match="GEMINI_API_KEY"):
        get_llm_client()


def test_unknown_backend(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "nope")
    with pytest.raises(LLMError, match="Unknown"):
        get_llm_client()


def test_ollama_connection_error_message(monkeypatch, capsys):
    monkeypatch.setenv("LLM_BACKEND", "ollama")
    client = get_llm_client()
    calls = {"n": 0}

    def _boom(*_a, **_k):
        calls["n"] += 1
        raise urllib.error.URLError("connection refused")

    with patch("internship_cli.llm.urllib.request.urlopen", side_effect=_boom):
        with pytest.raises(LLMError, match="Ollama not reachable"):
            client.complete([Message(role="user", content="hi")])

    assert calls["n"] == 2  # initial + one retry
    err = capsys.readouterr().err
    assert "Ollama not reachable — is it running?" in err
    assert "underlying: URLError" in err


def test_ollama_retries_once_then_succeeds(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "ollama")
    client = get_llm_client()
    body = json.dumps(
        {"choices": [{"message": {"content": "ok", "tool_calls": []}}]}
    ).encode()

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return body

    attempts = {"n": 0}

    def _flaky(*_a, **_k):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise TimeoutError("timed out")
        return _Resp()

    with patch("internship_cli.llm.urllib.request.urlopen", side_effect=_flaky):
        result = client.complete([Message(role="user", content="hi")])

    assert attempts["n"] == 2
    assert result.content == "ok"


def test_ollama_sends_keep_alive_and_long_timeout(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "ollama")
    client = get_llm_client()
    seen: dict = {}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(
                {"choices": [{"message": {"content": "hi", "tool_calls": []}}]}
            ).encode()

    def _capture(req, timeout=None):
        seen["timeout"] = timeout
        seen["body"] = json.loads(req.data.decode())
        return _Resp()

    with patch("internship_cli.llm.urllib.request.urlopen", side_effect=_capture):
        client.complete([Message(role="user", content="hi")])

    assert seen["timeout"] == OLLAMA_HTTP_TIMEOUT == 120
    assert seen["body"]["keep_alive"] == OLLAMA_KEEP_ALIVE == "10m"
    assert seen["body"]["options"]["keep_alive"] == "10m"


def test_ollama_parses_tool_calls(monkeypatch):
    payload = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "arguments": '{"q":"Acme"}',
                            },
                        }
                    ],
                }
            }
        ]
    }

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(payload).encode()

    monkeypatch.setenv("LLM_BACKEND", "ollama")
    client = get_llm_client()
    with patch("internship_cli.llm.urllib.request.urlopen", return_value=_Resp()):
        result = client.complete(
            [Message(role="user", content="hi")],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "description": "lookup",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        )

    assert isinstance(result, Completion)
    assert result.tool_calls == [ToolCall(id="call_1", name="lookup", arguments='{"q":"Acme"}')]


def test_gemini_and_ollama_share_complete_signature():
    import inspect

    gemini = inspect.signature(GeminiClient.complete)
    ollama = inspect.signature(OllamaClient.complete)
    assert list(gemini.parameters) == list(ollama.parameters)
    # Ollama defaults to a longer timeout; parameter surface stays the same.
    assert ollama.parameters["timeout"].default == OLLAMA_HTTP_TIMEOUT


def test_load_project_env_reads_dotenv(tmp_path, monkeypatch):
    import internship_cli.llm as llm

    env_file = tmp_path / ".env"
    env_file.write_text(
        "LLM_BACKEND=ollama\nOLLAMA_MODEL=qwen2.5:3b\nGEMINI_API_KEY=secret-from-env\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(llm, "_ENV_LOADED", False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LLM_BACKEND", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    loaded = llm.load_project_env()
    assert loaded == env_file
    assert os.environ["LLM_BACKEND"] == "ollama"
    assert os.environ["OLLAMA_MODEL"] == "qwen2.5:3b"
    assert os.environ["GEMINI_API_KEY"] == "secret-from-env"

    # Shell export wins: already-set vars are not overridden.
    monkeypatch.setattr(llm, "_ENV_LOADED", False)
    monkeypatch.setenv("LLM_BACKEND", "gemini")
    llm.load_project_env()
    assert os.environ["LLM_BACKEND"] == "gemini"

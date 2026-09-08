"""Agent tool schemas, caps, and turn loop (no live LinkedIn / LLM)."""

from __future__ import annotations

import json
from contextlib import contextmanager
from unittest.mock import patch

from internship_cli.agent import AGENT_TOOLS, Agent, AgentRuntime
from internship_cli.llm import Completion, Message, ToolCall


def _passthrough_gate(client, posts, *, log=print):  # noqa: ANN001
    return list(posts)


class _FakeClient:
    backend = "ollama"
    model = "qwen2.5"

    def __init__(self, scripted: list[Completion], on_complete=None) -> None:
        self._scripted = list(scripted)
        self.calls: list[dict] = []
        self._on_complete = on_complete

    def complete(self, messages, tools=None, **kwargs):
        self.calls.append({"messages": list(messages), "tools": tools, **kwargs})
        if self._on_complete:
            self._on_complete(self.calls[-1])
        if not self._scripted:
            return Completion(content="(no more scripted replies)")
        return self._scripted.pop(0)


def test_tool_schemas_include_scrape_and_company():
    names = {t["function"]["name"] for t in AGENT_TOOLS}
    assert names == {"scrape_posts", "company_posts"}
    scrape = next(t for t in AGENT_TOOLS if t["function"]["name"] == "scrape_posts")
    company = next(t for t in AGENT_TOOLS if t["function"]["name"] == "company_posts")
    assert "query" in scrape["function"]["parameters"]["required"]
    for key in (
        "max_age_days",
        "match_mode",
        "scrolls",
        "limit",
        "role_terms",
        "domain_terms",
        "hiring_terms",
        "resolve_links",
    ):
        assert key in scrape["function"]["parameters"]["properties"]
    for key in (
        "max_age_days",
        "match_mode",
        "scrolls",
        "role_query",
        "queries",
        "queries_by_company",
        "max_queries_per_company",
        "role_terms",
        "resolve_links",
        "max_companies",
    ):
        assert key in company["function"]["parameters"]["properties"]
    assert scrape["function"]["parameters"]["properties"]["match_mode"]["enum"] == [
        "all",
        "loose",
    ]


def test_system_prompt_mentions_retry_and_tuning():
    from internship_cli.agent import SYSTEM_PROMPT

    assert "match_mode" in SYSTEM_PROMPT
    assert "retry" in SYSTEM_PROMPT.lower()
    assert "scrolls" in SYSTEM_PROMPT


def test_agent_calls_tool_then_summarises():
    client = _FakeClient(
        [
            Completion(
                content="",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="scrape_posts",
                        arguments=json.dumps({"query": "AI intern", "limit": 2}),
                    )
                ],
            ),
            Completion(
                content=(
                    "Found 1 match:\n"
                    "- Acme — hiring:we're hiring — https://linkedin.com/x"
                )
            ),
        ]
    )

    class _Runtime(AgentRuntime):
        def execute(self, name, arguments):  # noqa: ANN001
            assert name == "scrape_posts"
            assert arguments["query"] == "AI intern"
            return {
                "tool": "scrape_posts",
                "count": 1,
                "results": [
                    {
                        "company": "Acme",
                        "why_matched": "hiring:we're hiring",
                        "post_url": "https://linkedin.com/x",
                    }
                ],
            }

    logs: list[str] = []
    bot = Agent(
        client=client,  # type: ignore[arg-type]
        runtime=_Runtime(page=None),
        max_tool_calls=3,
        log=logs.append,
    )
    with patch("internship_cli.agent.gate_hiring_posts", side_effect=_passthrough_gate):
        answer = bot.handle("Find AI intern posts")
    assert "Acme" in answer
    assert any(
        isinstance(m, Message) and m.role == "tool" and "Acme" in m.content
        for m in bot.history
    )
    assert len(client.calls) == 2
    assert bot.last_results and bot.last_results[0]["company"] == "Acme"
    assert any("qwen2.5" in line and ("summar" in line or "reviewing" in line) for line in logs)


def test_browser_closed_before_summary_llm_call():
    browser_open = {"value": False}
    events: list[str] = []

    @contextmanager
    def factory():
        browser_open["value"] = True
        events.append("open")
        try:
            yield object()
        finally:
            browser_open["value"] = False
            events.append("close")

    def on_complete(_call):
        events.append("llm")
        assert browser_open["value"] is False, "LLM must not run while browser is open"

    client = _FakeClient(
        [
            Completion(
                tool_calls=[
                    ToolCall(id="1", name="scrape_posts", arguments='{"query":"x"}')
                ]
            ),
            Completion(content="summary done"),
        ],
        on_complete=on_complete,
    )

    class _Runtime(AgentRuntime):
        def execute(self, name, arguments):  # noqa: ANN001
            events.append("tool")
            assert browser_open["value"] is True
            return {"tool": name, "count": 0, "results": []}

    bot = Agent(
        client=client,  # type: ignore[arg-type]
        runtime=_Runtime(browser_factory=factory),
        max_tool_calls=3,
        log=lambda _m: None,
    )
    with patch("internship_cli.agent.gate_hiring_posts", side_effect=_passthrough_gate):
        assert bot.handle("go") == "summary done"
    assert events == ["llm", "open", "tool", "close", "llm"]


def test_agent_caps_tool_calls():
    # Model keeps asking for tools; agent must cut off and force a summary turn.
    client = _FakeClient(
        [
            Completion(
                tool_calls=[
                    ToolCall(id="1", name="scrape_posts", arguments='{"query":"a"}')
                ]
            ),
            Completion(
                tool_calls=[
                    ToolCall(id="2", name="scrape_posts", arguments='{"query":"b"}')
                ]
            ),
            # Cap is 2; third complete should be without tools / summary.
            Completion(content="Stopped after cap; here is what I found."),
        ]
    )

    calls = {"n": 0}

    class _Runtime(AgentRuntime):
        def execute(self, name, arguments):  # noqa: ANN001
            calls["n"] += 1
            return {"tool": name, "count": 0, "results": []}

    bot = Agent(
        client=client,  # type: ignore[arg-type]
        runtime=_Runtime(page=None),
        max_tool_calls=2,
        log=lambda _m: None,
    )
    with patch("internship_cli.agent.gate_hiring_posts", side_effect=_passthrough_gate):
        answer = bot.handle("search a lot")
    assert calls["n"] == 2
    assert "Stopped after cap" in answer or "found" in answer.lower()
    # Final LLM call should disable tools once cap is hit.
    assert client.calls[-1]["tools"] is None

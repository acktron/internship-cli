"""Outbound link classification and redirect cache."""

from unittest.mock import patch

from internship_cli.links import (
    RedirectResolver,
    classify_final_url,
    classify_links,
    extract_urls_from_text,
)
from internship_cli.filters import heuristic_source_reject, source_spam_signals


def test_extract_lnkd_and_http():
    text = "Apply https://lnkd.in/abc123 or www.example.com/careers now"
    urls = extract_urls_from_text(text)
    assert any("lnkd.in/abc123" in u for u in urls)
    assert any("example.com/careers" in u for u in urls)


def test_classify_ats_and_careers_path():
    assert classify_final_url("https://boards.greenhouse.io/acme/jobs/1") == "careers"
    assert classify_final_url("https://jobs.lever.co/acme") == "careers"
    assert classify_final_url("https://jobs.ashbyhq.com/acme") == "careers"
    assert classify_final_url("https://acme.myworkdayjobs.com/careers") == "careers"
    assert classify_final_url("https://sarvam.ai/careers") == "careers"
    assert classify_final_url(
        "https://www.linkedin.com/jobs/view/123",
    ) == "careers"
    assert classify_final_url("https://sarvam.ai/blog", company="Sarvam AI") == "careers"


def test_classify_junk_hosts():
    assert classify_final_url("https://chat.whatsapp.com/inviteXXX") == "junk"
    assert classify_final_url("https://t.me/+abcd") == "junk"
    assert classify_final_url("https://discord.gg/jobs") == "junk"
    assert classify_final_url("https://www.linkedin.com/groups/123/invite") == "junk"
    assert classify_final_url("https://jobscans.in/alerts") == "junk"


def test_careers_beats_junk_in_the_same_post():
    verdict = classify_links(
        [
            "https://chat.whatsapp.com/xxx",
            "https://boards.greenhouse.io/acme/jobs/1",
        ],
        company="Acme",
    )
    assert verdict.kind == "careers"
    assert verdict.reject is False


def test_junk_only_rejects():
    verdict = classify_links(["https://chat.whatsapp.com/xxx"])
    assert verdict.kind == "junk"
    assert verdict.reject is True


def test_resolver_caches_and_follows():
    calls = {"n": 0}

    def fake_follow(url, timeout=8.0):
        calls["n"] += 1
        return "https://boards.greenhouse.io/acme/jobs/9"

    resolver = RedirectResolver(delay=0)
    with patch("internship_cli.links._http_follow", side_effect=fake_follow):
        a = resolver.resolve("https://lnkd.in/abc")
        b = resolver.resolve("https://lnkd.in/abc")
    assert a.endswith("/jobs/9")
    assert a == b
    assert calls["n"] == 1


def test_heuristic_rejects_hashtags_and_dm_referral():
    text = (
        "I'm hiring DM me for referral "
        "#jobs #hiring #intern #ai #ml #sde #backend #frontend #freshers #apply"
    )
    signals = source_spam_signals(text, link_kind="none")
    assert any(s.startswith("hashtags:") for s in signals)
    assert "dm-for-referral" in signals
    reject, reason = heuristic_source_reject(text, link_kind="none")
    assert reject is True
    assert "source-spam" in reason


def test_heuristic_keeps_company_careers():
    text = "We're hiring an AI intern. Apply on our careers page."
    reject, _ = heuristic_source_reject(
        text,
        poster="Acme",
        company="Acme",
        is_company_page=True,
        link_kind="careers",
    )
    assert reject is False


def test_junk_link_rejects_even_for_company_page():
    reject, reason = heuristic_source_reject(
        "Join our team intern hiring",
        is_company_page=True,
        link_kind="junk",
    )
    assert reject is True
    assert reason.startswith("junk-link")

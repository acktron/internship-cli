"""Relevance gate parsing and filtering."""

from __future__ import annotations

import json

from internship_cli.filters import classify_first_party, parse_follower_count
from internship_cli.gate import (
    format_result_blocks,
    gate_hiring_posts,
    normalize_agent_row,
)
from internship_cli.llm import Completion


class _FakeClient:
    backend = "ollama"
    model = "qwen2.5:3b"

    def __init__(self, content: str) -> None:
        self._content = content

    def complete(self, messages, tools=None, **kwargs):
        return Completion(content=self._content)


def test_gate_keeps_company_and_recruiter_drops_farmer():
    posts = [
        {
            "company": "Student",
            "poster": "Ada",
            "snippet": "Successfully completed my internship!",
            "why_matched": "hiring:internship",
            "post_url": "https://li/1",
        },
        {
            "company": "Acme",
            "poster": "Acme",
            "role": "AI Intern",
            "snippet": "We're hiring AI interns — apply now https://acme.com/careers",
            "why_matched": "hiring:we're hiring",
            "post_url": "https://www.linkedin.com/feed/update/urn:li:activity:2/",
            "link_class": "careers",
        },
        {
            "company": "Bob",
            "poster": "Bob",
            "snippet": "I'm hiring an intern on my personal project",
            "why_matched": "hiring:hiring",
            "post_url": "https://li/3",
        },
        {
            "company": "Acme",
            "poster": "JobAlerts Daily",
            "snippet": "DM me for referral. Join my WhatsApp community.",
            "why_matched": "hiring:hiring",
            "post_url": "https://li/4",
            "link_class": "junk",
        },
    ]
    payload = {
        "results": [
            {"id": 0, "keep": False, "reason": "celebration", "source_type": "spam"},
            {"id": 1, "keep": True, "reason": "open role + careers", "source_type": "company"},
            {"id": 2, "keep": True, "reason": "personal hire", "source_type": "recruiter"},
            {"id": 3, "keep": False, "reason": "group invite", "source_type": "farmer"},
        ]
    }
    kept = gate_hiring_posts(
        _FakeClient(json.dumps(payload)),  # type: ignore[arg-type]
        posts,
        log=lambda _m: None,
    )
    assert len(kept) == 2
    assert kept[0]["company"] == "Acme"
    assert kept[0]["source_type"] == "company"
    assert kept[1]["source_type"] == "recruiter"
    assert "gate:keep,source:company" in kept[0]["why_matched"]
    assert "open role" in kept[0]["gate_reason"]
    assert "poster:Acme" in kept[0]["why_matched"]


def test_gate_keeps_creator():
    posts = [
        {
            "company": "Microsoft",
            "poster": "Alex Chen",
            "poster_headline": "Sharing CS internships · 31k+ LinkedIn",
            "snippet": "Microsoft Software Engineering Intern. Apply: https://careers.microsoft.com",
            "why_matched": "hiring:hiring",
            "post_url": "https://www.linkedin.com/feed/update/urn:li:activity:9/",
        }
    ]
    payload = {
        "results": [
            {"id": 0, "keep": True, "reason": "real opening + apply link", "source_type": "creator"},
        ]
    }
    kept = gate_hiring_posts(
        _FakeClient(json.dumps(payload)),  # type: ignore[arg-type]
        posts,
        log=lambda _m: None,
    )
    assert len(kept) == 1
    assert kept[0]["source_type"] == "creator"
    assert "gate:keep,source:creator" in kept[0]["why_matched"]
    assert "poster:Alex Chen" in kept[0]["why_matched"]
    assert "followers:31k" in kept[0]["why_matched"]


def test_gate_drops_keep_true_aggregator():
    posts = [{"company": "Acme", "snippet": "50 intern openings DM me", "post_url": "x"}]
    payload = {
        "results": [
            {"id": 0, "keep": True, "reason": "listicle", "source_type": "aggregator"},
        ]
    }
    kept = gate_hiring_posts(
        _FakeClient(json.dumps(payload)),  # type: ignore[arg-type]
        posts,
        log=lambda _m: None,
    )
    assert kept == []


def test_gate_unparseable_keeps_all():
    posts = [{"company": "Acme", "snippet": "hiring intern", "post_url": "x"}]
    kept = gate_hiring_posts(
        _FakeClient("not-json"),  # type: ignore[arg-type]
        posts,
        log=lambda _m: None,
    )
    assert kept == posts


def test_legacy_hiring_now_schema_still_parses():
    posts = [
        {"company": "Acme", "snippet": "hiring intern", "post_url": "x"},
    ]
    payload = {
        "results": [
            {"id": 0, "hiring_now": True, "is_company_role": True, "reason": "ok"},
        ]
    }
    kept = gate_hiring_posts(
        _FakeClient(json.dumps(payload)),  # type: ignore[arg-type]
        posts,
        log=lambda _m: None,
    )
    assert len(kept) == 1
    assert kept[0]["source_type"] == "company"


def test_microsoft_spam_posters_rejected():
    cases = [
        (
            "Frontlines EduTech",
            "#Hiring Alert | IBM & Microsoft are Hiring Discover today's top hiring "
            "opportunities for Freshers. IBM — Technical Support Representative Intern.",
        ),
        (
            "Cyber Jobs Daily",
            "NEW #Cybersecurity JOBS IN THE USA Looking for SOC Analyst Intern – "
            "1️⃣ 2️⃣ 3️⃣ apply at jobright",
        ),
        (
            "Creator Coach",
            "𝗠𝗼𝘀𝘁 𝗷𝘂𝗻𝗶𝗼𝗿𝘀 𝗮𝘀𝗸 𝗺𝗲 𝗼𝗻𝗲 𝗾𝘂𝗲𝘀𝘁𝗶𝗼𝗻: 𝗜𝗻𝘁𝗲𝗿𝗻𝘀𝗵𝗶𝗽 𝗸𝗮𝗵𝗮 𝘀𝗲 𝗺𝗶𝗹𝗲𝗴𝗶? "
            "Your internship search shouldn’t start when placements begin.",
        ),
        (
            "Random Recruiter",
            "Junior AI Full Stack Developer Location: Noida Experience: 2+ Years "
            "(Internships & strong projects count). We're looking for a developer.",
        ),
        (
            "Fuse Integration",
            "UC San Diego Jacobs School of Engineering partner Fuse Integration is hiring "
            "in San Diego. Apply now for a software internship.",
        ),
        (
            "JobAlerts Daily",
            "Microsoft is Hiring Software Engineering Intern! Company: Dynamics Solution "
            "and Technology. We are looking for an intern in Lahore.",
        ),
    ]
    for poster, text in cases:
        source, keep, reason = classify_first_party(
            "Microsoft", poster=poster, text=text, strict=True
        )
        assert keep is False, (poster, source, reason)


def test_microsoft_employee_referral_passes():
    source, keep, reason = classify_first_party(
        "Microsoft",
        poster="Priya Shah",
        headline="Software Engineer at Microsoft",
        text=(
            "We're hiring a Software Engineering Intern on my team at Microsoft. "
            "DM me for a referral or comment below."
        ),
        strict=True,
    )
    assert keep is True, reason
    assert source == "employee"


def test_credible_creator_real_role_passes():
    headline = "Sharing CS internships · 31k+ LinkedIn"
    source, keep, reason = classify_first_party(
        "Microsoft",
        poster="Alex Chen",
        headline=headline,
        text=(
            "Microsoft Software Engineering Intern (Summer 2026). "
            "Apply: https://careers.microsoft.com/students/us/en/internship"
        ),
        strict=True,
    )
    assert keep is True, reason
    assert source == "creator"
    assert parse_follower_count(headline) == 31_000


def test_cybersecurity_usa_jobs_rejected():
    source, keep, reason = classify_first_party(
        "Microsoft",
        poster="Cyber Jobs Daily",
        text=(
            "NEW #Cybersecurity JOBS IN THE USA Looking for SOC Analyst Intern – "
            "1️⃣ 2️⃣ 3️⃣ apply at jobright"
        ),
        strict=True,
    )
    assert keep is False
    assert source == "aggregator", (source, reason)


def test_contentless_juniors_ask_me_rejected():
    source, keep, reason = classify_first_party(
        "Microsoft",
        poster="Creator Coach",
        text=(
            "Most juniors ask me one question: Internship kaha se milegi? "
            "Your internship search shouldn’t start when placements begin."
        ),
        strict=True,
    )
    assert keep is False
    assert "concrete opening" in reason or "off-topic" in reason


def test_microsoft_company_page_kept():
    source, keep, reason = classify_first_party(
        "Microsoft",
        poster="Microsoft",
        headline="Software · Technology",
        text="We're hiring software engineering interns. Apply now on our careers page.",
        strict=True,
    )
    assert keep is True
    assert source == "company"


def test_employee_headline_kept():
    source, keep, reason = classify_first_party(
        "Microsoft",
        poster="Ada Lovelace",
        headline="University Recruiter at Microsoft",
        text="We're hiring AI interns on my team. Apply now.",
        strict=True,
    )
    assert keep is True
    assert source == "recruiter"


def test_loose_mode_allows_aggregator_mention():
    source, keep, _ = classify_first_party(
        "Microsoft",
        poster="Jobs Daily Alerts",
        text="Microsoft is hiring an AI intern. Apply now.",
        strict=False,
    )
    assert keep is True
    assert source == "aggregator"
    row = normalize_agent_row(
        {
            "company": "Acme",
            "role": "AI Intern",
            "why_matched": "gate:keep,source:company",
            "post_url": "https://www.linkedin.com/feed/update/urn:li:activity:1/",
            "source_type": "company",
            "gate_reason": "first-party ATS link",
        }
    )
    text = format_result_blocks([row])
    assert "company: Acme" in text
    assert "role: AI Intern" in text
    assert "post_url: https://www.linkedin.com/feed/update/urn:li:activity:1/" in text
    assert "source_type: company" in text
    assert "first-party ATS link" in row["why_matched"]

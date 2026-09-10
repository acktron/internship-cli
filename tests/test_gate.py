"""Relevance gate parsing and filtering."""

from __future__ import annotations

import json

from internship_cli.filters import (
    author_affiliation_matches_target,
    classify_first_party,
    parse_follower_count,
    poster_employer_from_headline,
)
from internship_cli.gate import (
    apply_gate_labels,
    company_gate_profile,
    format_result_blocks,
    gate_company_posts_semantic,
    gate_hiring_posts,
    load_gate_history,
    normalize_agent_row,
    semantic_keep,
    write_gate_history,
)
from internship_cli.llm import Completion
from internship_cli.posts import _engagement_spam_signal


class _FakeClient:
    backend = "ollama"
    model = "qwen2.5:3b"

    def __init__(self, content: str) -> None:
        self._content = content
        self.calls = 0
        self.messages = []

    def complete(self, messages, tools=None, **kwargs):
        self.calls += 1
        self.messages.append(messages)
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


def test_semantic_gate_rejects_buildathon_student_seeking_internship():
    """Regression: target-company mentions plus intern keywords are not an offer."""
    post = {
        "company": "Razorpay",
        "poster": "Aarav Student",
        "poster_headline": "Computer Science Student | Open to Work",
        "snippet": (
            "I built X for Razorpay's Buildathon and I'm looking for an AI/ML "
            "internship. Please refer me; portfolio: github.com/aarav"
        ),
        "why_matched": "company:Razorpay; hiring:looking for; role:intern; domain:ai",
        "post_url": "https://linkedin.example/buildathon",
    }
    client = _FakeClient(
        json.dumps(
            {
                "results": [
                    {
                        "id": 0,
                        "author_is_hirer": False,
                        "is_offer_not_request": False,
                        "cta_is_application": False,
                        "role_named": False,
                        "intent": "candidate_seeking",
                        "reason": "Student is seeking an internship after a project showcase.",
                    }
                ]
            }
        )
    )
    kept = gate_company_posts_semantic(client, [post], log=lambda _m: None)
    assert kept == []
    # The deterministic pre-filter prevents an unnecessary model call.
    assert client.calls == 0


def test_semantic_gate_keeps_company_hiring_and_records_reason():
    post = {
        "company": "Razorpay",
        "poster": "Riya Shah",
        "poster_headline": "University Recruiter at Razorpay",
        "snippet": "We're hiring a Machine Learning Intern. Apply here: https://jobs.razorpay.com",
        "why_matched": "company:Razorpay; hiring:we're hiring",
        "post_url": "https://linkedin.example/opening",
    }
    payload = {
        "results": [
            {
                "id": 0,
                "author_is_hirer": True,
                "is_offer_not_request": True,
                "cta_is_application": True,
                "role_named": True,
                "intent": "company_hiring",
                "reason": "Target-company recruiter advertises a named role with an apply link.",
            }
        ]
    }
    post.update({
        "poster_employer": "Razorpay",
        "author_affiliation_matches_target": True,
        "link_categories": ["ats_or_careers"],
        "engagement": {"likes": 3, "comments": 1, "reposts": 0},
    })
    client = _FakeClient(json.dumps(payload))
    kept = gate_company_posts_semantic(client, [post], log=lambda _m: None)
    assert len(kept) == 1
    assert "semantic:company_hiring" in kept[0]["why_matched"]
    assert kept[0]["gate_reason"].startswith("Target-company recruiter")
    prompt = client.messages[0][0].content
    assert "link_categories: ['ats_or_careers']" in prompt
    assert "poster_employer: Razorpay" in prompt
    assert "engagement: {'likes': 3, 'comments': 1, 'reposts': 0}" in prompt


def test_single_company_many_hashtag_hiring_post_reaches_semantic_gate():
    post = {
        "company": "Hiver", "poster": "Hiver", "poster_headline": "Recruiting at Hiver",
        "snippet": "Hiver Hiring. Role: SDE Intern. Location: Bangalore. " + " ".join(f"#tag{i}" for i in range(16)),
        "post_url": "hiver-hashtags",
    }
    payload = {"results": [{
        "id": 0, "author_is_hirer": True, "is_offer_not_request": True,
        "cta_is_application": True, "role_named": True,
        "intent": "company_hiring", "reason": "Official company opening.",
    }]}
    client = _FakeClient(json.dumps(payload))
    assert len(gate_company_posts_semantic(client, [post], log=lambda _m: None)) == 1
    assert client.calls == 1


def test_semantic_gate_fails_closed_for_invalid_json():
    post = {"company": "Acme", "snippet": "We're hiring an AI intern.", "post_url": "x"}
    kept = gate_company_posts_semantic(_FakeClient("not-json"), [post], log=lambda _m: None)
    assert kept == []


def test_semantic_gate_retries_invalid_json_and_keeps_valid_retry():
    class _RetryClient(_FakeClient):
        def __init__(self, responses):
            super().__init__(responses[0])
            self.responses = responses

        def complete(self, messages, tools=None, **kwargs):
            self.messages.append(messages)
            response = self.responses[self.calls]
            self.calls += 1
            return Completion(content=response)

    valid = json.dumps({"results": [{
        "id": 0, "author_is_hirer": True, "is_offer_not_request": True,
        "cta_is_application": True, "role_named": True,
        "intent": "company_hiring", "reason": "Recruiter posted an application link.",
    }]})
    client = _RetryClient(["not JSON at all", valid])
    post = {"company": "Hiver", "poster": "Recruiter", "snippet": "Hiver is hiring an SDE Intern. Apply now.", "post_url": "hiver"}
    assert len(gate_company_posts_semantic(client, [post], log=lambda _m: None)) == 1
    assert client.calls == 2


def test_semantic_gate_rejects_model_labelled_candidate_showcase():
    post = {
        "company": "Razorpay",
        "poster": "Aarav Student",
        "poster_headline": "Computer Science Student",
        "snippet": "I built a fraud model for Razorpay's Buildathon. Hiring managers, please consider me for an AI internship.",
        "post_url": "x",
    }
    payload = {
        "results": [{
            "id": 0,
            "author_is_hirer": False,
            "is_offer_not_request": False,
            "cta_is_application": False,
            "role_named": True,
            "intent": "candidate_seeking",
            "reason": "Student showcases a project while seeking a role.",
        }]
    }
    kept = gate_company_posts_semantic(_FakeClient(json.dumps(payload)), [post], log=lambda _m: None)
    assert kept == []


def test_semantic_gate_reports_candidate_and_employer_mismatch_drops():
    candidate_tally = {}
    candidate = {
        "company": "Acme", "poster": "Student", "snippet": "I am looking for an Acme internship.",
        "post_url": "candidate",
    }
    assert gate_company_posts_semantic(
        _FakeClient("{}"), [candidate], drop_tally=candidate_tally, log=lambda _m: None,
    ) == []
    assert candidate_tally["Acme"]["gate:candidate_seeking"] == 1

    mismatch = {
        "company": "Acme", "poster": "Employee", "poster_employer": "OtherCo",
        "author_affiliation_matches_target": False, "snippet": "Acme is hiring an intern. Apply now.",
        "post_url": "mismatch",
    }
    payload = {"results": [{
        "id": 0, "author_is_hirer": True, "is_offer_not_request": True,
        "cta_is_application": True, "role_named": True,
        "intent": "company_hiring", "reason": "Model claims this is an offer.",
    }]}
    mismatch_tally = {}
    assert gate_company_posts_semantic(
        _FakeClient(json.dumps(payload)), [mismatch], drop_tally=mismatch_tally, log=lambda _m: None,
    ) == []
    assert mismatch_tally["Acme"]["gate:employer_mismatch"] == 1


def test_semantic_gate_batches_at_five():
    accepted = {
        "author_is_hirer": True,
        "is_offer_not_request": True,
        "cta_is_application": True,
        "role_named": True,
        "intent": "company_hiring",
        "reason": "Hiring post.",
    }

    class _BatchClient:
        model = "qwen"

        def __init__(self):
            self.calls = 0

        def complete(self, messages, tools=None, **kwargs):
            count = 5 if self.calls == 0 else 1
            self.calls += 1
            return Completion(content=json.dumps({
                "results": [{"id": i, **accepted} for i in range(count)]
            }))

    client = _BatchClient()
    posts = [{"company": "Acme", "snippet": f"We're hiring AI intern {i}", "post_url": str(i)} for i in range(6)]
    assert len(gate_company_posts_semantic(client, posts, log=lambda _m: None)) == 6
    assert client.calls == 2


def test_semantic_keep_rule_is_python_owned():
    base = {
        "intent": "company_hiring",
        "author_is_hirer": True,
        "is_offer_not_request": True,
        "cta_is_application": False,
        "role_named": True,
    }
    assert semantic_keep(base) is True
    assert semantic_keep({**base, "author_is_hirer": False}) is False
    assert semantic_keep({**base, "intent": "candidate_seeking"}) is False


def test_affiliation_parsing_matches_target_and_rejects_students():
    assert poster_employer_from_headline("SWE @ Razorpay") == "Razorpay"
    assert author_affiliation_matches_target("SWE @ Razorpay", "Razorpay") is True
    assert poster_employer_from_headline("Student | BITS '27") == ""
    assert author_affiliation_matches_target("SWE @ Stripe", "Razorpay") is False


def test_semantic_gate_rejects_high_engagement_self_promo_link():
    post = {
        "company": "Razorpay",
        "poster": "Aarav",
        "poster_employer": "",
        "author_affiliation_matches_target": False,
        "link_categories": ["poster_portfolio_or_github"],
        "engagement": {"likes": 2500, "comments": 110, "reposts": 40},
        "snippet": "Razorpay Buildathon project — check out my GitHub portfolio.",
        "post_url": "x",
    }
    payload = {"results": [{
        "id": 0, "author_is_hirer": True, "is_offer_not_request": True,
        "cta_is_application": True, "role_named": True,
        "intent": "company_hiring", "reason": "Model guess.",
    }]}
    assert gate_company_posts_semantic(_FakeClient(json.dumps(payload)), [post], log=lambda _m: None) == []


def test_engagement_is_negative_only_spam_context():
    assert _engagement_spam_signal(
        {"likes": 0, "comments": 0, "reposts": 0},
        ["https://lnkd.in/a", "https://lnkd.in/b"],
    ) == "zero_engagement_multiple_shorteners"
    assert _engagement_spam_signal(
        {"likes": 9999, "comments": 20, "reposts": 3},
        ["https://lnkd.in/a", "https://lnkd.in/b"],
    ) == ""


def test_gate_history_is_written_and_deduped_by_post_url(tmp_path):
    path = tmp_path / ".gate_history.jsonl"
    write_gate_history(
        [{"post_url": "https://li/1", "company": "Acme", "kept": False, "gate_reason": "old"}],
        path,
    )
    write_gate_history(
        [{"post_url": "https://li/1", "company": "Acme", "kept": True, "gate_reason": "new"}],
        path,
    )
    rows = load_gate_history(path)
    assert len(rows) == 1
    assert rows[0]["kept"] is True
    assert rows[0]["gate_reason"] == "new"


def test_company_profile_conservatively_trusts_and_blocks_posters(tmp_path):
    path = tmp_path / ".gate_history.jsonl"
    write_gate_history([
        {"post_url": "https://li/g1", "company": "Acme", "poster": "Good", "kept": True},
        {"post_url": "https://li/g2", "company": "Acme", "poster": "Good", "kept": True},
        {"post_url": "https://li/b1", "company": "Acme", "poster": "Bad", "kept": False},
        {"post_url": "https://li/b2", "company": "Acme", "poster": "Bad", "kept": False},
    ], path)
    profile = company_gate_profile(load_gate_history(path), "Acme")
    assert "good" in profile["trusted_posters"]
    assert "bad" in profile["blocked_posters"]

    payload = {"results": [{
        "id": 0, "author_is_hirer": True, "is_offer_not_request": True,
        "cta_is_application": True, "role_named": True,
        "intent": "company_hiring", "reason": "Recruiter application link.",
    }]}
    client = _FakeClient(json.dumps(payload))
    kept = gate_company_posts_semantic(
        client,
        [
            {"company": "Acme", "poster": "Good", "snippet": "Hiring intern", "post_url": "https://li/g3"},
            {"company": "Acme", "poster": "Bad", "snippet": "Hiring intern", "post_url": "https://li/b3"},
        ],
        history_path=path,
        log=lambda _m: None,
    )
    assert [row["poster"] for row in kept] == ["Good"]
    prompt = client.messages[0][0].content
    assert "profile_prior: trusted_poster" in prompt
    assert "poster='Bad'" not in prompt


def test_labelled_examples_are_company_scoped_and_injected(tmp_path):
    path = tmp_path / ".gate_history.jsonl"
    write_gate_history([
        {
            "post_url": "https://li/a", "company": "Acme", "poster": "Ana",
            "kept": True, "intent": "company_hiring", "gate_reason": "ATS",
        },
        {
            "post_url": "https://li/o", "company": "Other", "poster": "Omar",
            "kept": False, "intent": "candidate_seeking", "gate_reason": "student",
        },
    ], path)
    assert apply_gate_labels([{"post_url": "https://li/a", "label": "good"}], path) == 1
    payload = {"results": [{
        "id": 0, "author_is_hirer": True, "is_offer_not_request": True,
        "cta_is_application": True, "role_named": True,
        "intent": "company_hiring", "reason": "ATS role.",
    }]}
    client = _FakeClient(json.dumps(payload))
    gate_company_posts_semantic(
        client,
        [{"company": "Acme", "poster": "New", "snippet": "Hiring intern", "post_url": "https://li/new"}],
        history_path=path,
        log=lambda _m: None,
    )
    prompt = client.messages[0][0].content
    assert "User-labelled examples for this target company" in prompt
    assert "Ana" in prompt
    assert "Omar" not in prompt


def test_no_history_uses_generic_examples_and_matches_generic_gate(tmp_path):
    path = tmp_path / "missing.jsonl"
    payload = {"results": [{
        "id": 0, "author_is_hirer": True, "is_offer_not_request": True,
        "cta_is_application": True, "role_named": True,
        "intent": "company_hiring", "reason": "ATS role.",
    }]}
    post = {"company": "Acme", "poster": "Recruiter", "snippet": "Hiring intern", "post_url": "https://li/new"}
    client = _FakeClient(json.dumps(payload))
    kept = gate_company_posts_semantic(client, [post], history_path=path, log=lambda _m: None)
    assert len(kept) == 1
    assert "Generic examples:" in client.messages[0][0].content


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

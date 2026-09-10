"""Company-first STRONG matching: proximity + job-offer cues."""

import tempfile
from pathlib import Path

from internship_cli.filters import (
    HiringMatcher,
    load_companies,
    match_strong_company,
    text_mentions_company,
)
from internship_cli.output import make_snippet, write_results
from internship_cli.posts import (
    Post, _strong_hits_for_company, dedupe_posts_by_identity, funnel_counts_are_monotonic,
)

# --- Real Zomato false positives from results.txt (bag-of-words used to keep these) ---

ZOMATO_CUISINE_STORY = (
    '🚀 "Get me every cuisine in the world. Then every dish inside every cuisine." '
    "My first real task as an intern. I read it twice: Is this even possible? "
    "Seven months later, I have an answer. "
    "📅 In October 2025, my college internship coordinator connected me with Zomato. "
    "I spent months looking for the right opening into food-discovery AI systems, "
    "and the team finally let me ship. This is not a hiring post — just my story."
)

ZOMATO_KOLKATA_NEWSLETTER = (
    "𝐊𝐎𝐋𝐊𝐀𝐓𝐀 𝐑𝐀𝐃𝐀𝐑 | 𝐕𝐨𝐥. #19 | 𝟑𝐫𝐝 𝐀𝐮𝐠𝐮𝐬𝐭 𝟐𝟎𝟐𝟔 "
    "Big money, big moments — this week's roundup. "
    "From foreign investment landing in the state to Brazil's football giants, "
    "Kolkata and West Bengal had a week worth tracking. "
    "Also in this issue: Zomato expands dark stores and a full stack meetup downtown. "
    "HR desks floated campus hiring calendars for next quarter — no open roles listed. "
    + ("More city notes and event blurbs. " * 12)
    + "Elsewhere, one student intern shared an AI side project at a weekend hack. "
    "Subscribe for next week's radar."
)

ZOMATO_COURSE_AD = (
    "Launch Your Career as a Data Analyst with AI-Powered Learning! "
    "The demand for skilled Data Analysts is growing faster than ever. "
    "Our 6–8 Month AI-Powered Data Analyst Job Ready Program includes an internship "
    "module. Looking for motivated learners. Partners include case studies from "
    "Zomato and other consumer apps. Enrol today."
)

ZOMATO_REAL_HIRING = (
    "Zomato is hiring! We're looking for an AI/ML intern to join our search team. "
    "Role: Machine Learning Intern. Apply now or DM me your CV."
)


def test_text_mentions_company_word_boundary():
    assert text_mentions_company("Sarvam AI is hiring an ML intern", "Sarvam AI")
    assert text_mentions_company("Join Groww as an intern", "Groww")
    assert not text_mentions_company("She said apply for the role", "AI")


def test_text_mentions_company_strips_suffix():
    assert text_mentions_company(
        "Acme Technologies is hiring software interns",
        "Acme Technologies Pvt Ltd",
    )


def test_strong_keeps_real_ai_intern_offer():
    result = match_strong_company(ZOMATO_REAL_HIRING, "Zomato")
    assert result.matched is True
    assert any(r.startswith("company:Zomato") for r in result.reasons)
    assert "hiring:none" not in result.reasons
    assert "role:none" not in result.reasons
    assert "domain:none" not in result.reasons
    assert any(r.startswith("proximity:") and "none" not in r for r in result.reasons)


def test_strong_keeps_compact_hiring_line():
    text = (
        "We're hiring an AI intern at Razorpay to work on ML fraud models. "
        "Apply now."
    )
    assert match_strong_company(text, "Razorpay").matched is True


def test_zomato_cuisine_story_rejected():
    """BEFORE: bag-of-words kept this. AFTER: no hire~intern job-offer cluster."""
    # Prove the old shallow signals are present somewhere in the post.
    shallow = HiringMatcher(mode="all").match(ZOMATO_CUISINE_STORY)
    assert shallow.matched is True, "fixture must still trip the old bag-of-words filter"

    result = match_strong_company(ZOMATO_CUISINE_STORY, "Zomato")
    assert result.matched is False
    assert any("proximity:" in r for r in result.reasons)


def test_zomato_kolkata_newsletter_rejected():
    shallow = HiringMatcher(mode="all").match(ZOMATO_KOLKATA_NEWSLETTER)
    assert shallow.matched is True, "fixture must still trip the old bag-of-words filter"

    result = match_strong_company(ZOMATO_KOLKATA_NEWSLETTER, "Zomato")
    assert result.matched is False


def test_zomato_course_ad_rejected():
    shallow = HiringMatcher(mode="all").match(ZOMATO_COURSE_AD)
    assert shallow.matched is True, "fixture must still trip the old bag-of-words filter"

    result = match_strong_company(ZOMATO_COURSE_AD, "Zomato")
    assert result.matched is False


def test_strong_drops_without_company_mention():
    text = "We're hiring an AI/ML intern. Apply now — looking for students."
    result = match_strong_company(text, "Sarvam AI")
    assert result.matched is False
    assert "company:none" in result.reasons


def test_strong_drops_hr_intern_at_company():
    text = "Meesho is hiring an HR intern. Looking for someone to join people ops. Apply now."
    result = match_strong_company(text, "Meesho")
    assert result.matched is False
    assert "domain:none" in result.reasons or any(
        "proximity:no job-offer cue near intern+domain" in r for r in result.reasons
    )


def test_strong_allows_company_mention_via_poster_field():
    """Company-page authors often omit the name in the body."""
    body = "We're hiring an AI intern for our ML platform. Apply now."
    result = match_strong_company(
        body,
        "Postman",
        company_text=f"{body} Postman",
    )
    assert result.matched is True


def test_strong_allows_company_mention_via_poster_headline():
    """HR posts often name the employer only in the headline, not the body."""
    body = (
        "We're hiring a Data Science Intern to work on mobility datasets. "
        "Apply now."
    )
    result = match_strong_company(
        body,
        "Rapido",
        company_text=f"{body} Talent Acquisition at Rapido",
    )
    assert result.matched is True


def test_rapido_data_science_intern_fixture():
    text = (
        "We are Hiring | Data Science Intern\n"
        "Company: Rapido\n"
        "Location: Bangalore\n"
        "Rapido is looking for a passionate Data Science Intern. Apply Now"
    )
    assert match_strong_company(text, "Rapido").matched is True


def test_scattered_keywords_rejected():
    """Hiring in one paragraph, intern+AI in another, far apart."""
    text = (
        "Thanks everyone who came to the Zomato meetup. We are hiring across teams "
        "next quarter — details soon.\n\n"
        + ("Meanwhile community notes. " * 20)
        + "An intern demoed an AI side project yesterday. Cool stuff."
    )
    result = match_strong_company(text, "Zomato")
    assert result.matched is False


def test_loose_prefilter_sends_a_broader_candidate_net_than_strict():
    posts = [
        Post(poster="Riya", poster_headline="Recruiter at Acme", post_text="Acme is hiring a Software Intern. Apply now."),
        Post(poster="Riya", poster_headline="Recruiter at Acme", post_text="Acme is hiring an HR Intern. Apply now."),
        Post(poster="Student", post_text="I built for Acme and am looking for an internship."),
    ]
    strict = _strong_hits_for_company(
        posts, "Acme", HiringMatcher(mode="all"), {"weak": 0, "not_first_party": 0}, prefilter="strict",
    )
    loose = _strong_hits_for_company(
        posts, "Acme", HiringMatcher(mode="all"), {"weak": 0, "not_first_party": 0}, prefilter="loose",
    )
    assert len(loose) > len(strict)


def test_funnel_counts_are_monotonic():
    assert funnel_counts_are_monotonic({
        "seen": 36, "passed_prefilter": 12, "sent_to_gate": 10, "kept_by_gate": 4,
    })
    assert not funnel_counts_are_monotonic({
        "seen": 3, "passed_prefilter": 4, "sent_to_gate": 2, "kept_by_gate": 1,
    })


def test_strict_prefilter_without_gate_is_unchanged():
    post = Post(poster="Riya", poster_headline="Recruiter at Acme", post_text="Acme is hiring a Software Intern. Apply now.")
    default = _strong_hits_for_company(
        [post], "Acme", HiringMatcher(mode="all"), {"weak": 0, "not_first_party": 0},
    )
    explicit = _strong_hits_for_company(
        [post], "Acme", HiringMatcher(mode="all"), {"weak": 0, "not_first_party": 0}, prefilter="strict",
    )
    assert len(default) == len(explicit) == 1


def test_duplicate_posts_across_query_variations_are_collapsed_before_prefilter():
    first = Post(poster="Riya", post_text="Hiver is hiring an SDE Intern")
    duplicate = Post(poster="Riya", post_text="Hiver is  hiring an SDE Intern")
    other = Post(poster="Nia", post_text="Hiver is hiring a Backend Intern")
    assert dedupe_posts_by_identity([first, duplicate, other]) == [first, other]


def test_load_companies_preserves_order_and_skips_comments():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "companies.txt"
        path.write_text(
            "# header\n\nSarvam AI\nRazorpay\n# skip\nRazorpay\nGroww\n",
            encoding="utf-8",
        )
        assert load_companies(path) == ["Sarvam AI", "Razorpay", "Groww"]


def test_make_snippet_truncates():
    text = "word " * 100
    snippet = make_snippet(text, max_chars=40)
    assert len(snippet) <= 40
    assert snippet.endswith("…")


def test_write_results_groups_txt_by_company():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "out.txt"
        rows = [
            {
                "company": "Razorpay",
                "snippet": "hiring AI intern",
                "post_url": "https://example.com/a",
                "why_matched": "company:Razorpay; hiring:hiring; role:intern; domain:ai",
            },
            {
                "company": "Groww",
                "snippet": "ML internship open",
                "post_url": "https://example.com/b",
                "why_matched": "company:Groww; hiring:hiring; role:internship; domain:ml",
            },
            {
                "company": "Razorpay",
                "snippet": "second post",
                "post_url": "https://example.com/c",
                "why_matched": "company:Razorpay; hiring:hiring; role:intern; domain:software",
            },
        ]
        write_results(
            rows,
            path,
            fmt="txt",
            fields=("company", "snippet", "post_url", "why_matched"),
            title="Company posts",
            meta={"group_by": "company"},
        )
        body = path.read_text(encoding="utf-8")
        assert "=== Razorpay (2 posts) ===" in body
        assert "=== Groww (1 post) ===" in body
        assert body.index("=== Razorpay") < body.index("=== Groww")


if __name__ == "__main__":
    test_text_mentions_company_word_boundary()
    test_text_mentions_company_strips_suffix()
    test_strong_keeps_real_ai_intern_offer()
    test_strong_keeps_compact_hiring_line()
    test_zomato_cuisine_story_rejected()
    test_zomato_kolkata_newsletter_rejected()
    test_zomato_course_ad_rejected()
    test_strong_drops_without_company_mention()
    test_strong_drops_hr_intern_at_company()
    test_strong_allows_company_mention_via_poster_field()
    test_scattered_keywords_rejected()
    test_load_companies_preserves_order_and_skips_comments()
    test_make_snippet_truncates()
    test_write_results_groups_txt_by_company()
    print("ALL PASS")

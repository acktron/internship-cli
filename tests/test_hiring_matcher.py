"""Prove --match all requires hiring + internship + AI/ML domain signals."""

from internship_cli.filters import HiringMatcher


def test_match_all_keeps_ai_intern_hiring_post():
    matcher = HiringMatcher(mode="all")
    text = (
        "We're hiring an AI intern to work on LLM eval pipelines. "
        "Apply now — looking for students who can build."
    )
    result = matcher.match(text)
    assert result.matched is True
    assert "hiring:none" not in result.reasons
    assert "role:none" not in result.reasons
    assert "domain:none" not in result.reasons


def test_match_all_drops_hr_intern_post():
    """Hiring + internship alone is not enough without an AI/ML/software term."""
    matcher = HiringMatcher(mode="all")
    text = "We're hiring an HR intern. Looking for someone to join our people team. Apply now."
    result = matcher.match(text)
    assert result.matched is False
    assert "domain:none" in result.reasons
    assert "hiring:none" not in result.reasons
    assert "role:none" not in result.reasons


def test_match_all_drops_university_career_fair_post():
    """University / aggregator noise with hiring language but no AI/ML role."""
    matcher = HiringMatcher(mode="all")
    text = (
        "University Career Services is hiring student ambassadors and looking for "
        "interns across departments. Apply at the jobs fair this Friday."
    )
    result = matcher.match(text)
    assert result.matched is False
    assert "domain:none" in result.reasons


def test_match_all_drops_hiring_only_aggregator():
    matcher = HiringMatcher(mode="all")
    text = "We're hiring! 50 open roles this week — apply on our jobs board."
    result = matcher.match(text)
    assert result.matched is False
    assert "role:none" in result.reasons
    assert "domain:none" in result.reasons


def test_match_loose_keeps_hiring_plus_intern_without_domain():
    """loose = hiring AND (role OR domain); HR intern should pass here."""
    matcher = HiringMatcher(mode="loose")
    text = "We're hiring an HR intern. Apply now."
    assert matcher.match(text).matched is True


def test_match_all_rejects_html_false_positive_for_ml():
    matcher = HiringMatcher(mode="all")
    text = "We're hiring an intern for html work. Apply now."
    result = matcher.match(text)
    assert result.matched is False
    assert "domain:none" in result.reasons

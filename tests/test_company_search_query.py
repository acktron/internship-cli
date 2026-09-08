"""company multi-query helpers — structured boolean queries, not post bodies."""

from internship_cli.posts import (
    company_search_query,
    default_company_queries,
    is_post_body_query,
    sanitize_company_queries,
)


def test_default_role_query_is_boolean():
    q = company_search_query("Acme")
    assert q.startswith('"Acme"')
    assert "(intern OR internship)" in q
    assert "hiring" in q


def test_custom_role_is_quoted_into_structure():
    q = company_search_query("Acme", role_query="ML intern")
    assert '"Acme"' in q
    assert "(ML intern)" in q
    assert "(intern OR internship)" in q


def test_default_company_queries_are_structured_and_diverse():
    queries = default_company_queries("Sarvam AI", role_query="intern")
    assert len(queries) == 7
    assert queries[0] == '"Sarvam AI" intern'
    blob = " ".join(queries)
    assert all('"Sarvam AI"' in q for q in queries)
    assert any("(intern OR internship)" in q for q in queries)
    assert '"we\'re hiring"' in blob or "we're hiring" in blob
    assert "SDE" in blob and "backend" in blob
    assert "ML" in blob and "data" in blob
    # Not the old near-duplicate pair, and not a bare company-name search.
    assert '"Sarvam AI" hiring intern' not in queries
    assert '"Sarvam AI"' not in queries
    assert all(len(q) < 160 for q in queries)
    assert all(not is_post_body_query(q) for q in queries)


def test_default_company_queries_respects_cap():
    assert len(default_company_queries("Acme", max_queries=2)) == 2


def test_default_company_queries_can_reach_ceiling():
    assert len(default_company_queries("Acme", max_queries=7)) == 7
    assert len(default_company_queries("Acme", max_queries=15)) == 7


def test_rejects_scraped_post_body_as_query():
    body = (
        "We're hiring AI Forward Deployed Engineer Interns at Alumnx AI Labs. "
        "Apply now and send your resume. This is a long paragraph copied from "
        "a LinkedIn post and must never be pasted back into search."
    )
    assert is_post_body_query(body) is True
    assert is_post_body_query('"Acme" (intern OR internship) hiring') is False


def test_sanitize_drops_post_bodies_and_falls_back():
    body = "We're hiring. " * 40
    out = sanitize_company_queries(
        [body, '"Acme" intern hiring'],
        "Acme",
        max_queries=7,
    )
    assert body not in out
    assert any("Acme" in q for q in out)
    assert all(not is_post_body_query(q) for q in out)


def test_sanitize_empty_uses_defaults():
    out = sanitize_company_queries([], "Sarvam AI", max_queries=3)
    assert out == default_company_queries("Sarvam AI", max_queries=3)

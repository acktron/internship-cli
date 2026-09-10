"""Exact post_url extraction — timestamp / author anchor hrefs only."""

from pathlib import Path

from internship_cli.output import TXT_FIELDS, write_results
from internship_cli.posts import (
    Post,
    _absolute_post_url,
    _is_exact_post_url,
    _is_feed_fallback_url,
    _is_search_results_url,
    _normalise_activity_url,
    activity_permalink,
    extract_activity_id_from_href,
    extract_activity_id_from_html,
    permalink_from_href,
)

_VALID_ID = "7489706814855344128"
_OTHER_ID = "9998887776665554444"


def test_activity_id_from_timestamp_anchor_href():
    html = Path(__file__).parent.joinpath("fixtures/card_activity.html").read_text()
    assert extract_activity_id_from_html(html) == _VALID_ID
    assert activity_permalink(_VALID_ID).endswith(f"urn:li:activity:{_VALID_ID}/")


def test_data_urn_attr_is_not_used_without_anchor():
    html = (
        '<div data-chameleon-result-urn="urn:li:activity:5556667778889990000">'
        "<p>We're hiring software interns at Microsoft. Apply now.</p></div>"
    )
    assert extract_activity_id_from_html(html) == ""


def test_activity_id_from_sdui_cgsi_comment_tools_id():
    html = (
        '<div id="CgsIgoCyhP6+q5fQAQ-replaceableCommentTools'
        'Jb2_FnFvp1urZa02f0EhKzpAxThW1mVhO-zOdOofd_cFeedType_FLAGSHIP_SEARCH"></div>'
    )
    assert extract_activity_id_from_html(html) == ""


def test_sdui_proto_activity_url_is_not_a_real_permalink():
    fake = "https://www.linkedin.com/feed/update/urn:li:activity:15001071693790879744/"
    assert _is_exact_post_url(fake) is False
    assert activity_permalink("15001071693790879744") == ""


def test_wrong_digit_length_rejected():
    assert extract_activity_id_from_href(
        "https://www.linkedin.com/feed/update/urn:li:activity:99/"
    ) == ""
    assert extract_activity_id_from_href(
        "https://www.linkedin.com/feed/update/urn:li:activity:111222333444555/"
    ) == ""


def test_activity_id_from_reshare_timestamp_anchor():
    html = Path(__file__).parent.joinpath("fixtures/card_reshare.html").read_text()
    assert extract_activity_id_from_html(html) == _VALID_ID
    url, aid = permalink_from_href(
        "/feed/update/urn%3Ali%3Aactivity%3A7489706814855344128/?reshareId=1"
    )
    assert aid == _VALID_ID
    assert url == f"https://www.linkedin.com/feed/update/urn:li:activity:{_VALID_ID}/"


def test_exact_feed_update_url():
    url = f"https://www.linkedin.com/feed/update/urn:li:activity:{_VALID_ID}/"
    assert _is_exact_post_url(url) is True
    assert _is_feed_fallback_url(url) is False
    assert _is_search_results_url(url) is False


def test_exact_posts_slug_with_activity():
    url = f"https://www.linkedin.com/posts/sundeep-o_ai-interns-activity-{_VALID_ID}-AbCd"
    assert _is_exact_post_url(url) is True


def test_company_posts_feed_is_not_exact():
    url = "https://www.linkedin.com/company/liveuaejobs/posts/"
    assert _is_exact_post_url(url) is False
    assert _is_feed_fallback_url(url) is True


def test_search_results_url_is_never_a_permalink():
    url = (
        "https://www.linkedin.com/search/results/content/"
        "?keywords=%22Acme%22%20hiring%20intern"
    )
    assert _is_search_results_url(url) is True
    assert _is_exact_post_url(url) is False


def test_normalise_highlighted_update_urn():
    raw = (
        "https://www.linkedin.com/groups/13904524/"
        "?q=highlightedFeedForGroups"
        f"&highlightedUpdateUrn=urn%3Ali%3Aactivity%3A{_VALID_ID}"
    )
    assert _normalise_activity_url(raw) == (
        f"https://www.linkedin.com/feed/update/urn:li:activity:{_VALID_ID}/"
    )


def test_absolute_post_url_strips_query():
    href = f"/feed/update/urn:li:activity:{_VALID_ID}/?utm=1&trk=foo"
    assert _absolute_post_url(href) == (
        f"https://www.linkedin.com/feed/update/urn:li:activity:{_VALID_ID}/"
    )


def test_txt_output_blocks_use_permalink_not_search():
    out = Path("/tmp/_internship_posts_out.txt")
    posts = [
        Post(
            company="Sarvam AI",
            role="AI Intern",
            post_url=f"https://www.linkedin.com/feed/update/urn:li:activity:{_VALID_ID}/",
            activity_urn=f"urn:li:activity:{_VALID_ID}",
            why_matched="post_url:exact; hiring:hiring; role:intern; domain:ai",
        ),
        Post(
            company="LiveuaeJobs",
            role="",
            post_url="",
            why_matched="company:company-page",
        ),
    ]
    written = write_results(
        posts,
        out,
        fmt="txt",
        fields=TXT_FIELDS,
        title="LinkedIn hiring posts — AI intern",
        meta={"query": "AI intern", "match_mode": "all"},
    )
    body = written.read_text(encoding="utf-8")
    assert "post_url:exact" in body
    assert _VALID_ID in body
    assert "/search/results/content/" not in body
    assert "/company/liveuaejobs/posts/" not in body
    out.unlink(missing_ok=True)


def test_dedupe_key_prefers_permalink_then_poster_and_normalized_text():
    from internship_cli.posts import _post_dedupe_key

    post = Post(
        post_url=f"https://www.linkedin.com/feed/update/urn:li:activity:{_VALID_ID}/",
        activity_urn=f"urn:li:activity:{_VALID_ID}",
    )
    assert f"activity:{_VALID_ID}" in _post_dedupe_key(post)
    search = Post(
        post_url="https://www.linkedin.com/search/results/content/?keywords=x",
        activity_urn=f"urn:li:activity:{_OTHER_ID}",
        poster="Ada",
        post_text="Hiring   an intern",
    )
    same = Post(poster="Ada", post_text="Hiring an intern")
    assert _post_dedupe_key(search) == _post_dedupe_key(same)

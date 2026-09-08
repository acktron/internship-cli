"""Permalink fill via embed dialog (copy-link fallback)."""

from unittest.mock import MagicMock

from internship_cli.posts import (
    Post,
    _clear_post_url,
    _is_exact_post_url,
    _mark_resolved,
    _permalink_from_card_anchors,
    _snippet_tokens,
    activity_permalink,
    clean_permalink,
    extract_activity_id_from_html,
    permalink_from_embed_blob,
    canonical_post_permalink,
    _normalize_copied_url,
    resolve_post_links,
)

_VALID_ID = "7489706814855344128"
_VALID_URL = f"https://www.linkedin.com/feed/update/urn:li:activity:{_VALID_ID}/"
_SHARE_ID = "7388292563575123456"
_SHARE_URL = f"https://www.linkedin.com/feed/update/urn:li:share:{_SHARE_ID}/"


def test_permalink_from_embed_iframe_src():
    src = (
        "https://www.linkedin.com/embed/feed/update/"
        f"urn:li:share:{_SHARE_ID}?collapsed=1"
    )
    assert permalink_from_embed_blob(src) == _SHARE_URL


def test_permalink_from_embed_textarea():
    code = (
        f'<iframe src="https://www.linkedin.com/embed/feed/update/'
        f'urn:li:activity:{_VALID_ID}" height="399"></iframe>'
    )
    assert permalink_from_embed_blob(code) == _VALID_URL


def test_canonical_post_permalink_from_posts_slug():
    url = (
        "https://www.linkedin.com/posts/user_slug_salam-ugcPost-7500642672152526848-abc/"
        "?utm_source=share"
    )
    assert canonical_post_permalink(url) == (
        "https://www.linkedin.com/feed/update/urn:li:ugcPost:7500642672152526848/"
    )


def test_canonical_post_permalink_from_lnkd_in(monkeypatch):
    class _Resolver:
        def resolve(self, url: str) -> str:
            assert "lnkd.in" in url
            return (
                "https://www.linkedin.com/posts/user_slug_salam-ugcPost-7500642672152526848-abc/"
            )

    raw = "https://lnkd.in/p/gHZevthQ"
    assert _normalize_copied_url(raw, _Resolver()) == (
        "https://www.linkedin.com/feed/update/urn:li:ugcPost:7500642672152526848/"
    )


def test_clean_permalink_strips_tracking():
    raw = f"{_VALID_URL}?utm_source=share&trk=public_post"
    assert clean_permalink(raw) == _VALID_URL


def test_clean_permalink_rejects_search_url():
    assert clean_permalink(
        "https://www.linkedin.com/search/results/content/?keywords=hiring"
    ) == ""


def test_mark_resolved_sets_permalink_and_urn():
    post = Post(
        poster="Sundeep O.",
        post_url="",
        why_matched="hiring:hiring",
        _reasons=["hiring:hiring"],
    )
    _mark_resolved(post, _VALID_URL)
    assert post.post_url == _VALID_URL
    assert post.activity_urn == f"urn:li:activity:{_VALID_ID}"
    assert _is_exact_post_url(post.post_url)
    assert "post_url:exact" in post.why_matched


def test_mark_resolved_share_urn():
    post = Post(post_url="", _reasons=["hiring:hiring"])
    _mark_resolved(post, _SHARE_URL)
    assert post.activity_urn == f"urn:li:share:{_SHARE_ID}"
    assert _is_exact_post_url(post.post_url)


def test_mark_resolved_rejects_search_listing_url():
    post = Post(post_url="", _reasons=["hiring:hiring"])
    _mark_resolved(
        post,
        "https://www.linkedin.com/search/results/content/?keywords=hiring",
    )
    assert post.post_url == ""


def test_clear_post_url_does_not_emit_unresolved():
    post = Post(
        post_url="",
        why_matched="hiring:hiring",
        _reasons=["hiring:hiring"],
    )
    _clear_post_url(post)
    assert post.post_url == ""
    assert "post_url:" not in post.why_matched


def test_snippet_tokens_skips_boilerplate():
    tokens = _snippet_tokens(
        "We're hiring an AI Forward Deployed Engineer Intern at Alumnx",
        n=5,
    )
    assert "hiring" not in {t.lower() for t in tokens}
    assert any(t.lower() in {"forward", "deployed", "engineer", "alumnx", "ai"} for t in tokens)


def test_visible_text_is_not_an_activity_id_source():
    snippet = (
        "Microsoft is hiring a Software Engineering Intern. Apply now. "
        f"The number {_VALID_ID} is not a URN."
    )
    assert extract_activity_id_from_html(snippet) == ""


class _FakePage:
    url = "https://www.linkedin.com/search/results/content/?keywords=x"

    def __init__(self) -> None:
        self.context = MagicMock()

    def wait_for_timeout(self, _ms: int) -> None:
        return None

    def bring_to_front(self) -> None:
        return None


class _FakeCard:
    def evaluate(self, _js):
        return []

    def get_attribute(self, *_a, **_k):
        return None

    def scroll_into_view_if_needed(self, timeout=2000):
        return None


def test_urn_fast_path_skips_card_rescan(monkeypatch):
    post = Post(
        poster="Acme",
        post_url=activity_permalink(_VALID_ID),
        activity_urn=f"urn:li:activity:{_VALID_ID}",
        _reasons=["post_url:exact"],
        post_text="We're hiring an AI intern apply now",
    )
    calls: list[str] = []
    monkeypatch.setattr(
        "internship_cli.posts._find_card_for_post",
        lambda *a, **k: calls.append("found") or _FakeCard(),
    )
    tally = resolve_post_links(
        _FakePage(),
        [post],
        "https://www.linkedin.com/search/results/content/",
        resolve_delay=0,
        log=lambda _m: None,
    )
    assert calls == []
    assert tally["skipped_exact"] == 1
    assert tally["attempted"] == 0


def test_missing_url_resolved_from_embed_dialog(monkeypatch):
    post = Post(
        poster="Ada",
        post_url="",
        post_text="We're hiring an AI intern apply now at Acme",
        _reasons=["hiring:hiring"],
    )
    monkeypatch.setattr(
        "internship_cli.posts._find_card_for_post",
        lambda *_a, **_k: _FakeCard(),
    )
    monkeypatch.setattr(
        "internship_cli.posts._resolve_permalink_from_card",
        lambda *_a, **_k: (_SHARE_URL, "embed"),
    )
    tally = resolve_post_links(
        _FakePage(),
        [post],
        "https://www.linkedin.com/search/results/content/",
        resolve_delay=0,
        log=lambda _m: None,
    )
    assert tally["embedded"] == 1
    assert _SHARE_ID in post.post_url
    assert "post_url:exact" in post._reasons


def test_fake_proto_url_cleared_when_embed_and_copy_fail(monkeypatch):
    post = Post(
        poster="Ada",
        post_url="https://www.linkedin.com/feed/update/urn:li:activity:15001071693790879744/",
        post_text="We're hiring an AI intern apply now at Acme",
        _reasons=["hiring:hiring"],
    )
    monkeypatch.setattr("internship_cli.posts._find_card_for_post", lambda *_a, **_k: _FakeCard())
    monkeypatch.setattr(
        "internship_cli.posts._resolve_permalink_from_card",
        lambda *_a, **_k: ("", ""),
    )
    tally = resolve_post_links(
        _FakePage(),
        [post],
        "https://www.linkedin.com/search/results/content/",
        resolve_delay=0,
        log=lambda _m: None,
    )
    assert tally["failed"] == 1
    assert post.post_url == ""
    assert "15001071693790879744" not in post.post_url


def test_permalink_from_card_anchors_prefers_timestamp():
    class TimestampCard:
        def evaluate(self, _js):
            return [
                f"/feed/update/urn:li:activity:{_VALID_ID}/?trk=1",
            ]

    url, kind, aid = _permalink_from_card_anchors(TimestampCard())
    assert kind == "exact"
    assert aid == _VALID_ID
    assert url == _VALID_URL


def test_permalink_cache_skips_second_rescan(monkeypatch):
    a = Post(
        poster="Ada",
        post_url="",
        post_text="We're hiring an AI intern apply now",
        _reasons=["hiring:hiring"],
    )
    b = Post(
        poster="Ada",
        post_url="",
        post_text="We're hiring an AI intern apply now",
        _reasons=["hiring:hiring"],
    )
    scans = {"n": 0}

    def fake_find(*_a, **_k):
        scans["n"] += 1
        return _FakeCard()

    monkeypatch.setattr("internship_cli.posts._find_card_for_post", fake_find)
    monkeypatch.setattr(
        "internship_cli.posts._resolve_permalink_from_card",
        lambda *_a, **_k: (_VALID_URL, "embed"),
    )
    cache: dict[str, str] = {}
    resolve_post_links(
        _FakePage(), [a],
        "https://www.linkedin.com/search/results/content/",
        resolve_delay=0, log=lambda _m: None, cache=cache,
    )
    resolve_post_links(
        _FakePage(), [b],
        "https://www.linkedin.com/search/results/content/",
        resolve_delay=0, log=lambda _m: None, cache=cache,
    )
    assert scans["n"] == 1
    assert a.post_url == b.post_url
    assert _VALID_ID in b.post_url

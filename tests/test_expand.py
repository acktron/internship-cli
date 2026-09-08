"""See-more expansion before post-body reads."""

from internship_cli.posts import (
    _TEXT_SELECTORS,
    expanded_post_text,
    is_see_more_label,
    looks_truncated,
)


SHORT = "We're hiring an AI intern. …more"
FULL = (
    "We're hiring an AI intern. Apply now. Full details about the role, "
    "the team, and how to send your resume."
)


class _Node:
    def __init__(self, owner: "FakeCard") -> None:
        self.owner = owner

    def count(self) -> int:
        return 1

    @property
    def first(self) -> "_Node":
        return self

    def nth(self, _index: int) -> "_Node":
        return self

    def click(self, **_kwargs) -> None:
        self.owner.expanded = True

    def inner_text(self, timeout: int = 0) -> str:  # noqa: ARG002
        return self.owner.body

    def text_content(self, timeout: int = 0) -> str:  # noqa: ARG002
        return self.owner.body


class FakeCard:
    def __init__(self, *, has_toggle: bool = True) -> None:
        self.expanded = not has_toggle
        self.has_toggle = has_toggle
        self.js_clicks = 0

    @property
    def body(self) -> str:
        return FULL if self.expanded else SHORT

    def locator(self, selector: str) -> _Node:
        return _Node(self)

    def evaluate(self, _js, *_args):
        if self.has_toggle:
            self.expanded = True
            self.js_clicks += 1
            return 1
        return 0

    def inner_text(self, timeout: int = 0) -> str:  # noqa: ARG002
        return self.body


def test_see_more_labels():
    assert is_see_more_label("see more")
    assert is_see_more_label("…more")
    assert is_see_more_label("...more")
    assert is_see_more_label("See More")
    assert not is_see_more_label("see less")
    assert not is_see_more_label("We're hiring more interns")


def test_looks_truncated():
    assert looks_truncated(SHORT)
    assert not looks_truncated(FULL)


def test_expand_then_reread_full_text():
    card = FakeCard(has_toggle=True)
    assert looks_truncated(card.body)
    text = expanded_post_text(card)
    assert "Full details" in text
    assert not looks_truncated(text)
    assert card.expanded is True
    assert card.js_clicks == 1


def test_no_toggle_still_returns_body():
    card = FakeCard(has_toggle=False)
    text = expanded_post_text(card)
    assert "Full details" in text


def test_text_selectors_include_inline_show_more():
    assert ".feed-shared-inline-show-more-text" in _TEXT_SELECTORS

"""The Streamlit page renders (no model call until the button is clicked)."""

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


def test_app_renders_with_empty_tag():
    at = AppTest.from_file("../app.py", default_timeout=30).run()  # path is relative to this test file
    assert not at.exception
    assert at.button[0].disabled  # nothing to triage yet


def test_typing_a_note_enables_triage():
    at = AppTest.from_file("../app.py", default_timeout=30).run()
    at.text_area(key="note").input("Crushing chest pressure for 40 minutes.").run()
    assert not at.exception
    assert not at.button[0].disabled

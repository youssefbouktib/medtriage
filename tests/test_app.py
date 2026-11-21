"""The Streamlit page renders (no model call until the button is clicked)."""

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


def test_app_renders_and_loads_an_example():
    at = AppTest.from_file("../app.py", default_timeout=30).run()  # path is relative to this test file
    assert not at.exception
    assert at.title[0].value.startswith("🩺 MedTriage Agent")

    at.selectbox[0].select("g01").run()
    assert "chest pressure" in at.text_area[0].value
    assert not at.button[0].disabled

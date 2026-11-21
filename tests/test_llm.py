import pytest

from medtriage.llm import call_with_fallback, is_transient


class Overloaded(Exception):
    def __str__(self):
        return "503 UNAVAILABLE. This model is currently experiencing high demand."


class RemoteProtocolError(Exception):
    pass


def test_falls_back_to_next_model_when_overloaded():
    tried = []

    def call(model):
        tried.append(model)
        if model == "big":
            raise Overloaded()
        return "answer"

    assert call_with_fallback(call, ["big", "small"], sleep=lambda s: None) == ("small", "answer")
    assert tried == ["big", "small"]


def test_backs_off_only_after_a_full_pass():
    sleeps = []

    def call(model):
        raise Overloaded()

    with pytest.raises(Overloaded):
        call_with_fallback(call, ["a", "b"], attempts=5, sleep=sleeps.append)
    assert len(sleeps) == 2  # after attempts 2 and 4, not after every failure


def test_non_transient_error_is_raised_immediately():
    tried = []

    def call(model):
        tried.append(model)
        raise ValueError("400 INVALID_ARGUMENT")

    with pytest.raises(ValueError):
        call_with_fallback(call, ["a", "b"], sleep=lambda s: None)
    assert tried == ["a"]


def test_network_errors_are_transient():
    assert is_transient(RemoteProtocolError("Server disconnected"))
    assert not is_transient(KeyError("x"))

from medtriage.classifier import HFZeroShotClassifier


def test_classifier_scores_descriptions_and_returns_the_label():
    seen = {}

    def fake_pipe(text, candidate_labels, hypothesis_template, **kwargs):
        seen["candidates"] = candidate_labels
        return {"labels": ["the skin", "the heart"], "scores": [0.9, 0.1]}

    clf = HFZeroShotClassifier.__new__(HFZeroShotClassifier)  # skip loading the real model
    clf._pipe = fake_pipe
    clf.descriptions = {"dermatology": "the skin", "cardiology": "the heart"}

    assert clf.classify("A mole on the back.", ["cardiology", "dermatology"]) == ("dermatology", 0.9)
    assert seen["candidates"] == ["the heart", "the skin"]


def test_label_without_description_is_used_as_is():
    clf = HFZeroShotClassifier.__new__(HFZeroShotClassifier)
    clf._pipe = lambda text, candidate_labels, hypothesis_template, **kw: {"labels": candidate_labels, "scores": [1.0]}
    clf.descriptions = {}
    assert clf.classify("note", ["urology"]) == ("urology", 1.0)

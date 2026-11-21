from medtriage.red_flags import find_red_flags


def test_keyword_flag():
    assert "chest pain" in find_red_flags("Patient reports chest pain since this morning.")


def test_negated_keyword_is_ignored():
    assert find_red_flags("Denies chest pain or shortness of breath.") == []


def test_negation_only_applies_within_its_sentence():
    flags = find_red_flags("No fever. Now has chest tightness.")
    assert flags == ["chest pain"]


def test_vital_sign_thresholds():
    flags = find_red_flags("SpO2 86% on room air, BP 82/50, HR 140.")
    assert flags == ["hypoxia (SpO2 < 90%)", "hypotension (systolic < 90)", "abnormal heart rate"]


def test_normal_vitals_are_not_flagged():
    assert find_red_flags("BP 128/80, HR 72, SpO2 98%.") == []


def test_stroke_and_suicidality():
    flags = find_red_flags("Sudden slurred speech. He is suicidal with a plan.")
    assert flags == ["stroke signs", "suicidal ideation"]

import quality.crap as crap


def test_crap_score_equals_complexity_at_full_coverage():
    assert crap.crap_score(4, 1.0) == 4.0


def test_crap_score_increases_as_coverage_falls():
    assert crap.crap_score(4, 0.0) == 20.0


def test_function_coverage_counts_only_lines_in_the_function_span():
    function = {"lineno": 10, "endline": 12}
    file_coverage = {
        "executed_lines": [8, 10, 12],
        "missing_lines": [11, 14],
    }

    assert crap._function_coverage(function, file_coverage) == 2 / 3


def test_main_fails_when_a_function_reaches_the_threshold(monkeypatch):
    monkeypatch.setattr(
        crap,
        "_run_radon",
        lambda: [{
            "file": "tests/example.py",
            "name": "example",
            "complexity": 5,
            "lineno": 1,
            "endline": 1,
        }],
    )
    monkeypatch.setattr(crap, "_load_coverage", lambda: {})

    assert crap.main() == 1


def test_main_passes_when_functions_are_below_the_threshold(monkeypatch):
    monkeypatch.setattr(
        crap,
        "_run_radon",
        lambda: [{
            "file": "tests/example.py",
            "name": "example",
            "complexity": 4,
            "lineno": 1,
            "endline": 1,
        }],
    )
    monkeypatch.setattr(crap, "_load_coverage", lambda: {})

    assert crap.main() == 0
import scripts.run_qa as run_qa


def test_main_runs_qa_steps_from_the_project_root(monkeypatch):
    calls = []

    def record_run(command, cwd, check):
        calls.append((command, cwd, check))

    monkeypatch.setattr(run_qa.subprocess, "run", record_run)

    assert run_qa.main() == 0
    assert tuple((call[0][-1], call[1], call[2]) for call in calls) == (
        ("pytest", run_qa.PROJECT_ROOT, True),
        ("coverage.json", run_qa.PROJECT_ROOT, True),
        ("quality/crap.py", run_qa.PROJECT_ROOT, True),
    )
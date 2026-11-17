from io import BytesIO
from pathlib import Path

import pytest
from pypdf import PdfWriter
from streamlit.testing.v1 import AppTest

from app.streamlit_app import job_title_for_display, resume_text
from hiremind import agent, tools
from hiremind.schemas import Job, Judgement, Profile, ScoreBreakdown


def test_resume_text_validates_inputs():
    assert resume_text("resume.txt", b"\xef\xbb\xbfHello") == "Hello"
    with pytest.raises(ValueError, match="no extractable text"):
        resume_text("empty.txt", b"  ")
    with pytest.raises(ValueError, match="PDF or text"):
        resume_text("resume.docx", b"text")
    pdf = BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.write(pdf)
    with pytest.raises(ValueError, match="no extractable text"):
        resume_text("scan.pdf", pdf.getvalue())


def test_inconsistent_senior_title_is_explained():
    job = Job("1", "Senior Software Engineer", "We are looking for Junior Software Engineers.", "IT")
    assert "posting describes a junior role" in job_title_for_display(job)


def test_app_runs_matching_and_cover_letter(monkeypatch):
    import streamlit as st

    class Upload:
        name = "resume.txt"
        def getvalue(self):
            return b"Python developer"

    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: Upload())
    result = agent.Result(Profile("INFORMATION-TECHNOLOGY", skills=["Python"]),
                          [Judgement("1", 8, "Good match")])
    def run(text, progress=None, preferences=None, in_depth=False):
        progress({"phase": "parsing"})
        progress({"phase": "searching", "round": 1})
        progress({"phase": "scoring", "round": 1, "completed": 2, "total": 3, "remaining_seconds": 12})
        progress({"phase": "done"})
        return result
    monkeypatch.setattr(agent, "run", run)
    monkeypatch.setattr(tools, "get_job_details", lambda job_id: Job(job_id, "Developer", "Build software", "INFORMATION-TECHNOLOGY"))
    def cover_letter(profile, job_id, progress=None):
        progress({"phase": "drafting", "attempt": 1, "total": 2})
        progress({"phase": "checking", "attempt": 1, "total": 2})
        progress({"phase": "done", "attempt": 1, "total": 2})
        return "Dear Hiring Manager"
    monkeypatch.setattr(agent, "cover_letter", cover_letter)
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app/streamlit_app.py").run()
    assert not app.exception
    app.button[0].click().run()
    assert not app.exception
    assert any("Python" in item.value for item in app.markdown)
    assert app.expander[-1].label == "Developer — 8/10"
    app.button[1].click().run()
    assert not app.exception
    assert app.text_area[0].value == "Dear Hiring Manager"
    assert app.progress[0].value == 100


def test_preferences_and_in_depth_are_passed_and_breakdown_is_shown(monkeypatch):
    import streamlit as st
    class Upload:
        name = "resume.txt"
        def getvalue(self):
            return b"Python developer"
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: Upload())
    calls = []
    breakdown = ScoreBreakdown(requirements=8, experience=7, preferences=None, qualification=7.7,
                               coverage=.9, notes=["Salary unavailable"], mode="in-depth")
    def run(text, progress=None, preferences=None, in_depth=False):
        calls.append((preferences, in_depth))
        return agent.Result(Profile("IT"), [Judgement("1", 7.7, "Salary unavailable", breakdown)])
    monkeypatch.setattr(agent, "run", run)
    monkeypatch.setattr(tools, "get_job_details", lambda job_id: Job(job_id, "Developer", "", "IT"))
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app/streamlit_app.py").run()
    app.toggle[0].set_value(True)
    app.text_input[0].set_value("Boston")
    app.number_input[0].set_value(100000)
    app.button[0].click().run()
    assert not app.exception
    assert calls[0][1] is True and calls[0][0].location == "Boston" and calls[0][0].min_salary == 100000
    assert any("Requirements (60%)" in x.value and "Unknown" in x.value for x in app.markdown)

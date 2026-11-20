from io import BytesIO
from pathlib import Path

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
from streamlit.testing.v1 import AppTest

from app.streamlit_app import job_title_for_display, resume_text
from hiremind import agent, tools
from hiremind.schemas import Job, Judgement, Profile, ScoreBreakdown


def text_pdf(pages, font_name="/Helvetica"):
    writer = PdfWriter()
    name = NameObject
    font = DictionaryObject({name("/Type"): name("/Font"), name("/Subtype"): name("/Type1"),
                             name("/BaseFont"): name(font_name)})
    for lines in pages:
        page = writer.add_blank_page(width=600, height=800)
        page[name("/Resources")] = DictionaryObject({name("/Font"): DictionaryObject({name("/F1"): font})})
        stream = DecodedStreamObject()
        stream.set_data("\n".join(f"BT /F1 10 Tf {x} {y} Td ({line}) Tj ET" for x, y, line in lines).encode())
        page[name("/Contents")] = stream
    pdf = BytesIO()
    writer.write(pdf)
    return pdf.getvalue()


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
    with pytest.raises(ValueError, match="could not be read"):
        resume_text("broken.pdf", b"not a pdf")
    with pytest.raises(ValueError, match="UTF-8"):
        resume_text("legacy.txt", b"\xff")
    protected = BytesIO()
    writer.encrypt("fixture-password")
    writer.write(protected)
    with pytest.raises(ValueError, match="password protected"):
        resume_text("locked.pdf", protected.getvalue())


def test_inconsistent_senior_title_is_explained():
    job = Job("1", "Senior Software Engineer", "We are looking for Junior Software Engineers.", "IT")
    assert "posting describes a junior role" in job_title_for_display(job)


def test_pdf_upload_reads_both_pages_in_order():
    pdf = text_pdf([[(12, 30, "Alex Morgan")], [(12, 30, "Second page project")]])
    assert resume_text("resume.pdf", pdf).splitlines() == ["Alex Morgan", "Second page project"]


@pytest.mark.parametrize("sidebar_heading", ["SKILLS", "2020 to 2024"])
@pytest.mark.parametrize("font_name", ["/Courier", "/Helvetica"])
def test_pdf_upload_groups_clear_sidebar_but_preserves_date_column(sidebar_heading, font_name):
    lines = [(30, 760, "Alex Morgan"), (30, 720, "EXPERIENCE"), (330, 720, sidebar_heading),
             (30, 700, "Reporting Analyst"), (330, 700, "Python"),
             (30, 680, "Built weekly reports"), (330, 680, "SQL"),
             (30, 660, "EDUCATION"), (30, 640, "BS Statistics")]
    actual = resume_text("two-column.pdf", text_pdf([lines], font_name)).splitlines()
    if sidebar_heading == "SKILLS":
        assert actual == ["Alex Morgan", "EXPERIENCE", "Reporting Analyst", "Built weekly reports",
                          "EDUCATION", "BS Statistics", "SKILLS", "Python", "SQL"]
    else:
        assert actual == ["Alex Morgan", "EXPERIENCE 2020 to 2024", "Reporting Analyst Python",
                          "Built weekly reports SQL", "EDUCATION", "BS Statistics"]


def test_pdf_upload_preserves_full_width_section_after_sidebar():
    project = "Built a dashboard for tracking missing invoices and shared it with accounting."
    first = [(30, 760, "Alex Morgan"), (30, 720, "PROFESSIONAL EXPERIENCE"), (330, 720, "TECHNICAL SKILLS"),
             (30, 700, "Reporting Analyst"), (332, 700, "Python"),
             (30, 680, "Built weekly reports"), (330, 680, "SQL"),
             (30, 600, "SELECTED PROJECTS"), (30, 580, project)]
    second = [(30, 720, "CONTINUING EDUCATION"), (30, 700, "Completed a workshop")]
    text = resume_text("mixed-layout.pdf", text_pdf([first, second], "/Courier"))
    assert [line for line in text.splitlines() if line] == [
        "Alex Morgan", "PROFESSIONAL EXPERIENCE", "Reporting Analyst", "Built weekly reports",
        "TECHNICAL SKILLS", "Python", "SQL", "SELECTED PROJECTS", project,
        "CONTINUING EDUCATION", "Completed a workshop",
    ]


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
    labels = [panel.label for panel in app.expander]
    assert labels.index("Developer — 8/10") < labels.index("Search steps")
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

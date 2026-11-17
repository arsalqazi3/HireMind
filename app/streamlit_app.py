"""Local Streamlit interface for the HireMind agent."""
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from pypdf import PdfReader

from hiremind import agent, tools
from hiremind.schemas import Preferences


def resume_text(name: str, content: bytes) -> str:
    """Extract text without writing the uploaded resume to disk."""
    suffix = Path(name).suffix.lower()
    if suffix == ".txt":
        text = content.decode("utf-8-sig")
    elif suffix == ".pdf":
        text = "\n".join(page.extract_text() or "" for page in PdfReader(BytesIO(content)).pages)
    else:
        raise ValueError("Resume must be a PDF or text file")
    if not text.strip():
        raise ValueError("Resume contains no extractable text")
    return text


def job_title_for_display(job) -> str:
    """Explain a misleading senior dataset title when the posting says junior."""
    if (re.search(r"\b(?:senior|sr\.?)\b", job.title, re.I) and
            re.search(r"\b(?:seeking|looking for|hiring)\b[^.\n]{0,80}\b(?:junior|entry.level)\b", job.description[:1500], re.I)):
        return f"{job.title} (posting describes a junior role)"
    return job.title


def main() -> None:
    st.set_page_config(page_title="HireMind", page_icon="◈", layout="wide", initial_sidebar_state="collapsed")
    css = (Path(__file__).parent / "styles.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    st.markdown('''<div class="hm-header">
      <div class="hm-brand">HIREMIND <span>// AI Career Engine</span></div>
      <nav class="hm-nav" aria-label="Workspace sections">
        <a href="#workspace" target="_self">Workspace</a><a href="#job-matches" target="_self">Job Matcher</a><a href="#cover-letter" target="_self">Cover Letter</a>
      </nav><span class="hm-tag">Local workspace</span></div>
      <div class="hm-hero" id="workspace"><span class="hm-tag">Your career workspace</span>
      <h1>Find work that fits.</h1><p>Bring your experience into focus. Explore job matches and prepare a cover letter, all on your own device.</p></div>''', unsafe_allow_html=True)

    left, center, right = st.columns([0.87, 1.5, 1.1], gap="large")
    with left:
        with st.container(border=True):
            st.markdown('<div class="hm-label">01 / Your resume</div>', unsafe_allow_html=True)
            st.subheader("Start with your story")
            st.caption("Upload a PDF or text resume to find roles based on your experience.")
            uploaded = st.file_uploader("Upload your resume", type=["pdf", "txt"])
            in_depth = st.toggle("In-depth review", value=False,
                                 help="Use separate model checks for requirements, experience and preferences when enough information is available. This is slower.")
            with st.expander("Optional job preferences"):
                location = st.text_input("Preferred city or location", help="A simple text match, not a commute or eligibility check.")
                arrangement = st.selectbox("Preferred work arrangement", ["No preference", "Remote", "Hybrid", "Onsite"])
                min_salary = st.number_input("Minimum annual salary (optional)", min_value=0, value=0, step=1000,
                                             help="Salary is not structured in the current job data and cannot be scored yet.")
                benefits = st.text_input("Desired benefits (optional)", help="Benefits are not structured in the current job data and cannot be scored yet.")
                st.caption("Salary and benefits are not currently scored. Unknown details remain unscored.")
            preferences = Preferences(location=location.strip() or None,
                                      work_mode=None if arrangement == "No preference" else arrangement.lower(),
                                      min_salary=float(min_salary) if min_salary else None,
                                      benefits=benefits.strip() or None)
            search_clicked = st.button("Find matching jobs", type="primary", use_container_width=True, disabled=not uploaded)
        with st.container(border=True):
            st.markdown('<div class="hm-label">How matching works</div>', unsafe_allow_html=True)
            st.markdown("**60%** job requirements  \n**30%** experience and seniority  \n**10%** your preferences")
            st.caption("Unavailable details are excluded. Scores are provisional, not probabilities.")

    upload_key = sha256(uploaded.getvalue()).hexdigest() if uploaded else None
    if st.session_state.get("upload_key") != upload_key:
        for key in ("result", "letter", "letter_job", "letter_name", "applicant_name"):
            st.session_state.pop(key, None)
        st.session_state.upload_key = upload_key

    with center:
        st.markdown('<div id="job-matches"></div><div class="hm-label">02 / Your matches</div>', unsafe_allow_html=True)
        st.subheader("A closer look at your matches")
        st.caption("Review the fit, the evidence and the full job description before applying.")
        status = st.empty()
        bar_slot = st.empty()
        if uploaded and search_clicked:
            for key in ("result", "letter", "letter_job", "letter_name", "applicant_name"):
                st.session_state.pop(key, None)
            try:
                text = resume_text(uploaded.name, uploaded.getvalue())
                def show_progress(event):
                    phase = event["phase"]
                    if phase == "parsing":
                        status.info("Reading your resume and loading the model. The first run may take longer.")
                    elif phase == "searching":
                        bar_slot.empty()
                        status.info(f"Searching for jobs (round {event['round']}).")
                    elif phase == "scoring":
                        done, total = event["completed"], event["total"]
                        bar_slot.progress(done / total if total else 0, text=f"Reviewed {done} of {total} candidates in round {event['round']}")
                        remaining = event["remaining_seconds"]
                        if remaining is None:
                            status.info("Estimating review time after the first two candidates. Additional search rounds may be needed.")
                        else:
                            minutes, seconds = divmod(round(remaining), 60)
                            status.info(f"Estimated time for this round: {minutes}m {seconds:02d}s remaining. Additional search rounds may be needed.")
                    elif phase == "done":
                        bar_slot.empty()
                        status.empty()
                st.session_state.result = agent.run(text, progress=show_progress, preferences=preferences, in_depth=in_depth)
            except Exception as exc:
                bar_slot.empty()
                status.empty()
                st.error(f"Could not match this resume: {exc}")

        result = st.session_state.get("result")
        jobs = {}
        if result is None:
            with st.container(border=True):
                st.markdown('<div class="hm-empty"><strong>Your matches will appear here</strong>Upload your resume and start a search to see ranked jobs and the evidence behind each score.</div>', unsafe_allow_html=True)
        else:
            with st.expander("Search steps"):
                if not result.steps:
                    st.write("No search steps were recorded.")
                for step in result.steps:
                    st.write(f"**{step.action}**", step.args)
                    if step.feedback:
                        st.write(step.feedback)
                st.caption(f"Critic retries: {result.retries}")
            st.caption("Scores use 60% requirements, 30% experience and 10% preferences when available. Unknown components are excluded and weights rescaled. Results must meet the current 6/10 requirements and qualification cutoffs. Changes to preferences apply on the next search.")
            if result.profile.early_career:
                st.caption("Roles described as senior are excluded for this early-career search unless the posting describes a junior role. Titles and descriptions can be inaccurate.")
            if not result.top_jobs:
                st.warning("No jobs met the current matching threshold. Try a more detailed resume.")
            elif len(result.top_jobs) < 5:
                st.info(f"Found {len(result.top_jobs)} qualifying job(s) after {result.retries + 1} search round(s). Other candidates did not pass the checks or could not be scored.")
            for judgement in result.top_jobs:
                job = tools.get_job_details(judgement.job_id)
                jobs[job.job_id] = job
                with st.expander(f"{job_title_for_display(job)} — {judgement.score:g}/10"):
                    if judgement.breakdown:
                        b = judgement.breakdown
                        def display(value):
                            return f"{value:g}/10" if value is not None else "Unknown"
                        st.write(f"**Requirements (60%):** {display(b.requirements)}  |  **Experience (30%):** {display(b.experience)}  |  **Preferences (10%):** {display(b.preferences)}")
                        st.write(f"**Qualification match:** {display(b.qualification)}  |  **Score coverage:** {b.coverage:.0%}  |  **Review:** {b.mode}")
                        if b.preference_coverage:
                            st.caption(f"Preference information assessed: {b.preference_coverage:.0%} of the preferences you specified.")
                        st.caption("Unknown qualifications or preferences make a score partial. Review the evidence and original posting.")
                    st.write(judgement.reason)
                    st.write(f"**Category:** {job.category}  |  **Level:** {job.level or 'Not specified'}  |  **Location:** {job.location or 'Not specified'}")
                    st.write(job.description)

    with right:
        st.markdown('<div id="profile"></div><div class="hm-label">03 / Your profile</div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.subheader("Your experience")
            if result is None:
                st.caption("Your extracted profile will appear here after a search. You can check your name before writing a letter.")
            else:
                profile = result.profile
                name = st.text_input("Applicant name (check or edit)", value=profile.name or "", key="applicant_name")
                profile.name = name.strip() or None
                if st.session_state.get("letter_name") != profile.name:
                    st.session_state.pop("letter", None)
                    st.session_state.pop("letter_job", None)
                st.write(f"**Category:** {profile.category}")
                st.write(f"**Skills:** {', '.join(profile.skills) if profile.skills else 'None identified'}")
                if profile.years_experience is not None:
                    st.write(f"**Years of experience:** {profile.years_experience:g}")
                for label, values in (("Education", profile.education), ("Roles", profile.roles), ("Projects", profile.projects)):
                    if values:
                        st.write(f"**{label}:** {', '.join(values)}")
        st.markdown('<div id="cover-letter"></div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown('<div class="hm-label">Cover letter studio</div>', unsafe_allow_html=True)
            st.subheader("Make it personal")
            if not jobs:
                st.caption("Choose a qualifying job after your search to create an editable draft from your resume.")
            else:
                selected = st.selectbox("Choose a job for your cover letter", list(jobs),
                                        format_func=lambda job_id: f"{job_title_for_display(jobs[job_id])} ({job_id})")
                if st.button("Write cover letter", use_container_width=True):
                    st.session_state.pop("letter", None)
                    st.session_state.pop("letter_job", None)
                    try:
                        letter_status, letter_bar = st.empty(), st.empty()
                        def letter_progress(event):
                            attempt, total, phase = event["attempt"], event["total"], event["phase"]
                            if phase == "drafting":
                                letter_bar.progress((attempt - 1) / total, text=f"Draft attempt {attempt} of up to {total}")
                                letter_status.info("Building a draft from your profile and the job details.")
                            elif phase == "checking":
                                letter_status.info(f"Checking draft {attempt} for unsupported claims.")
                            else:
                                letter_bar.progress(1.0, text="Cover letter ready")
                                letter_status.empty()
                        st.session_state.letter = agent.cover_letter(profile, selected, progress=letter_progress)
                        st.session_state.letter_job = selected
                        st.session_state.letter_name = profile.name
                    except Exception as exc:
                        letter_bar.empty()
                        letter_status.empty()
                        st.error(f"Could not write a cover letter: {exc}")
                if st.session_state.get("letter_job") == selected and "letter" in st.session_state:
                    st.caption("Replace the bracketed contact details and review the draft before using it.")
                    st.text_area("Review and edit before sending", st.session_state.letter, height=300)
        with st.container(border=True):
            st.markdown('<div class="hm-label">Made for your review</div>', unsafe_allow_html=True)
            st.caption("Matching happens locally. Confirm job requirements, location, licensing and current availability before applying.")

    st.markdown('<div class="hm-footer">HIREMIND // LOCAL CAREER WORKSPACE</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()

"""Local Streamlit interface for the HireMind agent."""
from hashlib import sha256
from html import escape
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from hiremind import agent, tools
from hiremind.roles.parser import resume_text
from hiremind.schemas import Preferences


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
    st.markdown('''<header class="hm-header">
      <div class="hm-brand"><span class="hm-brand-mark" aria-hidden="true">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">
          <rect x="4" y="6" width="16" height="14" rx="2"/><path d="M9 6V4h6v2M4 11l8 2 8-2"/>
        </svg></span>HireMind</div>
      <nav class="hm-nav" aria-label="Workspace sections">
        <a href="#workspace" target="_self">Résumé</a><a href="#job-matches" target="_self">Job matches</a>
        <a href="#cover-letter" target="_self">Cover letter</a>
      </nav></header>
      <section class="hm-hero" id="workspace">
        <h1>Job matching workspace</h1>
        <p>Find roles that fit your résumé, review the evidence, and draft a cover letter for the job you choose.</p>
      </section>''', unsafe_allow_html=True)

    left, center, right = st.columns([0.9, 1.5, 1.1], gap="medium")
    with left:
        with st.container(border=True, key="upload-panel"):
            st.header("Your résumé")
            st.caption("Upload a text-based PDF or UTF-8 TXT resume to find roles based on your experience.")
            uploaded = st.file_uploader("Upload your resume", type=["pdf", "txt"])
            search_clicked = st.button("Find matching jobs", type="primary", use_container_width=True, disabled=not uploaded)
            if not uploaded:
                st.caption("Upload a résumé to start matching.")
            in_depth = st.toggle("In-depth review", value=False,
                                 help="Use separate model checks for requirements, experience and preferences when enough information is available. This is slower.")
            with st.expander("Job preferences"):
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
        with st.container(border=True, key="method-panel"):
            st.header("How scoring works")
            st.markdown('''<dl class="hm-weights">
              <div><dt>Job requirements</dt><dd>60%</dd></div>
              <div><dt>Experience</dt><dd>30%</dd></div>
              <div><dt>Your preferences</dt><dd>10%</dd></div>
            </dl>''', unsafe_allow_html=True)
            st.caption("Unavailable details are excluded. Scores are provisional, not probabilities.")

    upload_key = sha256(uploaded.getvalue()).hexdigest() if uploaded else None
    if st.session_state.get("upload_key") != upload_key:
        for key in ("result", "letter", "letter_job", "letter_name", "applicant_name"):
            st.session_state.pop(key, None)
        st.session_state.upload_key = upload_key

    with center, st.container(border=True, key="match-panel"):
        st.header("Job matches", anchor="job-matches")
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
            st.markdown('<div class="hm-empty"><strong>Your matches will appear here</strong>Upload your résumé and select Find matching jobs to see up to five ranked results.</div>', unsafe_allow_html=True)
        else:
            if result.top_jobs:
                count = len(result.top_jobs)
                top_score = result.top_jobs[0].score
                st.markdown(
                    f'<div class="hm-summary"><strong>{count} ranked {"match" if count == 1 else "matches"}</strong>'
                    f'<span class="hm-score">Top score {top_score:g}/10</span></div>',
                    unsafe_allow_html=True,
                )
            st.caption("Expand a result to see its score and evidence. Unknown components are excluded from the score.")
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
                        components = (
                            ("Requirements (60%)", b.requirements),
                            ("Experience (30%)", b.experience),
                            ("Preferences (10%)", b.preferences),
                        )
                        breakdown = "".join(
                            f"<div><dt>{label}</dt><dd>{f'{value:g}/10' if value is not None else 'Unknown'}</dd></div>"
                            for label, value in components
                        )
                        st.markdown(f'<dl class="hm-breakdown">{breakdown}</dl>', unsafe_allow_html=True)
                        qualification = f"{b.qualification:g}/10" if b.qualification is not None else "Unknown"
                        st.caption(f"Qualification match: {qualification} · Score coverage: {b.coverage:.0%} · Review: {b.mode}")
                        if b.preference_coverage:
                            st.caption(f"Preference information assessed: {b.preference_coverage:.0%} of the preferences you specified.")
                        st.caption("Unknown qualifications or preferences make a score partial. Review the evidence and original posting.")
                    st.write(judgement.reason)
                    st.write(f"**Category:** {job.category}  |  **Level:** {job.level or 'Not specified'}  |  **Location:** {job.location or 'Not specified'}")
                    st.markdown(
                        '<details class="hm-posting"><summary>Full job description</summary>'
                        f'<div>{escape(job.description)}</div></details>',
                        unsafe_allow_html=True,
                    )
            with st.expander("Search steps"):
                if not result.steps:
                    st.write("No search steps were recorded.")
                for step in result.steps:
                    st.write(f"**{step.action}**", step.args)
                    if step.feedback:
                        st.write(step.feedback)
                st.caption(f"Critic retries: {result.retries}")

    with right:
        with st.container(border=True, key="profile-panel"):
            st.header("Extracted profile")
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
        with st.container(border=True, key="letter-panel"):
            st.header("Cover letter studio")
            if not jobs:
                st.caption("Choose a qualifying job after your search to create an editable draft from your resume.")
            else:
                selected = st.selectbox("Choose a job for your cover letter", list(jobs),
                                        format_func=lambda job_id: f"{job_title_for_display(jobs[job_id])} ({job_id})")
                selected_title = job_title_for_display(jobs[selected])
                if len(selected_title) > 38:
                    st.caption(f"Selected: {selected_title}")
                if st.button("Write cover letter", type="primary", use_container_width=True):
                    st.session_state.pop("letter", None)
                    st.session_state.pop("letter_job", None)
                    try:
                        letter_status, letter_bar = st.empty(), st.empty()
                        def letter_progress(event):
                            attempt, total, phase = event["attempt"], event["total"], event["phase"]
                            if phase == "drafting":
                                letter_bar.progress((attempt - 1) / total, text="Drafting cover letter")
                                letter_status.info("Writing a draft around your experience and the selected posting.")
                            elif phase == "checking":
                                letter_status.info("Checking the draft’s claims against the résumé and posting.")
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
    st.markdown('<footer class="hm-footer">HireMind · Matching runs on this device. Check the original posting for current availability.</footer>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()

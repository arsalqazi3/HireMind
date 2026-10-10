"""HireMind demo app.

Run as: streamlit run app/streamlit_app.py

Upload a resume, see the parsed profile, watch the agent's steps, get five
ranked jobs with reasons, and write a cover letter for the one you pick.
"""
import streamlit as st

st.title("HireMind")
st.caption("A self-correcting job-matching agent")

uploaded = st.file_uploader("Upload your resume", type=["pdf", "txt"])
if uploaded:
    st.info("The agent is not wired in yet.")

import streamlit as st
import pandas as pd
from utils.helpers import analyze_resume, get_job_suggestions

st.title("AI Resume Analyzer & Job Suggester")

uploaded_file = st.file_uploader("Choose a resume file", type=["txt", "pdf", "docx"])

if uploaded_file is not None:
    try:
        resume_text = uploaded_file.read().decode("utf-8")  # Handle different encodings
        analysis_results = analyze_resume(resume_text)

        st.subheader("Resume Analysis")
        col1, col2, col3 = st.columns(3)
        col1.metric("Keywords Found", analysis_results["keyword_count"])
        col2.metric("Skills Identified", len(analysis_results["skills"]))
        col3.metric("Experience Years", analysis_results["experience"])

        st.subheader("Skills Breakdown")
        skill_counts = pd.DataFrame(
            {
                "Skill": analysis_results["skills"],
                "Count": [1] * len(analysis_results["skills"]),
            }
        )
        skill_counts = skill_counts.groupby("Skill").size().reset_index(name="Count")
        st.bar_chart(skill_counts.set_index("Skill"))

        st.subheader("Job Suggestions")
        suggested_jobs = get_job_suggestions(
            analysis_results["skills"], analysis_results["experience"]
        )
        if suggested_jobs:
            st.table(suggested_jobs)
        else:
            st.write("No job suggestions found based on your resume.")

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please upload a resume file.")

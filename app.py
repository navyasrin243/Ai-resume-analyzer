
import os
import tempfile
import streamlit as st

from analyzer import analyze
from ingest import ingest_resume

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide",
)

st.title("AI Resume Analyzer")
st.write("Upload your resume and compare it with a job description.")

resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
jd_text = st.text_area("Paste Job Description", height=220)

if st.button("Analyze Resume", type="primary"):
    if resume_file is None:
        st.error("Please upload a PDF resume.")
    elif not jd_text.strip():
        st.error("Please enter the job description.")
    elif not os.getenv("GROQ_API_KEY"):
        st.error("GROQ_API_KEY is missing. Configure it in your .env file.")
    else:
        temp_path = None
        try:
            with st.spinner("Reading resume and creating embeddings..."):
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".pdf"
                ) as temp:
                    temp.write(resume_file.getvalue())
                    temp_path = temp.name

                vectorstore, resume_text, num_chunks = ingest_resume(temp_path)

            with st.spinner("Analyzing skills and generating results..."):
                result = analyze(
                    jd_text=jd_text,
                    resume_text=resume_text,
                    vectorstore=vectorstore,
                )

            st.success("Analysis completed!")

            st.metric("Resume Match Score", f"{result.match_score}/100")
            st.progress(max(0, min(100, result.match_score)) / 100)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.subheader("Matched Skills")
                st.write(result.matched_skills or ["None identified"])
            with col2:
                st.subheader("Partial Skills")
                st.write(result.partial_skills or ["None identified"])
            with col3:
                st.subheader("Missing Skills")
                st.write(result.missing_skills or ["None identified"])

            st.subheader("Explanation")
            st.write(result.explanation)

            st.subheader("Resume Improvement Tips")
            for tip in result.resume_tips:
                st.markdown(f"- {tip}")

            st.subheader("Resume Evidence")
            if result.evidence:
                for skill, snippet in result.evidence.items():
                    with st.expander(skill):
                        st.write(snippet)
            else:
                st.info("No supporting snippets were retrieved.")

            st.subheader("Cover Letter")
            st.text_area(
                "Generated cover letter",
                value=result.cover_letter,
                height=250,
            )
            st.download_button(
                "Download Cover Letter",
                data=result.cover_letter,
                file_name="cover_letter.txt",
                mime="text/plain",
            )

            st.subheader("Interview Questions")
            for i, question in enumerate(result.interview_questions, 1):
                st.markdown(f"{i}. {question}")

            st.caption(f"Resume chunks processed: {num_chunks}")

        except Exception as e:
            st.error("Analysis failed. Check the error below.")
            st.exception(e)

        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

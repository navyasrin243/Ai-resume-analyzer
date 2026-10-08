import os
import tempfile
import streamlit as st
from ingest import ingest_resume
from analyzer import analyze

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="🚀",
    layout="wide",
)

st.markdown("""
<style>
.main-title{font-size:2.2rem;font-weight:bold;
            text-align:center;color:#1f77b4;}
</style>""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🚀 AI Resume Analyzer</p>',
            unsafe_allow_html=True)
st.markdown("##### Upload Resume + Paste JD → Instant AI Analysis!")
st.divider()

# ── Session state (per user, never shared) ────────────────────────────────────
for key in ("file_key", "vectorstore", "resume_text", "chunks"):
    if key not in st.session_state:
        st.session_state[key] = None

# ── Inputs ────────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    st.subheader("📄 Upload Resume")
    uploaded_file = st.file_uploader("PDF only", type=["pdf"])
with col2:
    st.subheader("💼 Job Description")
    jd_text = st.text_area(
        "Paste JD here", height=200,
        placeholder="We are looking for an AI Engineer with Python, ML...",
    )

st.divider()
btn = st.button("🔍 Analyze Resume", type="primary", use_container_width=True)

# ── Analysis ──────────────────────────────────────────────────────────────────
if btn:
    if not uploaded_file:
        st.error("❌ Upload resume PDF!")
    elif not jd_text.strip():
        st.error("❌ Paste Job Description!")
    else:
        try:
            # Phase 1: ingest only when the uploaded file actually changed
            file_bytes = uploaded_file.getvalue()
            file_key = f"{uploaded_file.name}-{len(file_bytes)}"

            if file_key != st.session_state.file_key:
                tmp_path = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(file_bytes)
                        tmp_path = tmp.name
                    with st.spinner("📄 Processing resume..."):
                        vs, text, n = ingest_resume(tmp_path)
                finally:
                    if tmp_path:
                        try:
                            os.unlink(tmp_path)
                        except OSError:
                            pass
                st.session_state.vectorstore = vs
                st.session_state.resume_text = text
                st.session_state.chunks = n
                st.session_state.file_key = file_key

            st.success(
                f"✅ {st.session_state.chunks} chunks indexed "
                f"({len(st.session_state.resume_text)} characters read)"
            )

            # Phase 2: analyze
            with st.spinner("🤖 Analyzing... (20-30 sec)"):
                result = analyze(
                    jd_text,
                    st.session_state.resume_text,
                    st.session_state.vectorstore,
                )

            st.divider()
            st.subheader("📊 Analysis Results")

            # ── Score ─────────────────────────────────────────────────────────
            score = result.match_score
            color = "#d4edda" if score >= 75 else "#fff3cd" if score >= 50 else "#f8d7da"
            tcolor = "#155724" if score >= 75 else "#856404" if score >= 50 else "#721c24"
            emoji = "🟢" if score >= 75 else "🟡" if score >= 50 else "🔴"

            st.markdown(f"""
            <div style="background:{color};color:{tcolor};
                        font-size:2.2rem;font-weight:bold;
                        text-align:center;padding:18px;
                        border-radius:10px;margin:10px 0;">
                {emoji} Match Score: {score}/100
            </div>""", unsafe_allow_html=True)

            st.info(f"💡 {result.explanation}")
            st.divider()

            # ── Skills Grid ───────────────────────────────────────────────────
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("### ✅ Matched")
                if result.matched_skills:
                    for s in result.matched_skills:
                        st.success(f"✅ {s}")
                else:
                    st.info("None")
            with c2:
                st.markdown("### ❌ Missing")
                if result.missing_skills:
                    for s in result.missing_skills:
                        st.error(f"❌ {s}")
                else:
                    st.info("None! 🎉")
            with c3:
                st.markdown("### ⚠️ Partial")
                if result.partial_skills:
                    for s in result.partial_skills:
                        st.warning(f"⚠️ {s}")
                else:
                    st.info("None!")

            # ── Evidence (RAG) ────────────────────────────────────────────────
            if result.evidence:
                with st.expander("🔎 Where these skills appear in your resume"):
                    for skill, snippet in result.evidence.items():
                        st.markdown(f"**{skill}** — _{snippet}_")

            st.divider()

            # ── Resume Tips ───────────────────────────────────────────────────
            st.subheader("📝 Resume Tips")
            for i, tip in enumerate(result.resume_tips, 1):
                st.info(f"**Tip {i}:** {tip}")

            st.divider()

            # ── Interview Questions ───────────────────────────────────────────
            st.subheader("🎯 Interview Questions")
            for i, q in enumerate(result.interview_questions, 1):
                st.markdown(f"**Q{i}.** {q}")

            st.divider()

            # ── Cover Letter ──────────────────────────────────────────────────
            st.subheader("✉️ Cover Letter")
            st.text_area("Copy:", value=result.cover_letter, height=250)
            st.download_button(
                "⬇️ Download Cover Letter",
                data=result.cover_letter,
                file_name="cover_letter.txt",
                mime="text/plain",
            )

        except Exception as e:
            st.error(f"❌ {str(e)}")
            st.info("💡 Check your inputs and try again.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<center>Built with ❤️ LangChain + ChromaDB + "
    "Groq + Streamlit | Navyasri Akula</center>",
    unsafe_allow_html=True,
)
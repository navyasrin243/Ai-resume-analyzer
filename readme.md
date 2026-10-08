# 🚀 AI Resume Analyzer

An AI-powered resume analyzer that matches your resume against any Job Description and provides instant, explainable analysis.

## 🔗 Live Demo
👉 [Try it here](https://ai-resume-analyzer-4bb8tdqh3mwuczjwrb2yz5.streamlit.app/)

## 🎯 What It Does

Upload your resume PDF + paste any Job Description →

- ✅ **Match Score** (rule-based, explainable — not LLM-generated!)
- ✅ **Matched Skills** — skills you already have
- ❌ **Missing Skills** — skills to learn/add
- ⚠️ **Partial Skills** — skills partially mentioned
- 📝 **Resume Tips** — how to improve your resume
- 🎯 **Interview Questions** — likely questions for this JD
- ✉️ **Cover Letter** — auto-generated, ready to use

## 🏗️ Architecture

    Resume PDF + Job Description
            ↓
    PyPDFLoader → Chunks (500/50)
            ↓
    HuggingFace MiniLM Embeddings
            ↓
    ChromaDB Vector Store
            ↓
    Hybrid Retrieval (BM25 + Dense)
            ↓
    Skill Extraction (Groq LLM)
            ↓
    Rule-Based Weighted Scoring
            ↓
    LLM Analysis (Groq)
            ↓
    Pydantic Structured Output
            ↓
    Streamlit UI

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| PDF Processing | PyPDFLoader (LangChain) |
| Chunking | RecursiveCharacterTextSplitter |
| Embeddings | HuggingFace MiniLM (free) |
| Vector DB | ChromaDB (local) |
| Retrieval | Hybrid (BM25 + Dense) |
| LLM | Groq (fast, free) |
| Output | Pydantic structured |
| UI | Streamlit |
| Deployment | Streamlit Cloud |

## 🚀 Run Locally

### Step 1: Clone
    git clone https://github.com/navyasrin243/Ai-resume-analyzer.git
    cd Ai-resume-analyzer

### Step 2: Virtual Environment
    python -m venv venv
    venv\Scripts\activate  # Windows
    source venv/bin/activate  # Mac/Linux

### Step 3: Install
    pip install -r requirements.txt

### Step 4: API Keys
Create .env file:
    GROQ_API_KEY=your_groq_key_here
    HF_TOKEN=your_hf_token_here

### Step 5: Run
    streamlit run app.py

## 📁 Project Structure

    ai-resume-analyzer/
    ├── app.py           ← Streamlit UI
    ├── analyzer.py      ← JD parsing + skill matching + LLM analysis
    ├── ingest.py        ← PDF load + chunk + embed + ChromaDB store
    ├── requirements.txt
    ├── .env             ← API keys (never commit!)
    └── .gitignore

## 💡 Key Design Decisions

### Why Rule-Based Scoring?
LLM-generated scores are not explainable.
My scoring uses direct text matching with aliases:

    Score = (matched x 1.0 + partial x 0.5) / total x 100

This makes every score defensible in interviews!

### Why Hybrid Retrieval?
BM25 alone gives false matches.
Dense alone misses exact keywords.
BM25 + Dense = best of both worlds!

### Why Groq over OpenAI?
- Free tier with high rate limits
- 500+ tokens/second inference
- No credit card required

## 👩‍💻 Author

**Navyasri Akula**
- 🎓 Integrated B.Sc-M.Sc Data Science, VIT-AP University
- 📧 navyasri.akula.2026@gmail.com
- 🔗 [LinkedIn](https://linkedin.com/in/navyasriakula)
- 🐙 [GitHub](https://github.com/navyasrin243)
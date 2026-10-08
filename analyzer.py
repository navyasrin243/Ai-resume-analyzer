import os
import re
import json
from typing import List, Dict

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

load_dotenv()


# ── Output model ──────────────────────────────────────────────────────────────
class AnalysisResult(BaseModel):
    match_score:         int            = Field(description="Score 0-100")
    matched_skills:      List[str]      = Field(description="Matched skills")
    missing_skills:      List[str]      = Field(description="Missing skills")
    partial_skills:      List[str]      = Field(description="Partial skills")
    explanation:         str            = Field(description="Explanation")
    resume_tips:         List[str]      = Field(description="Resume tips")
    cover_letter:        str            = Field(description="Cover letter")
    interview_questions: List[str]      = Field(description="Interview Qs")
    evidence:            Dict[str, str] = Field(
        default_factory=dict, description="Resume snippet backing each found skill"
    )


def get_llm():
    return ChatGroq(
        model="openai/gpt-oss-20b",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3,
    )


# ── Skill aliases (module level, built once) ──────────────────────────────────
ALIASES = {
    "machine learning": ["machine learning", "ml"],
    "deep learning":    ["deep learning", "dl"],
    "nlp":              ["nlp", "natural language processing"],
    "power bi":         ["power bi", "powerbi"],
    "scikit-learn":     ["scikit-learn", "scikit learn", "sklearn"],
    "tensorflow":       ["tensorflow", "tf"],
    "data analysis":    ["data analysis", "data analytics"],
    "eda":              ["eda", "exploratory data analysis"],
    "git":              ["git", "github", "gitlab"],
    "github":           ["github"],
    "sql":              ["sql", "mysql", "postgresql", "sqlite"],
    "python":           ["python"],
    "java":             ["java"],
    "javascript":       ["javascript", "js"],
    "html5":            ["html5", "html"],
    "css3":             ["css3", "css"],
    "react":            ["react", "react.js", "reactjs"],
    "node.js":          ["node.js", "nodejs", "node"],
    "aws":              ["aws", "amazon web services"],
    "docker":           ["docker"],
    "tableau":          ["tableau"],
    "mongodb":          ["mongodb"],
    "pytorch":          ["pytorch", "torch"],
}

# Words too generic to earn "partial" credit on their own
GENERIC_WORDS = {
    "data", "analysis", "analytics", "learning", "development", "developer",
    "engineering", "engineer", "design", "management", "software", "tools",
    "tool", "systems", "system", "skills", "experience", "knowledge",
    "programming", "framework", "frameworks", "application", "applications",
    "cloud", "web", "testing", "model", "models", "modeling",
}


def has_term(term: str, text: str) -> bool:
    """Whole-word match: 'java' will NOT match 'javascript', 'git' will NOT match 'digital'."""
    pattern = rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])"
    return re.search(pattern, text) is not None


# ── Parsing helpers ───────────────────────────────────────────────────────────
def _extract_json(raw: str, kind: str):
    """Pull a JSON array ('[') or object ('{') out of an LLM reply, however it is wrapped."""
    pattern = r"\[.*\]" if kind == "[" else r"\{.*\}"
    m = re.search(pattern, raw, re.S)
    if not m:
        raise ValueError(f"No JSON found in model reply: {raw[:200]}")
    return json.loads(m.group(0), strict=False)


def extract_jd_skills(jd_text: str, llm) -> List[Dict]:
    """Returns [{'skill': 'Python', 'required': True}, ...]"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Extract EVERY technical skill, tool, language, framework, platform
and domain requirement from the job description.
Mark each as "required" or "preferred" (preferred = nice to have / plus / bonus).
Return ONLY a JSON array, no markdown, no explanation. Example:
[{{"skill": "Python", "type": "required"}}, {{"skill": "AWS", "type": "preferred"}}]"""),
        ("human", "{jd}"),
    ])
    raw = (prompt | llm | StrOutputParser()).invoke({"jd": jd_text[:4000]})
    items = _extract_json(raw, "[")

    skills, seen = [], set()
    for it in items:
        if isinstance(it, str):
            name, required = it, True
        elif isinstance(it, dict):
            name = str(it.get("skill", "")).strip()
            required = str(it.get("type", "required")).lower() != "preferred"
        else:
            continue
        name = name.strip()
        if name and name.lower() not in seen:
            seen.add(name.lower())
            skills.append({"skill": name, "required": required})
    if not skills:
        raise ValueError("Could not find any skills in the job description.")
    return skills[:30]


# ── Skill matching + weighted score ───────────────────────────────────────────
def calculate_skill_match(skills: List[Dict], resume_full_text: str) -> Dict:
    text = re.sub(r"\s+", " ", resume_full_text.lower())
    matched, missing, partial = [], [], []
    earned, total = 0.0, 0.0

    for item in skills:
        skill = item["skill"]
        weight = 1.0 if item["required"] else 0.5
        key = skill.lower().strip()
        terms = ALIASES.get(key, [key])
        total += weight

        if any(has_term(t, text) for t in terms):
            matched.append(skill)
            earned += weight
            continue

        # Partial credit: multi-word skills only, on a distinctive word
        words = [w for w in re.split(r"[\s/,-]+", key)
                 if len(w) >= 4 and w not in GENERIC_WORDS]
        if len(key.split()) > 1 and words and any(has_term(w, text) for w in words):
            partial.append(skill)
            earned += weight * 0.5
        else:
            missing.append(skill)

    score = int(round(earned / total * 100)) if total else 0
    return {"matched": matched, "partial": partial, "missing": missing, "score": score}


# ── RAG evidence: where on the resume does each skill show up? ────────────────
def get_evidence(vectorstore, skills: List[str], limit: int = 8) -> Dict[str, str]:
    evidence = {}
    for skill in skills[:limit]:
        try:
            docs = vectorstore.similarity_search(skill, k=1)
            if docs:
                snippet = re.sub(r"\s+", " ", docs[0].page_content).strip()
                evidence[skill] = snippet[:220]
        except Exception as e:
            print(f"Evidence lookup failed for {skill}: {e}")
    return evidence


# ── Main entry point ──────────────────────────────────────────────────────────
def analyze(jd_text: str, resume_text: str, vectorstore=None) -> AnalysisResult:
    if not resume_text or not resume_text.strip():
        raise ValueError("Resume text is empty. Upload a resume first.")

    llm = get_llm()

    # Step 1: JD skills
    skills = extract_jd_skills(jd_text, llm)
    print(f"Skills ({len(skills)}): {[s['skill'] for s in skills]}")

    # Step 2: deterministic matching and score
    sm = calculate_skill_match(skills, resume_text)
    print(f"Matched: {sm['matched']}\nPartial: {sm['partial']}\n"
          f"Missing: {sm['missing']}\nScore: {sm['score']}")

    # Step 2b: RAG evidence for matched/partial skills
    evidence = {}
    if vectorstore is not None:
        evidence = get_evidence(vectorstore, sm["matched"] + sm["partial"])

    # Step 3: LLM writes the explanation, tips, cover letter, questions
    status = "\n".join(
        f"- {s['skill']} ({'required' if s['required'] else 'preferred'}): "
        + ("FOUND" if s["skill"] in sm["matched"]
           else "PARTIAL" if s["skill"] in sm["partial"] else "MISSING")
        for s in skills
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a senior HR analyst.
Return ONLY valid JSON, no markdown:
{{
  "explanation": "<2 sentences>",
  "resume_tips": ["tip1","tip2","tip3"],
  "cover_letter": "<cover letter, about 150 words, use \\n for line breaks>",
  "interview_questions": ["q1","q2","q3","q4","q5"]
}}"""),
        ("human", """JD: {jd}

Score: {score}/100
Skill status:
{status}"""),
    ])

    raw = (prompt | llm | StrOutputParser()).invoke({
        "jd": jd_text[:2000],
        "score": sm["score"],
        "status": status,
    })

    try:
        data = _extract_json(raw, "{")
    except Exception as e:
        print(f"LLM JSON parse failed: {e}")
        data = {}

    return AnalysisResult(
        match_score=sm["score"],
        matched_skills=sm["matched"],
        missing_skills=sm["missing"],
        partial_skills=sm["partial"],
        explanation=data.get("explanation", "Analysis completed; see skill breakdown below."),
        resume_tips=data.get("resume_tips", []),
        cover_letter=data.get("cover_letter", ""),
        interview_questions=data.get("interview_questions", []),
        evidence=evidence,
    )
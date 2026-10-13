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
        default_factory=dict,
        description="Resume snippet backing each found skill"
    )


def get_llm():
    return ChatGroq(
        model="qwen/qwen3.8-27b",  # Better JSON compliance
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3,
    )


# ── Skill aliases ─────────────────────────────────────────────────────────────
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
    "langchain":        ["langchain", "lang chain"],
    "rag":              ["rag", "retrieval augmented", "retrieval-augmented"],
    "chromadb":         ["chromadb", "chroma"],
    "streamlit":        ["streamlit"],
    "fastapi":          ["fastapi", "fast api"],
}

GENERIC_WORDS = {
    "data", "analysis", "analytics", "learning", "development", "developer",
    "engineering", "engineer", "design", "management", "software", "tools",
    "tool", "systems", "system", "skills", "experience", "knowledge",
    "programming", "framework", "frameworks", "application", "applications",
    "cloud", "web", "testing", "model", "models", "modeling",
}


def has_term(term: str, text: str) -> bool:
    pattern = rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])"
    return re.search(pattern, text) is not None


# ── Robust JSON extractor ─────────────────────────────────────────────────────
def _extract_json(raw: str, kind: str):
    """
    Robustly extract JSON array or object from LLM response.
    Handles: markdown fences, extra text, nested content.
    """
    if not raw or not raw.strip():
        raise ValueError("Empty response from model.")

    # Remove markdown fences
    text = re.sub(r"```json\s*", "", raw, flags=re.IGNORECASE)
    text = re.sub(r"```\s*", "", text)
    text = text.strip()

    if kind == "[":
        start = text.find("[")
        end   = text.rfind("]") + 1
    else:
        start = text.find("{")
        end   = text.rfind("}") + 1

    if start == -1 or end <= start:
        raise ValueError(f"No JSON {kind} found in model reply:\n{raw[:300]}")

    json_str = text[start:end]

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        try:
            return json.loads(json_str, strict=False)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}\nContent: {json_str[:200]}")


# ── Extract JD skills ─────────────────────────────────────────────────────────
def extract_jd_skills(jd_text: str, llm) -> List[Dict]:
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Extract EVERY technical skill, tool, language, framework,
platform and domain requirement from the job description.

Return ONLY a JSON array. No markdown. No explanation.
First character must be [ and last must be ].

Example:
[{{"skill": "Python", "type": "required"}}, {{"skill": "AWS", "type": "preferred"}}]"""),
        ("human", "Job Description:\n{jd}"),
    ])

    chain = prompt | llm | StrOutputParser()

    for attempt in range(2):
        try:
            raw = chain.invoke({"jd": jd_text[:4000]})
            items = _extract_json(raw, "[")
            skills, seen = [], set()
            for it in items:
                if isinstance(it, str):
                    name, required = it.strip(), True
                elif isinstance(it, dict):
                    name     = str(it.get("skill", "")).strip()
                    required = str(it.get("type", "required")).lower() != "preferred"
                else:
                    continue
                if name and name.lower() not in seen:
                    seen.add(name.lower())
                    skills.append({"skill": name, "required": required})
            if skills:
                return skills[:30]
            raise ValueError("No skills found.")
        except Exception as e:
            print(f"  Skill extraction attempt {attempt+1} failed: {e}")
            if attempt == 1:
                raise ValueError(f"Could not extract JD skills: {e}")

    raise ValueError("Skill extraction failed.")


# ── Skill matching ─────────────────────────────────────────────────────────────
def calculate_skill_match(skills: List[Dict], resume_full_text: str) -> Dict:
    text = re.sub(r"\s+", " ", resume_full_text.lower())
    matched, missing, partial = [], [], []
    earned, total = 0.0, 0.0

    for item in skills:
        skill  = item["skill"]
        weight = 1.0 if item["required"] else 0.5
        key    = skill.lower().strip()
        terms  = ALIASES.get(key, [key])
        total += weight

        if any(has_term(t, text) for t in terms):
            matched.append(skill)
            earned += weight
            continue

        words = [
            w for w in re.split(r"[\s/,-]+", key)
            if len(w) >= 4 and w not in GENERIC_WORDS
        ]
        if len(key.split()) > 1 and words and any(has_term(w, text) for w in words):
            partial.append(skill)
            earned += weight * 0.5
        else:
            missing.append(skill)

    score = int(round(earned / total * 100)) if total else 0
    return {"matched": matched, "partial": partial, "missing": missing, "score": score}


# ── RAG evidence ──────────────────────────────────────────────────────────────
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


# ── LLM analysis ─────────────────────────────────────────────────────────────
def generate_analysis(jd_text: str, skills: List[Dict], sm: Dict, llm) -> Dict:
    status = "\n".join(
        f"- {s['skill']} ({'required' if s['required'] else 'preferred'}): "
        + ("FOUND"   if s["skill"] in sm["matched"]
           else "PARTIAL" if s["skill"] in sm["partial"]
           else "MISSING")
        for s in skills
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a senior HR analyst and career coach.
You MUST return ONLY a raw JSON object.
No introduction. No explanation. No markdown.
First character MUST be {{ and last MUST be }}.

Exact structure required:
{{
  "explanation": "2 sentences about why the candidate matches or doesn't match",
  "resume_tips": [
    "Specific tip 1 to improve resume",
    "Specific tip 2 to improve resume",
    "Specific tip 3 to improve resume"
  ],
  "cover_letter": "Professional cover letter of 150 words",
  "interview_questions": [
    "Question 1 likely to be asked",
    "Question 2 likely to be asked",
    "Question 3 likely to be asked",
    "Question 4 likely to be asked",
    "Question 5 likely to be asked"
  ]
}}

IMPORTANT:
- resume_tips MUST be a list of 3 strings
- interview_questions MUST be a list of 5 strings
- Both must be actual content, not empty"""),
        ("human", """Job Description:
{jd}

Match Score: {score}/100
Matched: {matched}
Missing: {missing}

Skill Status:
{status}

Return JSON only:"""),
    ])

    chain = prompt | llm | StrOutputParser()

    for attempt in range(2):
        try:
            raw = chain.invoke({
                "jd":      jd_text[:1500],
                "score":   sm["score"],
                "matched": sm["matched"],
                "missing": sm["missing"],
                "status":  status,
            })
            data = _extract_json(raw, "{")

            # Validate lists are actually lists with content
            if not isinstance(data.get("resume_tips"), list) or \
               len(data.get("resume_tips", [])) == 0:
                raise ValueError("resume_tips missing or empty")

            if not isinstance(data.get("interview_questions"), list) or \
               len(data.get("interview_questions", [])) == 0:
                raise ValueError("interview_questions missing or empty")

            return data

        except Exception as e:
            print(f"  Analysis attempt {attempt+1} failed: {e}")
            if attempt == 1:
                # Fallback — never return empty
                return {
                    "explanation": (
                        f"Your resume matches {sm['score']}% of the job requirements. "
                        f"You have {len(sm['matched'])} matching skills and "
                        f"{len(sm['missing'])} skills to develop."
                    ),
                    "resume_tips": [
                        f"Add missing skills to your resume: {', '.join(sm['missing'][:3])}",
                        "Quantify your achievements with specific numbers and metrics.",
                        "Highlight relevant tools and technologies prominently in skills section."
                    ],
                    "cover_letter": (
                        f"Dear Hiring Manager,\n\n"
                        f"I am writing to express my interest in this role. "
                        f"With my background in {', '.join(sm['matched'][:3])}, "
                        f"I am confident in my ability to contribute effectively. "
                        f"I am eager to bring my skills and passion to your team.\n\n"
                        f"Best regards"
                    ),
                    "interview_questions": [
                        "Tell me about yourself and your relevant experience.",
                        f"How have you used {sm['matched'][0] if sm['matched'] else 'your skills'} in a project?",
                        "Describe a challenging problem you solved recently.",
                        "How do you stay updated with the latest technology trends?",
                        "Where do you see yourself in 5 years?"
                    ]
                }


# ── Main entry point ──────────────────────────────────────────────────────────
def analyze(jd_text: str, resume_text: str, vectorstore=None) -> AnalysisResult:

    if not resume_text or not resume_text.strip():
        raise ValueError("Resume text is empty. Upload a resume first.")
    if not jd_text or not jd_text.strip():
        raise ValueError("Job description is empty.")

    llm = get_llm()

    # Step 1: Extract JD skills
    print("\n🔍 Extracting JD skills...")
    skills = extract_jd_skills(jd_text, llm)
    print(f"✅ Found {len(skills)} skills")

    # Step 2: Rule-based matching
    print("\n📊 Matching skills...")
    sm = calculate_skill_match(skills, resume_text)
    print(f"✅ Score: {sm['score']}/100")

    # Step 3: RAG evidence
    evidence = {}
    if vectorstore is not None:
        evidence = get_evidence(vectorstore, sm["matched"] + sm["partial"])

    # Step 4: LLM analysis
    print("\n🤖 Generating analysis...")
    data = generate_analysis(jd_text, skills, sm, llm)
    print("✅ Done!")

    return AnalysisResult(
        match_score         = sm["score"],
        matched_skills      = sm["matched"],
        missing_skills      = sm["missing"],
        partial_skills      = sm["partial"],
        explanation         = data.get("explanation", ""),
        resume_tips         = data.get("resume_tips", []),
        cover_letter        = data.get("cover_letter", ""),
        interview_questions = data.get("interview_questions", []),
        evidence            = evidence,
    )
import os
import re
import json
from typing import List, Dict, Any

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

load_dotenv()

# ==================================================
# 1. ANALYSIS RESULT MODEL
# ==================================================

class AnalysisResult(BaseModel):
    match_score: int = Field(
        description="Resume and job description match score out of 100"
    )
    matched_skills: List[str] = Field(default_factory=list)
    missing_skills: List[str] = Field(default_factory=list)
    partial_skills: List[str] = Field(default_factory=list)
    explanation: str = ""
    resume_tips: List[str] = Field(default_factory=list)
    cover_letter: str = ""
    interview_questions: List[str] = Field(default_factory=list)
    evidence: Dict[str, str] = Field(default_factory=dict)

# ==================================================
# 2. INITIALIZE GROQ LLM
# ==================================================

def get_llm():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. Configure it in your .env file "
            "or Streamlit Cloud Secrets."
        )

    return ChatGroq(
        model="qwen/qwen3.8-27b",
        temperature=0.2,
        api_key=api_key,
    )

# ==================================================
# 3. JSON RESPONSE HANDLING
# ==================================================

def _clean_model_response(raw: str) -> str:
    """Remove common wrappers from the LLM response."""

    if not raw:
        raise ValueError("The AI returned an empty response.")

    cleaned = raw.strip()

    # Remove model reasoning tags if present
    cleaned = re.sub(
        r"<think>.*?</think>",
        "",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()

    # Remove Markdown code fences
    cleaned = re.sub(
        r"^\s*```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s*```\s*$", "", cleaned)

    return cleaned.strip()

def _extract_json(raw: str, kind: str = "{"):
    """Extract and parse a JSON object or array."""

    cleaned = _clean_model_response(raw)

    start_char, end_char = (
        ("{", "}") if kind == "{" else ("[", "]")
    )

    start = cleaned.find(start_char)
    end = cleaned.rfind(end_char)

    if start == -1 or end == -1 or end < start:
        raise ValueError(
            "No JSON found in model reply. Response preview: "
            + cleaned[:250]
        )

    json_text = cleaned[start:end + 1]

    try:
        result = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON returned by AI: {exc}"
        ) from exc

    expected_type = dict if kind == "{" else list

    if not isinstance(result, expected_type):
        raise ValueError(
            f"Expected a JSON {'object' if kind == '{' else 'array'}."
        )

    return result

# ==================================================
# 4. SKILL ALIASES
# ==================================================

# Only explicit, known equivalents are matched.
# Do not infer that one related technology proves another.

SKILL_ALIASES = {
    "deep learning": [
        "deep learning",
        "dl",
    ],
    "machine learning": [
        "machine learning",
        "ml",
    ],
    "artificial intelligence": [
        "artificial intelligence",
        "ai",
    ],
    "natural language processing": [
        "natural language processing",
        "nlp",
    ],
    "large language models": [
        "large language models",
        "large language model",
        "llms",
        "llm",
    ],
    "react": [
        "react",
        "react.js",
        "reactjs",
    ],
    "vue.js": [
        "vue.js",
        "vuejs",
        "vue",
    ],
    "node.js": [
        "node.js",
        "nodejs",
    ],
    "javascript": [
        "javascript",
    ],
    "typescript": [
        "typescript",
    ],
    "html5": [
        "html5",
        "html",
    ],
    "css3": [
        "css3",
        "css",
    ],
    "postgresql": [
        "postgresql",
        "postgres",
    ],
    "scikit-learn": [
        "scikit-learn",
        "scikit learn",
        "sklearn",
    ],
    "power bi": [
        "power bi",
        "powerbi",
    ],
    "rest apis": [
        "rest apis",
        "rest api",
        "restful apis",
        "restful api",
    ],
    "ci/cd": [
        "ci/cd",
        "continuous integration",
        "continuous delivery",
        "continuous deployment",
    ],
}

def _normalise_phrase(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold().strip())

def _skill_variants(skill: str) -> List[str]:
    """Get explicit variants of a skill."""

    normalized = _normalise_phrase(skill)

    variants = SKILL_ALIASES.get(
        normalized,
        [normalized],
    )

    result = []
    seen = set()

    for variant in list(variants) + [normalized]:
        variant = _normalise_phrase(variant)

        if variant and variant not in seen:
            result.append(variant)
            seen.add(variant)

    return result

def _phrase_present(phrase: str, text: str) -> bool:
    """Match a complete phrase, not an unrelated substring."""

    phrase = _normalise_phrase(phrase)

    if not phrase:
        return False

    escaped = re.escape(phrase).replace(r"\ ", r"\s+")

    pattern = (
        rf"(?<![a-zA-Z0-9+#])"
        rf"{escaped}"
        rf"(?![a-zA-Z0-9+#])"
    )

    return bool(
        re.search(pattern, text, flags=re.IGNORECASE)
    )

def skill_is_present(skill: str, resume_text: str) -> bool:
    """Check whether a skill or its explicit alias occurs in the resume."""

    return any(
        _phrase_present(variant, resume_text)
        for variant in _skill_variants(skill)
    )

# ==================================================
# 5. EXTRACT SKILLS FROM JOB DESCRIPTION
# ==================================================

def extract_jd_skills(
    jd_text: str,
    llm,
) -> List[Dict[str, Any]]:

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert technical recruiter.

Extract technical skills explicitly mentioned in the job description.

Return ONLY a valid JSON array. Each item must contain:
- skill: the skill name
- required: true if required, otherwise false

Include programming languages, frameworks, libraries, databases,
cloud platforms, developer tools, and technical concepts.

Do not invent skills.
Avoid duplicates.
Return [] if no technical skills are identifiable.

Example:
[
  {{"skill": "Python", "required": true}},
  {{"skill": "SQL", "required": true}},
  {{"skill": "Power BI", "required": false}}
]

Return valid JSON only, without Markdown fences."""
        ),
        (
            "human",
            "JOB DESCRIPTION:\n{jd}",
        ),
    ])

    chain = prompt | llm | StrOutputParser()

    raw = chain.invoke({
        "jd": jd_text[:7000],
    })

    skills = _extract_json(raw, "[")

    valid_skills = []
    seen = set()

    for item in skills:
        if not isinstance(item, dict):
            continue

        skill = item.get("skill")

        if not isinstance(skill, str) or not skill.strip():
            continue

        skill = skill.strip()
        key = skill.casefold()

        if key in seen:
            continue

        seen.add(key)

        valid_skills.append({
            "skill": skill,
            "required": bool(item.get("required", True)),
        })

    return valid_skills

# ==================================================
# 6. MATCH JD SKILLS AGAINST THE ENTIRE RESUME
# ==================================================

def calculate_skill_match(
    skills: List[Dict[str, Any]],
    resume_text: str,
) -> Dict[str, Any]:

    matched = []
    missing = []
    partial = []

    total_weight = 0
    earned_weight = 0

    for item in skills:
        skill = item["skill"]

        weight = (
            2 if item.get("required", True)
            else 1
        )

        total_weight += weight

        # Search the complete extracted resume text
        if skill_is_present(skill, resume_text):
            matched.append(skill)
            earned_weight += weight
        else:
            # No fuzzy matching based on individual words.
            # If the skill/alias is absent, mark it as missing.
            missing.append(skill)

    score = (
        round((earned_weight / total_weight) * 100)
        if total_weight
        else 0
    )

    return {
        "score": min(score, 100),
        "matched": matched,
        "missing": missing,
        "partial": partial,
    }

# ==================================================
# 7. SPLIT RESUME INTO EVIDENCE SNIPPETS
# ==================================================

def _resume_chunks(resume_text: str) -> List[str]:
    """
    Split PDF-extracted resume text into manageable snippets.

    PDF extraction sometimes puts an entire section on one line,
    so this function also splits long text into smaller chunks.
    """

    text = resume_text.replace("\r", "\n")

    chunks = re.split(
        r"\n+|(?<=[.!?])\s+|(?<=;)\s+|(?<=\|)\s*",
        text,
    )

    cleaned_chunks = []

    for chunk in chunks:
        chunk = re.sub(r"\s+", " ", chunk).strip(" \t•|-")

        if chunk:
            cleaned_chunks.append(chunk)

    result = []

    for chunk in cleaned_chunks:
        if len(chunk) <= 700:
            result.append(chunk)
        else:
            # Keep long lines from displaying the entire resume
            for start in range(0, len(chunk), 450):
                piece = chunk[start:start + 450].strip()

                if piece:
                    result.append(piece)

    return result

# ==================================================
# 8. EVIDENCE FOR MATCHED SKILLS ONLY
# ==================================================

def get_evidence(
    skills: List[Dict[str, Any]],
    resume_text: str,
    matched_skills: List[str],
) -> Dict[str, str]:

    evidence = {}

    chunks = _resume_chunks(resume_text)

    # Search only skills already confirmed as matched
    for skill in matched_skills:

        matching_chunks = [
            chunk
            for chunk in chunks
            if skill_is_present(skill, chunk)
        ]

        if matching_chunks:
            # Only relevant resume snippets are shown.
            evidence[skill] = " | ".join(
                matching_chunks[:3]
            )[:700]

    # Missing skills are deliberately excluded from evidence.
    return evidence

# ==================================================
# 9. GENERATE PERSONALIZED AI ANALYSIS
# ==================================================

def generate_analysis(
    jd_text: str,
    skills: List[Dict[str, Any]],
    sm: Dict[str, Any],
    llm,
    resume_text: str,
) -> Dict[str, Any]:

    status_lines = []

    for item in skills:
        skill = item["skill"]

        if skill in sm["matched"]:
            result = "MATCHED"
        else:
            result = "MISSING"

        requirement = (
            "required"
            if item.get("required", True)
            else "preferred"
        )

        status_lines.append(
            f"- {skill} ({requirement}): {result}"
        )

    status = (
        "\n".join(status_lines)
        or "No technical skills identified."
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert career coach and technical recruiter.

Analyze the candidate's actual resume against the job description.

Return ONLY one valid JSON object with these fields:
{{
  "explanation": "Two or three sentences explaining the candidate's fit",
  "resume_tips": ["Specific improvement 1", "Specific improvement 2"],
  "cover_letter": "A personalized professional cover letter",
  "interview_questions": ["Question 1", "Question 2", "Question 3"]
}}

COVER LETTER REQUIREMENTS:
1. Write approximately 180-250 words.
2. Start with Dear Hiring Manager,
3. Use 3-4 professional paragraphs.
4. Refer to the role in the job description when identifiable.
5. Connect actual resume skills, projects, education, and experience
   to the job requirements.
6. For entry-level candidates, emphasize relevant projects,
   academic work, practical skills, and willingness to learn.
7. Never invent employers, projects, qualifications, certifications,
   achievements, metrics, or years of experience.
8. Do not claim unsupported skills or experience.
9. Do not invent a candidate name or company name.
10. Do not use placeholders such as [Company Name] or [Your Name].
11. Use natural and professional language.
12. End with Thank you for your consideration. and Best regards,
    without inventing a signature name.

ANALYSIS REQUIREMENTS:
- Explain the fit using the supplied skill comparison.
- Give actionable resume improvement tips.
- Generate interview questions relevant to the job.
- Do not claim missing skills are present.
- Return valid JSON only, without Markdown fences or extra commentary."""
        ),
        (
            "human",
            """JOB DESCRIPTION:
{jd}

MATCH SCORE: {score}/100

MATCHED SKILLS:
{matched}

MISSING SKILLS:
{missing}

SKILL STATUS:
{status}

CANDIDATE RESUME:
{resume_text}

Generate a complete and personalized analysis using the information
provided above. Do not invent facts about the candidate."""
        ),
    ])

    chain = prompt | llm | StrOutputParser()

    for attempt in range(2):
        try:
            raw = chain.invoke({
                "jd": jd_text[:7000],
                "score": sm["score"],
                "matched": (
                    ", ".join(sm["matched"])
                    or "None"
                ),
                "missing": (
                    ", ".join(sm["missing"])
                    or "None"
                ),
                "status": status,
                "resume_text": resume_text[:8000],
            })

            data = _extract_json(raw, "{")

            explanation = data.get("explanation", "")
            tips = data.get("resume_tips", [])
            letter = data.get("cover_letter", "")
            questions = data.get("interview_questions", [])

            if not isinstance(explanation, str):
                raise ValueError(
                    "The explanation must be a string."
                )

            if not isinstance(tips, list):
                raise ValueError(
                    "Resume tips must be a list."
                )

            if not isinstance(letter, str) or not letter.strip():
                raise ValueError(
                    "The cover letter is empty."
                )

            if not isinstance(questions, list):
                raise ValueError(
                    "Interview questions must be a list."
                )

            return {
                "explanation": explanation,
                "resume_tips": [
                    str(item) for item in tips
                ],
                "cover_letter": letter.strip(),
                "interview_questions": [
                    str(item) for item in questions
                ],
            }

        except Exception as exc:
            print(
                f"Analysis attempt {attempt + 1} failed: {exc}"
            )

            if attempt == 1:
                raise ValueError(
                    "Could not generate the AI analysis. "
                    "Check the Groq API key, model availability, "
                    "and job description. "
                    f"Details: {exc}"
                ) from exc

    raise ValueError("AI analysis failed.")

# ==================================================
# 10. MAIN ANALYSIS FUNCTION
# ==================================================

def analyze(
    jd_text: str,
    resume_text: str,
    vectorstore=None,
) -> AnalysisResult:

    if not jd_text or not jd_text.strip():
        raise ValueError(
            "Please provide a job description."
        )

    if not resume_text or not resume_text.strip():
        raise ValueError(
            "The resume text is empty or could not be read."
        )

    llm = get_llm()

    # Step 1: Extract skills from the job description
    skills = extract_jd_skills(
        jd_text,
        llm,
    )

    # Step 2: Match every skill against the complete resume
    skill_match = calculate_skill_match(
        skills,
        resume_text,
    )

    # Step 3: Generate evidence only for matched skills
    evidence = get_evidence(
        skills=skills,
        resume_text=resume_text,
        matched_skills=skill_match["matched"],
    )

    # Step 4: Generate cover letter, tips, and interview questions
    generated = generate_analysis(
        jd_text=jd_text,
        skills=skills,
        sm=skill_match,
        llm=llm,
        resume_text=resume_text,
    )

    # Step 5: Return structured results to the Streamlit app
    return AnalysisResult(
        match_score=skill_match["score"],
        matched_skills=skill_match["matched"],
        missing_skills=skill_match["missing"],
        partial_skills=skill_match["partial"],
        explanation=generated.get("explanation", ""),
        resume_tips=generated.get("resume_tips", []),
        cover_letter=generated.get("cover_letter", ""),
        interview_questions=generated.get(
            "interview_questions",
            [],
        ),
        evidence=evidence,
    )

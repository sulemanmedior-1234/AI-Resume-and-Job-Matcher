"""
AI Resume and Job Matcher
Paste (or upload) a resume and a job description. The app shows:
  - an AI match score and a plain keyword-coverage score
  - matching skills, with evidence from the resume
  - missing requirements, with honest advice
  - suggested wording improvements (before / after)
  - the resume with the job's keywords highlighted

Uses the Groq REST API directly through `requests`, so the `groq` package is
not needed. PDF reading needs `pypdf` (listed in requirements.txt).

Run locally:  streamlit run app.py
"""

import html
import io
import json
import os
import re
from collections import Counter
from datetime import datetime

import requests
import streamlit as st

# --------------------------------------------------------------------------- #
# Page configuration
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="AI Resume Matcher", page_icon="📄", layout="wide")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
MAX_RESUME_CHARS = 12000
MAX_JD_CHARS = 8000
MIN_CHARS = 100

SAMPLE_RESUME = """Ayesha Khan
Lahore, Pakistan | ayesha.khan@example.com | linkedin.com/in/ayesha-khan-demo

SUMMARY
Recent Computer Science graduate who enjoys working with data. Looking for an analyst role.

EXPERIENCE
Data Intern, BrightMart Retail - Jun 2025 to Sep 2025
- Worked on sales data using Excel and some SQL queries.
- Helped the team make weekly reports.
- Cleaned data with Python and pandas.

Teaching Assistant, Department of Computer Science - Sep 2024 to May 2025
- Helped students with Python assignments.
- Graded quizzes and answered questions in lab sessions.

PROJECTS
Sales Dashboard - Built a dashboard in Power BI to show monthly sales by city.
Customer Survey Analysis - Used Python to analyze 400 survey responses and made charts with matplotlib.

EDUCATION
BS Computer Science, University of Lahore - 2021 to 2025, CGPA 3.4

SKILLS
Python, SQL, Excel, Power BI, pandas, matplotlib, communication, teamwork"""

SAMPLE_JD = """Junior Data Analyst - NorthPoint Analytics

About the role
We are looking for a Junior Data Analyst to join our growing analytics team. You will turn raw data into insights that help our business teams make better decisions.

Responsibilities
- Write SQL queries to extract and analyze data from relational databases.
- Build and maintain interactive dashboards using Tableau or Power BI.
- Clean, transform and validate large datasets using Python (pandas).
- Present findings and recommendations to non-technical stakeholders.
- Automate recurring reports and improve data quality processes.

Requirements
- Bachelor's degree in Computer Science, Statistics, Mathematics or a related field.
- 1+ years of experience in data analysis (internships count).
- Strong SQL and Python skills.
- Experience with data visualization tools such as Tableau or Power BI.
- Good communication skills and the ability to explain results clearly.
- Familiarity with A/B testing and basic statistics.

Nice to have
- Experience with cloud platforms such as AWS or Google Cloud.
- Knowledge of Git and version control."""

# HELPERS START
STOPWORDS = set(
    """a an the and or but if of to in on at by for from with about as is are was were be been being
    it its this that these those i me my we our you your he she they them their do does did can could
    should would will shall may might what which who whom when where why how any all there here have
    has had not no than then so such also into out up down over under again more most other some
    their per etc""".split()
)

# Words that appear in almost every job post and say nothing about skills.
FILLER = set(
    """experience experiences work working role roles team teams ability able strong good skill skills
    knowledge looking join growing responsibilities requirement requirements required include including
    candidate candidates years year plus nice have must will help helps make makes better using use
    used within across new related field degree bachelor company business teams position job
    opportunity responsible responsibility understand understanding excellent great ideal etc
    junior senior basic clearly clear build maintain improve drive support
    ensure provide deliver count counts""".split()
)

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+#.\-]*", re.IGNORECASE)


def clean_token(tok):
    return tok.lower().rstrip(".-")


def stem(word):
    """Very light stemming so 'dashboards' matches 'dashboard'."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 5 and word.endswith("sses"):
        return word[:-2]
    if len(word) > 5 and word.endswith("ing"):
        return word[:-3]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def tokens(text):
    out = []
    for raw in TOKEN_RE.findall(text):
        t = clean_token(raw)
        if t:
            out.append(t)
    return out


def jd_keywords(jd_text, limit=20):
    """Most important terms in the job description, as {stem: display word}.

    Terms count by frequency. Names that look like tools or technologies (capitalised
    in the middle of a sentence, ALL CAPS, or containing + # .) get a bonus, so
    'Tableau', 'AWS' and 'SQL' rank above ordinary words.
    """
    scores, display = Counter(), {}
    for m in TOKEN_RE.finditer(jd_text):
        raw = m.group(0)
        core = raw.rstrip(".-")
        t = clean_token(core)
        if not t or t in STOPWORDS or t in FILLER or len(t) < 2 or t[0].isdigit():
            continue
        s = stem(t)
        weight = 1.0
        before = jd_text[:m.start()].rstrip(" \t")
        at_sentence_start = (not before) or before[-1] in "\n-•*.:!?"
        if (core.isupper() and len(core) >= 2) or any(c in core for c in "+#."):
            weight += 2.0
        elif core[0].isupper() and not at_sentence_start:
            weight += 1.5
        scores[s] += weight
        display.setdefault(s, t)
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]
    return {s: display[s] for s, _ in ranked}


def keyword_coverage(resume_text, keywords):
    """Return (matched, missing) lists of display words."""
    resume_stems = {stem(t) for t in tokens(resume_text)}
    matched = [w for s, w in keywords.items() if s in resume_stems]
    missing = [w for s, w in keywords.items() if s not in resume_stems]
    return matched, missing


def highlight_resume(resume_text, keywords):
    """Return safe HTML with job keywords wrapped in <mark>."""
    wanted = set(keywords.keys())
    out, last = [], 0
    for m in TOKEN_RE.finditer(resume_text):
        raw = m.group(0)
        core = raw.rstrip(".-")
        trail = raw[len(core):]
        out.append(html.escape(resume_text[last:m.start()]))
        piece = html.escape(core)
        if core and stem(clean_token(core)) in wanted:
            piece = "<mark>" + piece + "</mark>"
        out.append(piece + html.escape(trail))
        last = m.end()
    out.append(html.escape(resume_text[last:]))
    return "".join(out).replace("\n", "<br>")


def build_messages(resume, jd):
    system_msg = (
        "You are an expert recruiter and resume coach. You compare a resume with a job "
        "description honestly and fairly. You never invent experience, skills, employers, "
        "dates or numbers. Reply with valid JSON only."
    )
    user_msg = (
        "Compare the RESUME with the JOB DESCRIPTION.\n\n"
        "Return a JSON object with exactly this shape:\n"
        "{\n"
        '  "match_score": 0-100 (integer, how well the resume fits the job),\n'
        '  "category_scores": {"Skills": 0-100, "Experience": 0-100, "Education": 0-100, "Keywords": 0-100},\n'
        '  "summary": "2-4 sentences: overall fit, biggest strength, biggest gap",\n'
        '  "matching_skills": [{"skill": "name", "evidence": "short quote or paraphrase from the resume"}],\n'
        '  "missing_requirements": [{"requirement": "what the job asks for", '
        '"importance": "Must-have" or "Nice-to-have", "advice": "honest, practical advice"}],\n'
        '  "wording_improvements": [{"original": "exact line from the resume", '
        '"improved": "stronger wording", "reason": "why it is better"}]\n'
        "}\n\n"
        "Rules:\n"
        "- matching_skills: only skills or requirements that are really shown in the resume.\n"
        "- missing_requirements: only things the job asks for that the resume does NOT show.\n"
        "- advice for a missing requirement: if the candidate has related experience, say how "
        "to present it truthfully; otherwise suggest a realistic way to learn or gain it. "
        "Never tell the candidate to claim something they do not have.\n"
        "- wording_improvements: 3 to 6 items. 'original' must be copied from the resume. "
        "Use strong action verbs and the job's keywords where they are truthful. "
        "If a number would help but is not in the resume, use a placeholder like [X%] or [N] "
        "instead of inventing one.\n"
        "- Be realistic with scores. Do not give 90+ unless nearly every requirement is met.\n\n"
        'RESUME:\n"""\n' + resume + '\n"""\n\n'
        'JOB DESCRIPTION:\n"""\n' + jd + '\n"""'
    )
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def extract_json(text):
    """Parse JSON even if the model wrapped it in code fences or extra text."""
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError("The AI did not return valid JSON.")


def clamp_score(value, default=0):
    try:
        return max(0, min(100, int(round(float(value)))))
    except Exception:
        return default


def clean_list(items, keys):
    """Keep only dict items; make every key a non-empty-safe string."""
    out = []
    for item in items or []:
        if isinstance(item, dict):
            row = {k: str(item.get(k, "")).strip() for k in keys}
            if row[keys[0]]:
                out.append(row)
        elif isinstance(item, str) and item.strip():
            row = {k: "" for k in keys}
            row[keys[0]] = item.strip()
            out.append(row)
    return out


def validate(data):
    """Turn the AI's JSON into a safe, predictable structure."""
    if not isinstance(data, dict):
        raise ValueError("The AI reply was not in the expected format.")
    cats = data.get("category_scores") or {}
    category_scores = {}
    for name in ("Skills", "Experience", "Education", "Keywords"):
        category_scores[name] = clamp_score(cats.get(name, 0)) if isinstance(cats, dict) else 0
    return {
        "match_score": clamp_score(data.get("match_score", 0)),
        "category_scores": category_scores,
        "summary": str(data.get("summary", "")).strip(),
        "matching_skills": clean_list(data.get("matching_skills"), ["skill", "evidence"]),
        "missing_requirements": clean_list(
            data.get("missing_requirements"), ["requirement", "importance", "advice"]
        ),
        "wording_improvements": [
            r
            for r in clean_list(data.get("wording_improvements"), ["original", "improved", "reason"])
            if r["improved"]
        ],
    }


def score_label(score):
    if score >= 80:
        return "Strong match"
    if score >= 60:
        return "Good match"
    if score >= 40:
        return "Partial match"
    return "Weak match"


def build_report(result, matched, missing, coverage):
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# Resume and Job Match Report",
        "",
        "_Generated " + stamp + "_",
        "",
        "**AI match score:** " + str(result["match_score"]) + "/100 (" + score_label(result["match_score"]) + ")  ",
        "**Keyword coverage:** " + str(coverage) + "%",
        "",
        "## Summary",
        result["summary"] or "-",
        "",
        "## Matching skills",
    ]
    for s in result["matching_skills"]:
        lines.append("- **" + s["skill"] + "**" + (": " + s["evidence"] if s["evidence"] else ""))
    lines += ["", "## Missing requirements"]
    for r in result["missing_requirements"]:
        tag = " (" + r["importance"] + ")" if r["importance"] else ""
        lines.append("- **" + r["requirement"] + "**" + tag + (": " + r["advice"] if r["advice"] else ""))
    lines += ["", "## Suggested wording improvements"]
    for w in result["wording_improvements"]:
        lines.append("- Before: " + w["original"])
        lines.append("  After: " + w["improved"])
        if w["reason"]:
            lines.append("  Why: " + w["reason"])
    lines += ["", "## Keywords", "Matched: " + (", ".join(matched) or "-"), "Missing: " + (", ".join(missing) or "-"), ""]
    return "\n".join(lines)


def friendly_error(exc):
    msg = str(exc)
    low = msg.lower()
    if "401" in msg or "invalid api key" in low or "invalid_api_key" in low:
        return "Your Groq API key looks invalid. Please check it and try again."
    if "429" in msg or "rate limit" in low:
        return "Groq rate limit reached. Please wait a minute and try again."
    if "decommissioned" in low or "does not exist" in low or "404" in msg:
        return (
            "That model is not available on your account. Choose another model in the "
            "sidebar. Details: " + msg[:300]
        )
    if "connection" in low or "timeout" in low or "timed out" in low:
        return "Could not reach the Groq API. Check your internet connection and retry."
    if "json" in low or "expected format" in low:
        return "The AI reply could not be read. Please click Analyze again."
    return "Something went wrong: " + msg
# HELPERS END


def get_api_key(sidebar_value):
    """Priority: sidebar input -> st.secrets -> environment variable."""
    if sidebar_value and sidebar_value.strip():
        return sidebar_value.strip()
    try:
        secret = st.secrets["GROQ_API_KEY"]
        if secret:
            return str(secret).strip()
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY", "").strip()


@st.cache_data(ttl=600, show_spinner=False)
def fetch_models(api_key):
    """Ask Groq which chat models this API key can use right now."""
    try:
        r = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": "Bearer " + api_key},
            timeout=20,
        )
        if r.status_code != 200:
            return []
        skip = ("whisper", "guard", "tts", "playai", "orpheus", "safeguard", "embed")
        ids = [
            m["id"]
            for m in r.json().get("data", [])
            if m.get("active", True) and not any(s in m["id"].lower() for s in skip)
        ]
        return sorted(ids)
    except Exception:
        return []


def call_groq(api_key, model, messages):
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.3,
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": "Bearer " + api_key, "Content-Type": "application/json"}
    resp = requests.post(GROQ_URL, headers=headers, json=payload, timeout=90)

    # Some models do not support JSON mode: retry once without it.
    if resp.status_code == 400 and "response_format" in resp.text:
        payload.pop("response_format")
        resp = requests.post(GROQ_URL, headers=headers, json=payload, timeout=90)

    if resp.status_code != 200:
        raise RuntimeError("Groq API error " + str(resp.status_code) + ": " + resp.text[:300])
    choices = resp.json().get("choices") or []
    if not choices:
        return ""
    return (choices[0].get("message", {}).get("content") or "").strip()


def read_resume_file(name, data):
    """Return the text of an uploaded resume (PDF, TXT or MD)."""
    if name.lower().endswith(".pdf"):
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ValueError("PDF support needs the 'pypdf' package. Add it to requirements.txt.")
        reader = PdfReader(io.BytesIO(data))
        text = "\n".join((p.extract_text() or "") for p in reader.pages).strip()
        if not text:
            raise ValueError("No readable text found in the PDF (it may be a scanned image).")
        return text
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return data.decode(enc).strip()
        except Exception:
            continue
    raise ValueError("Could not read the file.")


# --------------------------------------------------------------------------- #
# Session state and callbacks
# --------------------------------------------------------------------------- #
for key, value in {"resume_text": "", "jd_text": "", "file_error": "", "analysis": None}.items():
    if key not in st.session_state:
        st.session_state[key] = value


def load_example():
    st.session_state["resume_text"] = SAMPLE_RESUME
    st.session_state["jd_text"] = SAMPLE_JD
    st.session_state["file_error"] = ""


def load_resume_file():
    f = st.session_state.get("resume_file")
    if f is None:
        return
    try:
        st.session_state["resume_text"] = read_resume_file(f.name, f.getvalue())
        st.session_state["file_error"] = ""
    except Exception as exc:
        st.session_state["file_error"] = str(exc)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("⚙️ Settings")
    key_input = st.text_input(
        "Groq API key (optional)",
        type="password",
        help="Leave empty to use st.secrets or the GROQ_API_KEY environment variable.",
    )
    api_key = get_api_key(key_input)
    if api_key:
        st.success("API key found ✅")
    else:
        st.warning("No API key found yet")

    live_models = fetch_models(api_key) if api_key else []
    model_options = live_models or MODELS
    default_index = 0
    for preferred in ("llama-3.3-70b-versatile", "llama-3.1-8b-instant"):
        if preferred in model_options:
            default_index = model_options.index(preferred)
            break
    model = st.selectbox("Groq model", model_options, index=default_index)

    st.divider()
    st.button("📋 Load an example", on_click=load_example)
    st.caption("Fills both boxes with a sample resume and job post.")

# --------------------------------------------------------------------------- #
# Main page
# --------------------------------------------------------------------------- #
st.title("📄 AI Resume and Job Matcher")
st.write(
    "Paste your resume and a job description. The app shows what matches, what is "
    "missing, and how to word your resume more strongly, without inventing experience."
)

left, right = st.columns(2)
with left:
    st.subheader("Your resume")
    st.file_uploader(
        "Upload a PDF or text file (optional)",
        type=["pdf", "txt", "md"],
        key="resume_file",
        on_change=load_resume_file,
    )
    if st.session_state["file_error"]:
        st.error(st.session_state["file_error"])
    st.text_area("Resume text", height=380, key="resume_text", placeholder="Paste your resume here...")
with right:
    st.subheader("Job description")
    st.text_area("Job description text", height=470, key="jd_text", placeholder="Paste the job description here...")

if st.button("✨ Analyze match", type="primary"):
    resume = st.session_state["resume_text"].strip()
    jd = st.session_state["jd_text"].strip()

    if not api_key:
        st.error(
            "No Groq API key found. Enter it in the sidebar, or add GROQ_API_KEY "
            "to your Streamlit secrets."
        )
    elif not resume or not jd:
        st.warning("Please provide both a resume and a job description.")
    elif len(resume) < MIN_CHARS or len(jd) < MIN_CHARS:
        st.warning("Both texts are too short. Please provide at least " + str(MIN_CHARS) + " characters each.")
    else:
        if len(resume) > MAX_RESUME_CHARS:
            st.warning("Your resume was trimmed to " + str(MAX_RESUME_CHARS) + " characters.")
            resume = resume[:MAX_RESUME_CHARS]
        if len(jd) > MAX_JD_CHARS:
            st.warning("The job description was trimmed to " + str(MAX_JD_CHARS) + " characters.")
            jd = jd[:MAX_JD_CHARS]

        try:
            with st.spinner("Comparing your resume with the job..."):
                raw = call_groq(api_key, model, build_messages(resume, jd))
                result = validate(extract_json(raw))

            keywords = jd_keywords(jd)
            matched, missing = keyword_coverage(resume, keywords)
            coverage = round(len(matched) / len(keywords) * 100) if keywords else 0

            st.session_state["analysis"] = {
                "result": result,
                "keywords": keywords,
                "matched": matched,
                "missing": missing,
                "coverage": coverage,
                "resume": resume,
            }
        except Exception as exc:
            st.error(friendly_error(exc))

# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
analysis = st.session_state["analysis"]
if analysis is not None:
    result = analysis["result"]
    score = result["match_score"]

    st.divider()
    st.header("Results")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("AI match score", str(score) + " / 100", score_label(score), delta_color="off")
    m2.metric("Keyword coverage", str(analysis["coverage"]) + "%")
    m3.metric("Matching skills", len(result["matching_skills"]))
    m4.metric("Missing requirements", len(result["missing_requirements"]))
    st.progress(score / 100)
    st.caption(
        "The AI score is a judgement of overall fit. Keyword coverage is a simple count of the "
        "job's most frequent terms that also appear in your resume. Use both as guidance."
    )

    tabs = st.tabs(
        ["📝 Summary", "✅ Matching skills", "⚠️ Missing requirements", "✍️ Wording improvements", "🖍️ Highlighted resume"]
    )

    with tabs[0]:
        st.write(result["summary"] or "No summary was returned.")
        st.subheader("Score breakdown")
        st.bar_chart(result["category_scores"])

    with tabs[1]:
        if not result["matching_skills"]:
            st.info("No clearly matching skills were found.")
        for s in result["matching_skills"]:
            st.markdown("**✅ " + s["skill"] + "**")
            if s["evidence"]:
                st.caption("In your resume: " + s["evidence"])

    with tabs[2]:
        if not result["missing_requirements"]:
            st.success("No missing requirements were found.")
        for r in result["missing_requirements"]:
            tag = "🔴 Must-have" if r["importance"].lower().startswith("must") else "🟡 Nice-to-have"
            st.markdown("**" + r["requirement"] + "**  \n" + tag)
            if r["advice"]:
                st.caption(r["advice"])
            st.write("")

    with tabs[3]:
        if not result["wording_improvements"]:
            st.info("No wording suggestions were returned.")
        for i, w in enumerate(result["wording_improvements"], start=1):
            st.markdown("**Suggestion " + str(i) + "**")
            c1, c2 = st.columns(2)
            c1.markdown("Before")
            c1.error(w["original"] or "-")
            c2.markdown("After")
            c2.success(w["improved"])
            if w["reason"]:
                st.caption("Why: " + w["reason"])
        st.caption("Replace any [X%] or [N] placeholder with a real number from your own work, and use only wording that is true.")

    with tabs[4]:
        st.caption(
            "Highlighted words are terms from the job description that appear in your resume."
        )
        st.markdown(
            '<div style="line-height:1.7">' + highlight_resume(analysis["resume"], analysis["keywords"]) + "</div>",
            unsafe_allow_html=True,
        )
        k1, k2 = st.columns(2)
        k1.markdown("**Job keywords found in your resume**")
        k1.write(", ".join(analysis["matched"]) or "None")
        k2.markdown("**Job keywords not found**")
        k2.write(", ".join(analysis["missing"]) or "None")

    st.download_button(
        "⬇️ Download report (.md)",
        data=build_report(result, analysis["matched"], analysis["missing"], analysis["coverage"]),
        file_name="resume_match_report.md",
        mime="text/markdown",
    )

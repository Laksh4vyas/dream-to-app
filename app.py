import os
import io
import re
import json
import time
import base64
import zipfile
import nbformat
import subprocess
from datetime import datetime
from pathlib import Path
from textwrap import dedent

import streamlit as st
from dotenv import load_dotenv

# ---- Optional deps (graceful fallback) ---------------------------------------
try:
    import black  # code formatter
    HAS_BLACK = True
except Exception:
    HAS_BLACK = False

try:
    from flake8.api import legacy as flake8  # linter
    HAS_FLAKE8 = True
except Exception:
    HAS_FLAKE8 = False

try:
    import bcrypt  # simple auth hashing
    HAS_BCRYPT = True
except Exception:
    HAS_BCRYPT = False

# ---- LLM (Gemini via langchain) ----------------------------------------------
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    HAS_GEMINI = True
except Exception:
    HAS_GEMINI = False

# ==============================================================================
#                              CONFIG / CONSTANTS
# ==============================================================================
APP_NAME = "Dream-to-App Studio"
DATA_DIR = Path("data")
USERS_FILE = DATA_DIR / "users.json"
GALLERY_DIR = DATA_DIR / "gallery"
VERSIONS_DIR = DATA_DIR / "versions"
PROJECTS_DIR = DATA_DIR / "projects"

DEFAULT_PORT = 8501  # for preview subprocess
MAX_TOKENS_DEFAULT = 4096

CATEGORIES = {
    "📊 Dashboards": "Use pandas & matplotlib. Provide filters, KPI cards, and at least one chart.",
    "🤖 AI Utilities": "Demonstrate an LLM call or embedding use. Add prompt boxes and output area.",
    "🎮 Games": "Keep it simple (guessing game, tic-tac-toe). Use clean UI and session_state.",
    "📅 Productivity": "Include todos, notes, reminders, persistence (json file).",
}

SAMPLE_IDEAS = [
    "A Streamlit app that takes a sentence and counts words.",
    "A Weather Dashboard using OpenWeather API.",
    "A ToDo List app with login/logout functionality.",
    "An AI Resume Analyzer that gives job suggestions.",
    "A Personal Expense Tracker with charts.",
]

# Heuristic import->package fallback map
IMPORT_TO_PKG = {
    "streamlit": "streamlit",
    "pandas": "pandas",
    "numpy": "numpy",
    "matplotlib": "matplotlib",
    "plotly": "plotly",
    "requests": "requests",
    "scikit-learn": "scikit-learn",
    "sklearn": "scikit-learn",
    "PIL": "Pillow",
    "google.generativeai": "google-generativeai",
    "langchain": "langchain",
    "openai": "openai",
    "bs4": "beautifulsoup4",
    "seaborn": "seaborn",
    "altair": "altair",
}

# ==============================================================================
#                                  UTILITIES
# ==============================================================================
def ensure_dirs():
    for p in [DATA_DIR, GALLERY_DIR, VERSIONS_DIR, PROJECTS_DIR]:
        p.mkdir(parents=True, exist_ok=True)

def load_users():
    ensure_dirs()
    if USERS_FILE.exists():
        try:
            return json.loads(USERS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def save_users(users: dict):
    ensure_dirs()
    USERS_FILE.write_text(json.dumps(users, indent=2), encoding="utf-8")

def hash_pw(password: str) -> str:
    if HAS_BCRYPT:
        return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    # fallback NOT cryptographically strong (warn user)
    import hashlib
    return "sha256$" + hashlib.sha256(password.encode()).hexdigest()

def verify_pw(password: str, hashed: str) -> bool:
    if HAS_BCRYPT and hashed and not hashed.startswith("sha256$"):
        try:
            return bcrypt.checkpw(password.encode(), hashed.encode())
        except Exception:
            return False
    if hashed.startswith("sha256$"):
        import hashlib
        return hashed == "sha256$" + hashlib.sha256(password.encode()).hexdigest()
    return False

def user_dir(username: str) -> Path:
    d = DATA_DIR / "users" / username
    d.mkdir(parents=True, exist_ok=True)
    return d

def history_file(username: str) -> Path:
    return user_dir(username) / "history.json"

def load_history(username: str) -> list:
    f = history_file(username)
    if f.exists():
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []

def save_history(username: str, entries: list):
    f = history_file(username)
    f.write_text(json.dumps(entries, indent=2), encoding="utf-8")

def now_stamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def format_code(code: str) -> str:
    if HAS_BLACK:
        try:
            return black.format_str(code, mode=black.Mode())
        except Exception:
            return code
    return code

def lint_results(code: str):
    """Run flake8 if available; else heuristic."""
    if HAS_FLAKE8:
        style_guide = flake8.get_style_guide(ignore=[], max_line_length=100)
        # Write to a temp file to lint
        tmp = PROJECTS_DIR / "_lint_temp.py"
        tmp.write_text(code, encoding="utf-8")
        report = style_guide.check_files([str(tmp)])
        # Return total errors & human-ish score
        total = report.total_errors
        # ad-hoc score: 100 minus a penalty, clamp to [0,100]
        score = max(0, 100 - min(100, total * 5))
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        return total, score, "flake8"
    # Heuristic fallback
    too_long = sum(1 for ln in code.splitlines() if len(ln) > 100)
    tabs = code.count("\t")
    trailing_ws = sum(1 for ln in code.splitlines() if ln.rstrip() != ln)
    total = too_long + tabs + trailing_ws
    score = max(0, 95 - total * 3)
    return total, score, "heuristic"

def extract_imports(code: str):
    # very rough regex
    imports = set()
    for line in code.splitlines():
        m1 = re.match(r"\s*import\s+([a-zA-Z0-9_\.]+)", line)
        m2 = re.match(r"\s*from\s+([a-zA-Z0-9_\.]+)\s+import\s+", line)
        if m1:
            imports.add(m1.group(1).split(".")[0])
        if m2:
            imports.add(m2.group(1).split(".")[0])
    return sorted(list(imports))

def infer_requirements(code: str):
    reqs = set(["streamlit"])  # baseline
    for mod in extract_imports(code):
        reqs.add(IMPORT_TO_PKG.get(mod, mod))
    # Remove stdlib guesses
    stdlib_like = {"io","os","re","json","time","pathlib","datetime","subprocess","textwrap","random","string","sys","typing"}
    reqs = [r for r in reqs if r not in stdlib_like]
    return sorted(set(reqs))

def parse_multifile_blob(text: str):
    """
    Expect blocks like:
        ### FILE: app.py
        <code...>
        ### FILE: utils/helpers.py
        <code...>
    returns dict{path: code}
    """
    files = {}
    parts = re.split(r"(?m)^###\s*FILE:\s*(.+)$", text)
    if len(parts) <= 1:
        return None
    # parts = ["prefix", "path1", "code1", "path2", "code2", ...]
    for i in range(1, len(parts), 2):
        pth = parts[i].strip()
        code = parts[i+1]
        files[pth] = code.strip()
    return files or None

def build_project_zip(files_map: dict, project_name: str, requirements: list, readme: str = ""):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for rel, content in files_map.items():
            z.writestr(f"{project_name}/{rel}", content)
        z.writestr(f"{project_name}/requirements.txt", "\n".join(requirements) + "\n")
        if readme:
            z.writestr(f"{project_name}/README.md", readme)
        # minimal Dockerfile
        dockerfile = dedent(f"""
        FROM python:3.10-slim
        WORKDIR /app
        COPY requirements.txt .
        RUN pip install -r requirements.txt
        COPY . .
        EXPOSE 8501
        CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
        """).strip()
        z.writestr(f"{project_name}/Dockerfile", dockerfile)
        # HF Spaces helper
        z.writestr(f"{project_name}/Procfile", "web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0\n")
    buf.seek(0)
    return buf

def write_version(username: str, idea: str, files_map: dict):
    vdir = VERSIONS_DIR / username
    vdir.mkdir(parents=True, exist_ok=True)
    stamp = now_stamp()
    vfile = vdir / f"{stamp}.json"
    payload = {"idea": idea, "files": files_map, "timestamp": stamp}
    vfile.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return vfile

def start_preview_subprocess(script_path: Path, port: int):
    # Launch "streamlit run <path> --server.port=<port>"
    return subprocess.Popen(
        ["streamlit", "run", str(script_path), f"--server.port={port}"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )

def stop_preview_subprocess(proc):
    try:
        proc.terminate()
    except Exception:
        pass

# ==============================================================================
#                               LLM HELPERS
# ==============================================================================
def get_llm(model_name: str, temperature: float, max_tokens: int):
    if not HAS_GEMINI:
        st.error("langchain_google_genai not installed. `pip install -U langchain-google-genai`")
        return None
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("GEMINI_API_KEY not set in .env")
        return None
    os.environ["GOOGLE_API_KEY"] = api_key
    return ChatGoogleGenerativeAI(model=model_name, temperature=temperature, max_output_tokens=max_tokens)

def prompt_generate(idea: str, category_hint: str, multifile: bool):
    if multifile:
        return dedent(f"""
        You are an AI project generator.
        Create a COMPLETE Streamlit project for the idea:

        IDEA:
        {idea}

        CATEGORY BEST PRACTICES (merge appropriately, don't mention explicitly):
        {category_hint}

        Output as MULTI-FILE with clear file boundaries using the exact format:
        ### FILE: app.py
        <python code for app.py>
        ### FILE: utils/helpers.py
        <python code for helpers>
        ### FILE: assets/__init__.py
        <...>
        (Add as many files as needed, but keep structure simple and runnable.)

        Rules:
        - Only output pure code sections (NO markdown, NO backticks).
        - Ensure app runs with: `streamlit run app.py`.
        - Keep UI clean and user friendly.
        """).strip()

    return dedent(f"""
    You are an AI project generator.
    Generate a complete Python Streamlit app for this idea:

    IDEA:
    {idea}

    CATEGORY BEST PRACTICES (merge appropriately, don't mention explicitly):
    {category_hint}

    Rules:
    - Only output pure Python code (NO ``` fences, NO markdown).
    - App should run directly with 'streamlit run generated_app.py'.
    - Keep UI clean and user friendly.
    """).strip()

def prompt_explain(code: str):
    return dedent(f"""
    You are an expert Python tutor. Explain what the following Streamlit app code does.
    Provide a concise, readable, section-by-section explanation with bullet points:

    CODE:
    {code}
    """).strip()

def prompt_fix(code: str, error_text: str):
    return dedent(f"""
    You are a senior Python developer. The following Streamlit code is failing.
    Read the code and the error log, then suggest specific changes. Provide a short
    explanation and then the FULL corrected file(s).

    If multiple files are needed, output in this format:
    ### FILE: app.py
    <corrected code>
    ### FILE: utils/helpers.py
    <corrected code>

    ORIGINAL CODE:
    {code}

    ERROR LOG:
    {error_text}
    """).strip()

# ==============================================================================
#                                   UI
# ==============================================================================
st.set_page_config(page_title=APP_NAME, page_icon="💡", layout="wide", initial_sidebar_state="expanded")

ensure_dirs()

# --- THEME TOGGLE --------------------------------------------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "light"

def inject_theme_css():
    if st.session_state.theme == "dark":
        st.markdown("""
        <style>
        .stApp { background: #0f1117; color: #e5e7eb; }
        .stButton>button { background: linear-gradient(90deg,#9333ea,#6366f1); color: white; }
        .stTextArea textarea, .stTextInput input, .stSelectbox div, .stSlider { border-radius: 10px; }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        .stButton>button { background: linear-gradient(90deg,#4CAF50,#45a049); color: white; border-radius:8px; }
        .stTextArea textarea { border-radius:10px; border:2px solid #4CAF50; }
        </style>
        """, unsafe_allow_html=True)

inject_theme_css()

# --- AUTH ----------------------------------------------------------------------
users = load_users()
if "auth_user" not in st.session_state:
    st.session_state.auth_user = None

with st.sidebar:
    st.title("🔐 Auth")
    if st.session_state.auth_user:
        st.success(f"Logged in as **{st.session_state.auth_user}**")
        if st.button("Logout"):
            st.session_state.auth_user = None
            st.rerun()
    else:
        tab_login, tab_signup = st.tabs(["Login", "Sign up"])
        with tab_login:
            li_user = st.text_input("Username", key="li_user")
            li_pw = st.text_input("Password", type="password", key="li_pw")
            if st.button("Login"):
                if li_user in users and verify_pw(li_pw, users[li_user]["password"]):
                    st.session_state.auth_user = li_user
                    st.success("Logged in")
                    time.sleep(0.7)
                    st.rerun()
                else:
                    st.error("Invalid credentials")

        with tab_signup:
            su_user = st.text_input("New Username", key="su_user")
            su_pw = st.text_input("New Password", type="password", key="su_pw")
            if st.button("Create account"):
                if not su_user or not su_pw:
                    st.warning("Enter username & password")
                elif su_user in users:
                    st.error("User already exists")
                else:
                    users[su_user] = {"password": hash_pw(su_pw), "created": now_stamp()}
                    save_users(users)
                    st.success("Account created. Please login.")

    st.divider()
    st.subheader("🎛️ Model Controls")
    model_name = st.selectbox("Model", ["gemini-1.5-flash", "gemini-1.5-pro"], index=0)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.5, 0.05)
    max_tokens = st.slider("Max output tokens", 512, 8192, MAX_TOKENS_DEFAULT, 256)

    st.divider()
    st.subheader("🎨 Theme")
    theme_choice = st.radio("Select theme", ["light", "dark"], index=0 if st.session_state.theme=="light" else 1)
    if theme_choice != st.session_state.theme:
        st.session_state.theme = theme_choice
        st.rerun()

# Guard
if not st.session_state.auth_user:
    st.title(APP_NAME)
    st.info("Please **log in** (left sidebar) to use the studio.")
    st.stop()

# --- MAIN CONTENT --------------------------------------------------------------
st.title("💡 Dream-to-App Studio")

# Quick Start row
colA, colB = st.columns([1, 1])
with colA:
    st.subheader("✨ Quick Start Ideas")
    chosen_sample = st.selectbox("Pick an idea:", ["-- None --"] + SAMPLE_IDEAS)

with colB:
    st.subheader("📦 Category")
    category = st.selectbox("Select category", list(CATEGORIES.keys()))
    multifile = st.checkbox("Generate multi-file project", value=True)
    preview_port = st.number_input("Preview port (local)", min_value=1024, max_value=65535, value=DEFAULT_PORT, step=1)

st.subheader("🧠 Your Idea")
idea = st.text_area(
    "Describe the app you want:",
    value=("" if chosen_sample == "-- None --" else chosen_sample),
    placeholder="e.g., A Streamlit app that takes a sentence and counts words.",
    height=120,
)

# Generate
llm = get_llm(model_name, temperature, max_tokens)
if "last_files_map" not in st.session_state:
    st.session_state.last_files_map = None
if "preview_proc" not in st.session_state:
    st.session_state.preview_proc = None

col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    run_btn = st.button("🚀 Generate")
with col2:
    tutor_btn = st.button("🧑‍🏫 Explain Code (AI Tutor)", disabled=st.session_state.last_files_map is None)
with col3:
    debug_btn = st.button("🛠️ Debug & Fix", disabled=st.session_state.last_files_map is None)

if run_btn:
    if not idea.strip():
        st.error("Please enter an idea.")
    elif not llm:
        st.stop()
    else:
        with st.spinner("Generating project..."):
            prompt = prompt_generate(idea, CATEGORIES[category], multifile)
            resp = llm.invoke(prompt)
            raw = resp.content or ""

            files_map = None
            if multifile:
                files_map = parse_multifile_blob(raw)

            # Fallback to single file
            if not files_map:
                code = raw.replace("```python", "").replace("```", "").strip()
                files_map = {"app.py": code}

            # Format files & compute reqs from main entry
            formatted = {}
            for pth, code in files_map.items():
                formatted[pth] = format_code(code)

            main_code = formatted.get("app.py", next(iter(formatted.values())))
            total_issues, score, engine = lint_results(main_code)
            reqs = infer_requirements("\n".join(formatted.values()))

            # Save to per-user project path
            user_home = user_dir(st.session_state.auth_user)
            proj_name = f"project_{now_stamp()}"
            proj_dir = user_home / proj_name
            proj_dir.mkdir(parents=True, exist_ok=True)
            for rel, content in formatted.items():
                out_path = proj_dir / rel
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(content, encoding="utf-8")
            (proj_dir / "requirements.txt").write_text("\n".join(reqs) + "\n", encoding="utf-8")

            # Save version record
            write_version(st.session_state.auth_user, idea, formatted)

            # Update history
            hist = load_history(st.session_state.auth_user)
            hist.append({
                "timestamp": now_stamp(),
                "project": proj_name,
                "idea": idea,
                "category": category,
                "multifile": multifile,
                "requirements": reqs,
                "issues": total_issues,
                "score": score,
            })
            save_history(st.session_state.auth_user, hist)

            st.session_state.last_files_map = formatted

        st.success(f"✅ Generated **{proj_name}** — Code health ({engine}): **{score}/100** (issues: {total_issues})")

if st.session_state.last_files_map:
    st.divider()
    st.subheader("📁 Project Files")
    tabs = st.tabs(list(st.session_state.last_files_map.keys()))
    for (fname, code), tab in zip(st.session_state.last_files_map.items(), tabs):
        with tab:
            st.code(code, language="python")
            # Inline edit
            edited = st.text_area(f"Edit {fname}", code, height=300, key=f"edit_{fname}")
            if st.button(f"Save edits to {fname}", key=f"save_{fname}"):
                st.session_state.last_files_map[fname] = edited
                st.success("Saved (in session). Re-download to persist.")

    # Requirements + export
    st.subheader("📦 Requirements & Export")
    combined_code = "\n\n".join(st.session_state.last_files_map.values())
    reqs = infer_requirements(combined_code)
    st.write("**Detected requirements:**")
    st.code("\n".join(reqs) or "streamlit", language="text")

    # Download single main file if present
    if "app.py" in st.session_state.last_files_map:
        st.download_button(
            "📥 Download app.py",
            st.session_state.last_files_map["app.py"],
            file_name="app.py",
            mime="text/x-python",
        )

    # Export as Notebook
    nb = nbformat.v4.new_notebook()
    nb.cells.append(nbformat.v4.new_code_cell(st.session_state.last_files_map.get("app.py","")))
    nb_bytes = nbformat.writes(nb).encode("utf-8")
    st.download_button("📓 Download Jupyter Notebook", nb_bytes, file_name="generated_app.ipynb", mime="application/x-ipynb+json")

    # Zip full project
    proj_zip = build_project_zip(
        st.session_state.last_files_map,
        project_name="generated_app",
        requirements=reqs,
        readme=f"# Generated by {APP_NAME}\n\nIdea: {idea}\n"
    )
    st.download_button("📦 Download Full Project (ZIP)", proj_zip.getvalue(), file_name="generated_app.zip", mime="application/zip")

    # Launch (local) preview
    st.subheader("🔍 Local Preview (experimental)")
    st.caption("Starts a local Streamlit subprocess using your environment. Works on your machine only.")
    preview_col1, preview_col2 = st.columns(2)
    with preview_col1:
        if st.session_state.preview_proc is None and st.button("▶️ Start preview"):
            # write temp app to projects dir
            temp_dir = PROJECTS_DIR / f"preview_{st.session_state.auth_user}"
            temp_dir.mkdir(parents=True, exist_ok=True)
            (temp_dir / "app.py").write_text(st.session_state.last_files_map.get("app.py", ""), encoding="utf-8")
            st.session_state.preview_proc = start_preview_subprocess(temp_dir / "app.py", int(preview_port))
            st.toast(f"Preview started on http://localhost:{int(preview_port)}")
        elif st.session_state.preview_proc is not None:
            st.info(f"Running at: http://localhost:{int(preview_port)}")
    with preview_col2:
        if st.session_state.preview_proc is not None and st.button("⏹ Stop preview"):
            stop_preview_subprocess(st.session_state.preview_proc)
            st.session_state.preview_proc = None
            st.toast("Preview stopped")

    # Tutor
    if tutor_btn:
        if not llm:
            st.stop()
        with st.spinner("Explaining code..."):
            exp = llm.invoke(prompt_explain(st.session_state.last_files_map.get("app.py", ""))).content
        st.subheader("🧑‍🏫 AI Tutor — Code Explanation")
        st.write(exp)

    # Debug & Fix
    if debug_btn:
        st.subheader("🛠️ Debug & Fix Mode")
        err = st.text_area("Paste error log here:", height=180, key="errlog")
        if st.button("🔧 Suggest fix"):
            if not err.strip():
                st.warning("Paste an error first.")
            elif not llm:
                st.stop()
            else:
                with st.spinner("Analyzing & proposing fix..."):
                    fix = llm.invoke(prompt_fix(st.session_state.last_files_map.get("app.py",""), err)).content
                # Try parse
                maybe_files = parse_multifile_blob(fix)
                if maybe_files:
                    st.session_state.last_files_map.update(maybe_files)
                    st.success("Applied multi-file fix to session (not saved to disk yet).")
                    st.subheader("Proposed changes")
                    for pth, c in maybe_files.items():
                        st.markdown(f"**{pth}**")
                        st.code(c, language="python")
                else:
                    st.subheader("Suggested fix (manual apply)")
                    st.code(fix, language="python")

st.divider()

# --- HISTORY / GALLERY / VERSIONING -------------------------------------------
st.subheader("🖼️ Gallery & Versioning")
hist = load_history(st.session_state.auth_user)
if not hist:
    st.info("No history yet. Generate your first project!")
else:
    for i, item in enumerate(reversed(hist), start=1):
        with st.expander(f"{i}. {item['timestamp']} — {item['idea'][:70]}"):
            st.write(f"**Project:** {item['project']} • **Category:** {item['category']} • **Score:** {item['score']}")
            st.code("Requirements:\n" + "\n".join(item.get("requirements", [])), language="text")
            # reload button
            if st.button("🔄 Load files into editor", key=f"load_{i}"):
                # load from disk to session
                proj_path = user_dir(st.session_state.auth_user) / item["project"]
                files_map = {}
                for p in proj_path.rglob("*"):
                    if p.is_file() and p.suffix in {".py",".json",".txt"}:
                        files_map[str(p.relative_to(proj_path)).replace("\\","/")] = p.read_text(encoding="utf-8")
                if files_map:
                    st.session_state.last_files_map = files_map
                    st.success("Loaded into editor area above.")
                    st.experimental_rerun()
            # rollback to this snapshot by copying files to session_state
            if st.button("⏪ Rollback to this version (session only)", key=f"rb_{i}"):
                # find nearest version json by timestamp
                vfile = VERSIONS_DIR / st.session_state.auth_user / f"{item['timestamp'].replace(':','-').replace(' ','_')}.json"
                # history timestamp not exactly same; we stored version at generation time; fallback: find latest
                if not vfile.exists():
                    # pick the newest version file
                    vers = sorted((VERSIONS_DIR / st.session_state.auth_user).glob("*.json"))
                    if vers:
                        vfile = vers[-1]
                if vfile.exists():
                    payload = json.loads(vfile.read_text(encoding="utf-8"))
                    st.session_state.last_files_map = payload.get("files", {})
                    st.success("Rolled back (in session).")
                    st.experimental_rerun()
                else:
                    st.warning("No version snapshot found for this item.")

# --- DEPLOYMENT HELPERS --------------------------------------------------------
st.divider()
st.subheader("🚀 Deploy / Share")
st.markdown("""
- **Hugging Face Spaces**: Upload the ZIP you downloaded. It already includes `requirements.txt`, `Dockerfile`, and `Procfile`.
- **Docker**: `docker build -t myapp .` then `docker run -p 8501:8501 myapp`
- **Render/Railway**: Create a new web service, use `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`.
""")

st.caption("Tip: Install optional tools for best results: `pip install black flake8 bcrypt langchain-google-genai nbformat`")

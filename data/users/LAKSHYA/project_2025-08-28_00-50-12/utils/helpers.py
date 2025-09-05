import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger")

# Sample job data (replace with your actual job data)
job_data = {
    "Software Engineer": [
        "python",
        "java",
        "programming",
        "software",
        "development",
        3,
    ],
    "Data Scientist": ["python", "pandas", "machine learning", "data analysis", 2],
    "Data Analyst": ["sql", "excel", "data analysis", "tableau", 1],
    "Project Manager": ["project management", "agile", "communication", 5],
}


def analyze_resume(resume_text):
    # Basic keyword extraction
    keywords = [
        "python",
        "java",
        "machine learning",
        "data analysis",
        "sql",
        "project management",
        "communication",
        "agile",
        "software",
        "development",
        "tableau",
        "excel",
        "pandas",
    ]
    keyword_count = sum(
        1
        for keyword in keywords
        if re.search(r"\b" + keyword + r"\b", resume_text, re.IGNORECASE)
    )

    # Basic skill extraction (replace with more robust NLP techniques)
    tokens = word_tokenize(resume_text)
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [
        w.lower() for w in tokens if not w.lower() in stop_words and w.isalnum()
    ]
    skills = [
        word for word, pos in nltk.pos_tag(filtered_tokens) if pos.startswith("NN")
    ]  # Extract nouns as skills

    # Basic experience extraction (replace with more robust NLP techniques)
    experience_match = re.findall(
        r"(\d+)\s*(?:year|years)\s*experience", resume_text, re.IGNORECASE
    )
    experience = int(experience_match[0]) if experience_match else 0

    return {"keyword_count": keyword_count, "skills": skills, "experience": experience}


def get_job_suggestions(skills, experience):
    suggestions = []
    for job, required_skills in job_data.items():
        skill_match = sum(
            1
            for skill in skills
            if skill.lower() in [s.lower() for s in required_skills[:-1]]
        )
        if (
            skill_match >= len(required_skills[:-1]) / 2
            and experience >= required_skills[-1]
        ):
            suggestions.append({"Job Title": job})
    return pd.DataFrame(suggestions) if suggestions else None

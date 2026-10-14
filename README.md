# AI Resume ATS Checker

Upload a resume (PDF, DOCX or TXT), optionally paste a job description, and get:

- An overall **ATS score** (0-100)
- Category scores (keywords, formatting, experience impact, skills, readability)
- Prioritised **improvements** with exact fixes
- **Missing keywords** for the target job
- **Rewritten bullet points**
- A downloadable JSON report

Built with [Streamlit](https://streamlit.io) and Google Gemini Flash.

## How the score works

`Final score = 70% AI content score + 30% rule-based format checks`

The rule-based checks (email, phone, core sections, length, bullets, metrics) run locally and are deterministic. The AI score covers keyword match and content quality. This is an estimate, since every real ATS behaves differently.

## Run locally

```bash
pip install -r requirements.txt
export GEMINI_API_KEY="your_key_here"     # Windows PowerShell: $env:GEMINI_API_KEY="your_key_here"
streamlit run app.py
```

Get a free API key at https://aistudio.google.com/apikey. You can also paste the key into the app's sidebar.

## Deploy on Streamlit Community Cloud

1. Push `app.py`, `requirements.txt` and `README.md` to a GitHub repo.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app**, pick the repo, branch `main`, main file `app.py`.
4. Open **Advanced settings > Secrets** and add:
   ```toml
   GEMINI_API_KEY = "your_key_here"
   ```
5. Click **Deploy**.

Never commit your API key to GitHub.

## Notes

- Scanned/image-only resumes can't be read (ATS systems can't read them either). Use a text-based PDF or DOCX.
- The model name is editable in the sidebar (default `gemini-flash-latest`).
- Resume text is sent to the Gemini API for analysis.

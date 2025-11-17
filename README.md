# Skin Condition Triage

Streamlit application that sends uploaded skin images to a local vision model (OpenAI-compatible endpoint on `http://localhost:8080/v1`) and, when a disease is detected, uses Linkup to find dermatologists and curated doctor-made YouTube videos for that condition. The videos are displayed using the `streamlit-carousel` component for an interactive viewing experience.

LOOM: https://www.loom.com/share/c1ad083f6ca04983a3c23b905507b523

## Setup

1. (Optional) Create/activate a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure your local OpenAI-style vision model is running on port 8080. Override defaults with:
   - `OPENAI_BASE_URL`
   - `OPENAI_API_KEY`
   - `OPENAI_MODEL`
4. Export your Linkup API key so the doctor search and video gallery can run:
   ```bash
   export LINKUP_API_KEY="your-key"
   ```
5. (Optional) Provide a Groq API key if you want dynamic educational summaries for each diagnosis:
   ```bash
   export GROQ_API_KEY="your-groq-key"
   ```
   - Override `GROQ_MODEL` to use another Groq-hosted LLM.

## Run the app

```bash
streamlit run app.py
```

Upload an image, click **Analyze**, and review the detected condition. The app displays an educational snippet (powered by Groq when configured), a gallery of dermatologist-created YouTube videos (via Linkup), and, if the result is not **Normal Skin**, queries Linkup for dermatology specialists (optionally scoped to the location in the sidebar).

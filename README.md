# Skin Condition Triage

Streamlit application that sends uploaded skin images to a local vision model (OpenAI-compatible endpoint on `http://localhost:8080/v1`) and, when a disease is detected, queries Linkup for dermatologists who treat that condition.

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
4. Export your Linkup API key so the doctor search can run:
   ```bash
   export LINKUP_API_KEY="your-key"
   ```

## Run the app

```bash
streamlit run app.py
```

Upload an image, click **Analyze**, and review the detected condition. If the result is not **Normal Skin**, Linkup will be queried for dermatology specialists (optionally scoped to the location you enter in the sidebar).

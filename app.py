import base64
import json
import os
from typing import Any, Dict, Optional, Tuple

import streamlit as st
from openai import OpenAI

try:
    from linkup import LinkupClient
except ImportError:  # pragma: no cover - Linkup SDK might not be installed locally yet.
    LinkupClient = None  # type: ignore


OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:8080/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "not-needed")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "lfm2-vl")
LINKUP_API_KEY = os.getenv("LINKUP_API_KEY")

DIAGNOSES = [
    "Acne",
    "Eczema (Atopic Dermatitis)",
    "Psoriasis",
    "Rosacea",
    "Normal Skin",
]


@st.cache_resource(show_spinner=False)
def get_openai_client() -> OpenAI:
    return OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)


@st.cache_resource(show_spinner=False)
def get_linkup_client() -> Optional[LinkupClient]:
    if not LINKUP_API_KEY or LinkupClient is None:
        return None
    return LinkupClient(api_key=LINKUP_API_KEY)


def parse_model_payload(payload: str) -> Dict[str, Any]:
    """Extract JSON payload from the model and return a dictionary."""
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        start = payload.find("{")
        end = payload.rfind("}")
        if start != -1 and end != -1:
            snippet = payload[start : end + 1]
            return json.loads(snippet)
    raise ValueError("Model response was not valid JSON.")


def classify_skin(image_bytes: bytes) -> Dict[str, Any]:
    client = get_openai_client()
    image_data = base64.b64encode(image_bytes).decode("utf-8")

    system_text = (
        "You are a dermatology assistant. Exactly one of the following conditions is present "
        "in each image: Acne, Eczema (Atopic Dermatitis), Psoriasis, Rosacea, or Normal Skin. "
        "Identify the most likely condition."
    )
    user_text = (
        "Classify this skin condition. Respond ONLY in JSON with keys: "
        '"label" (one of: Acne, Eczema (Atopic Dermatitis), Psoriasis, Rosacea, Normal Skin), '
        '"confidence" (0-1 float), and "explanation" (short text).'
    )

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_text},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}" }},
                ],
            },
        ],
        max_tokens=256,
    )

    payload = response.choices[0].message.content
    if isinstance(payload, list):
        payload = "".join(chunk.get("text", "") for chunk in payload if isinstance(chunk, dict))
    return parse_model_payload(payload)


def search_specialists(label: str, location: str) -> Tuple[Optional[Any], Optional[str]]:
    client = get_linkup_client()
    if client is None:
        return None, "Linkup SDK not installed or LINKUP_API_KEY missing."

    query_location = f"in {location.strip()}" if location.strip() else ""
    query = (
        f"Find dermatologists or clinics specializing in {label} {query_location}. "
        "Return actionable contact details if possible."
    )

    try:
        response = client.search(query=query, depth="deep", output_type="sourcedAnswer")
        return response, None
    except Exception as exc:  # pragma: no cover - depends on external service.
        return None, str(exc)


def main() -> None:
    st.set_page_config(page_title="Skin Condition Triage", page_icon="🩺")
    st.title("Skin Condition Triage")
    st.write(
        "Upload a close-up image of the affected skin area. "
        "The local vision model (running on port 8080) will suggest the most likely condition. "
        "If an issue is detected, we will query Linkup for relevant dermatologists."
    )

    with st.sidebar:
        st.header("Linkup Search")
        location = st.text_input("Preferred location", placeholder="City, state, or leave blank")
        if not LINKUP_API_KEY:
            st.info("Set the LINKUP_API_KEY environment variable to enable doctor search.")

    uploaded = st.file_uploader("Skin image", type=["jpg", "jpeg", "png"])
    image_bytes: Optional[bytes] = None
    if uploaded:
        image_bytes = uploaded.getvalue()
        st.image(image_bytes, caption="Uploaded image", use_column_width=True)

    if image_bytes and st.button("Analyze"):
        with st.spinner("Contacting the vision model..."):
            try:
                result = classify_skin(image_bytes)
            except Exception as exc:
                st.error(f"Model call failed: {exc}")
                return

        label = result.get("label", "Unknown")
        confidence = result.get("confidence")
        explanation = result.get("explanation", "")

        st.subheader("Model Assessment")
        st.write(f"**Condition:** {label}")
        if confidence is not None:
            st.write(f"**Confidence:** {confidence:.2f}")
        if explanation:
            st.write(f"**Explanation:** {explanation}")

        if label.lower() == "normal skin" or label.lower() == "normal":
            st.success("The model did not detect a skin disease. No specialist search triggered.")
            return

        st.divider()
        st.subheader("Recommended Specialists")
        response, error = search_specialists(label, location)
        if error:
            st.error(f"Could not fetch specialists: {error}")
        elif response is None:
            st.warning("No results returned.")
        else:
            if isinstance(response, dict):
                st.json(response)
            elif hasattr(response, "dict"):
                st.json(response.dict())
            else:
                st.code(str(response))


if __name__ == "__main__":
    main()

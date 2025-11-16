import base64
import json
import os
import urllib.parse
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from openai import OpenAI
from streamlit_carousel import carousel

try:
    from linkup import LinkupClient
except ImportError:  # pragma: no cover - Linkup SDK might not be installed locally yet.
    LinkupClient = None  # type: ignore

try:
    from groq import Groq
except ImportError:  # pragma: no cover - Groq SDK optional.
    Groq = None  # type: ignore


OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:8080/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "not-needed")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "lfm2-vl")
LINKUP_API_KEY = os.getenv("LINKUP_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192")

DIAGNOSES = [
    "Acne",
    "Eczema (Atopic Dermatitis)",
    "Psoriasis",
    "Rosacea",
    "Normal Skin",
]

ACNE_EDUCATION_TEXT = (
    "What is acne?\n"
    "Our system sees a pattern that looks similar to acne: clogged pores, bumps, and redness.\n"
    "Acne is a very common skin condition where pores get blocked by oil and dead skin cells, "
    "sometimes with bacteria, leading to blackheads, whiteheads, and pimples.\n\n"
    "General steps that may help (not medical advice):"
)

DEFAULT_VIDEO_THUMBNAIL = "https://placehold.co/600x360?text=Dermatology+Video"


@st.cache_resource(show_spinner=False)
def get_openai_client() -> OpenAI:
    return OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)


@st.cache_resource(show_spinner=False)
def get_linkup_client() -> Optional[LinkupClient]:
    if not LINKUP_API_KEY or LinkupClient is None:
        return None
    return LinkupClient(api_key=LINKUP_API_KEY)


@st.cache_resource(show_spinner=False)
def get_groq_client() -> Optional[Groq]:
    if not GROQ_API_KEY or Groq is None:
        return None
    return Groq(api_key=GROQ_API_KEY)

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


def _extract_json_snippet(value: str) -> str:
    start = value.find("{")
    end = value.rfind("}")
    if start != -1 and end != -1:
        return value[start : end + 1]
    return value


def _extract_text_from_linkup(response: Any) -> str:
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        for key in ("output_text", "answer", "content", "text", "output"):
            val = response.get(key)
            if isinstance(val, str):
                return val
        return json.dumps(response)
    for attr in ("output_text", "answer", "content", "text", "output"):
        if hasattr(response, attr):
            val = getattr(response, attr)
            if isinstance(val, str):
                return val
    if hasattr(response, "dict"):
        try:
            return json.dumps(response.dict())
        except Exception:
            pass
    return str(response)


def search_condition_videos(label: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    client = get_linkup_client()
    if client is None:
        return [], "Linkup SDK not installed or LINKUP_API_KEY missing."

    query = (
        "List up to six YouTube videos created by board-certified dermatologists "
        f"that explain or treat {label}. "
        "Respond ONLY in JSON with a top-level key 'videos', where each entry contains: "
        "title, url, doctorName, channelName, publishedDate, thumbnailUrl, and summary."
    )

    try:
        response = client.search(query=query, depth="deep", output_type="sourcedAnswer")
    except Exception as exc:  # pragma: no cover
        return [], str(exc)

    raw_text = _extract_text_from_linkup(response)
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        snippet = _extract_json_snippet(raw_text)
        try:
            data = json.loads(snippet)
        except json.JSONDecodeError:
            return [], "Could not parse Linkup video response."

    videos_raw: Any = []
    if isinstance(data, dict):
        videos_raw = data.get("videos") or data.get("results") or data.get("items", [])
    elif isinstance(data, list):
        videos_raw = data

    videos: List[Dict[str, Any]] = []
    if isinstance(videos_raw, list):
        for entry in videos_raw:
            if not isinstance(entry, dict):
                continue
            videos.append(
                {
                    "title": entry.get("title") or entry.get("name"),
                    "url": entry.get("url") or entry.get("link"),
                    "doctor": entry.get("doctorName") or entry.get("doctor"),
                    "channel": entry.get("channelName") or entry.get("channel"),
                    "summary": entry.get("summary") or entry.get("description"),
                    "published": entry.get("publishedDate") or entry.get("published"),
                    "thumbnail": entry.get("thumbnailUrl") or entry.get("thumbnail"),
                }
            )

    return videos, None


def _extract_youtube_id(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    parsed = urllib.parse.urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path.strip("/")

    if "youtu.be" in host:
        return path or None
    if "youtube.com" in host:
        if path.startswith("shorts/"):
            return path.split("/", 1)[1] if "/" in path else path.replace("shorts/", "", 1)
        query = urllib.parse.parse_qs(parsed.query)
        if "v" in query and query["v"]:
            return query["v"][0]
    return None


def _resolve_video_thumbnail(url: Optional[str], provided: Optional[str]) -> str:
    if provided:
        return provided
    video_id = _extract_youtube_id(url)
    if video_id:
        return f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"
    return DEFAULT_VIDEO_THUMBNAIL


def build_carousel_items(videos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for video in videos[:6]:
        link = video.get("url") or video.get("link") or "#"
        title = video.get("title") or "Dermatology video"
        items.append(
            {
                "img": _resolve_video_thumbnail(link, video.get("thumbnail")),
                "title": "",
                "text": "",
                "link": link,
            }
        )
    return items


def render_video_details(videos: List[Dict[str, Any]]) -> None:
    st.caption("Tap a slide to open YouTube. Details:")
    for video in videos[:6]:
        title = video.get("title") or "Dermatology video"
        link = video.get("url") or video.get("link") or "#"
        summary = video.get("summary")
        doctor = video.get("doctor")
        channel = video.get("channel")
        published = video.get("published")

        st.markdown(f"**[{title}]({link})**")
        meta_parts = list(filter(None, [doctor, channel, published]))
        if meta_parts:
            st.caption(" • ".join(meta_parts))
        if summary:
            st.write(summary)
        st.divider()


def fetch_condition_info(label: str) -> Tuple[Optional[str], Optional[str]]:
    """Return an educational summary for the detected condition via Groq."""
    label_clean = label.strip()
    client = get_groq_client()
    if client is None:
        if label_clean.lower() == "acne":
            return ACNE_EDUCATION_TEXT, None
        return None, "Groq SDK not installed or GROQ_API_KEY missing."

    messages = [
        {
            "role": "system",
            "content": (
                "You are a dermatology educator creating succinct, friendly explanations. "
                "Provide a heading in the form 'What is CONDITION?' followed by two sentences "
                "describing what the system sees and what the condition is. "
                "End with the line 'General steps that may help (not medical advice):' and "
                "optionally add a short bulleted list beneath it. "
                "When the condition is Acne, respond EXACTLY with the following text:\n"
                f"{ACNE_EDUCATION_TEXT}\n"
                "Do not add extra commentary."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Condition: {label_clean}. Describe it following the instructions. "
                "If not acne, adapt the structure but keep it under 100 words."
            ),
        },
    ]

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            max_tokens=300,
            temperature=0.3,
        )
    except Exception as exc:  # pragma: no cover - depends on Groq availability.
        if label_clean.lower() == "acne":
            return ACNE_EDUCATION_TEXT, None
        return None, str(exc)

    text = response.choices[0].message.content
    return text, None


def main() -> None:
    st.set_page_config(page_title="Skin Condition Triage", page_icon="🩺", layout="wide")
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
        st.image(image_bytes, caption="Uploaded image", use_container_width=True)

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

        info_col, video_col = st.columns([1.15, 1])

        with info_col:
            st.subheader("Model Assessment")
            st.write(f"**Condition:** {label}")
            if confidence is not None:
                st.write(f"**Confidence:** {confidence:.2f}")
            if explanation:
                st.write(f"**Explanation:** {explanation}")

            st.subheader("Condition Overview")
            info_text, info_error = fetch_condition_info(label)
            if info_error:
                st.info(f"Educational snippet unavailable: {info_error}")
            elif info_text:
                st.markdown(info_text)

        with video_col:
            st.subheader("Doctor Video Gallery")
            videos, video_error = search_condition_videos(label)
            if video_error:
                st.info(f"Video recommendations unavailable: {video_error}")
            elif not videos:
                st.write("No dermatologist-created videos found right now.")
            else:
                items = build_carousel_items(videos)
                if not items:
                    st.write("No dermatologist-created videos found right now.")
                else:
                    carousel(
                        items=items,
                        controls=True,
                        indicators=True,
                        interval=6000,
                        pause="hover",
                        container_height=460,
                        width=1.0,
                        key=f"video-carousel-{label.lower().replace(' ', '-')}",
                    )
                    render_video_details(videos)

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

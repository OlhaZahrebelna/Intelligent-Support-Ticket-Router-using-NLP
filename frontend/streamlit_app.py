import os

import requests
import streamlit as st

API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "https://intelligent-support-ticket-router.onrender.com",
).rstrip("/")
PREDICT_URL = f"{API_BASE_URL}/predict"
HEALTH_URL = f"{API_BASE_URL}/health"
REVIEW_URL = f"{API_BASE_URL}/review/confirm"

st.set_page_config(page_title="Support Ticket Router", page_icon="🎫", layout="centered")
st.title("🎫 Intelligent Support Ticket Router")
st.write(
    "High-margin tickets are routed automatically. Low-margin predictions "
    "are escalated for human review."
)
st.divider()

with st.sidebar:
    st.header("About")
    st.write("Primary model: TF-IDF + Linear SVM")
    st.write("Review policy: top-2 SVM margin")
    st.write("Backend: FastAPI")

    if st.button("Check API health"):
        try:
            response = requests.get(HEALTH_URL, timeout=20)
            response.raise_for_status()
            data = response.json()
            st.success(f"API status: {data.get('status')}")
            st.write(f"Model loaded: {data.get('model_loaded')}")
            st.write(f"Review threshold: `{data.get('review_margin_threshold', 'N/A')}`")
        except requests.exceptions.RequestException as exc:
            st.error(f"Could not connect to API: {exc}")

st.subheader("Enter a support ticket")

examples = {
    "Billing issue": "I was charged twice for my monthly subscription.",
    "Login issue": "I cannot login to my account and need to reset my password.",
    "Technical issue": "The app crashes every time I try to open it.",
    "VPN issue": "I cannot connect to the corporate VPN from my work laptop.",
}

selected_example = st.selectbox(
    "Choose an example or write your own ticket:",
    ["Custom"] + list(examples.keys()),
)

ticket_text = st.text_area(
    "Ticket text",
    value=examples.get(selected_example, ""),
    height=160,
)

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "submitted_ticket" not in st.session_state:
    st.session_state.submitted_ticket = ""

if st.button("Route ticket", type="primary"):
    if not ticket_text.strip():
        st.warning("Please enter a ticket text first.")
    else:
        try:
            response = requests.post(
                PREDICT_URL,
                json={"text": ticket_text.strip()},
                timeout=30,
            )
            response.raise_for_status()
            st.session_state.prediction_result = response.json()
            st.session_state.submitted_ticket = ticket_text.strip()
        except requests.exceptions.Timeout:
            st.error("The API request timed out. Retry after the Render service starts.")
        except requests.exceptions.RequestException as exc:
            st.error(f"Could not complete prediction: {exc}")

result = st.session_state.prediction_result

if result:
    prediction = result.get("prediction", "Unknown")
    margin = result.get("margin")
    threshold = result.get("review_threshold")
    needs_review = bool(result.get("needs_review"))

    if needs_review:
        st.warning("⚠️ Human review required")
        st.write(
            "The top-2 SVM margin is below the review threshold "
            f"`{threshold:.3f}`."
        )
    else:
        st.success("✅ Ticket can be routed automatically")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Predicted queue", prediction)
    with col2:
        st.metric("Top-2 margin", f"{margin:.4f}" if margin is not None else "N/A")

    top_classes = result.get("top_classes", [])
    if top_classes:
        st.subheader("Suggested queues")
        for rank, candidate in enumerate(top_classes, start=1):
            st.write(
                f"**{rank}. {candidate.get('queue', 'Unknown')}** — "
                f"decision score `{candidate.get('decision_score', 0):.4f}`"
            )

    keywords = result.get("keywords", [])
    if keywords:
        st.subheader("Influential text features")
        st.write(" · ".join(f"`{keyword}`" for keyword in keywords))
        st.caption(
            "These TF-IDF features make the largest positive contribution "
            "to the predicted SVM class."
        )

    if needs_review and top_classes:
        st.divider()
        st.subheader("Reviewer decision")

        candidate_queues = [
            candidate.get("queue")
            for candidate in top_classes
            if candidate.get("queue")
        ]

        selected_queue = st.radio(
            "Choose the correct queue:",
            candidate_queues,
            index=0,
        )
        reviewer_note = st.text_input("Reviewer note (optional)")

        if st.button("Confirm reviewed queue"):
            try:
                review_response = requests.post(
                    REVIEW_URL,
                    json={
                        "text": st.session_state.submitted_ticket,
                        "model_prediction": prediction,
                        "selected_queue": selected_queue,
                        "margin": margin,
                        "reviewer_note": reviewer_note or None,
                    },
                    timeout=20,
                )
                review_response.raise_for_status()
                review_result = review_response.json()

                if review_result.get("was_overridden"):
                    st.success(
                        f"Review confirmed. Final queue: **{selected_queue}** "
                        "(model prediction overridden)."
                    )
                else:
                    st.success(
                        f"Review confirmed. Final queue: **{selected_queue}**."
                    )

                st.info(
                    "Demo limitation: the API acknowledges the decision but "
                    "does not yet persist it in a database."
                )
            except requests.exceptions.RequestException as exc:
                st.error(f"Could not submit review decision: {exc}")

    with st.expander("Raw API response"):
        st.json(result)

st.divider()
st.caption(f"Backend API documentation: {API_BASE_URL}/docs")

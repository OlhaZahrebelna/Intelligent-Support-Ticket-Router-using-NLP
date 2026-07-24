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

REQUIRED_PREDICTION_FIELDS = {
    "prediction",
    "margin",
    "review_threshold",
    "needs_review",
    "top_classes",
    "keywords",
}


st.set_page_config(
    page_title="Support Ticket Router",
    page_icon="🎫",
    layout="centered",
)

st.title("🎫 Intelligent Support Ticket Router")
st.write(
    "High-margin tickets are routed automatically. Low-margin predictions "
    "are escalated for human review."
)
st.divider()


with st.sidebar:
    st.header("About")
    st.write("Primary model: TF-IDF + Linear SVM")
    st.write("Review signal: top-2 SVM decision margin")
    st.write("Backend: FastAPI")
    st.write("Deployment: Render")

    if st.button("Check API health"):
        try:
            response = requests.get(HEALTH_URL, timeout=20)
            response.raise_for_status()
            data = response.json()

            st.success(f"API status: {data.get('status')}")
            st.write(f"Model loaded: {data.get('model_loaded')}")
            st.write(f"API version: {data.get('api_version', 'unknown')}")
            st.write(
                "Review threshold: "
                f"`{data.get('review_margin_threshold', 'N/A')}`"
            )
        except requests.exceptions.RequestException as exc:
            st.error(f"Could not connect to API: {exc}")


st.subheader("Enter a support ticket")

examples = {
    "Billing issue": "I was charged twice for my monthly subscription.",
    "Login issue": "I cannot log in and need to reset my password.",
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
    placeholder="Example: I was charged twice for my subscription.",
)

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if "submitted_ticket" not in st.session_state:
    st.session_state.submitted_ticket = ""

if "review_submitted" not in st.session_state:
    st.session_state.review_submitted = False


if st.button("Route ticket", type="primary"):
    if not ticket_text.strip():
        st.warning("Please enter a ticket text first.")
    else:
        st.session_state.review_submitted = False

        with st.spinner("Sending ticket to the FastAPI model..."):
            try:
                response = requests.post(
                    PREDICT_URL,
                    json={"text": ticket_text.strip()},
                    timeout=30,
                )
                response.raise_for_status()

                result = response.json()
                missing_fields = (
                    REQUIRED_PREDICTION_FIELDS - set(result.keys())
                )

                if missing_fields:
                    st.session_state.prediction_result = None
                    st.error(
                        "The backend API is incompatible with this frontend. "
                        "Missing fields: "
                        + ", ".join(sorted(missing_fields))
                    )
                else:
                    st.session_state.prediction_result = result
                    st.session_state.submitted_ticket = ticket_text.strip()

            except requests.exceptions.Timeout:
                st.error(
                    "The API request timed out. A sleeping Render service "
                    "may need one additional request after startup."
                )
            except requests.exceptions.RequestException as exc:
                st.error(f"Could not complete prediction: {exc}")


result = st.session_state.prediction_result

if result:
    prediction = result["prediction"]
    margin = float(result["margin"])
    threshold = float(result["review_threshold"])
    needs_review = bool(result["needs_review"])

    if needs_review:
        st.warning("⚠️ Human review required")
        st.write(
            "The difference between the two strongest SVM scores is below "
            f"the threshold `{threshold:.3f}`."
        )
    else:
        st.success("✅ Ticket can be routed automatically")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Predicted queue", prediction)

    with col2:
        st.metric("Top-2 margin", f"{margin:.4f}")

    top_classes = result.get("top_classes", [])

    if top_classes:
        st.subheader("Suggested queues")

        for rank, candidate in enumerate(top_classes, start=1):
            st.write(
                f"**{rank}. {candidate['queue']}** — "
                f"decision score `{candidate['decision_score']:.4f}`"
            )

    keywords = result.get("keywords", [])

    if keywords:
        st.subheader("Influential text features")
        st.write(" · ".join(f"`{keyword}`" for keyword in keywords))
        st.caption(
            "TF-IDF features with the largest positive contribution to the "
            "predicted SVM class."
        )

    if needs_review and top_classes:
        st.divider()
        st.subheader("Reviewer decision")

        model_candidates = [
            candidate["queue"]
            for candidate in top_classes
        ]
        all_queues = [
            "Billing and Payments",
            "Customer Service",
            "General Inquiry",
            "Human Resources",
            "IT Support",
            "Product Support",
            "Returns and Exchanges",
            "Sales and Pre-Sales",
            "Service Outages and Maintenance",
            "Technical Support",
        ]
        review_options = model_candidates + [
            queue for queue in all_queues
            if queue not in model_candidates
        ]

        selected_queue = st.selectbox(
            "Choose the correct queue:",
            review_options,
        )

        reviewer_note = st.text_input(
            "Reviewer note (optional)",
            placeholder="Why was this queue selected?",
        )

        if st.button(
            "Confirm reviewed queue",
            disabled=st.session_state.review_submitted,
        ):
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
                st.session_state.review_submitted = True

                if review_result.get("was_overridden"):
                    st.success(
                        "Review confirmed. The model prediction was "
                        f"overridden with **{selected_queue}**."
                    )
                else:
                    st.success(
                        "Review confirmed. The reviewer accepted "
                        f"**{selected_queue}**."
                    )

                st.info(
                    "P0 limitation: the API acknowledges the decision but "
                    "does not store it in a database yet."
                )

            except requests.exceptions.RequestException as exc:
                st.error(f"Could not submit review decision: {exc}")

    with st.expander("Raw API response"):
        st.json(result)


st.divider()
st.caption(f"Backend API documentation: {API_BASE_URL}/docs")

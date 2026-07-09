import requests
import streamlit as st

API_URL = "https://intelligent-support-ticket-router.onrender.com/predict"
HEALTH_URL = "https://intelligent-support-ticket-router.onrender.com/health"

st.set_page_config(
    page_title="Support Ticket Router",
    page_icon="🎫",
    layout="centered",
)

st.title("🎫 Intelligent Support Ticket Router")
st.write(
    "This app classifies customer support tickets into the most likely support category "
    "using a deployed FastAPI machine learning API."
)

st.divider()

with st.sidebar:
    st.header("About")
    st.write("Model: TF-IDF + Linear SVM")
    st.write("Backend: FastAPI")
    st.write("Deployment: Render")

    if st.button("Check API health"):
        try:
            response = requests.get(HEALTH_URL, timeout=20)
            if response.status_code == 200:
                data = response.json()
                st.success(f"API status: {data.get('status')}")
                st.write(f"Model loaded: {data.get('model_loaded')}")
            else:
                st.error(f"API returned status code {response.status_code}")
        except requests.exceptions.RequestException as e:
            st.error(f"Could not connect to API: {e}")

st.subheader("Enter a support ticket")

examples = {
    "Billing issue": "I was charged twice for my monthly subscription.",
    "Login issue": "I cannot login to my account and I need to reset my password.",
    "Technical issue": "The app crashes every time I try to open it.",
    "Cancellation request": "I want to cancel my subscription and close my account.",
}

selected_example = st.selectbox(
    "Choose an example or write your own ticket:",
    ["Custom"] + list(examples.keys()),
)

if selected_example != "Custom":
    default_text = examples[selected_example]
else:
    default_text = ""

ticket_text = st.text_area(
    "Ticket text",
    value=default_text,
    height=160,
    placeholder="Example: I was charged twice for my monthly subscription.",
)

if st.button("Predict category", type="primary"):
    if not ticket_text.strip():
        st.warning("Please enter a ticket text first.")
    else:
        with st.spinner("Sending ticket to the FastAPI model..."):
            try:
                response = requests.post(
                    API_URL,
                    json={"text": ticket_text},
                    timeout=30,
                )

                if response.status_code == 200:
                    result = response.json()

                    st.success("Prediction completed")

                    st.metric(
                        label="Predicted category",
                        value=result.get("prediction", "Unknown"),
                    )

                    score = result.get("score")
                    if score is not None:
                        st.write(f"Model score: `{score:.4f}`")
                    else:
                        st.write("Model score: `N/A`")

                    with st.expander("Raw API response"):
                        st.json(result)

                else:
                    st.error(f"API error: status code {response.status_code}")
                    st.write(response.text)

            except requests.exceptions.Timeout:
                st.error(
                    "The API request timed out. "
                    "If the Render service was sleeping, wait a moment and try again."
                )
            except requests.exceptions.RequestException as e:
                st.error(f"Could not connect to the API: {e}")

st.divider()

st.caption(
    "Backend API: https://intelligent-support-ticket-router.onrender.com/docs"
)

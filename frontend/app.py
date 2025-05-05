import streamlit as st
import requests

st.set_page_config(page_title="GrammaticaX - Essay Evaluator", layout="wide")

st.title("📝 GrammaticaX - Automated Essay Scoring")

# Input options
input_mode = st.radio("Choose Input Method", ["Type Essay", "Upload Essay"])

if input_mode == "Type Essay":
    essay_text = st.text_area("Enter your essay below:", height=300)
else:
    uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])
    if uploaded_file:
        essay_text = uploaded_file.read().decode("utf-8")
    else:
        essay_text = ""

# Button to evaluate essay
if st.button("🔍 Evaluate Essay") and essay_text.strip():
    with st.spinner("Scoring your essay..."):
        # Replace with your FastAPI endpoint
        response = requests.post("http://localhost:8000/predict", json={"essay": essay_text})

        if response.status_code == 200:
            result = response.json()

            st.success("✅ Essay Evaluated Successfully!")

            # Display scores
            st.metric("Final Score", result["final_score"])
            st.metric("Grammar Score", result["grammar_score"])
            st.metric("Content Score", result["content_score"])
            st.metric("Structure Score", result["structure_score"])
            st.metric("Sentiment Polarity", result["sentiment_polarity"])
            st.metric("Sentiment Subjectivity", result["sentiment_subjectivity"])

            # Feedback
            st.markdown("### 💬 Feedback")
            st.info(result["feedback"])
        else:
            st.error("❌ Error evaluating essay.")

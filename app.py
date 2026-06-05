import streamlit as st
import pickle
from helper import preprocess_text
import nltk
  
# 1. Page Configuration
st.set_page_config(
    page_title="MindCare | NLP Text Analytics",
    page_icon="🌿",
    layout="centered"
)

# 2. High-Readability Dark Theme CSS
st.markdown("""
    <style>
    /* 1. Main Background: Deep Black/Charcoal */
    .stApp {
        background-color: #0A0F14 !important;
    }
    
    /* 2. Global Text: Force all standard prose to be Crisp White/Off-White */
    html, body, [class*="css"], p, span, label {
        color: #E0E6ED !important; 
    }
    
    /* 3. Headers: High-Visibility Neon Mint Green */
    h1, h2, h3 {
        color: #2ECC71 !important;
        text-align: center;
        font-weight: 700;
    }
    
    /* 4. Text Area: Dark background, bright white text, neon border when typing */
    .stTextArea textarea {
        background-color: #141B22 !important;
        color: #FFFFFF !important;
        border: 2px solid #2C3E50 !important;
        border-radius: 12px;
        padding: 15px;
        font-size: 16px;
    }
    .stTextArea textarea:focus {
        border-color: #2ECC71 !important;
    }
    
/* 5. Predict Button: Signature MindCare Mint */
    .stButton>button {
        background-color: #2ECC71 !important; /* Bright Mint Green */
        color: #0A0F14 !important; /* Dark charcoal text for perfect contrast */
        border-radius: 25px;
        width: 100%;
        border: none;
        font-weight: bold;
        font-size: 18px;
        padding: 10px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #27AE60 !important; /* Deeper green on hover */
        color: #FFFFFF !important;
        box-shadow: 0 4px 15px rgba(46, 204, 113, 0.4);
    }
    
    /* 6. Custom Alert Boxes optimized for Dark Backgrounds */
    div[data-testid="stError"] {
        background-color: #2C1A1D !important;
        color: #FF6B6B !important;
        border: 1px solid #E74C3C !important;
        border-left: 6px solid #E74C3C !important;
    }
    div[data-testid="stSuccess"] {
        background-color: #142C1E !important;
        color: #2ECC71 !important;
        border: 1px solid #2ECC71 !important;
        border-left: 6px solid #2ECC71 !important;
    }
    
    /* 7. Metric Styling for Dark Mode */
    div[data-testid="stMetricValue"] {
        color: #00D2FF !important;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Load Resources
@st.cache_resource
def load_resources():
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
    return tfidf, model

try:
    tfidf, model = load_resources()
except FileNotFoundError:
    st.error("Model files missing! Please ensure 'vectorizer.pkl' and 'model.pkl' are in the directory.")
    st.stop()

# 4. Header Section
st.markdown("<h1>🌿 MindCare Analytics</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px; color: #A0AAB5 !important;'>An empathetic NLP tool for early detection of depressive linguistic patterns.</p>", unsafe_allow_html=True)
st.markdown("<hr style='border-color: #2C3E50;'>", unsafe_allow_html=True)

# 5. Input Section
user_input = st.text_area(
    "How are you feeling? / Paste chat context below:",
    height=180,
    placeholder="Type your thoughts, journal entry, or chat snippet here..."
)

# 6. Prediction Logic & UI
if st.button("Analyze Text"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing linguistic patterns..."):
            # Preprocess & Vectorize
            transformed_text = preprocess_text(user_input)
            vector_input = tfidf.transform([transformed_text])

            # Prediction & Probability
            prediction = model.predict(vector_input)[0]
            probability = model.predict_proba(vector_input)[0][1]

            st.markdown("<hr style='border-color: #2C3E50;'>", unsafe_allow_html=True)
            st.subheader("Analysis Results")

            col1, col2 = st.columns([1, 1])

            with col1:
                if prediction == 1:
                    st.error("🚨 **Result:** Indicators Detected")
                    st.write("The text contains linguistic markers commonly aligned with depressive expressions.")
                else:
                    st.success("🌱 **Result:** No Strong Indicators")
                    st.write("The text does not show significant statistical signs of depressive patterns.")

            with col2:
                st.metric(label="Model Confidence", value=f"{probability * 100:.1f}%")
                if prediction == 1:
                    st.progress(probability)
                else:
                    st.progress(1.0 - probability)

# 7. Ethical Disclaimer Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.info(
    "**⚠️ Disclaimer:** This application is a machine learning experiment intended for educational purposes only. "
    "It is not a clinical diagnostic tool. If you or someone you know is struggling, please reach out to a professional mental health service."
)

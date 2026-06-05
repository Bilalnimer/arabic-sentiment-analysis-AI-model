import sys
import asyncio

# Fix for Python 3.13 + Windows + Streamlit asyncio bug
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


import streamlit as st
import torch
from transformers import pipeline


@st.cache_resource
def load_model():
    """
    Loads the fine-tuned MARBERTv2 pipeline. 
    Cached to prevent reloading on every button click.
    """
    device = 0 if torch.cuda.is_available() else -1
    model_path = "./my_best_arabic_sentiment_model"
    
    pipe = pipeline(
        task="text-classification", 
        model=model_path,
        device=device
    )
    return pipe

def predict_sentiment(text, pipe):
    """Processes the text and formats the output."""
    id_to_sentiment = {
        "LABEL_0": ("سلبي (Negative) 😡", "#ff4b4b"),
        "LABEL_1": ("محايد (Neutral) 😐", "#808495"),
        "LABEL_2": ("إيجابي (Positive) 😍", "#00cc96")
    }
    
    result = pipe(text)[0]
    label_text, color = id_to_sentiment[result['label']]
    confidence = result['score'] * 100
    
    return label_text, color, confidence

# ==========================================
# USER INTERFACE (GUI)
# ==========================================

def render_ui():
    # Force Right-to-Left (RTL) alignment for Arabic text
    st.markdown("""
        <style>
        .stTextArea textarea {direction: RTL; text-align: right;}
        </style>
    """, unsafe_allow_html=True)

    st.title("Arabic Sentiment Analysis")
    st.markdown("---")

    # Load model in the background
    with st.spinner("Loading AI weights..."):
        sentiment_pipe = load_model()

    # User Input
    user_text = st.text_area(
        "Enter your Arabic text here (أدخل النص هنا):", 
        height=150,
        placeholder="مثال: الخدمة كانت ممتازة جداً والتوصيل سريع..."
    )

    # Action Button
    if st.button("Analyze Sentiment", type="primary"):
        if not user_text.strip():
            st.warning("Please enter some text first.")
        else:
            with st.spinner("Analyzing context..."):
                label, color, conf = predict_sentiment(user_text, sentiment_pipe)
                
                st.markdown("---")
                st.subheader("Results")
                
                # Display metrics in a clean row
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Detected Sentiment", label.split(" ")[0])
                with col2:
                    st.metric("Confidence", f"{conf:.1f}%")
                
                # Visual confidence bar
                st.markdown(f"**Confidence Level:**")
                st.progress(int(conf))

if __name__ == "__main__":
    st.set_page_config(
        page_title="Arabic Sentiment UI", 
        page_icon="🧠", 
        layout="centered"
    )
    render_ui()
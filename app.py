import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForQuestionAnswering, pipeline
import matplotlib.pyplot as plt
from model import Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource(show_spinner="⏳ Loading models…")
def load_models():
    # Emotion model
    emotion_model_name = "cardiffnlp/twitter-roberta-base-emotion"
    emotion_tokenizer = AutoTokenizer.from_pretrained(emotion_model_name)
    emotion_model = AutoModelForSequenceClassification.from_pretrained(emotion_model_name)
    emotion_pipeline = pipeline("text-classification", model=emotion_model, tokenizer=emotion_tokenizer, device=0 if torch.cuda.is_available() else -1)

    # QA model
    qa_model_name = "distilbert-base-uncased-distilled-squad"
    qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
    qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)
    qa_pipeline = pipeline("question-answering", model=qa_model, tokenizer=qa_tokenizer, device=0 if torch.cuda.is_available() else -1)

    return emotion_pipeline, qa_pipeline, qa_tokenizer, qa_model

# --- ESTILOS PERSONALIZADOS ---

st.markdown("""
    <style>
        .stApp, .main, .block-container {
            background-color: #fff !important;
        }
        .stTextInput>div>div>input,
        .stFileUploader>div>div,
        .stButton>button,
        .stDataFrame,
        .stFileUploader .stFileUploaderFileName {
            color: #000 !important;
        }
        .stTextInput>div>div>input {
            background-color: #26262f !important; /* o el color oscuro que uses */
            color: #fff !important;               /* letra blanca */
            border-radius: 8px !important;
}
        .stTable {
            background-color: #fff !important;
            color: #222 !important;
            border-radius: 8px !important;
            border: 1px solid #ddd !important;
        }
        /* Selectbox general */
        .stSelectbox>div>div>div,
        .stSelectbox>div>div>div>div {
            background-color: #000 !important;
            color: #fff !important;
            border-radius: 8px !important;
            border: 1px solid #ddd !important;
        }
        /* Selectbox específico en negro (caja cerrada y desplegada) */
        label[for^="Select the column where the comments to analyze are located"] + div > div > div,
        label[for^="Select the column where the comments to analyze are located"] + div > div > div > div,
        label[for^="Select the column where the comments to analyze are located"] + div input {
            background-color: #222 !important;
            color: #fff !important;
            border-radius: 8px !important;
            border: 1px solid #222 !important;
        }
        div[data-baseweb="popover"] ul {
            background-color: #222 !important;
            color: #fff !important;
        }
        div[data-baseweb="popover"] ul li {
            background-color: #222 !important;
            color: #fff !important;
        }
        div[data-baseweb="popover"] ul li[aria-selected="true"] {
            background-color: #444 !important;
            color: #fff !important;
        }
        label[for^="Select the column where the comments to analyze are located"] {
            color: #222 !important;
            font-weight: bold;
        }
        .stButton>button {
            background-color: #222 !important;
            color: #fff !important;
            border-radius: 8px !important;
            border: none !important;
            padding: 0.5em 1.5em !important;
            font-weight: 600 !important;
        }
        .stButton>button:hover {
            background-color: #444 !important;
            color: #fff !important;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #222 !important;
        }
        .stDataFrame, .stTable {
            background-color: #fff !important;
        }
        label, .stTextInput label, .css-1c7y2kd, .css-1n76uvr, .stTextInput>label {
            color: #222 !important;
        }
    </style>
""", unsafe_allow_html=True)

# --- CABECERA ---
st.markdown("<h1 style='color:#222;'>📝 Comment Analyzer</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='color:#444;font-size:1.1em;'>Welcome! Upload a CSV file with your comments and discover insights, word clouds, and emotional analysis. You can also ask questions about your data!</p>",
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("Upload your CSV file with comments", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown("#### Preview of your data:")
    st.dataframe(df.head())

    comment_column = st.selectbox("Pick the column containing the comments to analyze", df.columns)

    if comment_column != "Unnamed: 0":
        model = Model(df, comment_column)
        model.processing()

        st.markdown("### ☁️ Word Cloud")
        wordcloud_fig = model.wordcloud()
        st.pyplot(wordcloud_fig)

        st.markdown("### 📊 Most Frequent Words")
        bar_chart_fig = model.bar_chart()
        st.pyplot(bar_chart_fig)

        # Load models (emotion and QA)
        emotion_pipeline, qa_pipeline, qa_tokenizer, qa_model = load_models()

        # Emotion classification using the pipeline
        model.classification(emotion_pipeline)

        st.markdown("### 😃 Emotional Classification")
        results_chart_fig = model.results_chart()
        st.pyplot(results_chart_fig)

        question = st.text_input("Ask something about your comments:")

        if st.button("Get answer"):
            context_text = " ".join(df[comment_column].astype(str).tolist())
            try:
                answer = qa_pipeline(question=question, context=context_text)
                st.markdown(f"<div style='background:#000 ;padding:1em;border-radius:8px;'><b>Answer:</b> {answer['answer']}</div>", unsafe_allow_html=True)
            except Exception as e:
                st.markdown(f"<div style='color:red;'>Sorry, there was an error answering your question: {e}</div>", unsafe_allow_html=True)
    else:
        st.warning("Please select a valid column with comments to continue.")

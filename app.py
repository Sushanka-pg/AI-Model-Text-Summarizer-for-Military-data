import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    pipeline
)
import torch
import base64
import os
import requests
import threading
import socket 
import uuid
import streamlit.components.v1 as components
from PyPDF2 import PdfReader, PdfWriter

# ---- API KEYS (set from env in real usage) ----
MISTRAL_API_KEY = st.secrets["MISTRAL_API_KEY"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

# ---- Local Directories and Checkpoints ----
MODEL_PATHS = {
    "Flan-T5-Large": ("./local_model_dir/flan_t5_large", "google/flan-t5-large"),
    "LaMini-Flan-T5-248M": ("./local_model_dir/lamini_flan_t5_248m", "MBZUAI/LaMini-Flan-T5-248M"),
    "DistilBART-CNN-12-6": ("./local_model_dir/distilbart_cnn_12_6", "sshleifer/distilbart-cnn-12-6"),
    "Pegasus": ("./local_model_dir/pegasus_xsum", "google/pegasus-xsum")
}

stop_flag = threading.Event()

class ModelLoader:
    def __init__(self, name):
        self.name = name
        if name not in MODEL_PATHS:
            raise ValueError(f"Model '{name}' is not defined in MODEL_PATHS.")
        self.local_dir, self.checkpoint = MODEL_PATHS[name]

    def load(self):
        if not os.path.exists(self.local_dir):
            os.makedirs(self.local_dir, exist_ok=True)
            tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
            model = AutoModelForSeq2SeqLM.from_pretrained(self.checkpoint)
            tokenizer.save_pretrained(self.local_dir)
            model.save_pretrained(self.local_dir)
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.local_dir)
            model = AutoModelForSeq2SeqLM.from_pretrained(self.local_dir)
        return tokenizer, model

class PDFProcessor:
    def __init__(self, path):
        self.path = path

    def extract_chunks(self, num_chunks=10):
        loader = PyPDFLoader(self.path)
        pages = loader.load_and_split()
        full_text = ''.join([p.page_content for p in pages])
        chunk_size = max(300, len(full_text) // num_chunks)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_size//5
        )
        texts = splitter.split_documents(pages)
        return [t.page_content for t in texts]

class OfflineSummarizer:
    def __init__(self, tokenizer, model, length_choice, model_name):
        self.tokenizer = tokenizer
        self.model = model
        self.length_choice = length_choice
        self.model_name = model_name

    def get_length_params(self):
        choice = self.length_choice
        if choice == "Small": min_ratio, max_ratio = 0.1, 0.2; min_cap, max_cap = 20, 50
        elif choice == "Medium": min_ratio, max_ratio = 0.2, 0.5; min_cap, max_cap = 40, 100
        else: min_ratio, max_ratio = 0.4, 0.8; min_cap, max_cap = 80, 200
        return min_ratio, max_ratio, min_cap, max_cap

    def summarize(self, chunks):
        summaries = []
        min_r, max_r, min_c, max_c = self.get_length_params()

        if self.model_name == "Pegasus":
            for chunk in chunks:
                if stop_flag.is_set():
                    break
                prompt = (
                    "Summarize the following text with a concise paragraph. "
                    "If appropriate, add 3-5 key points as bullet points after the paragraph.\n\n"
                    f"{chunk}"
                )
                inputs = self.tokenizer(prompt, max_length=512, truncation=True, return_tensors="pt")
                with torch.no_grad():
                    summary_ids = self.model.generate(
                        inputs["input_ids"],
                        num_beams=2,
                        max_length=80,
                        min_length=25,
                        early_stopping=True
                    )
                summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                summaries.append(summary)
        else:
            pipe = pipeline('summarization', model=self.model, tokenizer=self.tokenizer)
            for chunk in chunks:
                if stop_flag.is_set():
                    break
                input_len = len(self.tokenizer(chunk, truncation=True)['input_ids'])
                max_len = min(max_c, int(input_len * max_r))
                min_len = max(min_c, int(input_len * min_r))
                result = pipe(chunk, truncation=True, max_length=max_len, min_length=min_len)
                summaries.append(result[0]['summary_text'])

        return "\n\n".join(summaries)

class OnlineSummarizer:
    def __init__(self, model_name, length_choice):
        self.model_name = model_name
        self.length_choice = length_choice

    def summarize(self, chunks):
        text = " ".join(chunks)
        if stop_flag.is_set():
            return "[Stopped by user]"
        if self.model_name == "Mistral":
            return self._summarize_mistral(text)
        elif self.model_name == "Gemini":
            return self._summarize_gemini(text)
        elif self.model_name == "BART":
            return self._summarize_bart(chunks)
        elif self.model_name == "Pegasus":
            return self._summarize_pegasus(chunks)
        return "[ERROR] Unknown model"

    def _summarize_mistral(self, text):
        word_limit = {"Small": 100, "Medium": 200, "Large": 350}.get(self.length_choice, 200)
        url = "https://api.mistral.ai/v1/chat/completions"
        prompt = f"Write a summary (max {word_limit} words). Paragraph + bullet points.\n\nText:\n{text[:3000]}"
        payload = {
            "model": "mistral-large-latest",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": word_limit * 2,
            "temperature": 0.5
        }
        headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"}
        try:
            res = requests.post(url, headers=headers, json=payload)
            res.raise_for_status()
            return res.json()['choices'][0]['message']['content']
        except Exception as e:
            return f"[Mistral API ERROR]: {e}"

    def _summarize_gemini(self, text):
        word_limit = {"Small": 100, "Medium": 200, "Large": 350}.get(self.length_choice, 200)
        url = "https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent"
        prompt = f"Write a summary (max {word_limit} words). Paragraph + bullet points.\n\nText:\n{text[:3000]}"
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        params = {"key": GEMINI_API_KEY}
        try:
            res = requests.post(url, params=params, headers={"Content-Type": "application/json"}, json=payload)
            res.raise_for_status()
            return res.json()['candidates'][0]['content']['parts'][0]['text']
        except Exception as e:
            return f"[Gemini API ERROR]: {e}"

    def _summarize_bart(self, chunks):
        tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
        model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")
        min_ratio, max_ratio, min_cap, max_cap = OfflineSummarizer(tokenizer, model, self.length_choice, "BART").get_length_params()
        pipe = pipeline('summarization', model=model, tokenizer=tokenizer)
        all_summaries = []
        for chunk in chunks:
            if stop_flag.is_set():
                break
            input_len = len(tokenizer(chunk, truncation=True)['input_ids'])
            max_len = min(max_cap, int(input_len * max_ratio), 512)
            min_len = max(min_cap, int(input_len * min_ratio))
            result = pipe(chunk, truncation=True, max_length=max_len, min_length=min_len)
            all_summaries.append(result[0]['summary_text'])
        return "\n\n".join(all_summaries)

    def _summarize_pegasus(self, chunks):
        tokenizer = AutoTokenizer.from_pretrained("google/pegasus-xsum")
        model = AutoModelForSeq2SeqLM.from_pretrained("google/pegasus-xsum")
        all_summaries = []
        for chunk in chunks:
            if stop_flag.is_set():
                break
            prompt = (
                "Summarize the following text with a concise paragraph. "
                "If appropriate, add 3-5 key points as bullet points after the paragraph.\n\n"
                f"{chunk}"
            )
            inputs = tokenizer(prompt, max_length=512, truncation=True, return_tensors="pt")
            with torch.no_grad():
                summary_ids = model.generate(
                    inputs["input_ids"],
                    num_beams=2,
                    max_length=80,
                    min_length=25,
                    early_stopping=True
                )
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            all_summaries.append(summary)
        return "\n\n".join(all_summaries)

def is_connected():
    try:
        socket.create_connection(("1.1.1.1", 53), timeout=2)
        return True
    except OSError:
        return False

# ---- UI Start ----
st.set_page_config(layout="wide")
st.title("📄 Document Summarization App (Offline or Online Model)")

mode_choice = st.radio("Select summarization mode:", ("Offline", "Online"), index=0)
offline_model = None
online_model = None

if mode_choice == "Offline":
    offline_model = st.selectbox(
        "Select offline model:",
        ("LaMini-Flan-T5-248M", "DistilBART-CNN-12-6", "Flan-T5-Large", "Pegasus"),
        index=0
    )
else:
    online_model = st.selectbox("Select online model:", ("Mistral", "Gemini", "BART", "Pegasus"), index=0)

summary_length = st.radio("Select summary length:", ("Small", "Medium", "Large"), index=1)
uploaded_file = st.file_uploader("📤 Upload your PDF file", type=['pdf'])

@st.cache_data(max_entries=3, ttl=3600)
def encode_pdf_bytes(file_bytes):
    return base64.b64encode(file_bytes).decode("utf-8")

# If file is uploaded
if uploaded_file is not None:
    os.makedirs("data", exist_ok=True)
    unique_id = str(uuid.uuid4())[:8]
    filename = f"{unique_id}_{uploaded_file.name if isinstance(uploaded_file.name, str) else 'uploaded.pdf'}"
    filepath = os.path.join("data", filename)

    file_bytes = uploaded_file.read()
    with open(filepath, "wb") as f:
        f.write(file_bytes)

    st.download_button(
        label="📥 Download Uploaded PDF",
        data=open(filepath, "rb").read(),
        file_name=filename,
        mime="application/pdf"
    )

    summarize_clicked = st.button("🧠 Summarize")
    if summarize_clicked:
        stop_flag.clear()
        processor = PDFProcessor(filepath)
        chunks = processor.extract_chunks()

        st.markdown("---")
        with st.spinner("Generating summary..."):
            if mode_choice == "Offline":
                loader = ModelLoader(offline_model)
                tokenizer, model = loader.load()
                summarizer = OfflineSummarizer(tokenizer, model, summary_length, offline_model)
                summary = summarizer.summarize(chunks)
            else:
                if not is_connected():
                    components.html(
                        """<script>alert("⚠️ No internet connection! Please turn on your network.");</script>""",
                        height=0,
                    )
                    summary = None
                else:
                    summarizer = OnlineSummarizer(online_model, summary_length)
                    summary = summarizer.summarize(chunks)

            if summary:
                st.session_state["last_summary"] = summary   

                # ---- Save summary automatically in same folder ----
                base_name, _ = os.path.splitext(filename)
                summary_path = os.path.join("data", f"{base_name}_summary.txt")
                with open(summary_path, "w", encoding="utf-8") as f:
                    f.write(summary)
                st.success(f"✅ Summary saved at: {summary_path}")

# ---- Summary Output ----
if "last_summary" in st.session_state and st.session_state["last_summary"]:
    summary = st.session_state["last_summary"]
    safe_summary = summary.encode("utf-8", "replace").decode("utf-8")

    # Download as Save As option
    st.download_button(
        label="💾 Save As (Download .txt)",
        data=safe_summary,
        file_name="summary.txt",
        mime="text/plain"
    )

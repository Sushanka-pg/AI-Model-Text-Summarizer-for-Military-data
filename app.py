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

MISTRAL_API_KEY = st.secrets["MISTRAL_API_KEY"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

@st.cache_resource
def get_cached_offline_model(model_name):
    try:
        if model_name == "LaMini-Flan-T5-248M":
            tokenizer = AutoTokenizer.from_pretrained("MBZUAI/LaMini-Flan-T5-248M")
            model = AutoModelForSeq2SeqLM.from_pretrained("MBZUAI/LaMini-Flan-T5-248M")
            return tokenizer, model
        elif model_name == "DistilBART-CNN-12-6":
            return None, None  # Use pipeline instead for speed
        elif model_name == "Flan-T5-Large":
            st.warning("⚠️ Flan-T5-Large is very slow for demos. Consider LaMini instead.")
            tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
            model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
            return tokenizer, model
        elif model_name == "Pegasus":
            tokenizer = AutoTokenizer.from_pretrained("google/pegasus-xsum")
            model = AutoModelForSeq2SeqLM.from_pretrained("google/pegasus-xsum")
            return tokenizer, model
    except Exception as e:
        st.error(f"Failed to load {model_name}: {e}")
        return None, None
    return None, None

@st.cache_resource
def get_cached_pipeline(model_name):
    try:
        if model_name == "DistilBART-CNN-12-6":
            return pipeline('summarization', 
                          model="sshleifer/distilbart-cnn-6-6",  # Smaller, faster version
                          device=0 if torch.cuda.is_available() else -1)
        elif model_name == "online_bart":
            return pipeline('summarization', 
                          model="sshleifer/distilbart-cnn-6-6",
                          device=0 if torch.cuda.is_available() else -1)
    except Exception as e:
        st.error(f"Failed to load pipeline for {model_name}: {e}")
        return None

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
        cached_result = get_cached_offline_model(self.name)
        if cached_result[0] is not None:
            return cached_result

        with st.spinner(f"First-time loading {self.name}... This may take 2-3 minutes"):
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

    def extract_chunks(self, num_chunks=5):  # REDUCED from 10 to 5 for speed
        loader = PyPDFLoader(self.path)
        pages = loader.load_and_split()
        full_text = ''.join([p.page_content for p in pages])
        
        # SPEED FIX: Limit very long documents
        if len(full_text) > 8000:
            full_text = full_text[:8000]
            st.info(" Document truncated to 8000 chars for faster processing")
            
        chunk_size = max(300, len(full_text) // num_chunks)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_size//10  # REDUCED overlap for speed
        )
        texts = splitter.split_documents(pages)
        chunks = [t.page_content for t in texts]
        
        # SPEED FIX: Limit number of chunks processed
        if len(chunks) > num_chunks:
            chunks = chunks[:num_chunks]
            st.info(f"Processing first {num_chunks} chunks for speed")
            
        return chunks

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

        progress_bar = st.progress(0)
        st.write(f"🔄 Processing {len(chunks)} chunks...")

        if self.model_name == "DistilBART-CNN-12-6":
            pipe = get_cached_pipeline("DistilBART-CNN-12-6")
            if pipe is None:
                st.error("Failed to load DistilBART pipeline")
                return "Error loading model"
                
            for i, chunk in enumerate(chunks):
                if stop_flag.is_set():
                    break
                progress_bar.progress((i + 1) / len(chunks))
                
                input_len = len(chunk.split())  # Faster than tokenizer
                max_len = min(max_c, max(min_c, int(input_len * max_r)))
                min_len = max(min_c, int(input_len * min_r))
                
                try:
                    result = pipe(chunk[:1000], truncation=True, max_length=max_len, min_length=min_len, do_sample=False)
                    summaries.append(result[0]['summary_text'])
                except Exception as e:
                    st.warning(f"Chunk {i+1} failed: {str(e)[:50]}")
                    continue

        elif self.model_name == "Pegasus":
            for i, chunk in enumerate(chunks):
                if stop_flag.is_set():
                    break
                progress_bar.progress((i + 1) / len(chunks))
                
                inputs = self.tokenizer(chunk[:800], max_length=512, truncation=True, return_tensors="pt")
                with torch.no_grad():
                    summary_ids = self.model.generate(
                        inputs["input_ids"],
                        num_beams=2,
                        max_length=60,  
                        min_length=20,
                        early_stopping=True
                    )
                summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                summaries.append(summary)
        else:
            pipe = pipeline('summarization', model=self.model, tokenizer=self.tokenizer)
            for i, chunk in enumerate(chunks):
                if stop_flag.is_set():
                    break
                progress_bar.progress((i + 1) / len(chunks))
                
                try:
                    input_len = len(self.tokenizer(chunk, truncation=True)['input_ids'])
                    max_len = min(max_c, int(input_len * max_r))
                    min_len = max(min_c, int(input_len * min_r))
                    result = pipe(chunk, truncation=True, max_length=max_len, min_length=min_len)
                    summaries.append(result[0]['summary_text'])
                except Exception as e:
                    st.warning(f"Chunk {i+1} failed: {str(e)[:50]}")
                    continue

        progress_bar.progress(1.0)
        return "\n\n".join(summaries) if summaries else "No summary generated"

class OnlineSummarizer:
    def __init__(self, model_name, length_choice):
        self.model_name = model_name
        self.length_choice = length_choice

    def summarize(self, chunks):
        text = " ".join(chunks)
        # SPEED FIX: Limit input size for faster API calls
        text = text[:4000]  # Reduced from unlimited
        
        if stop_flag.is_set():
            return "[Stopped by user]"
        if self.model_name == "Mistral":
            return self._summarize_mistral(text)
        elif self.model_name == "Gemini":
            return self._summarize_gemini(text)
        elif self.model_name == "BART":
            return self._summarize_bart(chunks[:3])  # SPEED FIX: Limit chunks
        elif self.model_name == "Pegasus":
            return self._summarize_pegasus(chunks[:3])  # SPEED FIX: Limit chunks
        return "[ERROR] Unknown model"

    def _summarize_mistral(self, text):
        word_limit = {"Small": 100, "Medium": 200, "Large": 350}.get(self.length_choice, 200)
        url = "https://api.mistral.ai/v1/chat/completions"
        prompt = f"Write a concise military document summary (max {word_limit} words):\n\n{text}"
        payload = {
            "model": "mistral-small-latest",  # SPEED FIX: Use smaller, faster model
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": word_limit + 50,  # REDUCED for speed
            "temperature": 0.3  # REDUCED for faster, more focused responses
        }
        headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"}
        try:
            with st.spinner("🚀 Calling Mistral API..."):
                res = requests.post(url, headers=headers, json=payload, timeout=30)  # Added timeout
                res.raise_for_status()
                return res.json()['choices'][0]['message']['content']
        except Exception as e:
            return f"[Mistral API ERROR]: {e}"

    def _summarize_gemini(self, text):
        word_limit = {"Small": 100, "Medium": 200, "Large": 350}.get(self.length_choice, 200)
        url = "https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent"
        prompt = f"Write a concise military document summary (max {word_limit} words):\n\n{text}"
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        params = {"key": GEMINI_API_KEY}
        try:
            with st.spinner("🚀 Calling Gemini API..."):
                res = requests.post(url, params=params, headers={"Content-Type": "application/json"}, 
                                  json=payload, timeout=30)  # Added timeout
                res.raise_for_status()
                return res.json()['candidates'][0]['content']['parts'][0]['text']
        except Exception as e:
            return f"[Gemini API ERROR]: {e}"

    def _summarize_bart(self, chunks):
        # SPEED FIX: Use cached pipeline
        pipe = get_cached_pipeline("online_bart")
        if pipe is None:
            return "[ERROR] Failed to load BART pipeline"
            
        min_ratio, max_ratio, min_cap, max_cap = OfflineSummarizer(None, None, self.length_choice, "BART").get_length_params()
        
        all_summaries = []
        progress_bar = st.progress(0)
        
        for i, chunk in enumerate(chunks):
            if stop_flag.is_set():
                break
            progress_bar.progress((i + 1) / len(chunks))
            
            try:
                input_len = len(chunk.split())  # Faster estimation
                max_len = min(max_cap, int(input_len * max_ratio), 150)  # CAPPED for speed
                min_len = max(min_cap, int(input_len * min_ratio))
                result = pipe(chunk[:1000], truncation=True, max_length=max_len, min_length=min_len, do_sample=False)
                all_summaries.append(result[0]['summary_text'])
            except Exception as e:
                st.warning(f"BART chunk {i+1} failed: {str(e)[:50]}")
                continue
                
        progress_bar.progress(1.0)
        return "\n\n".join(all_summaries) if all_summaries else "BART summarization failed"

    def _summarize_pegasus(self, chunks):
        # SPEED FIX: Use cached model
        tokenizer, model = get_cached_offline_model("Pegasus")
        if tokenizer is None:
            return "[ERROR] Failed to load Pegasus model"
            
        all_summaries = []
        progress_bar = st.progress(0)
        
        for i, chunk in enumerate(chunks):
            if stop_flag.is_set():
                break
            progress_bar.progress((i + 1) / len(chunks))
            
            try:
                # SPEED FIX: Simplified processing
                inputs = tokenizer(chunk[:800], max_length=512, truncation=True, return_tensors="pt")
                with torch.no_grad():
                    summary_ids = model.generate(
                        inputs["input_ids"],
                        num_beams=2,
                        max_length=60,  # REDUCED for speed
                        min_length=20,
                        early_stopping=True
                    )
                summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                all_summaries.append(summary)
            except Exception as e:
                st.warning(f"Pegasus chunk {i+1} failed: {str(e)[:50]}")
                continue
                
        progress_bar.progress(1.0)
        return "\n\n".join(all_summaries) if all_summaries else "Pegasus summarization failed"

def is_connected():
    try:
        socket.create_connection(("1.1.1.1", 53), timeout=2)
        return True
    except OSError:
        return False

# ---- UI Start ----
st.set_page_config(layout="wide")
st.title("Document Summarization App (Offline or Online Model)")

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
uploaded_file = st.file_uploader("Upload your PDF file", type=['pdf'])

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
        label="Download Uploaded PDF",
        data=open(filepath, "rb").read(),
        file_name=filename,
        mime="application/pdf"
    )

    summarize_clicked = st.button("Summarize")
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
        label="Save As (Download .txt)",
        data=safe_summary,
        file_name="summary.txt",
        mime="text/plain"
    )

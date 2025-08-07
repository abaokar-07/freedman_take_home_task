from pathlib import Path
import pickle
import torch
import gradio as gr

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
import logging

# ✅ Configs
PDF_PATH = Path("machine_learning.pdf")
CACHE_DIR = Path("cache/cache")
CACHE_DIR.mkdir(exist_ok=True)

DOCS_PICKLE_PATH = CACHE_DIR / "documents.pkl"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "tiiuae/falcon-h1-1.5b-deep-instruct"
VECTOR_DB_NAME = "vectorstore"
VECTOR_DB_PATH = CACHE_DIR / f"{VECTOR_DB_NAME}.faiss"

from huggingface_hub import login
login("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")

# ✅ Load & Chunk PDF
def load_and_split_pdf(pdf_path):
    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_documents(docs)

if DOCS_PICKLE_PATH.exists():
    with open(DOCS_PICKLE_PATH, "rb") as f:
        documents = pickle.load(f)
else:
    documents = load_and_split_pdf(PDF_PATH)
    with open(DOCS_PICKLE_PATH, "wb") as f:
        pickle.dump(documents, f)

# ✅ Load or Download Embeddings
EMBEDDING_MODEL_CACHE_DIR = CACHE_DIR / "embedding_model"
EMBEDDING_MODEL_CACHE_DIR.mkdir(exist_ok=True)

if any(EMBEDDING_MODEL_CACHE_DIR.iterdir()):
    embeddings = HuggingFaceEmbeddings(model_name=str(EMBEDDING_MODEL_CACHE_DIR))
else:
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    model.save(str(EMBEDDING_MODEL_CACHE_DIR))
    embeddings = HuggingFaceEmbeddings(model_name=str(EMBEDDING_MODEL_CACHE_DIR))

# ✅ Load or Create FAISS Vector DB
if VECTOR_DB_PATH.exists():
    vector_store = FAISS.load_local(
        folder_path=str(CACHE_DIR),
        index_name=VECTOR_DB_NAME,
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )
else:
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local(str(CACHE_DIR), index_name=VECTOR_DB_NAME)

# ✅ Load or Download LLM
HF_MODEL_CACHE_DIR = CACHE_DIR / "falcon_deep_instruct"
if HF_MODEL_CACHE_DIR.exists():
    # tokenizer = AutoTokenizer.from_pretrained(str(HF_MODEL_CACHE_DIR), use_fast=False)
    tokenizer = AutoTokenizer.from_pretrained(str(HF_MODEL_CACHE_DIR))
    model = AutoModelForCausalLM.from_pretrained(
        str(HF_MODEL_CACHE_DIR),
        torch_dtype=torch.float32,
        # device_map="auto"
    )
    model.to("cpu")
else:
    # tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME, use_fast=False)
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME,
            torch_dtype=torch.float32,
            # device_map="auto",
            cache_dir=str(HF_MODEL_CACHE_DIR)
        )
    model.to("cpu")
    tokenizer.save_pretrained(str(HF_MODEL_CACHE_DIR))
    model.save_pretrained(str(HF_MODEL_CACHE_DIR))

import re

def clean_context(text):
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if re.search(r"(www\.|http[s]?://)", line, re.IGNORECASE):
            continue
        if re.match(r"\s*(ijrar|volume|issue|issn|page\s*\d+)", line, re.IGNORECASE):
            continue
        cleaned.append(line)
    return " ".join(cleaned)

# ✅ Context and Prompt Building
def get_context_from_query(query, k, max_context_tokens=600):
    docs = vector_store.similarity_search(query, k=k)
    # reranked_docs = rerank_passages(query, docs)
    # top_5_docs = reranked_docs[:5]
    combined_text = "\n\n".join([clean_context(doc.page_content) for doc in docs])
    tokens = tokenizer(combined_text, return_tensors="pt", truncation=False).input_ids[0]

    if len(tokens) > max_context_tokens:
        tokens = tokens[:max_context_tokens]
        combined_text = tokenizer.decode(tokens, skip_special_tokens=True)

    return combined_text, docs

def build_prompt(context, question):
    return f"""You are an expert machine learning assistant. Based on the context provided below, answer the user's question in a clear and concise way.

    Context:
    {context}

    Question:
    {question}

    Answer:"""

# ✅ QA Function (Non-streaming)
def ask_question_safe(query):
    try:
        context, _ = get_context_from_query(query, k=5)
        prompt = build_prompt(context, query)

        input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.to(model.device)
        input_length = input_ids.shape[-1]

        # max_model_len = model.config.max_position_embeddings
        # max_new_tokens = 512

        # if input_length + max_new_tokens > max_model_len:
        #     max_new_tokens = max_model_len - input_length
        #     if max_new_tokens <= 0:
        #         return "Prompt too long. Please ask a shorter question."

        max_new_tokens = 512

        print("\n=== PROMPT ===\n")
        print(prompt)
        print("\n==============\n")

        output_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=512,
        temperature=0.9,              # Encourage more exploration
        top_p=0.95,                   # Nucleus sampling
        do_sample=False,              # Enable sampling
        repetition_penalty=1.1,      # Discourage repeating short answers
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

        # answer = tokenizer.decode(output_ids[0][input_length:], skip_special_tokens=True)
        answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return answer

    except Exception as e:
        return f"Error during generation: {e}"

# ✅ Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 📘 PDF Q&A Assistant")

    with gr.Row():
        with gr.Column():
            question_input = gr.Textbox(lines=1, label="❓ Ask a question")
            answer_output = gr.Textbox(lines=10, label="💬 Answer", interactive=False)
            question_input.submit(fn=ask_question_safe, inputs=question_input, outputs=answer_output)

        import base64

        def encode_pdf_to_base64(pdf_path):
            with open(pdf_path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
            return f"data:application/pdf;base64,{encoded}"

        with gr.Column():
            gr.Markdown("### 📄 PDF Preview")

            encoded_pdf = encode_pdf_to_base64(PDF_PATH)
            pdf_iframe = f"""
            <iframe
                src="{encoded_pdf}"
                width="100%"
                height="600px"
                style="border: none;">
            </iframe>
            """
            gr.HTML(pdf_iframe)

demo.launch()
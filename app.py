from pathlib import Path
import pickle
import torch
import gradio as gr
# from gradio.components import PDF

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, AutoConfig, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer

# ✅ Configs
PDF_PATH = Path("machine_learning.pdf")
CACHE_DIR = Path("cache/cache")
CACHE_DIR.mkdir(exist_ok=True)

DOCS_PICKLE_PATH = CACHE_DIR / "documents.pkl"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "google/flan-t5-large"
VECTOR_DB_NAME = "vectorstore"
VECTOR_DB_PATH = CACHE_DIR / f"{VECTOR_DB_NAME}.faiss"
MODEL_FOLDER_NAME = "google_flan_t5_large"

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
HF_MODEL_CACHE_DIR = CACHE_DIR / MODEL_FOLDER_NAME
model_config = AutoConfig.from_pretrained(LLM_MODEL_NAME)

if HF_MODEL_CACHE_DIR.exists():
    tokenizer = AutoTokenizer.from_pretrained(str(HF_MODEL_CACHE_DIR), use_fast=False)
    if not model_config.is_encoder_decoder:
        model = AutoModelForCausalLM.from_pretrained(
            str(HF_MODEL_CACHE_DIR),
            torch_dtype=torch.float16,
            # device_map="auto"
        )
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
        str(HF_MODEL_CACHE_DIR),
        torch_dtype=torch.float16,
        # device_map="auto"
    )
    model.to("cpu")
else:
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME, use_fast=False)
    if not model_config.is_encoder_decoder:
        model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME,
            torch_dtype=torch.float16,
            # device_map="auto",
            cache_dir=str(HF_MODEL_CACHE_DIR)
        )
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            LLM_MODEL_NAME,
            torch_dtype=torch.float16,
            # device_map="auto",
            cache_dir=str(HF_MODEL_CACHE_DIR)
        )
    model.to("cpu")
    tokenizer.save_pretrained(str(HF_MODEL_CACHE_DIR))
    model.save_pretrained(str(HF_MODEL_CACHE_DIR))

if getattr(model.config, "is_encoder_decoder", True):
    print("######### This is encoder decoder model: " + LLM_MODEL_NAME + " #########")
else:
    print("######### This is sequence to sequence model: " + LLM_MODEL_NAME + " #########")

# ✅ Context and Prompt Building
def get_context_from_query(query, k=3, max_context_tokens=600):
    docs = vector_store.similarity_search(query, k=k)
    combined_text = "\n\n".join([doc.page_content for doc in docs])
    tokens = tokenizer(combined_text, return_tensors="pt", truncation=False).input_ids[0]

    if len(tokens) > max_context_tokens:
        tokens = tokens[-max_context_tokens:]
        combined_text = tokenizer.decode(tokens, skip_special_tokens=True)

    return combined_text, docs

def build_prompt(context, question):
    return f"""You are a helpful assistant. Use the following context to answer the question.

Context:
{context}

Question: {question}

Answer:"""

# ✅ QA Function (Non-streaming)
def ask_question_safe(query):
    try:
        if not model_config.is_encoder_decoder:
            context, _ = get_context_from_query(query, k=3)
            prompt = build_prompt(context, query)

            input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.to(model.device)
            input_length = input_ids.shape[-1]

            max_model_len = model.config.max_position_embeddings
            max_new_tokens = 512

            if input_length + max_new_tokens > max_model_len:
                max_new_tokens = max_model_len - input_length
                if max_new_tokens <= 0:
                    return "Prompt too long. Please ask a shorter question."

            output_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            answer = tokenizer.decode(output_ids[0][input_length:], skip_special_tokens=True)
        else:
            context, _ = get_context_from_query(query, k=5)
            prompt = build_prompt(context, query)

            input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.to(model.device)
            input_length = input_ids.shape[-1]
            max_new_tokens = 512

            output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=512,
            temperature=0.9,              # Encourage more exploration
            top_p=0.95,                   # Nucleus sampling
            do_sample=True,              # Enable sampling
            repetition_penalty=1.1,      # Discourage repeating short answers
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
            
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
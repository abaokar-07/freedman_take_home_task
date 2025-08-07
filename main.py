from config_qa import Config
from load_split_pdf import Load_And_Split_PDF
from load_dwnld_embed import Load_And_Dwnld_Embed
from load_create_vector_db import Load_Create_Vector_DB
from load_dwnld_llm import Load_And_Dwnld_LLM
from context_prompt_builder import Context_Prompt_Builder
from huggingface_hub import login
import gradio as gr
from transformers import AutoConfig
import logging, base64

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

config = Config()

PDF_PATH = config.PDF_PATH
CACHE_DIR = config.CACHE_DIR
CACHE_DIR.mkdir(exist_ok=True)

DOCS_PICKLE_PATH = config.DOCS_PICKLE_PATH
EMBEDDING_MODEL_NAME = config.EMBEDDING_MODEL_NAME
LLM_MODEL_NAME = config.LLM_MODEL_NAME
VECTOR_DB_NAME = config.VECTOR_DB_NAME
VECTOR_DB_PATH = config.VECTOR_DB_PATH
DEVICE = config.device
MAX_CONTENT_TOKENS = config.max_content_tokens
HF_TOKEN = config.HF_TOKEN
MODEL_FOLDER_NAME = config.MODEL_FOLDER_NAME

login(HF_TOKEN)

load_split_pdf = Load_And_Split_PDF(DOCS_PICKLE_PATH)
if DOCS_PICKLE_PATH.exists():
    documents = load_split_pdf.load()
else:
    documents = load_split_pdf.create_and_dump()

load_dwnld_embed = Load_And_Dwnld_Embed(CACHE_DIR)
if any(load_dwnld_embed.EMBEDDING_MODEL_CACHE_DIR.iterdir()):
    embeddings = load_dwnld_embed.load_embed()
else:
    embeddings = load_dwnld_embed.dwnld_and_cache_embed(EMBEDDING_MODEL_NAME)

load_create_vector_db = Load_Create_Vector_DB(CACHE_DIR, VECTOR_DB_NAME, embeddings)

if VECTOR_DB_PATH.exists():
    vector_store = load_create_vector_db.load_db()
else:
    vector_store = load_create_vector_db.create_db(documents)

load_dwnld_llm = Load_And_Dwnld_LLM(CACHE_DIR, LLM_MODEL_NAME, MODEL_FOLDER_NAME)
if load_dwnld_llm.HF_MODEL_CACHE_DIR.exists():
    tok_mod = load_dwnld_llm.load_llm()
    tokenizer, model = tok_mod[0], tok_mod[1]
else:
    tok_mod = load_dwnld_llm.dwnld_and_cache_llm()
    tokenizer, model = tok_mod[0], tok_mod[1]

def get_context_from_query(query, k, max_context_tokens=MAX_CONTENT_TOKENS):
    logging.debug("Doing the similarity search for query: %s", query)
    docs = vector_store.similarity_search(query, k=k)
    combined_text = "\n\n".join([doc.page_content for doc in docs])
    tokens = tokenizer(combined_text, return_tensors="pt", truncation=False).input_ids[0]

    if len(tokens) > max_context_tokens:
        tokens = tokens[:max_context_tokens]
        combined_text = tokenizer.decode(tokens, skip_special_tokens=True)
        logging.debug("Context truncated to %d tokens", max_context_tokens)

    return combined_text, docs

def build_prompt(context, question):
    logging.debug("Building prompt")
    return f"""Context: {context}

Question: {question}

Answer:"""

model_config = AutoConfig.from_pretrained(LLM_MODEL_NAME)

def ask_question_safe(query):
    try:
        if not model_config.is_encoder_decoder:
            logging.info("Received query: %s", query)
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
            
            logging.debug("Generating answer with the model.")
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

with gr.Blocks() as demo:
    gr.Markdown("## PDF Q&A Assistant")
    
    with gr.Row():
        with gr.Column():
            question_input = gr.Textbox(lines=1, label="Ask a question")
            answer_output = gr.Textbox(lines=10, label="Answer", interactive=False)
            question_input.submit(fn=ask_question_safe, inputs=question_input, outputs=answer_output)

        def encode_pdf_to_base64(pdf_path):
            logging.info("Encoding the PDF file for preview purposes!")
            with open(pdf_path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
            return f"data:application/pdf;base64,{encoded}"

        with gr.Column():
            gr.Markdown("### PDF Preview")

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

logging.info("Launching the gradio app!")
demo.launch()
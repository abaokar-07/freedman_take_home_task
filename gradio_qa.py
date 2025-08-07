import gradio as gr
import base64
from qa_function import QA_Function

class Gradio_QA:

    def __init__(self, model, tokenizer, pdf_path, k, max_content_tokens, vector_db):
        self.pdf_path = pdf_path
        self.model = model
        self.tokenizer = tokenizer
        self.k = k
        self.max_content_tokens = max_content_tokens
        self.db = vector_db
    
    def launch(self):
        with gr.Blocks() as demo:
            gr.Markdown("## 📘 PDF Q&A Assistant")
            
            with gr.Row():
                with gr.Column():
                    question_input = gr.Textbox(lines=1, label="❓ Ask a question")
                    answer_output = gr.Textbox(lines=10, label="💬 Answer", interactive=False)
                    qa_func = QA_Function(self.model, self.tokenizer, self.k, self.max_content_tokens, self.db)
                    question_input.submit(fn=qa_func.ask_question_safe, inputs=question_input, outputs=answer_output)

                def encode_pdf_to_base64(pdf_path):
                    with open(pdf_path, "rb") as f:
                        encoded = base64.b64encode(f.read()).decode("utf-8")
                    return f"data:application/pdf;base64,{encoded}"

                with gr.Column():
                    gr.Markdown("### 📄 PDF Preview")

                    encoded_pdf = encode_pdf_to_base64(self.pdf_path)
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
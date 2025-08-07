from context_prompt_builder import Context_Prompt_Builder
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class QA_Function:

    def __init__(self, model, tokenizer, k, max_content_tokens, vector_db):
        # self.context = context
        self.model = model
        self.tokenizer = tokenizer
        self.k = k
        self.max_content_tokens = max_content_tokens
        self.db = vector_db

    def ask_question_safe(self, query):
        try:
            cpb = Context_Prompt_Builder(query, self.k, self.max_content_tokens, self.db, self.tokenizer)
            context, _ = cpb.get_context_from_query()
            prompt = cpb.build_prompt(context, query)

            input_ids = self.tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.to(self.model.device)
            input_length = input_ids.shape[-1]

            # max_model_len = self.model.config.max_position_embeddings
            # max_new_tokens = 512

            # if input_length + max_new_tokens > max_model_len:
            #     max_new_tokens = max_model_len - input_length
            #     if max_new_tokens <= 0:
            #         return "Prompt too long. Please ask a shorter question."
            max_new_tokens = 512

            # output_ids = self.model.generate(
            #     input_ids=input_ids,
            #     max_new_tokens=max_new_tokens,
            #     temperature=0.7,
            #     do_sample=True,
            #     pad_token_id=self.tokenizer.eos_token_id,
            #     eos_token_id=self.tokenizer.eos_token_id,
            # )
            output_ids = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=512,
            temperature=0.9,              # Encourage more exploration
            top_p=0.95,                   # Nucleus sampling
            do_sample=True,              # Enable sampling
            repetition_penalty=1.1,      # Discourage repeating short answers
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )


            # answer = self.tokenizer.decode(output_ids[0][input_length:], skip_special_tokens=True)
            answer = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            logging.info("Answer: %s", answer)
            return answer

        except Exception as e:
            return f"Error during generation: {e}"
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoConfig
import torch
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class Load_And_Dwnld_LLM:

    def __init__(self, cache_path, model_name, folder_store):
        self.cache_path = cache_path
        self.model_name = model_name
        self.folder_store = folder_store
        self.HF_MODEL_CACHE_DIR = self.cache_path / self.folder_store
    
    def load_llm(self):
        logging.info("Model already cached. Loading the model")
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.HF_MODEL_CACHE_DIR), use_fast=False)
        self.model_config = AutoConfig.from_pretrained(self.model_name)
        if not self.model_config.is_encoder_decoder:
            print("######### This is decoder only model: " + self.model_name + " #########")
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.HF_MODEL_CACHE_DIR),
                torch_dtype=torch.float16,
                # device_map="auto"
            )
        else:
            print("######### This is encoder decoder model: " + self.model_name + " #########")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                str(self.HF_MODEL_CACHE_DIR),
                torch_dtype=torch.float16,
                # device_map="auto"
            )
        self.model.to("cpu") if not torch.cuda.is_available() else self.model.to('cuda')
        return [self.tokenizer, self.model]
    
    def dwnld_and_cache_llm(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        logging.info("Model not cached, downloading and caching it now")
        if not self.model_config.is_encoder_decoder:
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.HF_MODEL_CACHE_DIR),
                torch_dtype=torch.float16,
                # device_map="auto"
            )
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                str(self.HF_MODEL_CACHE_DIR),
                torch_dtype=torch.float16,
                # device_map="auto"
            )
        self.model.to("cpu") if not torch.cuda.is_available() else self.model.to('cuda')
        self.tokenizer.save_pretrained(str(self.HF_MODEL_CACHE_DIR))
        self.model.save_pretrained(str(self.HF_MODEL_CACHE_DIR))
        return [self.tokenizer, self.model]
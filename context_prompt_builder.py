class Context_Prompt_Builder:

    def __init__(self, query, k, max_content_tokens, vector_store, tokenizer):
        self.query = query
        self.k = k
        self.max_content_tokens = max_content_tokens
        self.vector_store = vector_store
        self.tokenizer = tokenizer
    
    def get_context_from_query(self):
        self.docs = self.vector_store.similarity_search(self.query, k=self.k)
        self.combined_text = "\n\n".join([doc.page_content for doc in self.docs])
        self.tokens = self.tokenizer(self.combined_text, return_tensors="pt", truncation=False).input_ids[0]

        if len(self.tokens) > self.max_content_tokens:
            self.tokens = self.tokens[-self.max_content_tokens:]
            self.combined_text = self.tokenizer.decode(self.tokens, skip_special_tokens=True)
        
        return [self.combined_text, self.docs]
    
    def build_prompt(self, context, question):
        return f"""You are a helpful assistant. Use the following context to answer the question.

        Context:
        {context}

        Question: {question}

        Answer:"""
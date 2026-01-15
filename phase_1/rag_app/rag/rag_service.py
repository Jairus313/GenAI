class RAGService:
    def __init__(self, embedder, vector_store, llm):
        self.embedder = embedder
        self.vector_store = vector_store
        self.llm = llm

    def answer(self, question: str) -> str:
        query_embedding = self.embedder.embed([question])[0]
        contexts = self.vector_store.search(query_embedding, k=3)

        context_block = "\n".join(contexts)

        prompt = f"""
            SYSTEM:
            You are a cafe assistant.
            Answer ONLY using the provided context, 
            incase of final bill use the percent 
            mentioned in the doc and calculate it.
            If the answer is not found, say "I don't know",
            Also be nice to the user.

            CONTEXT:
            {context_block}

            USER:
            {question}
        """

        return self.llm.generate(prompt)

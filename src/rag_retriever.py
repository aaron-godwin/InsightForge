from vector_store import SimpleVectorStore
from rag_docs import kb_to_text_chunks

class RAGRetriever:
    def __init__(self, embed_fn):
        df, kb = load_data_and_kb()
        self.df = df
        self.kb = kb

        chunks = kb_to_text_chunks(kb)
        self.vstore = SimpleVectorStore(embed_fn)
        self.vstore.add_texts(chunks)

    def retrieve(self, query, k=5):
        return self.vstore.similarity_search(query, k=k)
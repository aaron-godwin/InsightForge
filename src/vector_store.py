import numpy as np

class SimpleVectorStore:
    def __init__(self, embed_fn):
        self.embed_fn = embed_fn
        self.vectors = []
        self.texts = []

    def add_texts(self, texts):
        embeddings = self.embed_fn(texts)
        self.vectors.extend(embeddings)
        self.texts.extend(texts)

    def similarity_search(self, query, k=5):
        q_vec = self.embed_fn([query])[0]
        sims = []

        for vec, text in zip(self.vectors, self.texts):
            sim = np.dot(q_vec, vec) / (np.linalg.norm(q_vec) * np.linalg.norm(vec))
            sims.append((sim, text))

        sims.sort(reverse=True, key=lambda x: x[0])
        return [t for _, t in sims[:k]]
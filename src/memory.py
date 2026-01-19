class MemoryManager:
    """
    Simple memory system for InsightForge.
    Stores and retrieves relevant context from past interactions.
    """

    def __init__(self):
        self.memory = []

    def add(self, entry):
        """Store a new memory entry."""
        self.memory.append(entry)

    def retrieve(self, query):
        """
        Retrieve memory entries relevant to the current query.
        Simple keyword matching for now.
        """
        query_lower = query.lower()
        relevant = []

        for m in self.memory:
            if any(word in query_lower for word in m["keywords"]):
                relevant.append(m["text"])

        return relevant[-5:]  # return the most recent relevant memories
from prompting import (
    build_interpretation_prompt,
    build_insight_prompt,
    build_refinement_prompt
)
from retriever import InsightRetriever
from rag_retriever import RAGRetriever
from memory import MemoryManager


class InsightChain:
    """
    A hybrid RAG + statistics + memory reasoning pipeline for InsightForge.
    Orchestrates:
    1. Query interpretation
    2. Statistics-based retrieval (pandas)
    3. Semantic retrieval (vector store / RAG)
    4. Memory retrieval
    5. Insight generation
    6. Refinement of the final answer
    """

    def __init__(self, df, kb, llm, embed_fn):
        """
        Parameters:
        - df: Raw pandas DataFrame
        - kb: Structured knowledge base (dict)
        - llm: A function that takes a prompt and returns an LLM response
        - embed_fn: Embedding function used by the RAG retriever
        """
        self.stats_retriever = InsightRetriever(df, kb)
        self.rag_retriever = RAGRetriever(embed_fn)
        self.memory = MemoryManager()
        self.llm = llm

    def run(self, query):
        """
        Execute the full hybrid chain for a given user query.
        Returns a dictionary containing all intermediate steps.
        """

        # ---------------------------------------------------------
        # Step 1: Interpret the query
        # ---------------------------------------------------------
        interpretation_prompt = build_interpretation_prompt(query)
        interpretation_output = self.llm(interpretation_prompt)

        # ---------------------------------------------------------
        # Step 2: Retrieve structured statistics
        # ---------------------------------------------------------
        stats = self.stats_retriever.retrieve(query)

        # ---------------------------------------------------------
        # Step 3: Retrieve semantic context (RAG)
        # ---------------------------------------------------------
        rag_context = self.rag_retriever.retrieve(query, k=5)

        # ---------------------------------------------------------
        # Step 4: Retrieve relevant memory
        # ---------------------------------------------------------
        memory_context = self.memory.retrieve(query)

        # ---------------------------------------------------------
        # Step 5: Generate grounded insight
        # ---------------------------------------------------------
        insight_prompt = build_insight_prompt(
            query,
            {
                "stats": stats,
                "rag_context": rag_context,
                "memory_context": memory_context
            }
        )
        raw_insight = self.llm(insight_prompt)

        # ---------------------------------------------------------
        # Step 6: Refine the final answer
        # ---------------------------------------------------------
        refinement_prompt = build_refinement_prompt(raw_insight)
        final_insight = self.llm(refinement_prompt)

        # ---------------------------------------------------------
        # Step 7: Store new memory
        # ---------------------------------------------------------
        self.memory.add({
            "keywords": query.lower().split(),
            "text": final_insight
        })

        return {
            "interpretation": interpretation_output,
            "retrieved_stats": stats,
            "rag_context": rag_context,
            "memory_context": memory_context,
            "raw_insight": raw_insight,
            "final_insight": final_insight
        }
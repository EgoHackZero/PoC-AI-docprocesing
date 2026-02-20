"""
CrewAI orchestration for the document Q&A pipeline.

Agents (sequential):
    1. QueryRewriter  — fixes typos and expands the question into multiple search queries
    2. Retriever      — runs every query variant against ChromaDB, deduplicates results
    3. Evaluator      — decides whether the retrieved content actually answers the question
    4. Specialist     — writes the final answer, or responds "not in database" if evaluator
                        flagged the results as insufficient

Flow:
    raw question
        → QueryRewriter  (corrected question + search query variants)
        → Retriever      (chunks from all query variants, deduplicated)
        → Evaluator      (sufficient / insufficient verdict)
        → Specialist     (final answer or not-found response)
"""

import json
import os

import chromadb
from crewai import Agent, Crew, Process, Task
from crewai.tools import BaseTool
from pydantic import Field
from sentence_transformers import SentenceTransformer

import vectorstore as vs


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

class SearchManualTool(BaseTool):
    """Queries ChromaDB with a single search string.

    The Retriever agent calls this multiple times — once per query variant
    produced by the QueryRewriter — to maximise recall.
    """

    name: str = "search_manual"
    description: str = (
        "Search the indexed service manuals for relevant content. "
        "Use a specific technical term, error code (e.g. E12), symptom description, "
        "or component name. Call this tool multiple times with different queries "
        "to get broader coverage. Returns the top matching excerpts with "
        "document name, page number, and relevance score."
    )
    db_path: str = Field(default=vs.DB_PATH)
    _embedder: SentenceTransformer = None

    def _run(self, query: str, top_k: int = 5) -> str:
        if self._embedder is None:
            self._embedder = SentenceTransformer(vs.EMBEDDING_MODEL)

        embedding = self._embedder.encode(query, normalize_embeddings=True).tolist()
        client = chromadb.PersistentClient(path=self.db_path)
        collection = client.get_collection(vs.COLLECTION_NAME)
        results = collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        hits = [
            {
                "doc": meta["doc"],
                "page": meta["page"],
                "score": round(1 - dist, 4),
                "text": doc,
            }
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]
        return json.dumps(hits, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Crew factory
# ---------------------------------------------------------------------------

def build_crew(db_path: str = vs.DB_PATH) -> Crew:

    search_tool = SearchManualTool(db_path=db_path)

    # ------------------------------------------------------------------
    # Agent 1 — Query Rewriter
    # ------------------------------------------------------------------
    query_rewriter = Agent(
        role="Query Rewriter",
        goal=(
            "Correct spelling mistakes in the user's question and produce "
            "3 to 5 distinct search query variants that cover different angles "
            "of the same topic, so the retriever can find the most relevant content."
        ),
        backstory=(
            "You are a linguistics expert specialised in technical appliance vocabulary. "
            "You recognise misspelled error codes, component names, and symptom descriptions "
            "and know how to rephrase a question into multiple precise search queries."
        ),
        llm="gpt-4o-mini",
        verbose=False,
    )

    # ------------------------------------------------------------------
    # Agent 2 — Retriever
    # ------------------------------------------------------------------
    retriever = Agent(
        role="Service Manual Retriever",
        goal=(
            "Run every search query variant provided by the Query Rewriter against "
            "the service manual database. Call the search tool once per query variant "
            "and collect all unique results."
        ),
        backstory=(
            "You are an expert at searching technical documentation databases. "
            "You execute searches methodically — one query at a time — and compile "
            "all unique results, avoiding duplicates based on document and page number."
        ),
        tools=[search_tool],
        llm="gpt-4o-mini",
        verbose=False,
    )

    # ------------------------------------------------------------------
    # Agent 3 — Evaluator
    # ------------------------------------------------------------------
    evaluator = Agent(
        role="Relevance Evaluator",
        goal=(
            "Determine whether the retrieved manual excerpts contain enough relevant "
            "information to answer the user's original question. "
            "Output a clear verdict: 'SUFFICIENT' or 'INSUFFICIENT', with a short reason."
        ),
        backstory=(
            "You are a quality assurance specialist. You read retrieved document excerpts "
            "critically and judge whether they actually address the user's question, "
            "not just share some keywords with it."
        ),
        llm="gpt-4o-mini",
        verbose=False,
    )

    # ------------------------------------------------------------------
    # Agent 4 — Specialist (final answer)
    # ------------------------------------------------------------------
    specialist = Agent(
        role="Technical Support Specialist",
        goal=(
            "If the evaluator verdict is SUFFICIENT: write a clear, structured answer "
            "based strictly on the retrieved manual content with source citations. "
            "If the verdict is INSUFFICIENT: respond that the database does not contain "
            "relevant information for this question."
        ),
        backstory=(
            "You are a senior appliance service technician. You only answer based on "
            "what is in the provided manual excerpts. You never guess or make up "
            "information that is not explicitly in the source material."
        ),
        llm="gpt-4o-mini",
        verbose=False,
    )

    # ------------------------------------------------------------------
    # Tasks
    # ------------------------------------------------------------------
    rewrite_task = Task(
        description=(
            "The user asked: '{question}'\n\n"
            "1. Correct any spelling or grammar mistakes in the question.\n"
            "2. Identify the core technical topic (error code, component, symptom).\n"
            "3. Generate 3 to 5 distinct search query variants that approach the topic "
            "from different angles (e.g. error code alone, symptom description, "
            "component name, action to take).\n"
            "Output the corrected question and the list of query variants."
        ),
        expected_output=(
            "Corrected question and a numbered list of 3-5 search query variants, "
            "each on its own line."
        ),
        agent=query_rewriter,
    )

    retrieve_task = Task(
        description=(
            "Using the search query variants from the previous task, search the manual "
            "database. Call the search_manual tool ONCE for EACH query variant. "
            "Collect all results and remove duplicates (same document + page = duplicate). "
            "Return the full deduplicated list of excerpts."
        ),
        expected_output=(
            "A deduplicated list of manual excerpts, each with: "
            "document name, page number, relevance score, and text content."
        ),
        agent=retriever,
        context=[rewrite_task],
    )

    evaluate_task = Task(
        description=(
            "Original question: '{question}'\n\n"
            "Review the retrieved excerpts from the previous task and decide:\n"
            "- SUFFICIENT: the excerpts contain clear, direct information that answers "
            "the question.\n"
            "- INSUFFICIENT: the excerpts are off-topic, too vague, or do not address "
            "the question at all.\n"
            "State your verdict and a one-sentence reason."
        ),
        expected_output=(
            "A verdict of either 'SUFFICIENT' or 'INSUFFICIENT' followed by "
            "a one-sentence justification."
        ),
        agent=evaluator,
        context=[rewrite_task, retrieve_task],
    )

    answer_task = Task(
        description=(
            "Original question: '{question}'\n\n"
            "Read the evaluator verdict from the previous task:\n"
            "- If SUFFICIENT: write a clear answer using ONLY the retrieved excerpts. "
            "Format troubleshooting steps as a numbered list. "
            "Cite document name and page number for every piece of information.\n"
            "- If INSUFFICIENT: respond with exactly this and nothing else: "
            "'There is no information about this topic in the indexed documents.'"
        ),
        expected_output=(
            "Either a structured answer with numbered steps and source citations, "
            "or the not-found message if the evaluator verdict was INSUFFICIENT."
        ),
        agent=specialist,
        context=[rewrite_task, retrieve_task, evaluate_task],
    )

    return Crew(
        agents=[query_rewriter, retriever, evaluator, specialist],
        tasks=[rewrite_task, retrieve_task, evaluate_task, answer_task],
        process=Process.sequential,
        verbose=False,
    )


def run(question: str, db_path: str = vs.DB_PATH) -> str:
    """Run the full crew pipeline for a question and return the final answer."""
    crew = build_crew(db_path=db_path)
    result = crew.kickoff(inputs={"question": question})
    return str(result)

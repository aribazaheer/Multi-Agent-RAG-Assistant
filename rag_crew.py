from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from crewai import Task, Crew, Process
from agents import fact_checker_agent
from agents import answer_writer_agent

def create_answer_task(question, context, fact_check_result):
    """Create the final answer task using verified document information."""

    return Task(
        description=f"""
        Answer the user's question using ONLY the verified information
        from the provided documents.

        User Question:
        {question}

        Retrieved Context:
        {context}

        Fact-Checker Report:
        {fact_check_result}

        Rules:
        - Do not use outside knowledge.
        - Do not invent facts.
        - If the fact-checker marks information as unsupported,
          do not include that information.
        - Keep the answer clear and concise.
        - Include citations using the document source names.
        """,
        expected_output=(
            "A clear and accurate answer based only on the provided "
            "documents, with source citations."
        ),
        agent=answer_writer_agent
    )

def create_fact_check_task(question, context):
    """Create a fact-checking task using retrieved context."""

    return Task(
        description=f"""
        Verify the answer to the following question using ONLY
        the retrieved document context.

        Question:
        {question}

        Retrieved Context:
        {context}

        Check every important claim.

        Do not use outside knowledge.

        If the context does not support a claim, mark it as
        unsupported.

        Return:
        1. Verified facts
        2. Unsupported claims
        3. Overall verification result
        """,
        expected_output=(
            "A fact-check report containing verified facts, "
            "unsupported claims, and an overall verification result."
        ),
        agent=fact_checker_agent
    )
CHROMA_DIR = "chroma_db"


# Local embedding model
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


# Connect to ChromaDB
vectorstore = Chroma(
    collection_name="day15_documents",
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR,
)


def retrieve_documents(query, k=3):
    """Retrieve relevant documents from ChromaDB."""

    results = vectorstore.similarity_search_with_score(
        query,
        k=k
    )

    return results


def get_relevant_context(query, k=3):
    """Create a formatted context from retrieved documents."""

    results = retrieve_documents(query, k)

    if not results:
        return "", results

    context_parts = []

    for document, score in results:

        source = document.metadata.get(
            "source",
            "Unknown"
        )

        context_parts.append(
            f"Source: {source}\n"
            f"Content: {document.page_content}\n"
            f"Similarity Score: {score:.4f}"
        )

    context = "\n\n---\n\n".join(context_parts)

    return context, results

print("=" * 60)
print("DAY 15 MULTI-AGENT RAG")
...

def create_hierarchical_crew(tasks):
    """Create a hierarchical CrewAI workflow."""

    from agents import retriever_agent, fact_checker_agent, answer_writer_agent

    return Crew(
        agents=[
            retriever_agent,
            fact_checker_agent,
            answer_writer_agent
        ],
        tasks=tasks,
        process=Process.hierarchical,
        manager_llm="ollama/llama3.2:3b",
        verbose=True
    )
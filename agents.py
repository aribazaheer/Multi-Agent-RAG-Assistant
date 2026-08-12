from crewai import Agent
from langchain_ollama import ChatOllama


# Local Ollama model
llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0.1
)


# Retriever Agent
retriever_agent = Agent(
    role="Document Retriever",
    goal=(
        "Find relevant information from the provided document context "
        "and return only information that is supported by the documents."
    ),
    backstory=(
        "You are a retrieval specialist. Your job is to identify the "
        "most relevant information from the local document knowledge base. "
        "You must not invent information that is not present in the context."
    ),
    llm="ollama/llama3.2:3b",
    verbose=True
)


# Fact-Checker Agent
fact_checker_agent = Agent(
    role="Fact Checker",
    goal=(
        "Verify the retrieved information against the provided document "
        "context and identify unsupported or questionable claims."
    ),
    backstory=(
        "You are a careful fact checker. You verify claims using only "
        "the retrieved document context and reject unsupported information."
    ),
    llm="ollama/llama3.2:3b",
    verbose=True
)


# Answer Writer Agent
answer_writer_agent = Agent(
    role="Answer Writer",
    goal=(
        "Write a clear answer using only verified information from the "
        "retrieved documents and include document citations."
    ),
    backstory=(
        "You are a professional answer writer. You create concise and "
        "accurate answers grounded entirely in the provided documents. "
        "You never hallucinate information."
    ),
    llm="ollama/llama3.2:3b",
    verbose=True
)
# Day 15 - Multi-Agent RAG Main Application

from crewai import Crew, Process

from memory import save_conversation, get_previous_conversations

from rag_crew import (
    get_relevant_context,
    create_fact_check_task,
    create_answer_task
)

from guardrail import is_relevant_context, refusal_message


def main():
    print("=" * 60)
    print("DAY 15 - MULTI-AGENT RAG")
    print("=" * 60)

    # Load previous conversations
    previous_conversations = get_previous_conversations()

    if previous_conversations:
        print("\nPrevious Conversations:")

        for conversation in previous_conversations:
            print("Question:", conversation["question"])
            print("Answer:", conversation["answer"])
            print("-" * 60)

    question = input("\nEnter your question: ")

    # Prepare memory context
    memory_context = ""

    if previous_conversations:
        memory_context = "\nPrevious Conversation Context:\n"

        for conversation in previous_conversations[-3:]:
            memory_context += (
                f"Previous Question: {conversation['question']}\n"
                f"Previous Answer: {conversation['answer']}\n\n"
            )

    # Retrieve documents
    print("\n[1] Retrieving relevant documents...")

    context, results = get_relevant_context(
        question,
        k=3
    )

    if not is_relevant_context(results):
        print("\n" + refusal_message())
        return

    print("[✓] Relevant documents found.")

    # Combine memory with document context
    if memory_context:
        context = (
            memory_context
            + "\nCurrent Document Context:\n"
            + context
        )

    # Fact Checker
    print("\n[2] Running Fact Checker...")

    fact_check_task = create_fact_check_task(
        question,
        context
    )

    fact_checker_crew = Crew(
        agents=[fact_check_task.agent],
        tasks=[fact_check_task],
        process=Process.sequential,
        verbose=True
    )

    fact_check_result = fact_checker_crew.kickoff()

    print("\n[✓] Fact checking completed.")

    # Answer Writer
    print("\n[3] Running Answer Writer...")

    answer_task = create_answer_task(
        question,
        context,
        fact_check_result
    )

    answer_writer_crew = Crew(
        agents=[answer_task.agent],
        tasks=[answer_task],
        process=Process.sequential,
        verbose=True
    )

    final_answer = answer_writer_crew.kickoff()

    # Final Answer
    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)

    print(final_answer)

    # Save conversation
    save_conversation(
        question,
        final_answer
    )

    print("\n[✓] Conversation saved to memory.")


if __name__ == "__main__":
    main()
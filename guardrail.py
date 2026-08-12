def is_relevant_context(results, threshold=1.2):
    """
    Check whether retrieved documents are relevant enough.
    Chroma similarity scores are distance scores, so lower is better.
    """

    if not results:
        return False

    best_score = min(score for _, score in results)

    return best_score <= threshold


def refusal_message():
    return (
        "I’m sorry, but I don’t have enough relevant information "
        "in the provided documents to answer this question."
    )
if __name__ == "__main__":

    empty_results = []

    if not is_relevant_context(empty_results):
        print(refusal_message())
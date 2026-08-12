# Day 15 - Conversation Memory

import json
from pathlib import Path

MEMORY_FILE = Path("conversation_memory.json")


def load_memory():
    if not MEMORY_FILE.exists():
        return []

    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, FileNotFoundError):
        return []


def save_conversation(question, answer):
    memory = load_memory()

    memory.append({
        "question": question,
        "answer": str(answer)
    })

    with open(MEMORY_FILE, "w", encoding="utf-8") as file:
        json.dump(memory, file, indent=4, ensure_ascii=False)


def get_previous_conversations():
    return load_memory()
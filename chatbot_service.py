import os
from dotenv import load_dotenv

load_dotenv()

from flask import Flask, jsonify, request
from google import genai


MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")

app = Flask(__name__)


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


def get_client():
    api_key = GEMINI_API_KEY
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not configured")

    return genai.Client(api_key=api_key)


def extract_text(message):
    if not isinstance(message, dict):
        return ""

    text = message.get("text")
    if isinstance(text, str):
        return text.strip()

    parts = message.get("parts")
    if not isinstance(parts, list):
        return ""

    texts = []
    for part in parts:
        if not isinstance(part, dict):
            continue

        part_text = part.get("text")
        if isinstance(part_text, str) and part_text.strip():
            texts.append(part_text.strip())

    return "\n".join(texts)


def normalize_role(role):
    return "assistant" if role in {"assistant", "model"} else "user"


def normalize_history(chat_history):
    if not isinstance(chat_history, list):
        return []

    normalized_history = []

    for entry in chat_history:
        text = extract_text(entry)
        if not text:
            continue

        normalized_history.append(
            {
                "role": normalize_role(entry.get("role")),
                "text": text,
            }
        )

    return normalized_history

def normalize_courses(courses):
    if not isinstance(courses, list):
        return []

    normalized_courses = []

    for entry in courses:
        if not isinstance(entry, dict):
            continue

        name = str(entry.get("name", "")).strip()
        if not name:
            continue

        normalized_courses.append(
            {
                "name": name,
                "description": str(entry.get("description", "")).strip(),
                "price": entry.get("price"),
                "tag": str(entry.get("tag", "")).strip(),
                "rate": entry.get("rate"),
                "total_rate": entry.get("total_rate"),
            }
        )

    return normalized_courses

def build_prompt(message, chat_history, courses):
    prompt_lines = [
        "You are the chatbot for the OLP learning platform.",
        "Answer clearly and keep context from the prior conversation.",
        "Use only the provided course catalog context when answering course-specific questions.",
    ]

    if courses:
        prompt_lines.append("Course catalog context:")
        for idx, course in enumerate(courses, start=1):
            prompt_lines.append(
                (
                    f"{idx}. Name: {course['name']} | "
                    f"Tag: {course['tag'] or 'N/A'} | "
                    f"Price: {course['price']} | "
                    f"Rate: {course['rate']} | "
                    f"Total rate: {course['total_rate']}"
                )
            )
            if course["description"]:
                prompt_lines.append(f"   Description: {course['description']}")

    if chat_history:
        prompt_lines.append("Conversation so far:")
        for entry in chat_history:
            role = "Assistant" if entry["role"] == "assistant" else "User"
            prompt_lines.append(f"{role}: {entry['text']}")

    prompt_lines.append(f"User: {message}")
    prompt_lines.append("Assistant:")

    return "\n".join(prompt_lines)


def generate_response(message, chat_history, courses):
    client = get_client()
    prompt = build_prompt(message, chat_history, courses)
    result = client.models.generate_content(model=MODEL_NAME, contents=prompt)
    response_text = getattr(result, "text", "")

    if not response_text:
        raise RuntimeError("Gemini returned an empty response")

    return response_text.strip()


@app.post("/chat")
def chat():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()
    chat_history = normalize_history(data.get("chat_history", []))
    courses = normalize_courses(data.get("course_knowledge", []))

    if not message:
        return jsonify({"error": "message is required"}), 400

    try:
        response_text = generate_response(message, chat_history, courses)
    except Exception as exc:
        app.logger.exception("Failed to generate chatbot response")
        return jsonify({"error": str(exc)}), 503

    updated_history = chat_history + [
        {"role": "user", "text": message},
        {"role": "assistant", "text": response_text},
    ]

    return jsonify(
        {
            "response": response_text,
            "chat_history": updated_history,
        }
    )


@app.get("/health")
def health():
    return jsonify({"status": "ok"})
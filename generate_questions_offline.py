import os
import json
import re
import time
import requests
import nltk
from tqdm import tqdm

# === SETUP ===
nltk.download('punkt', quiet=True)
from nltk.tokenize import sent_tokenize

# === CONFIGURATION ===
TXT_FILE = "parse-pdf/merged-chapters/merged-chapters.txt"
JSONL_FILE = "parse-pdf/merged-chapters/merged-chapters.jsonl"
RAW_LOG_FILE = "raw_output.jsonl"
CHECKPOINT_FILE = "progress_checkpoint.json"

LLAMA_SERVER_URL = "http://127.0.0.1:8080/completion"

CHUNK_MIN_WORDS = 30
CHUNK_MAX_WORDS = 100

CHAPTER_REGEX = re.compile(r'^\s*(chapter\s+\d+|CHAPTER\s+\d+)\s*$', re.IGNORECASE | re.MULTILINE)

MAX_QUESTIONS_PER_PARAGRAPH = 3
MAX_QUESTIONS_PER_CHAPTER = 6

REQUEST_TIMEOUT = 120
N_PREDICT = 256


# === CLEAN TEXT ===
def clean_text(text):
    text = re.sub(r'-\n', '', text)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r'\n{2,}', '\n\n', text)
    return text.strip()


# === CHAPTER SPLITTING ===
def split_into_chapters(text):
    """Splits the full book into chapters. Falls back to one chapter if no
    markers are found."""
    matches = list(CHAPTER_REGEX.finditer(text))
    if not matches:
        return [("Whole Book", text)]

    chapters = []
    for i, match in enumerate(matches):
        title = match.group().strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            chapters.append((title, body))
    return chapters


# === PARAGRAPH SPLITTING ===
def split_into_paragraphs(text):
    paragraphs = []
    for block in text.split('\n\n'):
        block = block.strip()
        if not block:
            continue
        sentences = sent_tokenize(block)
        chunk = ""
        word_count = 0
        for sentence in sentences:
            words = sentence.split()
            word_count += len(words)
            chunk += " " + sentence
            if CHUNK_MIN_WORDS <= word_count <= CHUNK_MAX_WORDS:
                paragraphs.append(chunk.strip())
                chunk = ""
                word_count = 0
        if chunk:
            paragraphs.append(chunk.strip())
    return paragraphs


# === CALL LLAMA SERVER ===
def call_llama_server(prompt, n_predict=N_PREDICT, retries=2):
    payload = {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0.7,
        "stop": ["</s>"],
    }

    for attempt in range(retries + 1):
        try:
            response = requests.post(LLAMA_SERVER_URL, json=payload, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            return data.get("content", "").strip()
        except (requests.RequestException, json.JSONDecodeError) as e:
            if attempt < retries:
                time.sleep(2)
                continue
            print(f"⚠️ Server call failed after {retries + 1} attempts: {e}")
            return ""


def log_raw(prompt, output):
    with open(RAW_LOG_FILE, "a", encoding="utf-8") as log:
        log.write(json.dumps({
            "instruction": prompt.strip(),
            "input": "",
            "output": output
        }, ensure_ascii=False) + "\n")


def parse_questions(raw_text):
    questions = []
    for q in raw_text.split("\n"):
        q = q.strip()
        q = re.sub(r'^\d+(\.\d+)*[\).]?\s*', '', q)
        q = re.sub(r'^[-*]\s*', '', q)
        if q.endswith("?") and len(q.split()) >= 3 and not q.lower().startswith(("of", "and", "the")):
            questions.append(q)
    return list(dict.fromkeys(questions))  # dedupe, preserve order


# === PROMPTS FOR DIFFERENT QUESTION TYPES ===

def paragraph_prompt(paragraph, book_title, max_q):
    return f"""You are helping build a Q&A training set based on a book called *{book_title}*.

Read the passage below and write up to {max_q} different natural-language questions
that a reader could ask which are directly answered by this passage. Mix the style:
include at least one purely factual/detail question and, if the passage supports it,
one question that asks "why" or "how" something happened.

Write each question on its own line. No numbering, no explanations, no extra text.

Passage:
{paragraph}
"""


def chapter_prompt(chapter_title, chapter_text, book_title, max_q):
    # Keep chapter text bounded so we don't blow past context size.
    trimmed = chapter_text[:6000]
    return f"""You are helping build a Q&A training set based on a book called *{book_title}*.

Below is the text of "{chapter_title}". Write up to {max_q} varied questions a reader
might ask about this chapter, covering a MIX of these types:
- A summary question (e.g. "What happens in this chapter?")
- A thematic question (e.g. "What does this chapter suggest about X?")
- A character motivation question (e.g. "Why does [character] decide to...?")
- A cause-and-effect or comparison question relating events in this chapter

Write each question on its own line. No numbering, no explanations, no extra text.

Chapter text:
{trimmed}
"""


# === CHECKPOINT HANDLING ===
def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"completed_chapters": [], "completed_paragraph_idx": -1}


def save_checkpoint(state):
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f)


# === MAIN ===
def main():
    if not os.path.exists(TXT_FILE):
        print(f"❌ File not found: {TXT_FILE}")
        return

    book_title = os.path.splitext(os.path.basename(TXT_FILE))[0]

    with open(TXT_FILE, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    cleaned = clean_text(raw_text)
    chapters = split_into_chapters(cleaned)
    print(f"📚 Found {len(chapters)} chapter(s).")

    state = load_checkpoint()
    written_count = 0
    skipped_count = 0

    # Open in append mode so a resumed run doesn't wipe prior progress.
    out = open(JSONL_FILE, 'a', encoding='utf-8')

    try:
        # --- Chapter-level (broad) questions ---
        for chapter_title, chapter_text in tqdm(chapters, desc="Chapters"):
            if chapter_title in state["completed_chapters"]:
                continue

            raw = call_llama_server(
                chapter_prompt(chapter_title, chapter_text, book_title, MAX_QUESTIONS_PER_CHAPTER)
            )
            log_raw(f"[CHAPTER] {chapter_title}", raw)
            questions = parse_questions(raw)

            # Use a trimmed version of the chapter as the "answer" context.
            answer_context = chapter_text.strip()[:2000]

            for question in questions:
                entry = {"instruction": question, "input": "", "output": answer_context}
                out.write(json.dumps(entry, ensure_ascii=False) + '\n')
                written_count += 1

            state["completed_chapters"].append(chapter_title)
            save_checkpoint(state)

        # --- Paragraph-level (detail) questions across the whole book ---
        all_paragraphs = []
        for _, chapter_text in chapters:
            all_paragraphs.extend(split_into_paragraphs(chapter_text))

        print(f"📖 Found {len(all_paragraphs)} paragraphs to process.")

        start_idx = state["completed_paragraph_idx"] + 1
        for i in tqdm(range(start_idx, len(all_paragraphs)), desc="Paragraphs", initial=start_idx, total=len(all_paragraphs)):
            paragraph = all_paragraphs[i]
            try:
                raw = call_llama_server(paragraph_prompt(paragraph, book_title, MAX_QUESTIONS_PER_PARAGRAPH))
                log_raw(paragraph, raw)
                questions = parse_questions(raw)

                if not questions:
                    skipped_count += 1
                else:
                    for question in questions:
                        entry = {"instruction": question, "input": "", "output": paragraph.strip()}
                        out.write(json.dumps(entry, ensure_ascii=False) + '\n')
                        written_count += 1

            except Exception as e:
                print(f"[{i}] ❌ Error: {e}")
                skipped_count += 1

            state["completed_paragraph_idx"] = i
            if i % 20 == 0:
                out.flush()
                save_checkpoint(state)

    finally:
        out.close()
        save_checkpoint(state)

    print(f"\n📦 Done: {written_count} questions saved to '{JSONL_FILE}'")
    print(f"⚠️ Skipped: {skipped_count} paragraphs with no usable questions")


# === ENTRY ===
if __name__ == "__main__":
    main()
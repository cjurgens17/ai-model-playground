import pdfplumber
import re

#Metadata
PAGE_OFFSET = 22
PDF_LOCATION = "../Designing Data Intensive Applications by Martin Kleppmann.pdf"
TOC_START_IDENTIFIER = "Foundations of data systems"
TOC_END_IDENTIFIER = "glossary"
END_OF_LAST_CHAPTER = 544


def _extract_chapter_page_number(line: str) -> int | None:
    line = line.strip()

    starts_with_digits = re.match(r"^(\d+)\.\s+(.+)$", line)
    if not starts_with_digits:
        return None

    rest = starts_with_digits.group(2)

    page_match = re.search(r"(\d+)\s*$", rest)
    if not page_match:
        return None

    return int(page_match.group(1))

def _return_chapter_page_numbers() -> list[int]:
    chapter_pages = []
    #Table Of Contents
    TOC_START = None
    TOC_END = None
    with pdfplumber.open(PDF_LOCATION) as pdf:
        for i ,page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            if TOC_START_IDENTIFIER.lower() in text.lower():
                TOC_START = i
            if TOC_END_IDENTIFIER.lower() in text.lower():
                TOC_END = i
                break
        
        if TOC_START is None or TOC_END is None:
            raise ValueError("Could not find ToC start or end markers")

        for i in range(TOC_START, TOC_END + 1):
            page = pdf.pages[i]
            text = page.extract_text() or ""
            for line in text.split("\n"):
                page_number = _extract_chapter_page_number(line)
                if page_number is not None:
                    chapter_pages.append(page_number + PAGE_OFFSET)
    return chapter_pages

def write_chapters_to_text() -> None:
    chapter_page_numbers = _return_chapter_page_numbers()
    with pdfplumber.open(PDF_LOCATION) as pdf:
        for i, page_number in enumerate(chapter_page_numbers):
            start_idx = page_number - 1
            end_idx = (chapter_page_numbers[i + 1] - 1) if i + 1 < len(chapter_page_numbers) else END_OF_LAST_CHAPTER + PAGE_OFFSET

            with open(f"./ddia-chapters/Chapter-{i + 1}.txt", "w", encoding="utf-8") as f:
                for i in range(start_idx, end_idx):
                    page = pdf.pages[i]
                    text = page.extract_text() or ""
                    f.write(text)
                    f.write("\n")
                    
#Write all Chapters to text files
write_chapters_to_text()

            



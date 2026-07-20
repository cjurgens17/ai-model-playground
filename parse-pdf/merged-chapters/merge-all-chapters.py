from pathlib import Path


OUTPUT_FILE = Path("merged-chapters.txt")
INPUT_DIR = Path("../ddia-chapters")

def merge_all_chapters():
    with OUTPUT_FILE.open("w", encoding="utf-8") as outfile:
        for txt_file in sorted(INPUT_DIR.glob("*.txt")):
            outfile.write(txt_file.read_text(encoding="utf-8"))
            outfile.write("\n")
            
if __name__ == "__main__":
    merge_all_chapters()

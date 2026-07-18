# convert_to_gguf.py
#
# Robust HuggingFace → GGUF conversion using subprocess
# Run: python convert_to_gguf.py

import subprocess
from pathlib import Path
import sys

# -------- CONFIG --------
LLAMA_CPP_DIR = Path(r"../llama.cpp")   # 👈 CHANGE THIS
HF_MODEL_DIR  = Path(r"../randomLLM")     # 👈 CHANGE THIS
OUTPUT_GGUF   = Path(r"randomLLM.gguf")   # 👈 CHANGE THIS
OUTTYPE       = "f16"
# ------------------------

CONVERTER = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"

if not CONVERTER.exists():
    raise FileNotFoundError(f"convert_hf_to_gguf.py not found at {CONVERTER}")

cmd = [
    sys.executable,
    str(CONVERTER),
    str(HF_MODEL_DIR),
    "--outfile", str(OUTPUT_GGUF),
    "--outtype", OUTTYPE,
]

print("Running GGUF conversion:")
print(" ".join(cmd))
print()

subprocess.check_call(cmd)

print("\n✅ GGUF conversion complete")
print("Output:", OUTPUT_GGUF)
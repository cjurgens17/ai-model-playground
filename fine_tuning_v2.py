from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
import torch

# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------
# NOTE: AMD GPUs (Radeon RX series) do not use CUDA. On Windows, the only
# well-supported acceleration path for Hugging Face `Trainer` right now is
# CPU (or CUDA, if you ever run this on an Nvidia machine / cloud GPU).
# DirectML support exists for inference in some setups, but Trainer's
# fp16/bf16 mixed-precision + gradient handling assumes CUDA or MPS, so
# training on an AMD GPU via DirectML is not reliable here. This script
# runs safely on CPU and will automatically use CUDA if it's available.

if torch.cuda.is_available():
    device_type = "cuda"
    use_fp16 = True
    print(f"CUDA GPU detected: {torch.cuda.get_device_name(0)}")
else:
    device_type = "cpu"
    use_fp16 = False  # fp16 is unreliable/unsupported on CPU
    print("No CUDA GPU detected — running on CPU. This will be slow; "
          "see notes in the script header for AMD GPU options (WSL2+ROCm "
          "or a cloud GPU) if you want real acceleration.")

# ---------------------------------------------------------------------------
# Model + tokenizer
# ---------------------------------------------------------------------------
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float32,  # float32 on CPU; fp16 causes issues without CUDA
)

# ---------------------------------------------------------------------------
# LoRA setup
# ---------------------------------------------------------------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
dataset = load_dataset("json", data_files="parse-pdf/merged-chapters/merged-chapters.jsonl")["train"]


def format_prompt(example):
    return {
        "text": (
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Input:\n{example['input']}\n\n"
            f"### Response:\n{example['output']}"
        )
    }


dataset = dataset.map(format_prompt)


def tokenize(example):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=256,
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


tokenized_dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
training_args = TrainingArguments(
    output_dir="tinyllama-finetuned",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,   # effective batch size of 8 without more memory
    num_train_epochs=4,              # lowered from 30 — appropriate for a few thousand examples
    save_strategy="epoch",
    logging_steps=1,                 # frequent logging so you can see it's alive
    fp16=use_fp16,                   # only True if CUDA is actually available
    dataloader_pin_memory=(device_type == "cuda"),  # avoids the pin_memory warning on CPU
    report_to="none",
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

if __name__ == "__main__":
    trainer.train()
    trainer.model.save_pretrained("tinyllama-finetuned")
    tokenizer.save_pretrained("tinyllama-finetuned")
    print("✅ Fine-tuned adapter saved to ./tinyllama-finetuned")
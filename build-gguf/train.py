from transformers import (
    DataCollatorForSeq2Seq,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    Trainer,
    TrainingArguments
)
from datasets import Dataset

# -----------------------
# Tokenizer
# -----------------------
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
# Use default GPT-2 EOS and set PAD to EOS for causal LM training
tokenizer.pad_token = tokenizer.eos_token

# -----------------------
# Build regular prompt→completion dataset (no roles)
# -----------------------
def build_example(prompt, completion):
    # Compose sample without any role prefixes.
    # We keep instruction tuning style by masking the prompt, so the model learns
    # to generate the completion given the prompt.
    text = f"{prompt}\n{completion}{tokenizer.eos_token}"

    tokens = tokenizer(text, truncation=True)
    input_ids = tokens["input_ids"]

    # Mask the prompt part; only learn on the completion and EOS
    labels = [-100] * len(input_ids)
    prompt_prefix = f"{prompt}\n"
    prefix_ids = tokenizer(prompt_prefix, truncation=True)["input_ids"]
    start = len(prefix_ids)
    labels[start:] = input_ids[start:]

    return {
        "input_ids": input_ids,
        "attention_mask": tokens["attention_mask"],
        "labels": labels,
    }

def load_examples_from_file(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split on blank lines to get blocks, no reliance on any role markers
    blocks = [b.strip() for b in content.split("\n\n")]
    examples = []

    for block in blocks:
        if not block:
            continue

        lines = [line.strip() for line in block.splitlines() if line.strip()]

        # Expect exactly two lines per example: first = prompt, second = completion
        if len(lines) < 2:
            continue

        prompt = lines[0]
        completion = lines[1]

        examples.append(build_example(prompt, completion))

    return examples


examples = load_examples_from_file("build-gguf/train.txt")
dataset = Dataset.from_list(examples)

# -----------------------
# Model
# -----------------------
config = GPT2Config(
    vocab_size=len(tokenizer),
    n_positions=256,
    n_ctx=256,
    n_embd=384,
    n_layer=6,
    n_head=6,
)

model = GPT2LMHeadModel(config)
model.resize_token_embeddings(len(tokenizer))
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,  # helps it handle any model-specific padding needs
    label_pad_token_id=-100,
    padding=True,
)

# -----------------------
# Training
# -----------------------
args = TrainingArguments(
    output_dir="randomLLM",
    per_device_train_batch_size=2,
    num_train_epochs=200,
    learning_rate=5e-4,
    logging_steps=10,
    save_steps=500,
    save_total_limit=1,
    fp16=False,
    report_to="none",
    dataloader_num_workers=0,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
    data_collator=data_collator
)

trainer.train()

model.save_pretrained("randomLLM")
tokenizer.save_pretrained("randomLLM")

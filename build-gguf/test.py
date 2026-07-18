from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="./randomLLM",
    tokenizer="./randomLLM",
)

prompt = "User: Are You a Machine?\nAssistant:"
prompt_two = "User: Are you intelligent?\nAssistant:"
prompt_three = "User: Are you in the Terminator?\nAssistant:"
prompt_four = "User: What is the color of the sky?\nAssistant:"
prompt_five = "User: How many eggs are in a dozen?\nAssistant:"
prompts = [prompt, prompt_two, prompt_three,prompt_four, prompt_five]
out = pipe(prompts, max_new_tokens=40, do_sample=False)

for i, result in enumerate(out):
    print(f"---Prompt: ${i+1}")
    print(result[0]["generated_text"])
    print()
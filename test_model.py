from qalign.model import RemoteVLLM


# google/gemma-3-1b-it


model = RemoteVLLM(
    server_url="http://g3119.hyak.local:8252",
    model_path="Qwen/Qwen3-1.7B-Base",
    temperature=1.0,
    max_new_tokens=2048,
    max_prompt_length=1024,
)

print("starting endpoint")

t= [{"role": "user", "content": "Solve the following math problem step-by-step: Convert the point $(0,3)$ in rectangular coordinates to polar coordinates.  Enter your answer in the form $(r,\theta),$ where $r > 0$ and $0 \le \theta < 2 \pi.$\\n\\nPresent the answer in LaTex format: \\boxed{Your answer}"}]


print(t)

print("Testing ancestral method:")

response, logprobs = model.ancestral([t],n=8)
print(f"Response: {response[0][0]}")
# print(f"Logprobs: {logprobs[0][:10]}... (showing first 10 tokens)")
import pdb; pdb.set_trace()

print("\nTesting continuation method:")
# Apply chat template to t first
prompt_text = model.tokenizer.apply_chat_template(
    t,
    tokenize=False,
    add_generation_prompt=True,
)
print(f"Chat template applied: {prompt_text[:400]}... (showing first 100 chars)")

# Then tokenize it
tokenized_prompt = model.tokenize([prompt_text])
print(f"Tokenized prompt: {tokenized_prompt[0][:20]}... (showing first 20 tokens)")


completions, logprobs = model.continuation(tokenized_prompt)
print(f"Completion tokens: {completions[0][:20]}... (showing first 20 tokens)")

# Decode to see the actual text
completion_text = model.decode_tokenize(completions)
print(f"Completion text: {completion_text[0]}")

print("logprompts:")
responses = [t + [{"role": "assistant", "content": response[0][0]}]]

print(model.logprobs(responses))
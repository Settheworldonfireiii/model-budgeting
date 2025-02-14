from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")


messages = [{"role": "user", "content": "Find all ordered pairs $(m,n)$ where $m$ and $n$ are positive integers such that $\frac {n^3 + 1}{mn - 1}$ is an integer"}]

prompt = torch.tensor([tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)])


text = tokenizer.decode(model.generate(prompt, max_length = 35000)[0])

print(text)

tokenizer(text, return_tensors="pt")["input_ids"].shape[1]

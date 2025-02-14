from transformers import pipeline, StoppingCriteria, StoppingCriteriaList, AutoTokenizer
import torch

# ----------------------------
# Global model and pipeline initialization:
# Create the pipeline only once at application startup.
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
GLOBAL_PIPELINE = pipeline("text-generation", model=MODEL_NAME)
# You can also access the tokenizer as:
GLOBAL_TOKENIZER = GLOBAL_PIPELINE.tokenizer

# ----------------------------
# Custom stopping criteria class remains unchanged.
class TokenThresholdStopping(StoppingCriteria):
    """
    Custom stopping criteria that stops generation after reaching a token threshold
    and injects a trigger text for continuation.
    """
    def __init__(self, threshold, trigger_text, tokenizer):
        self.threshold = threshold
        self.trigger_text = trigger_text
        self.tokenizer = tokenizer
        self.injected = False
        self.original_text = ""

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids.shape[1] >= self.threshold and not self.injected:
            self.original_text = self.tokenizer.decode(input_ids[0])
            self.injected = True
            return True
        return False

# ----------------------------
# Function that reuses the global pipeline for generation.
def generate_with_early_stopping(messages, token_threshold=600, max_final_tokens=500):
    # Use the global pipeline and its tokenizer.
    pipe = GLOBAL_PIPELINE
    tokenizer = GLOBAL_TOKENIZER

    # Format input messages into a single string.
    input_text = "\n".join([f"<|{msg['role']}|>{msg['content']}" for msg in messages])

    # Define the trigger text and create the stopping criteria.
    trigger_text = "<|im_end|>" + " Final Answer:"
    stopping_criteria = TokenThresholdStopping(token_threshold, trigger_text, tokenizer)

    # Configure generation parameters for the first stage.
    generation_config = {
        "max_new_tokens": 32768,
        "return_full_text": False,
        "do_sample": True,
        "temperature": 0.4,
        "top_p": 0.9,
        "num_beams": 2,
        "stopping_criteria": [stopping_criteria]
    }
    initial_output = pipe(input_text, **generation_config)

    # If early stopping was triggered, continue generation with the trigger text.
    if stopping_criteria.injected:
        new_prompt = stopping_criteria.original_text + trigger_text
        print("MAX FINAL TOKENS ", max_final_tokens)
        final_config = {
            "max_new_tokens": max_final_tokens,
            "return_full_text": False,
            "do_sample": True,
            "temperature": 0.4,
            "top_p": 0.9,
            "num_beams": 2
        }
        final_output = pipe(new_prompt, **final_config)
        generated_text = new_prompt + final_output[0]['generated_text']
    else:
        generated_text = initial_output[0]['generated_text']

    # Get token count for reporting.
    token_count = tokenizer(generated_text, return_tensors="pt")["input_ids"].shape[1]
    return generated_text, token_count

# ----------------------------
# Optional cleanup function to explicitly delete the pipeline and free GPU memory.
def cleanup_pipeline():
    global GLOBAL_PIPELINE, GLOBAL_TOKENIZER
    del GLOBAL_PIPELINE
    del GLOBAL_TOKENIZER
    torch.cuda.empty_cache()

# ----------------------------
def main():
    messages = [
        {"role": "user", "content": "Who are you?"},
        {"role": "user", "content": "Find all ordered pairs $(m,n)$ where $m$ and $n$ are positive integers such that $\\frac{n^3 + 1}{mn - 1}$ is an integer."}
    ]
    generated_text, token_count = generate_with_early_stopping(
        messages=messages,
        token_threshold=15000,
        max_final_tokens=200
    )

    # Save the generated text to a file.
    with open("solution_output_15000.txt", "w", encoding="utf-8") as f:
        f.write(generated_text)

    print(f"Generated text saved. Token count: {token_count}")

    # Uncomment the following line if you need to reinitialize the pipeline later:
    # cleanup_pipeline()

if __name__ == "__main__":
    main()


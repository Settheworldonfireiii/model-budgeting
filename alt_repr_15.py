from transformers import pipeline, StoppingCriteria, StoppingCriteriaList, AutoTokenizer
import torch

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

def generate_with_early_stopping(messages, model_name, token_threshold=600, max_final_tokens=500):
    """
    Generate text with early stopping and continuation after threshold.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        model_name: Name of the model to use
        token_threshold: Number of tokens before early stopping
        max_final_tokens: Maximum tokens for final answer generation
    """
    # Initialize pipeline
    pipe = pipeline("text-generation", model=model_name)
    tokenizer = pipe.tokenizer

    # Ensure proper chat template: using control tokens as defined in the model's config.
    input_text = "\n".join([f"<|{msg['role']}|>{msg['content']}" for msg in messages])
    
    # Set up stopping criteria
    trigger_text = "<|im_end|>" + " Final Answer:"
    stopping_criteria = TokenThresholdStopping(token_threshold, trigger_text, tokenizer)

    # Configure generation parameters for first stage.
    generation_config = {
        "max_new_tokens": 32768,
        "return_full_text": False,
        "do_sample": True,
        "temperature": 0.4,
        "top_p": 0.9,
        "num_beams": 2,
        # Added repetition_penalty to discourage repeated prompts.
        "repetition_penalty": 1.2,
        "stopping_criteria": [stopping_criteria]
    }
    initial_output = pipe(input_text, **generation_config)

    if stopping_criteria.injected:
        new_prompt = stopping_criteria.original_text + trigger_text
        print("MAX FINAL TOKENS", max_final_tokens)
        final_config = {
            "max_new_tokens": max_final_tokens,
            "return_full_text": False,
            "do_sample": True,
            "temperature": 0.4,
            "top_p": 0.9,
            "num_beams": 2,
            "repetition_penalty": 1.2,
        }
        final_output = pipe(new_prompt, **final_config)
        generated_text = new_prompt + final_output[0]['generated_text']
    else:
        generated_text = initial_output[0]['generated_text']

    # Count tokens
    token_count = tokenizer(generated_text, return_tensors="pt")["input_ids"].shape[1]
    return generated_text, token_count

def main():
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # Use Qwen-1.5B checkpoint.
    messages = [
        {"role": "user", "content": "Who are you?"},
        {"role": "user", "content": "Find all ordered pairs $(m,n)$ where $m$ and $n$ are positive integers such that $\\frac{n^3 + 1}{mn - 1}$ is an integer."}
    ]

    generated_text, token_count = generate_with_early_stopping(
        messages=messages,
        model_name=model_name,
        token_threshold=3000,
        max_final_tokens=200
    )

    with open("solution_output_3000.txt", "w", encoding="utf-8") as f:
        f.write(generated_text)

    print(f"Generated text saved. Token count: {token_count}")

if __name__ == "__main__":
    main()


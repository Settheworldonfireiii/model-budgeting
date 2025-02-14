from transformers import pipeline




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


def main():
    messages = [
     {"role": "user", "content": "Alice and Bob are each given $2000 to invest. Alice puts all of her money in the stock market and increases her money by a certain factor. Bob invests in real estate and makes five times more money than he invested. Bob has $8000 more than Alice now. What is the ratio of Alice's final amount to her initial investment?"},
    ] 


    pipe = pipeline("text-generation", model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

    print(pipe(messages, temperature = 0.2, num_beams = 2, max_new_tokens = 14000)[0]['generated_text'][1]['content'])


if __name__ == "__main__":
    main()


from transformers import pipeline, StoppingCriteria, StoppingCriteriaList
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from copy import deepcopy
import sys


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

def generate_with_early_stopping(messages, model_name, token_threshold=1000, max_final_tokens=200, early_stopping = True, longer_thinking = False, min_tokens = 0):
    """
    Generate text with early stopping and continuation after threshold.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        model_name: Name of the model to use
        token_threshold: Number of tokens before early stopping
        max_final_tokens: Maximum tokens for final answer generation
    """
    # Initialize pipeline
    print(model_name)

    pipe = pipeline("text-generation", model=model_name)
    

    tokenizer = pipe.tokenizer

    # Format input messages
    input_text = messages
    
    # Set up stopping criteria
    trigger_text =  "<|im_end|>" + " Final Answer:"
    stopping_criteria = TokenThresholdStopping(token_threshold, trigger_text, tokenizer)

    # First generation up to threshold
    if early_stopping:
        generation_config = {
            "max_new_tokens": token_threshold,
            "return_full_text": False,
            "do_sample": True,
            "temperature": 0.4,
            "top_p": 0.9,
            "num_beams": 2,
            "stopping_criteria": [stopping_criteria]
        }
        initial_output = pipe(input_text, **generation_config)
        # If stopped at threshold, continue with trigger text
        if stopping_criteria.injected:
            new_prompt = stopping_criteria.original_text + trigger_text
            #print("STOPPING CRITERIA ORIGINAL TEXT ", new_prompt)
            #exit()
            #print("MAX FINAL TOKENS ", max_final_tokens)    
            # Generate final answer with limited tokens
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
            print("AAARGH!")
            generated_text = initial_output[0]['generated_text']
    elif  (not longer_thinking and not early_stopping):
        generated_text = pipe(input_text, num_beams = 2, max_new_tokens = token_threshold, return_full_text = False, do_sample = True, temperature = 0.4, top_p = 0.9)[0]['generated_text']
    else:
        generated_text_ = pipe(input_text, num_beams = 2, max_new_tokens = 900, return_full_text = False, do_sample = True, temperature = 0.4, top_p = 0.9)
        generated_text =  generated_text_[0]['generated_text']
        #print(generated_text_)
        tokencount = tokenizer(generated_text, return_tensors="pt")["input_ids"].shape[1]
        while (tokencount < min_tokens):
            print(input_text)
            print("====================================================================")
            #x =  "<｜begin▁of▁sentence｜><｜User｜>Find all ordered pairs $(m,n)$ where $m$ and $n$ are positive integers such that $ rac {n^3 + 1}{mn - 1}$ is an integer.<｜Assistant｜><think>"
            x =  "<｜begin▁of▁sentence｜><｜User｜>Alice and Bob are each given $2000 to invest. Alice puts all of her money in the stock market and increases her money by a certain factor. Bob invests in real estate and makes five times more money than he invested. Bob has $8000 more than Alice now. What is the ratio of Alice's final amount to her initial investment?<｜Assistant｜><think>"

            input_text_added = input_text + generated_text + "Wait " + "<think>"
            print(input_text_added)
            #increment by chunks of 900
            generated_text += pipe(input_text_added, num_beams = 2, max_new_tokens = 900, return_full_text = False, do_sample = True, temperature = 0.4, top_p = 0.9)[0]['generated_text']
            tokencount = tokenizer(generated_text, return_tensors="pt")["input_ids"].shape[1]
            print("token count ", tokencount)
            print(" ++++++++++++++++++++ ", generated_text) 

    
    # Get token count
    token_count = tokenizer(generated_text, return_tensors="pt")["input_ids"].shape[1]
    
    return generated_text, token_count

def main():
    print(type(sys.argv[1]))
    #model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    model_name = sys.argv[1]
    print(model_name)
    """ 
    messages = [ 
        {"role": "user", "content": "Alice and Bob are each given $2000 to invest. Alice puts all of her money in the stock market and increases her money by a certain factor. Bob invests in real estate and makes five times more money than he invested. Bob has $8000 more than Alice now. What is the ratio of Alice's final amount to her initial investment?"}
    ]
    """
    
    #messages = "<｜begin▁of▁sentence｜><｜User｜>Alice and Bob are each given $2000 to invest. Alice puts all of her money in the stock market and increases her money by a certain factor. Bob invests in real estate and makes five times more money than he invested. Bob has $8000 more than Alice now. What is the ratio of Alice's final amount to her initial investment?<｜Assistant｜><think>"
    messages = "<｜begin▁of▁sentence｜><｜User｜>" + sys.argv[2]+"<｜Assistant｜><think>"
    print(messages)
    """
    
    messages = [
        {"role": "user", "content": "Find all ordered pairs $(m,n)$ where $m$ and $n$ are positive integers such that $\frac {n^3 + 1}{mn - 1}$ is an integer."}
    ]
    """
    mode = sys.argv[3]
    if mode == "1":
        min_tokens = sys.argv[4]
        token_threshold = sys.argv[5]
        max_final_tokens = sys.argv[6]
        generated_text, token_count = generate_with_early_stopping(
            messages=messages,
            model_name=model_name,
            token_threshold=token_threshold,
            max_final_tokens=max_final_tokens,
            early_stopping = False,
            longer_thinking = True,
            min_tokens = min_tokens
        )
    elif mode == "2":
        min_tokens = sys.argv[4]
        token_threshold = sys.argv[5]
        max_final_tokens = sys.argv[6]
        generated_text, token_count = generate_with_early_stopping(
            messages=messages,
            model_name=model_name,
            token_threshold=token_threshold,
            max_final_tokens=max_final_tokens,
            early_stopping = False,
            longer_thinking = False,
            min_tokens = min_tokens
        )

    elif mode == "3"
        min_tokens = sys.argv[4]
        token_threshold = sys.argv[5]
        max_final_tokens = sys.argv[6]
        generated_text, token_count = generate_with_early_stopping(
            messages=messages,
            model_name=model_name,
            token_threshold=token_threshold,
            max_final_tokens=max_final_tokens,
            early_stopping = True,
            longer_thinking = False,
            min_tokens = min_tokens
        )




    # Save output
    with open("solution_output_4000_1B_bob_alice.txt", "w", encoding="utf-8") as f:
        f.write(generated_text)
    
    print(f"Generated text saved. Token count: {token_count}")

if __name__ == "__main__":
    main()

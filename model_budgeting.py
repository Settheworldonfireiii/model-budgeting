from transformers import pipeline, StoppingCriteria, StoppingCriteriaList
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

import torch
from copy import deepcopy
import sys
import pdb

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

def generate_with_early_stopping(messages, model_name, token_threshold=1000, max_final_tokens=200, early_stopping = True, longer_thinking = False, min_tokens = 0, temperature = 0.2, top_p = 0.9, num_beams = 2):
    """
    Generate text with early stopping and continuation after threshold.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        model_name: Name of the model to use
        token_threshold: Number of tokens before early stopping
        max_final_tokens: Maximum tokens for final answer generation
    """
    # Initialize pipeline

    pipe = pipeline("text-generation", model=model_name, device="cuda")
    

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
            "temperature": temperature,
            "top_p": top_p,
            "num_beams": num_beams,
            "stopping_criteria": [stopping_criteria]
        }
        initial_output = pipe(input_text, **generation_config)
        # If stopped at threshold, continue with trigger text
        if stopping_criteria.injected:
            new_prompt = stopping_criteria.original_text + trigger_text
            final_config = {
                "max_new_tokens": max_final_tokens,
                "return_full_text": False,
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
                "num_beams": num_beams
            }
        
            final_output = pipe(new_prompt, **final_config)
            generated_text = new_prompt + final_output[0]['generated_text']
        else:
           
            generated_text = initial_output[0]['generated_text']
    elif  (not longer_thinking and not early_stopping):
        generated_text = pipe(input_text, num_beams = num_beams, max_new_tokens = token_threshold, return_full_text = False, do_sample = True, temperature = temperature, top_p = top_p)[0]['generated_text']
    elif longer_thinking:
        generated_text = pipe(input_text, num_beams = num_beams, max_new_tokens = token_threshold, return_full_text = False, do_sample = True, temperature = temperature, top_p = top_p)[0]['generated_text']

        
        tokencount = tokenizer(generated_text, return_tensors="pt")["input_ids"].shape[1]        
        while (tokencount < min_tokens):

            input_text_added = input_text + generated_text + "Wait, let's reconsider " + "<think>"
           
           
            generated_text += pipe(input_text_added, num_beams = 2, max_new_tokens = token_threshold, return_full_text = False, do_sample = True, temperature = 0.4, top_p = 0.9)[0]['generated_text']
            tokencount = tokenizer(generated_text, return_tensors="pt")["input_ids"].shape[1]
            
        input_text_added =input_text + generated_text + "<|im_end|>" + " Final Answer:"
        generated_text += pipe(input_text_added, num_beams = 2, max_new_tokens = max_final_tokens, return_full_text = False, do_sample = True, temperature = 0.4, top_p = 0.9)[0]['generated_text']
        tokencount = tokenizer(generated_text, return_tensors="pt")["input_ids"].shape[1]

    # Get token count
    token_count = tokenizer(generated_text, return_tensors="pt")["input_ids"].shape[1]
    
    return generated_text, token_count

def main():
  
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',  help="model name, including vendor", type= str)
    parser.add_argument('--prompt',  help="the task we are giving the model to solve. it is imperative to put it in double quotes for the model to work properly", type= str)
    parser.add_argument('--mode',  help="this argument controls whether we are doing extended thinking, early interrupted thinking or just letting the model think for the amount of tokens it thinks appropriate", type= str)
    parser.add_argument('--min_tokens',  help="minimal amount of tokens that the model must think for when we extending its reasoning time", type= int)
    parser.add_argument('--max_tokens',  help="token threshold, or maximal amount of tokens that we allow before we interrupt the model and force it to give the answer", type= int)
    parser.add_argument('--max_final_tokens',  help="maximum amount of tokens the model is allowed to output after its thinking was interrupted and it is forced to give a final answer", type= int)
    parser.add_argument('--output_filename',  help="output_filename", type= str)
    parser.add_argument('--temperature',  help="generation temperature", type= float)
    parser.add_argument('--top_p',  help="share of the words the model considers when generating stuff", type= float)
    parser.add_argument('--num_beams',  help="the default method is beam search. number of beams to keep at the nth step when proceeding to n+1th step", type= int)




    args=parser.parse_args()



    
    messages = "<｜begin▁of▁sentence｜><｜User｜>" + args.prompt+"<｜Assistant｜><think>"

    mode = args.mode
    if mode == "extend-thinking":
        generated_text, token_count = generate_with_early_stopping(
            messages=messages,
            model_name=args.model_name,
            token_threshold= args.max_tokens,
            max_final_tokens=args.max_final_tokens,
            early_stopping = False,
            longer_thinking = True,
            min_tokens = args.min_tokens,
            temperature = args.temperature,
            top_p = args.top_p,
            num_beams = args.num_beams

        )
    elif mode == "plain":

        generated_text, token_count = generate_with_early_stopping(
            messages=messages,
            model_name=args.model_name,
            token_threshold=args.max_tokens,
            max_final_tokens=args.max_final_tokens,
            early_stopping = False,
            longer_thinking = False,
            min_tokens = args.min_tokens,
            temperature = args.temperature,
            top_p = args.top_p,
            num_beams = args.num_beams

        )

    elif mode == "limit-thinking":

        generated_text, token_count = generate_with_early_stopping(
            messages=messages,
            model_name=args.model_name,
            token_threshold=args.max_tokens,
            max_final_tokens=args.max_final_tokens,
            early_stopping = True,
            longer_thinking = False,
            min_tokens = args.min_tokens,
            temperature = args.temperature,
            top_p = args.top_p,
            num_beams = args.num_beams

        )




    # Save output
    output_file = args.output_filename
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(generated_text)
    
    print(f"Generated text saved. Token count: {token_count}")
    print(generated_text)
if __name__ == "__main__":
    main()

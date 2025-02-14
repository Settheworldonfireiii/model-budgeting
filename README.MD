How to use this script?

An example:

python model_budgeting.py --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --prompt "Alice and Bob are each given \$2000 to invest. Alice puts all of her money in the stock market and increases her money by a certain factor. Bob invests in real estate and makes five times more money than he invested. Bob has \$8000 more than Alice now. What is the ratio of Alice&apos;s final amount to her initial investment?" --mode limit-thinking --min_tokens 300 --max_tokens 800 --max_final_tokens 200 --output_filename mein_ausgabe

model_name -- model name. Can be any version of distilled R1 to Qwen

prompt -- prompt. The task you want it to solve. Put necessarily in quotes. Later, for larger scale experiments, I will make it optional so that one can choose whether to insert prompt like this, or load some dataset

mode -- mode of interference with model's thinking. limit-thinking is for early interruption, plain is for normal thinking whatever model nees to think, extend-thinking is for forcing the model to think for at least n tokens

min_tokens -- minimal amount of tokens the model will think. if extend-thinking or plain mode is chosen, it will not affect anything

max_tokens -- token threshold. If limit-thinking mode is chosen, it is the maximal amount of tokens the model is allowed to think before the answer is given. If plain mode is chosen, the same, but defaults to a higher value of 32768 and user input.

max_final_tokens -- max final tokens. How many tokens you can use for the answer, after thinking is finished.
 
output_filename -- the name of the file you want to write the thinking and the answer to

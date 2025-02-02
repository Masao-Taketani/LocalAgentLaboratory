import time, tiktoken
import openai
import os, anthropic, json
from transformers import set_seed

TOKENS_IN = dict()
TOKENS_OUT = dict()

encoding = tiktoken.get_encoding("cl100k_base")

def curr_cost_est():
    costmap_in = {
        "gpt-4o": 2.50 / 1000000,
        "gpt-4o-mini": 0.150 / 1000000,
        "o1-preview": 15.00 / 1000000,
        "o1-mini": 3.00 / 1000000,
        "claude-3-5-sonnet": 3.00 / 1000000,
        "deepseek-chat": 1.00 / 1000000,
        "o1": 15.00 / 1000000,
    }
    costmap_out = {
        "gpt-4o": 10.00/ 1000000,
        "gpt-4o-mini": 0.6 / 1000000,
        "o1-preview": 60.00 / 1000000,
        "o1-mini": 12.00 / 1000000,
        "claude-3-5-sonnet": 12.00 / 1000000,
        "deepseek-chat": 5.00 / 1000000,
        "o1": 60.00 / 1000000,
    }
    return sum([costmap_in[_]*TOKENS_IN[_] for _ in TOKENS_IN]) + sum([costmap_out[_]*TOKENS_OUT[_] for _ in TOKENS_OUT])

def query_model(platform, model_or_pipe, prompt, system_prompt, tries=5, timeout=5.0, temp=None, show_r1_thought=False):
    for _ in range(tries):
        if temp is None: temp = 1.0
        try:
            if model_str == "gpt-4o-mini" or model_str == "gpt4omini" or model_str == "gpt-4omini" or model_str == "gpt4o-mini":
                model_str = "gpt-4o-mini"
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}]

                    client = OpenAI()
                    completion = client.chat.completions.create(
                        model="gpt-4o-mini-2024-07-18", messages=messages, temperature=temp)
                answer = completion.choices[0].message.content

            if platform == "ollama":
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}]

                    client = openai.OpenAI(base_url='http://localhost:11434/v1/',
                                            api_key='ollama', # required but ignored
                                            )
                    completion = client.chat.completions.create(
                        model=model_or_pipe, messages=messages, temperature=temp)
                answer = completion.choices[0].message.content
                if "deepseek-r1" in model_or_pipe.lower(): 
                    thought, answer = answer.split("</think>")
                    answer = answer[2:]
                    if show_r1_thought:
                        print("\n\n[DeepSeek R1's thought process starts here]")
                        print(thought.split("<think>")[1][1:])
                        print("[DeepSeek R1's thought process ends here]\n\n")

            elif platform == "huggingface":
                prompt = model_or_pipe.tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ], 
                    tokenize=False, 
                    add_generation_prompt=True
                )

                set_seed(0)
                response = model_or_pipe(prompt,
                                         do_sample=True,
                                         temperature=temp,
                                         max_new_tokens=MAX_NUM_TOKENS,
                )
                answer = response[0]["generated_text"][len(prompt):]
                if "deepseek-r1" in model_or_pipe.tokenizer.name_or_path.lower(): 
                    thought, answer = answer.split("</think>")
                    answer = answer[2:]
                    if show_r1_thought:
                        print("\n\n[DeepSeek R1's thought process starts here]")
                        print(thought.split("<think>")[1][1:])
                        print("[DeepSeek R1's thought process ends here]\n\n")
            else:
                raise ValueError(f"Platform {platform} is not supported.")

            return answer
        except Exception as e:
            print("Inference Exception:", e)
            time.sleep(timeout)
            continue
    raise Exception("Max retries: timeout")


#print(query_model(model_str="o1-mini", prompt="hi", system_prompt="hey"))
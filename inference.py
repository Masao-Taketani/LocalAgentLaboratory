import time, tiktoken
from openai import OpenAI
import os, json, subprocess
from transformers import set_seed
from transformers.pipelines.text_generation import TextGenerationPipeline
import logging


def query_model(platform, model_or_pipe, prompt, system_prompt, tries=5, timeout=5.0, temp=None, show_r1_thought=False, 
                max_length=None, num_ctx=None):
    if temp is None: temp = 1.0
    print('temp:', temp)
    for _ in range(tries):
        try:
            if platform == "ollama":
                # Disable OpenAI API since num_ctx cannot be adjusted. 
                # For details, https://github.com/ollama/ollama/blob/main/docs/openai.md#setting-the-context-size
                #messages = [
                #    {"role": "system", "content": system_prompt},
                #    {"role": "user", "content": prompt}
                #]
                #
                #client = OpenAI(base_url='http://localhost:11434/v1/',
                #                api_key='ollama', # required but ignored
                #                )
                #completion = client.chat.completions.create(
                #    model=model_or_pipe, messages=messages, temperature=temp)
                #answer = completion.choices[0].message.content
                if num_ctx is None:
                    num_ctx = 32768 if "qwen2.5" in model_or_pipe.lower() else 131072
                system_prompt_dic = {"role": "system", "content": f"""{system_prompt}"""}
                prompt_dic = {"role": "user", "content": f"""{prompt}"""}
                system_prompt_json = json.dumps(system_prompt_dic)
                prompt_json = json.dumps(prompt_dic)
                messages = rf"""[{system_prompt_json}, {prompt_json}]"""
                args = rf"""{{"model": "{model_or_pipe}", "messages": {messages}, "options": {{"num_ctx": {num_ctx}}}, "stream": false}}"""
                with open("tmp_args.txt","w") as f:
                    f.write(args)
                command = ["curl", "http://localhost:11434/api/chat", "-d", "@tmp_args.txt"]
                rlt = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                answer = json.loads(rlt.stdout)["message"]["content"]
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
                do_sample = True if temp != 0 else False
                if max_length is None:
                    max_length = 32768 if "qwen2.5" in model_or_pipe.tokenizer.name_or_path.lower() else 131072
                response = model_or_pipe(prompt,
                                         do_sample=do_sample,
                                         temperature=temp,
                                         max_length=max_length,
                                         truncation=True
                )
                answer = response[0]["generated_text"][len(prompt):]
            else:
                raise ValueError(f"Platform {platform} is not supported.")

            if (isinstance(model_or_pipe, str) and "deepseek-r1" in model_or_pipe.lower()) \
            or (isinstance(model_or_pipe, TextGenerationPipeline) and "deepseek-r1" in model_or_pipe.tokenizer.name_or_path.lower()):
                thought, answer = answer.split("</think>")
                answer = answer[2:]
                if show_r1_thought:
                    print("\n\n[DeepSeek R1's thought process starts here]")
                    print(thought.split("<think>")[1][1:])
                    print("[DeepSeek R1's thought process ends here]\n\n")

            return answer
        except Exception as e:
            print("Inference Exception:", e)
            logging.exception("An unexpected error just happened.")
            time.sleep(timeout)
            continue
    raise Exception("Max retries: timeout")
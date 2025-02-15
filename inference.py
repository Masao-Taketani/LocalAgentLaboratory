import time, tiktoken
from openai import OpenAI
import os, anthropic, json
from transformers import set_seed
from transformers.pipelines.text_generation import TextGenerationPipeline


def query_model(platform, model_or_pipe, prompt, system_prompt, tries=5, timeout=5.0, temp=None, show_r1_thought=False, max_length=131072):
    for _ in range(tries):
        if temp is None: temp = 1.0
        try:
            if platform == "ollama":
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]

                client = OpenAI(base_url='http://localhost:11434/v1/',
                                api_key='ollama', # required but ignored
                                )
                completion = client.chat.completions.create(
                    model=model_or_pipe, messages=messages, temperature=temp)
                answer = completion.choices[0].message.content

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
            time.sleep(timeout)
            continue
    raise Exception("Max retries: timeout")
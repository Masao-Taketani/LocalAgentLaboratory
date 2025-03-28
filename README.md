# Local Agent Laboratory: Using Local LLM Agents as Research Assistants

<p align="center">
  <img src="media/LocalAgentLabMainPic.jpg" alt="Demonstration of the flow of AgentClinic" style="width: 60%;">
</p>

## Table of Contents

1. [Introduction](#introduction)
2. [Modification](#modification)
3. [Overview](#overview)
   - [How Does Local Agent Laboratory Work?](#-how-does-local-agent-laboratory-work)
   - [Supported Platforms and Models](#supported-platforms-and-models) 
4. [Environmental Setup](#environmental-setup)
5. [Program Execution](#program-execution)
   - [Now run Local Agent Laboratory!](#now-run-local-agent-laboratory)
   - [Co-Pilot Mode](#co-pilot-mode)
6. [Tips for Better Research Outcomes](#tips-for-better-research-outcomes)
   - [Regarding Local LLMs](#regarding-local-llms)
      - [[Tip #1] 🌡️ Adjust proper temperature for each phase! 🌡️](#tip-1-️-adjust-proper-temperature-for-each-phase-️)
      - [[Tip #2] 🤖 Hugging Face for accuracy, speed for Ollama! 🤖](#tip-2--hugging-face-for-accuracy-speed-for-ollama-)
      - [[Tip #3] 🤖 Qwen2.5-72B-Instruct for non-coding and DeepSeek-R1-Distill-Llama-70B for coding phases! 🤖](#tip-3--qwen25-72b-instruct-for-non-coding-and-deepseek-r1-distill-llama-70b-for-coding-phases-)
   - [Regarding LLMs Overvall](#regarding-llms-overvall)
      - [[Tip #4] 📝 Make sure to write extensive notes! 📝](#tip-4--make-sure-to-write-extensive-notes-)
      - [[Tip #5] 🚀 Using more powerful models generally leads to better research 🚀](#tip-5--using-more-powerful-models-generally-leads-to-better-research-)
      - [[Tip #6] ✅ You can load previous saves from checkpoints ✅](#tip-6--you-can-load-previous-saves-from-checkpoints-)
      - [[Tip #7] 🈯 If you are running in a language other than English 🈲](#tip-7--if-you-are-running-in-a-language-other-than-english-)
      - [[Tip #8] 🌟 There is a lot of room for improvement 🌟](#tip-8--there-is-a-lot-of-room-for-improvement-)
7. [📜 License](#-license)
8. [Reference](#reference)


## Introduction
This repository is based on [AgentLaboratory](https://github.com/SamuelSchmidgall/AgentLaboratory), but this repository supports local LLMs, 
which the original repo does not. It is especially good if you or your organization has enough GPUs for you to use. Some of possible 
advantages are as follows.

- Don't need to send your data to cloud LLMs.
- Once you download a model, you don't need to access the internet to utilize LLMs.
- Can flexibly and locally finetune a LLM model if you want to.
- Don't need to spend money on pay-as-you-go APIs, which are ambiguous and hard to estimate the total costs that you need to pay.
Besides, even if you decide to utilize one of the cloud LLMs later on, using local LLMs first can give you an approximation of the costs 
before using the cloud ones, such as how many tokens a LLM would produce in order to solve your problem. 
- It is also good if you would like to investigate and experiment what kind of outcomes you can expect when you feed your data for internal 
investigation purposes.


## Modification
I have made some modifications in this repo from the original one.
- Enable you to use local LLMs, which includes recently announced reasoning models such as  [**DeepSeek R1**](https://github.com/deepseek-ai/DeepSeek-R1) or [**QwQ**](https://qwenlm.github.io/blog/qwq-32b/), instead of cloud ones
- Fixed some prompts for clearer instructions
- Made arguments and some crucial parameters configurable using a JSON config file. For details, please check [config.json](config.json)
- Enable you to override arguments such as a model selection for each phase using the config file explained above even when you restart with a previously saved state file. The original repo does not allow that unless you modify the code since all the instance variables of `LaboratoryWorkflow` that contain those arguments are all set when the class is instantiated and are saved as part of the state files. Due to that, when you restart with one of those state files, the instance still uses the same arguments that were set when instantiated
- Made clear the import dependencies because the original code frequently uses `import *`, which is ambiguous and not recommended
- Created [Dockerfile](Dockerfile) in order to locally and efficiently build a development environment. For details, please check [Environmental Setup](#environmental-setup)
- Include some examples that were created using Local LLMs. Please refer to [examples](examples/) directory for details

> [!Tip]
> The [examples](examples/) were created using end-to-end autonomous mode, which means no human intervention from start to finish. In order to get better results, one way to do it is to include human intervention in some or all of the phases, which is called co-pilot mode in the paper. For more details, please check [Co-Pilot Mode](#co-pilot-mode).

> [!Tip]
> Other ways to get better results are to adjust temperature and prompts, conduct various trials with saved states, and so on. For more details, please check [Tips for Better Research Outcomes](#tips-for-better-research-outcomes).

## 📖 Overview

- **Local Agent Laboratory** is an end-to-end autonomous research workflow meant to assist **you** as the human researcher toward **implementing your research ideas**. Agent Laboratory consists of specialized agents driven by large language models to support you through the entire research workflow—from conducting literature reviews and formulating plans to executing experiments and writing comprehensive reports. 
- This system is not designed to replace your creativity but to complement it, enabling you to focus on ideation and critical thinking while automating repetitive and time-intensive tasks like coding and documentation. By accommodating varying levels of computational resources and human involvement, Local Agent Laboratory aims to accelerate scientific discovery and optimize your research productivity.

<p align="center">
  <img src="media/AgentLab.png" alt="Demonstration of the flow of AgentClinic" style="width: 99%;">
</p>

### 🔬 How Does Local Agent Laboratory Work?

- Local Agent Laboratory consists of three primary phases that systematically guide the research process: (1) Literature Review, (2) Experimentation, and (3) Report Writing. During each phase, specialized agents driven by LLMs collaborate to accomplish distinct objectives, integrating external tools like arXiv, Hugging Face, Python, and LaTeX to optimize outcomes. This structured workflow begins with the independent collection and analysis of relevant research papers, progresses through collaborative planning and data preparation, and results in automated experimentation and comprehensive report generation. Details on specific agent roles and their contributions across these phases are discussed in the paper.

<p align="center">
  <img src="media/AgentLabWF.png" alt="Demonstration of the flow of AgentClinic" style="width: 99%;">
</p>

### Supported Platforms and Models
For this repo, any models from Ollama or Hugging Face, including the recently-announced reasoning models, such as DeepSeek R1 and QwQ-32B, are supported to be used as local LLMs. 
So, pick a platform either `huggingface` or `ollama` with [`platform`](config.json#L24) argument.

If you'd like to check thought processes when you use one of the aforementioned reasoning models, set a flag named [`show_thought`](config.json#L46) as `true`. That way, you can see the thought processes in your console!


## Environmental Setup

> [!Tip]
> Remove #s for the [two lines](Dockerfile#L4) after setting your time zone, so that you can avoid an interruption in the process of building the environment.

Follow the installation steps below.
```
(Type the following commands at host)
git clone https://github.com/Masao-Taketani/LocalAgentLaboratory.git
docker build -t agentlab .
docker run -it --rm --gpus '"device=[device id(s)]"' -v .:/work agentlab:latest

(If you decide to use Ollama platform, type the following commands after starting the container)
(Start a screen session in order to start Ollama in another session)
screen -S ollama
ollama serve
(Press [Ctrl+a+d] to get out of the screen session)
ollama pull [ollama model name]
```


## Program Execution

### Now run Local Agent Laboratory!

Execute the following command. As for `[your config path]`, please refer to [config.json](config.json).
```
python ai_lab_repo.py --config_path [your config path]
```

### Co-Pilot Mode

If you would like to do co-pilot mode, modify the provided config file. You can intervene any phase(s) you want. In order to do that, modify [here](config.json#L14).


## Tips for Better Research Outcomes

### Regarding Local LLMs

#### [Tip #1] 🌡️ Adjust proper temperature for each phase! 🌡️

Since local LLMs' capabilities are not on par with cloud LLMs' such as GPT-4o, adjusting temperature is crucial. As I have experienced several times during experiments with this repo, I have encountered so many errors especially when LLMs are dealing with writing code and paper. Oftentimes adjusting temperature for those phases would work well although intial setting of temperature would not. As I said, `data preparation`, `running experiments`, and `report writing` phases are the most notorious ones! So, be patient, and conduct grid search or whatever you feel like. For reference, I have tried temperature from 0.0 to 1.0. It sometime worked and sometime not. So, see it for yourself! You can adjust each temperature [here](config.json#L36).

-----

#### [Tip #2] 🤖 Hugging Face for accuracy, speed for Ollama! 🤖

As far as I've experimented, I can say that performance of models coming from `huggingface` platform are better than ones from `ollama`. For example, `Qwen/Qwen2.5-72B-Instruct` from `huggingface` is better than `qwen2.5:72b-instruct-fp16` (which presumably is the best and non-quantized Qwen2.5 model available from Ollama) from `ollama`. What I exactly meant here is that models from `ollama` do not follow given instructions as much as the ones from `huggingface`. The reason seems to come from the fact that even the most accurate models from Ollama are half-precision (FP16). So, unless you have strict computational restrictions, I suggest you use models from `huggingface`, preferably models as capable as (or even better than) `Qwen/Qwen2.5-72B-Instruct`. As for inference speed, ones from Ollama perform pretty well, especially compared to Hugging Face's counterparts. Thus, if you want to save some machine power, use Ollama models for faster inference where you don't need precise execution such as phases that can only be done with conversations between agents.

-----

#### [Tip #3] 🤖 Qwen2.5-72B-Instruct for non-coding and DeepSeek-R1-Distill-Llama-70B for coding phases! 🤖

Also as far as I've experimented, `Qwen2.5-72B-Instruct` follows given instructions very well if the phases are non-coding, but not so much for coding ones. On the other hand, `DeepSeek-R1-Distill-Llama-70B` does good job when it comes to coding, but sometimes does not correctly follow non-coding instructions. Those are the things that I've found so far. So, if things don't work out for your case, please try this tip. By the way, as for the best configuration I've found when it comes to model selections, it is written [here](config.json#L26).

-----

### Regarding LLMs Overvall

#### [Tip #4] 📝 Make sure to write extensive notes! 📝

**Writing extensive notes is important** for helping your agent understand what you're looking to accomplish in your project, as well as any style preferences. Notes can include any experiments you want the agents to perform, providing API keys, certain plots or figures you want included, or anything you want the agent to know when performing research.

This is also your opportunity to let the agent know **what compute resources it has access to**, e.g. GPUs (how many, what type of GPU, how many GBs), CPUs (how many cores, what type of CPUs), storage limitations, and hardware specs.

In order to add notes, you must modify the [task_notes_LLM](ai_lab_repo.py#L694) structure inside of `ai_lab_repo.py`. 

#### [Tip #5] 🚀 Using more powerful models generally leads to better research 🚀

When conducting research, **the choice of model can significantly impact the quality of results**. More powerful models tend to have higher accuracy, better reasoning capabilities, and better report generation. If computational resources allow, prioritize the use of advanced models such as Qwen2.5-72B-Instruct or similar state-of-the-art local LLMs.

However, **it’s important to balance performance and cost-effectiveness**. While powerful models may yield better results, they are often more expensive and time-consuming to run. Consider using them selectively—for instance, for key experiments or final analyses—while relying on smaller, more efficient models for iterative tasks or initial prototyping.

When resources are limited, **optimize by fine-tuning smaller models** on your specific dataset or combining pre-trained models with task-specific prompts to achieve the desired balance between performance and computational efficiency.

-----

#### [Tip #6] ✅ You can load previous saves from checkpoints ✅

**If you lose progress, or if a subtask fails, you can always load from a previous state.** All of your progress is saved by default in the `state_saves` variable, which stores each individual checkpoint. Just set `load_existing` as `true`, which can be found [here](config.json#L3), and pass your saved state [here](config.json#L6) when running `ai_lab_repo.py`

-----

#### [Tip #7] 🈯 If you are running in a language other than English 🈲

If you are running Agent Laboratory in a language other than English, no problem, just make sure to provide a language flag to the agents to perform research in your preferred language. Note that we have not extensively studied running Local Agent Laboratory in other languages, so be sure to report any problems you encounter. You can adjust the language [here](config.json#L48).

----


#### [Tip #8] 🌟 There is a lot of room for improvement 🌟

There is a lot of room to improve this codebase, so if you end up making changes and want to help the community, please feel free to share the changes you've made! We hope this tool helps you!


## 📜 License

Source Code Licensing: This repository's source code is licensed under the MIT License. This license permits the use, modification, and distribution of the code, subject to certain conditions outlined in the MIT License.

## Reference

[SamuelSchmidgall/AgentLaboratory](https://github.com/SamuelSchmidgall/AgentLaboratory)

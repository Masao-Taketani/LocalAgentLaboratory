from copy import copy
import torch
from torch.backends.mkl import verbose
import argparse
import pickle
import time
import os
import gc
import json

from agents import ReviewersAgent, PhDStudentAgent, PostdocAgent, ProfessorAgent, MLEngineerAgent, SWEngineerAgent, init_hf_pipe, AVAILABLE_PLATFORMS
from utils import extract_prompt, save_to_file, remove_figures, remove_directory, read_jsonc
from mlesolver import MLESolver
from papersolver import PaperSolver
from tools import ArxivSearch, HFDataSearch, execute_code


DEFAULT_PLATFORM = "huggingface"
DEFAULT_LLM_BACKBONE = "Qwen/Qwen2.5-72B-Instruct"


class LaboratoryWorkflow:
    def __init__(self, research_topic, platform, agent_model_backbone, temps, show_r1_thought, out_dirpath, 
                 max_steps=100, num_papers_lit_review=5, notes=list(), human_in_loop_flag=None, compile_pdf=True, 
                 mlesolver_max_steps=3, papersolver_max_steps=5):
        """
        Initialize laboratory workflow
        @param research_topic: (str) description of research idea to explore
        @param max_steps: (int) max number of steps for each phase, i.e. compute tolerance budget
        @param num_papers_lit_review: (int) number of papers to include in the lit review
        @param agent_model_backbone: (dict) model backbone to use for agents
        @param notes: (list) notes for agent to follow during tasks
        """

        self.notes = notes
        self.max_steps = max_steps
        self.compile_pdf = compile_pdf
        self.research_topic = research_topic
        self.num_papers_lit_review = num_papers_lit_review
        self.platform = platform
        self.temps = temps

        self.show_r1_thought = show_r1_thought
        self.out_dirpath = out_dirpath

        self.print_cost = True
        self.review_override = True # should review be overridden?
        self.review_ovrd_steps = 0 # review steps so far
        self.arxiv_paper_exp_time = 3
        self.reference_papers = list()

        ##########################################
        ####### COMPUTE BUDGET PARAMETERS ########
        ##########################################
        self.num_ref_papers = 1
        self.review_total_steps = 0 # num steps to take if overridden
        self.arxiv_num_summaries = 5
        self.mlesolver_max_steps = mlesolver_max_steps
        self.papersolver_max_steps = papersolver_max_steps

        self.phases = [
            ("literature review", ["literature review"]),
            ("plan formulation", ["plan formulation"]),
            ("experimentation", ["data preparation", "running experiments"]),
            ("results interpretation", ["results interpretation", "report writing", "report refinement"]),
        ]
        self.phase_status = dict()
        for phase, subtasks in self.phases:
            for subtask in subtasks:
                self.phase_status[subtask] = False

        self.set_phase_models(agent_model_backbone)

        self.human_in_loop_flag = human_in_loop_flag

        self.statistics_per_phase = {
            "literature review":      {"time": 0.0, "steps": 0.0,},
            "plan formulation":       {"time": 0.0, "steps": 0.0,},
            "data preparation":       {"time": 0.0, "steps": 0.0,},
            "running experiments":    {"time": 0.0, "steps": 0.0,},
            "results interpretation": {"time": 0.0, "steps": 0.0,},
            "report writing":         {"time": 0.0, "steps": 0.0,},
            "report refinement":      {"time": 0.0, "steps": 0.0,},
        }

        self.save = True
        self.verbose = True
        # Following instantiations are not used
        self.reviewers = ReviewersAgent()
        self.phd = PhDStudentAgent(max_steps=self.max_steps)
        self.postdoc = PostdocAgent(max_steps=self.max_steps)
        self.professor = ProfessorAgent(max_steps=self.max_steps)
        self.ml_engineer = MLEngineerAgent(max_steps=self.max_steps)
        self.sw_engineer = SWEngineerAgent(max_steps=self.max_steps)

        # remove previous files
        remove_figures()
        remove_directory(os.path.join(self.out_dirpath, "research_dir"))
        # make src and research directory
        if not os.path.exists(os.path.join(self.out_dirpath, "state_saves")):
            os.mkdir(os.path.join(self.out_dirpath, "state_saves"))
        os.mkdir(os.path.join(self.out_dirpath, "research_dir"))
        os.mkdir(os.path.join(self.out_dirpath, "research_dir/src"))
        os.mkdir(os.path.join(self.out_dirpath, "research_dir/tex"))

    def set_phase_models(self, agent_model_backbone):
        for phase, subtasks in self.phases:
            for subtask in subtasks:
                assert subtask in agent_model_backbone, f"{subtask} is not in agent_model_backbone dict."
                val = agent_model_backbone[subtask]
                assert isinstance(val, str) and val, f"value of agent_model_backbone has to be str and not empty. What you input: {val}"
        self.phase_models = agent_model_backbone

    def set_model(self, model):
        if self.platform == "huggingface":
            self.model_or_pipe = init_hf_pipe(model)
        elif self.platform == "ollama":
            self.model_or_pipe = model

    def set_model_for_task(self, task):
        self.clear_gpu_mem_used_by_hf()
        if task in self.phase_models:
            self.set_model(self.phase_models[task])
        else:
            self.platform = DEFAULT_PLATFORM 
            self.set_model(f"{DEFAULT_LLM_BACKBONE}")

    def save_state(self, phase):
        """
        Save state for phase
        @param phase: (str) phase string
        @return: None
        """
        if self.platform == "huggingface":
            tmp = self.model_or_pipe
            self.model_or_pipe = None
        phase = phase.replace(" ", "_")
        with open(os.path.join(self.out_dirpath, f"state_saves/{phase}.pkl"), "wb") as f:
            pickle.dump(self, f)
        if self.platform == "huggingface": 
            self.model_or_pipe = tmp
            del tmp

    def set_agent_attr(self, attr, obj, incl_rev=False):
        """
        Set attribute for all agents
        @param attr: (str) agent attribute
        @param obj: (object) object attribute
        @return: None
        """
        if incl_rev: setattr(self.reviewers, attr, obj)
        setattr(self.phd, attr, obj)
        setattr(self.postdoc, attr, obj)
        setattr(self.professor, attr, obj)
        setattr(self.ml_engineer, attr, obj)
        setattr(self.sw_engineer, attr, obj)

    def reset_agents(self):
        """
        Reset all agent states
        @return: None
        """
        self.phd.reset()
        self.postdoc.reset()
        self.professor.reset()
        self.ml_engineer.reset()
        self.sw_engineer.reset()

    def clear_gpu_mem_used_by_hf(self):
        if getattr(self, "model_or_pipe", None):
            del self.model_or_pipe
            gc.collect()
            torch.cuda.empty_cache()

    def perform_research(self):
        """
        Loop through all research phases
        @return: None
        """
        for phase, subtasks in self.phases:
            phase_start_time = time.time()  # Start timing the phase
            if self.verbose: print(f"{'*'*50}\nBeginning phase: {phase}\n{'*'*50}")
            for subtask in subtasks:
                if self.phase_status[subtask]: 
                    print(f"{'='*40}\nSkip {subtask}\n{'='*40}")
                    continue
                if self.verbose: print(f"{'&'*30}\nBeginning subtask: {subtask}\n{'&'*30}")
                
                if (subtask not in self.phase_status or not self.phase_status[subtask]) and subtask == "literature review":
                    self.set_model_for_task(subtask)
                    repeat = True
                    while repeat: repeat = self.literature_review()
                    self.phase_status[subtask] = True
                if (subtask not in self.phase_status or not self.phase_status[subtask]) and subtask == "plan formulation":
                    self.set_model_for_task(subtask)
                    repeat = True
                    while repeat: repeat = self.plan_formulation()
                    self.phase_status[subtask] = True
                if (subtask not in self.phase_status or not self.phase_status[subtask]) and subtask == "data preparation":
                    self.set_model_for_task(subtask)
                    repeat = True
                    while repeat: repeat = self.data_preparation()
                    self.phase_status[subtask] = True
                if (subtask not in self.phase_status or not self.phase_status[subtask]) and subtask == "running experiments":
                    self.set_model_for_task(subtask)
                    repeat = True
                    while repeat: repeat = self.running_experiments()
                    self.phase_status[subtask] = True
                if (subtask not in self.phase_status or not self.phase_status[subtask]) and subtask == "results interpretation":
                    self.set_model_for_task(subtask)
                    repeat = True
                    while repeat: repeat = self.results_interpretation()
                    self.phase_status[subtask] = True
                if (subtask not in self.phase_status or not self.phase_status[subtask]) and subtask == "report writing":
                    self.set_model_for_task(subtask)
                    repeat = True
                    while repeat: repeat = self.report_writing()
                    self.phase_status[subtask] = True
                if (subtask not in self.phase_status or not self.phase_status[subtask]) and subtask == "report refinement":
                    self.set_model_for_task(subtask)
                    return_to_exp_phase = self.report_refinement()

                    if not return_to_exp_phase:
                        if self.save: self.save_state(subtask)
                        return

                    self.set_agent_attr("second_round", return_to_exp_phase)
                    self.set_agent_attr("prev_report", copy(self.phd.report))
                    self.set_agent_attr("prev_exp_results", copy(self.phd.exp_results))
                    self.set_agent_attr("prev_results_code", copy(self.phd.results_code))
                    self.set_agent_attr("prev_interpretation", copy(self.phd.interpretation))

                    self.phase_status["plan formulation"] = False
                    self.phase_status["data preparation"] = False
                    self.phase_status["running experiments"] = False
                    self.phase_status["results interpretation"] = False
                    self.phase_status["report writing"] = False
                    self.phase_status["report refinement"] = False
                    self.perform_research()
                if self.save: self.save_state(subtask)
                # Calculate and print the duration of the phase
                phase_end_time = time.time()
                phase_duration = phase_end_time - phase_start_time
                print(f"Subtask '{subtask}' completed in {phase_duration:.2f} seconds.")
                self.statistics_per_phase[subtask]["time"] = phase_duration

    def report_refinement(self):
        """
        Perform report refinement phase
        @return: (bool) whether to repeat the phase
        """
        subtask_name = "report refinement"
        reviews = self.reviewers.inference(self.phd.plan, self.phd.report, self.platform, self.model_or_pipe,
                                           self.show_r1_thought)
        print("Reviews:", reviews)
        if self.human_in_loop_flag[subtask_name]:
            print(f"Provided are reviews from a set of three reviewers: {reviews}")
            input("Would you like to be completed with the project or should the agents go back and improve their experimental results?\n (y) to go back (n) to complete project: ")
        else:
            review_prompt = f"Provided are reviews from a set of three reviewers: {reviews}. Would you like to be completed with the project or do you want to go back to the planning phase and improve your experiments?\n Type y and nothing else to go back, type n and nothing else for complete project."
            self.phd.phases.append(subtask_name)
            if self.review_override:
                if self.review_total_steps == self.review_ovrd_steps:
                    response = "n"
                else:
                    response = "y"
                    self.review_ovrd_steps += 1
            else:
                response = self.phd.inference(
                    research_topic=self.research_topic, phase=subtask_name, feedback=review_prompt, step=0, 
                    temp=self.temps[subtask_name], platform=self.platform, model_or_pipe=self.model_or_pipe, 
                    show_r1_thought=self.show_r1_thought, notes=self.notes)
            if len(response) == 0:
                raise Exception("Model did not respond")
            response = response.lower().strip()[0]
            if response == "n":
                if verbose: print("*"*40, "\n", "REVIEW COMPLETE", "\n", "*"*40)
                return False
            elif response == "y":
                self.set_agent_attr("reviewer_response", f"Provided are reviews from a set of three reviewers: {reviews}.")
                return True
            else: raise Exception("Model did not respond")

    def report_writing(self):
        """
        Perform report writing phase
        @return: (bool) whether to repeat the phase
        """
        subtask_name = "report writing"
        # experiment notes
        report_notes = [_note["note"] for _note in self.notes if subtask_name in _note["phases"]]
        report_notes = f"Notes for the task objective: {report_notes}\n" if len(report_notes) > 0 else ""
        self.reference_papers = []
        # instantiate paper-solver
        solver = PaperSolver(platform=self.platform, model_or_pipe=self.model_or_pipe, temp=self.temps[subtask_name], 
                             show_r1_thought=self.show_r1_thought, notes=report_notes, 
                             max_steps=self.papersolver_max_steps, plan=lab.phd.plan, 
                             exp_code=lab.phd.results_code, exp_results=lab.phd.exp_results, 
                             insights=lab.phd.interpretation, lit_review=lab.phd.lit_review, 
                             ref_papers=self.reference_papers, topic=research_topic, 
                             compile_pdf=compile_pdf)
        # run initialization for solver
        solver.initial_solve(self.out_dirpath)
        # run solver for N mle optimization steps
        for _ in range(self.papersolver_max_steps-1):
            solver.solve()
        # get best report results
        report = "\n".join(solver.best_report[0][0])
        score = solver.best_report[0][1]
        if self.verbose: print(f"Report writing completed, reward function score: {score}")
        if self.human_in_loop_flag[subtask_name]:
            retry = self.human_in_loop(subtask_name, report)
            if retry: return retry
        self.set_agent_attr("report", report)
        readme = self.professor.generate_readme(self.platform, self.model_or_pipe, self.show_r1_thought)
        save_to_file(os.path.join(self.out_dirpath, "research_dir"), "README.md", readme)
        save_to_file(os.path.join(self.out_dirpath, "research_dir"), "report.txt", report)
        self.reset_agents()
        return False

    def results_interpretation(self):
        """
        Perform results interpretation phase
        @return: (bool) whether to repeat the phase
        """
        subtask_name = "results interpretation"
        max_tries = self.max_steps
        dialogue = str()
        # iterate until max num tries to complete task is exhausted
        for _i in range(max_tries):
            resp = self.postdoc.inference(self.research_topic, subtask_name, feedback=dialogue, step=_i, 
                                          temp=self.temps[subtask_name], platform=self.platform, 
                                          model_or_pipe=self.model_or_pipe, show_r1_thought=self.show_r1_thought, 
                                          notes=self.notes)
            if self.verbose: print("Postdoc: ", resp, "\n~~~~~~~~~~~")
            dialogue = str()
            if "```DIALOGUE" in resp:
                dialogue = extract_prompt(resp, "DIALOGUE")
                dialogue = f"The following is dialogue produced by the postdoctoral researcher: {dialogue}"
                if self.verbose: print("#"*40, "\n", "Postdoc Dialogue:", dialogue, "\n", "#"*40)
            if "```INTERPRETATION" in resp:
                interpretation = extract_prompt(resp, "INTERPRETATION")
                if self.human_in_loop_flag[subtask_name]:
                    retry = self.human_in_loop(subtask_name, interpretation)
                    if retry: return retry
                self.set_agent_attr("interpretation", interpretation)
                # reset agent state
                self.reset_agents()
                self.statistics_per_phase[subtask_name]["steps"] = _i
                return False
            resp = self.phd.inference(self.research_topic, subtask_name, feedback=dialogue, step=_i, 
                                      temp=self.temps[subtask_name], platform=self.platform, 
                                      model_or_pipe=self.model_or_pipe, show_r1_thought=self.show_r1_thought, 
                                      notes=self.notes)
            if self.verbose: print("PhD Student: ", resp, "\n~~~~~~~~~~~")
            dialogue = str()
            if "```DIALOGUE" in resp:
                dialogue = extract_prompt(resp, "DIALOGUE")
                dialogue = f"The following is dialogue produced by the PhD student: {dialogue}"
                if self.verbose: print("#"*40, "\n", "PhD Dialogue:", dialogue, "#"*40, "\n")
        raise Exception("Max tries during phase: Results Interpretation")

    def running_experiments(self):
        """
        Perform running experiments phase
        @return: (bool) whether to repeat the phase
        """
        subtask_name = "running experiments"
        # experiment notes
        experiment_notes = [_note["note"] for _note in self.notes if subtask_name in _note["phases"]]
        experiment_notes = f"Notes for the task objective: {experiment_notes}\n" if len(experiment_notes) > 0 else ""
        # instantiate mle-solver
        solver = MLESolver(dataset_code=self.ml_engineer.dataset_code, platform=self.platform, 
                           model_or_pipe=self.model_or_pipe, temp=self.temps[subtask_name], 
                           show_r1_thought=self.show_r1_thought, notes=experiment_notes, 
                           insights=self.ml_engineer.lit_review_sum, max_steps=self.mlesolver_max_steps, 
                           plan=self.ml_engineer.plan)
        # run initialization for solver
        solver.initial_solve()
        # run solver for N mle optimization steps
        for _ in range(self.mlesolver_max_steps-1):
            solver.solve()
        # get best code results
        code = "\n".join(solver.best_codes[0][0])
        # regenerate figures from top code
        execute_code(code)
        score = solver.best_codes[0][1]
        exp_results = solver.best_codes[0][2]
        if self.verbose: print(f"Running experiments completed, reward function score: {score}")
        if self.human_in_loop_flag[subtask_name]:
            retry = self.human_in_loop(subtask_name, code)
            if retry: return retry
        save_to_file(os.path.join(self.out_dirpath, "research_dir/src"), "run_experiments.py", code)
        self.set_agent_attr("results_code", code)
        self.set_agent_attr("exp_results", exp_results)
        # reset agent state
        self.reset_agents()
        return False

    def data_preparation(self):
        """
        Perform data preparation phase
        @return: (bool) whether to repeat the phase
        """

        subtask_name = "data preparation"
        max_tries = self.max_steps
        ml_feedback = str()
        ml_dialogue = str()
        swe_feedback = str()
        ml_command = str()
        hf_engine = HFDataSearch()
        # iterate until max num tries to complete task is exhausted
        for _i in range(max_tries):
            if ml_feedback != "":
                ml_feedback_in = "Feedback provided to the ML agent: " + ml_feedback
            else: ml_feedback_in = ""
            resp = self.sw_engineer.inference(self.research_topic, subtask_name, 
                        feedback=f"{ml_dialogue}\nFeedback from previous command: {swe_feedback}\n{ml_command}{ml_feedback_in}", 
                        step=_i, temp=self.temps[subtask_name], platform=self.platform, model_or_pipe=self.model_or_pipe, 
                        show_r1_thought=self.show_r1_thought, notes=self.notes)
            swe_feedback = str()
            swe_dialogue = str()
            if "```DIALOGUE" in resp:
                dialogue = extract_prompt(resp, "DIALOGUE")
                swe_dialogue = f"\nThe following is dialogue produced by the SW Engineer: {dialogue}\n"
                if self.verbose: print("#"*40, f"\nThe following is dialogue produced by the SW Engineer: {dialogue}", "\n", "#"*40)
            if "```SUBMIT_CODE" in resp:
                final_code = extract_prompt(resp, "SUBMIT_CODE")
                code_resp = execute_code(final_code, timeout=60)
                if self.verbose: print("!"*100, "\n", f"CODE RESPONSE: {code_resp}")
                swe_feedback += f"\nCode Response: {code_resp}\n"
                if "[CODE EXECUTION ERROR]" in code_resp:
                    swe_feedback += "\nERROR: Final code had an error and could not be submitted! You must address and fix this error.\n"
                else:
                    if self.human_in_loop_flag[subtask_name]:
                        retry = self.human_in_loop(subtask_name, final_code)
                        if retry: return retry
                    save_to_file(os.path.join(self.out_dirpath, "research_dir/src"), "load_data.py", final_code)
                    self.set_agent_attr("dataset_code", final_code)
                    # reset agent state
                    self.reset_agents()
                    self.statistics_per_phase[subtask_name]["steps"] = _i
                    return False

            if ml_feedback != "":
                ml_feedback_in = "Feedback from previous command: " + ml_feedback
            else:
                ml_feedback_in = ""
            resp = self.ml_engineer.inference(
                self.research_topic, subtask_name,
                feedback=f"{swe_dialogue}\n{ml_feedback_in}", step=_i, temp=self.temps[subtask_name], 
                platform=self.platform, model_or_pipe=self.model_or_pipe, 
                show_r1_thought=self.show_r1_thought, notes=self.notes)
            #if self.verbose: print("ML Engineer: ", resp, "\n~~~~~~~~~~~")
            ml_feedback = str()
            ml_dialogue = str()
            ml_command = str()
            if "```DIALOGUE" in resp:
                dialogue = extract_prompt(resp, "DIALOGUE")
                ml_dialogue = f"\nThe following is dialogue produced by the ML Engineer: {dialogue}\n"
                if self.verbose: print("#" * 40, f"\nThe following is dialogue produced by the ML Engineer: {dialogue}", "#" * 40, "\n")
            if "```python" in resp:
                code = extract_prompt(resp, "python")
                code = self.ml_engineer.dataset_code + "\n" + code
                code_resp = execute_code(code, timeout=120)
                ml_command = f"Code produced by the ML agent:\n{code}"
                ml_feedback += f"\nCode Response: {code_resp}\n"
                if self.verbose: print("!"*100, "\n", f"CODE RESPONSE: {code_resp}")
            if "```SEARCH_HF" in resp:
                hf_query = extract_prompt(resp, "SEARCH_HF")
                hf_res = "\n".join(hf_engine.results_str(hf_engine.retrieve_ds(hf_query)))
                ml_command = f"HF search command produced by the ML agent:\n{hf_query}"
                ml_feedback += f"Huggingface results: {hf_res}\n"
        raise Exception("Max tries during phase: Data Preparation")

    def plan_formulation(self):
        """
        Perform plan formulation phase
        @return: (bool) whether to repeat the phase
        """
        
        subtask_name = "plan formulation"
        max_tries = self.max_steps
        dialogue = str()
        # iterate until max num tries to complete task is exhausted
        for _i in range(max_tries):
            # inference postdoc to
            resp = self.postdoc.inference(self.research_topic, subtask_name, feedback=dialogue, step=_i,
                                          platform=self.platform, model_or_pipe=self.model_or_pipe, temp=self.temps[subtask_name],
                                          show_r1_thought=self.show_r1_thought, notes=self.notes)
            if self.verbose: print("Postdoc: ", resp, "\n~~~~~~~~~~~")
            dialogue = str()

            if "```DIALOGUE" in resp:
                dialogue = extract_prompt(resp, "DIALOGUE")
                dialogue = f"The following is dialogue produced by the postdoctoral researcher: {dialogue}"
                if self.verbose: print("#"*40, "\n", "Postdoc Dialogue:", dialogue, "\n", "#"*40)

            if "```PLAN" in resp:
                plan = extract_prompt(resp, "PLAN")
                if self.human_in_loop_flag[subtask_name]:
                    retry = self.human_in_loop(subtask_name, plan)
                    if retry: return retry
                self.set_agent_attr("plan", plan)
                # reset agent state
                self.reset_agents()
                self.statistics_per_phase[subtask_name]["steps"] = _i
                return False

            resp = self.phd.inference(self.research_topic, subtask_name, feedback=dialogue, step=_i, 
                                      platform=self.platform, model_or_pipe=self.model_or_pipe,  temp=self.temps[subtask_name],
                                      show_r1_thought=self.show_r1_thought, notes=self.notes)
            if self.verbose: print("PhD Student: ", resp, "\n~~~~~~~~~~~")

            dialogue = str()
            if "```DIALOGUE" in resp:
                dialogue = extract_prompt(resp, "DIALOGUE")
                dialogue = f"The following is dialogue produced by the PhD student: {dialogue}"
                if self.verbose: print("#"*40, "\n", "PhD Dialogue:", dialogue, "#"*40, "\n")
        raise Exception("Max tries during phase: Plan Formulation")

    def literature_review(self):
        """
        Perform literature review phase
        @return: (bool) whether to repeat the phase
        """
        subtask_name = "literature review"
        arx_eng = ArxivSearch()
        max_tries = self.max_steps * 5 # lit review often requires extra steps
        # get initial response from PhD agent
        resp = self.phd.inference(self.research_topic, subtask_name, step=0, temp=self.temps[subtask_name], 
                                  platform=self.platform, model_or_pipe=self.model_or_pipe, 
                                  show_r1_thought=self.show_r1_thought, notes=self.notes)
        if self.verbose: print(resp, "\n~~~~~~~~~~~")
        # iterate until max num tries to complete task is exhausted
        for _i in range(max_tries):
            feedback = str()

            # grab summary of papers from arxiv
            if "```SUMMARY" in resp:
                query = extract_prompt(resp, "SUMMARY")
                papers = arx_eng.find_papers_by_str(query, N=self.arxiv_num_summaries)
                feedback = f"You requested arXiv papers related to the query {query}, here was the response\n{papers}"

            # grab full text from arxiv ID
            elif "```FULL_TEXT" in resp:
                query = extract_prompt(resp, "FULL_TEXT")
                # expiration timer so that paper does not remain in context too long
                arxiv_paper = f"```EXPIRATION {self.arxiv_paper_exp_time}\n" + arx_eng.retrieve_full_paper_text(query) + "```"
                feedback = arxiv_paper

            # if add paper, extract and add to lit review, provide feedback
            elif "```ADD_PAPER" in resp:
                query = extract_prompt(resp, "ADD_PAPER")
                feedback, text = self.phd.add_review(query, arx_eng)
                if len(self.reference_papers) < self.num_ref_papers:
                    self.reference_papers.append(text)

            # completion condition
            if len(self.phd.lit_review) >= self.num_papers_lit_review:
                # generate formal review
                lit_review_sum = self.phd.format_review()
                # if human in loop -> check if human is happy with the produced review
                if self.human_in_loop_flag[subtask_name]:
                    retry = self.human_in_loop(subtask_name, lit_review_sum)
                    # if not happy, repeat the process with human feedback
                    if retry:
                        self.phd.lit_review = []
                        return retry
                # otherwise, return lit review and move on to next stage
                if self.verbose: print(self.phd.lit_review_sum)
                # set agent
                self.set_agent_attr("lit_review_sum", lit_review_sum)
                # reset agent state
                self.reset_agents()
                self.statistics_per_phase[subtask_name]["steps"] = _i
                return False
            resp = self.phd.inference(self.research_topic, subtask_name, feedback=feedback, step=_i + 1, 
                                      temp=self.temps[subtask_name], platform=self.platform, model_or_pipe=self.model_or_pipe, 
                                      show_r1_thought=self.show_r1_thought, notes=self.notes)
            if self.verbose: print(resp, "\n~~~~~~~~~~~")
        raise Exception("Max tries during phase: Literature Review")

    def human_in_loop(self, phase, phase_prod):
        """
        Get human feedback for phase output
        @param phase: (str) current phase
        @param phase_prod: (str) current phase result
        @return: (bool) whether to repeat the loop
        """
        print("\n\n\n\n\n")
        print(f"Presented is the result of the phase [{phase}]: {phase_prod}")
        y_or_no = None
        # repeat until a valid answer is provided
        while y_or_no not in ["y", "n"]:
            y_or_no = input("\n\n\nAre you happy with the presented content? Respond Y or N: ").strip().lower()
            # if person is happy with feedback, move on to next stage
            if y_or_no == "y": pass
            # if not ask for feedback and repeat
            elif y_or_no == "n":
                # ask the human for feedback
                notes_for_agent = input("Please provide notes for the agent so that they can try again and improve performance: ")
                # reset agent state
                self.reset_agents()
                # add suggestions to the notes
                self.notes.append({
                    "phases": [phase],
                    "note": notes_for_agent})
                return True
            else: print("Invalid response, type Y or N")
        return False


def parse_arguments():
    parser = argparse.ArgumentParser(description="AgentLaboratory Research Workflow")
    parser.add_argument(
        '--config_path',
        type=str,
        default="config.json",
        help='Path to load config file used for AgentLaboratory Research Workflow.'
    )
    return parser.parse_args()

def adjust_phase_status(lab, load_path):
    # Update phase status based on the loaded state
    basename = os.path.basename(load_path)
    basename = basename.replace("_", " ")
    done_up_to_this_task = basename.split(".pkl")[0]
    task_exist = False
    for _, subtasks in lab.phases:
        for subtask in subtasks:
            if done_up_to_this_task == subtask: 
                task_exist = True
                break
    assert task_exist, f"task {load_path} does not exist. You need to specify correct pkl file."
    for _, subtasks in lab.phases:
        for subtask in subtasks:
            lab.phase_status[subtask] = True
            if done_up_to_this_task == subtask: 
                break
    return lab


if __name__ == "__main__":
    args = parse_arguments()
    cfg = read_jsonc(args.config_path)

    out_dirpath = cfg["out_dirpath"]
    os.makedirs(out_dirpath, exist_ok=True)
    platform = cfg["platform"]
    compile_pdf = cfg["compile_latex"]
    load_existing = cfg["load_existing"]
    show_r1_thought = cfg["show_r1_thought"]
    num_papers_lit_review = cfg["num_papers_lit_review"]
    assert isinstance(num_papers_lit_review, int), "'num_papers_lit_review' must be a valid integer!"
    papersolver_max_steps = cfg["papersolver_max_steps"]
    assert isinstance(papersolver_max_steps, int), "'papersolver_max_steps' must be a valid integer!"
    mlesolver_max_steps = cfg["mlesolver_max_steps"]
    assert isinstance(mlesolver_max_steps, int), "'mlesolver_max_steps' must be a valid integer!"

    ##########################################################
    # Research question that the agents are going to explore #
    ##########################################################
    research_topic = cfg["research_topic"]

    ####################################################
    ###  Stages where human input will be requested  ###
    ####################################################
    human_in_loop = cfg["human_in_loop"]

    ###################################################
    ###  LLM Backend used for the different phases  ###
    ###################################################
    agent_models = cfg["agent_models"]

    ############################################################
    ### INference temperatures used for the different phases ###
    ############################################################
    temps = cfg["temps"]

    if platform == "ollama":
        if "deepseek-r1" in agent_models["running experiments"]:
            call_model_prompt = f'from openai import OpenAI\nclient = OpenAI(base_url="http://localhost:11434/v1/", api_key="ollama")\ncompletion = client.chat.completions.create(model="{agent_models["running experiments"]}", messages=messages)\nanswer = completion.choices[0].message.content\n_, answer = answer.split("</think>")\nanswer = answer[2:]'
        else:
            call_model_prompt = f'from openai import OpenAI\nclient = OpenAI(base_url="http://localhost:11434/v1/", api_key="ollama")\ncompletion = client.chat.completions.create(model="{agent_models["running experiments"]}", messages=messages)\nanswer = completion.choices[0].message.content'
    elif platform == "huggingface":
        max_length = 131072
        if "deepseek-r1" in agent_models["running experiments"]:
            call_model_prompt = f'from transformers import pipeline\nimport torch\npipe = pipeline("text-generation", model="{agent_models["running experiments"]}", device_map="auto")\nprompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)\nresponse = pipe(prompt, do_sample=True, max_length={max_length}, truncation=True)\nanswer = response[0]["generated_text"][len(prompt):]\n_, answer = answer.split("</think>")\nanswer = answer[2:]'
        else:
            call_model_prompt = f'from transformers import pipeline\nimport torch\npipe = pipeline("text-generation", model="{agent_models["running experiments"]}", device_map="auto")\nprompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)\nresponse = pipe(prompt, do_sample=True, max_length={max_length}, truncation=True)\nanswer = response[0]["generated_text"][len(prompt):]'

    task_notes_LLM = [
        {"phases": ["plan formulation"],
         "note": f"You should come up with a plan for TWO experiments."},

        {"phases": ["plan formulation"],
         "note": f'Please plan to use {agent_models["data preparation"]} for your data preparation and {agent_models["running experiments"]} for your experiments if you decide to use a LLM on those phases.'},

        {"phases": ["data preparation"],
         "note": f'Please use {agent_models["data preparation"]} if you decide to use a LLM on this phase.'},

        {"phases": ["running experiments"],
         "note": f'Please use {agent_models["running experiments"]} if you decide to use a LLM on this phase. If you do so, Use the following code to inference {agent_models["running experiments"]}: \n{call_model_prompt}\nHowever, do not use too many inferences.'},

        {"phases": ["running experiments"],
         "note": "I would recommend using a small dataset (approximately only 100 data points) to run experiments in order to save time. Do not use much more than this unless you have to or are running the final tests."},

        {"phases": ["data preparation", "running experiments"],
         "note": "You are running programs on Ubuntu."},

        {"phases": ["data preparation", "running experiments"],
         "note": "Generate figures with very colorful and artistic design."},
    ]

    task_notes_LLM.append(
        {"phases": ["literature review", "plan formulation", "data preparation", "running experiments", "results interpretation", "report writing", "report refinement"],
        "note": f'You should always write in the following language to converse and to write the report {cfg["language"]}'},
    )

    if load_existing:
        load_path = cfg["load_existing_path"]
        if load_path is None:
            raise ValueError("Please provide path to load existing state.")
        with open(load_path, "rb") as f:
            lab = pickle.load(f)
            # Override instance variables of lab
            lab.notes = task_notes_LLM
            lab.platform = platform
            lab.set_phase_models(agent_models)
            lab.human_in_loop_flag = human_in_loop
            lab.compile_pdf = compile_pdf
            lab.num_papers_lit_review = num_papers_lit_review
            lab.papersolver_max_steps = papersolver_max_steps
            lab.mlesolver_max_steps = mlesolver_max_steps
            lab.show_r1_thought = show_r1_thought
            lab.out_dirpath = out_dirpath
            lab = adjust_phase_status(lab, load_path)
    else:
        lab = LaboratoryWorkflow(
            research_topic=research_topic,
            notes=task_notes_LLM,
            platform=platform,
            agent_model_backbone=agent_models,
            temps=temps,
            show_r1_thought=show_r1_thought,
            human_in_loop_flag=human_in_loop,
            compile_pdf=compile_pdf,
            num_papers_lit_review=num_papers_lit_review,
            papersolver_max_steps=papersolver_max_steps,
            mlesolver_max_steps=mlesolver_max_steps,
            out_dirpath=out_dirpath,
        )

    lab.perform_research()







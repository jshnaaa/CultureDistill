import os
import re
import yaml
from collections import Counter
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class ReconcileMAS:
    """
    RECONCILE-style multi-agent debate for cultural alignment tasks.

    Direction B: agents provide cultural reasoning paths only (no answer decision).
    The Judge reads all reasoning and answers the question independently.

    5 heterogeneous agents (Asian / European / North American / Latin American / African).
    Output format mirrors AgentArk LLM Debate:
      ===== Solution 1 ===== ... ===== Solution 5 ===== (agents)
      ===== Solution 6 ===== (Judge final answer)
    """

    def __init__(self, model_name, tensor_parallel_size=1, config_path=None,
                 temperature=0.7, max_tokens=1024):
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "configs", "reconcile_config.yaml")
        cfg = load_config(config_path)

        self.culture_roles = cfg["culture_roles"]
        self.num_agents = len(self.culture_roles)
        self.num_debate_rounds = cfg["num_debate_rounds"]
        self.judge_system_prompt = cfg["judge"]["system_prompt"].strip()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model_name = model_name

        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            gpu_memory_utilization=0.85,
            dtype="bfloat16",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stop=["<|eot_id|>", "<|end_of_text|>", "</s>"],
        )

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def _apply_chat(self, system, user):
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _build_initial_prompt(self, agent_idx, question):
        """
        Agent only provides cultural reasoning — no final answer required.
        This preserves the model's own knowledge for the Judge stage.
        """
        system = self.culture_roles[agent_idx]["system_prompt"].strip()
        user = (
            f"{question}\n\n"
            "From your cultural perspective, analyze how the target culture specified "
            "in the question would approach this topic. Focus on specific cultural norms, "
            "habits, and values relevant to each option. "
            "Provide your cultural analysis only — do NOT give a final answer number.\n\n"
            "Format your response as:\n"
            "Reasoning: <your cultural analysis of the options>"
        )
        return self._apply_chat(system, user)

    def _build_debate_prompt(self, agent_idx, question, other_responses):
        """
        Agent reviews other perspectives and refines its cultural analysis.
        Still no final answer — reasoning paths only.
        """
        system = self.culture_roles[agent_idx]["system_prompt"].strip()

        others_text = ""
        for name, resp in other_responses:
            others_text += f"\n[{name}]:\n{resp}\n"

        user = (
            f"{question}\n\n"
            "Other cultural experts have provided these analyses:\n"
            f"{others_text}\n"
            "Review these perspectives and refine your own cultural analysis. "
            "You may agree, disagree, or add new insights. "
            "Do NOT give a final answer number — provide cultural reasoning only.\n\n"
            "Format your response as:\n"
            "Reasoning: <your refined cultural analysis>"
        )
        return self._apply_chat(system, user)

    def _build_judge_prompt(self, question, agent_responses):
        """
        Judge reads all cultural reasoning paths and answers the question independently.
        The Judge is not influenced by any agent's answer — only by their reasoning.
        """
        responses_text = ""
        for name, resp in agent_responses:
            responses_text += f"\n[{name}]:\n{resp}\n"

        user = (
            f"{question}\n\n"
            "Five cultural experts have analyzed this question from different perspectives:\n"
            f"{responses_text}\n"
            "Using the cultural insights above as reference, answer the question yourself. "
            "Read the question precisely and give the single best answer.\n\n"
            "Reasoning: <your reasoning>\n"
            "Answer: <number>"
        )
        return self._apply_chat(self.judge_system_prompt, user)

    # ------------------------------------------------------------------
    # Answer extraction (Judge only)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_answer(text):
        m = re.search(r"Answer\s*:\s*([1-4])", text, re.IGNORECASE)
        if m:
            return m.group(1)
        digits = re.findall(r"\b([1-4])\b", text)
        return digits[-1] if digits else None

    # ------------------------------------------------------------------
    # Single-sample inference
    # ------------------------------------------------------------------

    def inference(self, sample):
        question = sample["query"]

        agent_responses = [""] * self.num_agents

        # Round 0: independent cultural analysis (no answers)
        prompts = [self._build_initial_prompt(i, question) for i in range(self.num_agents)]
        outputs = self.llm.generate(prompts, self.sampling_params)
        for i, out in enumerate(outputs):
            agent_responses[i] = out.outputs[0].text.strip()

        # Debate rounds: refine cultural analysis
        for _ in range(self.num_debate_rounds):
            prompts = []
            for i in range(self.num_agents):
                others = [
                    (self.culture_roles[j]["name"], agent_responses[j])
                    for j in range(self.num_agents) if j != i
                ]
                prompts.append(self._build_debate_prompt(i, question, others))
            outputs = self.llm.generate(prompts, self.sampling_params)
            for i, out in enumerate(outputs):
                agent_responses[i] = out.outputs[0].text.strip()

        # Judge: answers the question based on all cultural reasoning
        all_responses = [
            (self.culture_roles[i]["name"], agent_responses[i])
            for i in range(self.num_agents)
        ]
        judge_prompt = self._build_judge_prompt(question, all_responses)
        judge_output = self.llm.generate([judge_prompt], self.sampling_params)
        judge_response = judge_output[0].outputs[0].text.strip()

        formatted = ""
        for i, resp in enumerate(agent_responses):
            formatted += f"===== Solution {i + 1} =====\n{resp}\n"
        formatted += f"===== Solution {self.num_agents + 1} =====\n{judge_response}\n"

        return {"response": formatted}

    # ------------------------------------------------------------------
    # Batch inference
    # ------------------------------------------------------------------

    def inference_batch(self, samples):
        """
        Batch all agent×sample prompts per round to maximise GPU utilisation.
        Rounds are sequential; within each round all prompts run in parallel.
        """
        n = len(samples)
        questions = [s["query"] for s in samples]

        agent_responses = [[""] * self.num_agents for _ in range(n)]

        # Round 0
        prompts, meta = [], []
        for si in range(n):
            for ai in range(self.num_agents):
                prompts.append(self._build_initial_prompt(ai, questions[si]))
                meta.append((si, ai))
        outputs = self.llm.generate(prompts, self.sampling_params)
        for out, (si, ai) in zip(outputs, meta):
            agent_responses[si][ai] = out.outputs[0].text.strip()

        # Debate rounds
        for _ in range(self.num_debate_rounds):
            prompts, meta = [], []
            for si in range(n):
                for ai in range(self.num_agents):
                    others = [
                        (self.culture_roles[j]["name"], agent_responses[si][j])
                        for j in range(self.num_agents) if j != ai
                    ]
                    prompts.append(self._build_debate_prompt(ai, questions[si], others))
                    meta.append((si, ai))
            outputs = self.llm.generate(prompts, self.sampling_params)
            new_responses = [[""] * self.num_agents for _ in range(n)]
            for out, (si, ai) in zip(outputs, meta):
                new_responses[si][ai] = out.outputs[0].text.strip()
            agent_responses = new_responses

        # Judge: batch all judge prompts
        judge_prompts = []
        for si in range(n):
            all_responses = [
                (self.culture_roles[ai]["name"], agent_responses[si][ai])
                for ai in range(self.num_agents)
            ]
            judge_prompts.append(self._build_judge_prompt(questions[si], all_responses))

        judge_outputs = self.llm.generate(judge_prompts, self.sampling_params)

        results = []
        for si in range(n):
            judge_response = judge_outputs[si].outputs[0].text.strip()

            formatted = ""
            for ai in range(self.num_agents):
                formatted += f"===== Solution {ai + 1} =====\n{agent_responses[si][ai]}\n"
            formatted += f"===== Solution {self.num_agents + 1} =====\n{judge_response}\n"

            results.append({"response": formatted})

        return results

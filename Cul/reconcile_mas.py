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

    4 heterogeneous agents (Asian / Western / Latin American / African),
    each injected with a cultural system prompt. Runs num_debate_rounds
    rounds of debate, then a Judge agent selects the final answer.

    Output format mirrors AgentArk LLM Debate:
      ===== Solution 1 ===== (Agent 0)
      ...
      ===== Solution 4 ===== (Agent 3)
      ===== Solution 5 ===== (Judge consensus)
    """

    def __init__(self, model_name, tensor_parallel_size=1, config_path=None,
                 temperature=0.7, max_tokens=1024, num_debate_rounds=None,
                 include_judge=True):
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "configs", "reconcile_config.yaml")
        cfg = load_config(config_path)

        self.culture_roles = cfg["culture_roles"]
        self.num_agents = len(self.culture_roles)
        # Command-line argument overrides config value
        self.num_debate_rounds = num_debate_rounds if num_debate_rounds is not None else cfg["num_debate_rounds"]
        self.judge_system_prompt = cfg["judge"]["system_prompt"].strip()
        self.include_judge = include_judge
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
        stop_tokens = ["<|eot_id|>", "<|end_of_text|>", "</s>"]
        # Agents use high temperature for diverse reasoning paths
        self.sampling_params = SamplingParams(
            temperature=0.9,
            max_tokens=self.max_tokens,
            stop=stop_tokens,
        )
        # Judge uses low temperature for stable, accurate answer selection
        self.judge_sampling_params = SamplingParams(
            temperature=0.3,
            max_tokens=self.max_tokens,
            stop=stop_tokens,
        )

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def _apply_chat(self, system, user):
        """Apply Llama-3 chat template."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _build_initial_prompt(self, agent_idx, question):
        system = self.culture_roles[agent_idx]["system_prompt"].strip()
        user = (
            f"{question}\n\n"
            "Identify the TARGET CULTURE in the question. "
            "Reason about its specific customs and values first, "
            "then use your own cultural perspective as a reference if helpful.\n\n"
            "Reasoning: <your reasoning>\n"
            "Answer: <number>"
        )
        return self._apply_chat(system, user)

    def _build_debate_prompt(self, agent_idx, question, other_responses):
        """other_responses: list of (agent_name, response_text)"""
        system = self.culture_roles[agent_idx]["system_prompt"].strip()

        others_text = ""
        for name, resp in other_responses:
            others_text += f"\n[{name}]:\n{resp}\n"

        user = (
            f"{question}\n\n"
            "Other cultural experts have responded:\n"
            f"{others_text}\n"
            "Review their reasoning. Update your answer if you find stronger evidence, "
            "otherwise defend your original. Stay focused on the TARGET CULTURE.\n\n"
            "Reasoning: <your updated reasoning>\n"
            "Answer: <number>"
        )
        return self._apply_chat(system, user)

    def _build_judge_prompt(self, question, agent_responses):
        """Judge weighs agent perspectives against verifiable cultural facts."""
        responses_text = ""
        for name, resp in agent_responses:
            responses_text += f"\n[{name}]:\n{resp}\n"

        user = (
            f"{question}\n\n"
            "Five cultural expert agents have responded:\n"
            f"{responses_text}\n"
            "Determine the correct answer using verifiable facts about the TARGET CULTURE. "
            "Reference the agents' reasoning as supporting evidence, "
            "but do not simply follow the majority.\n\n"
            "Reasoning: <your reasoning, referencing agents as needed>\n"
            "Answer: <number>"
        )
        return self._apply_chat(self.judge_system_prompt, user)

    # ------------------------------------------------------------------
    # Answer extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_answer(text):
        """Extract numeric answer 1-4 from agent/judge response."""
        # Pattern 1: "Answer: 3" or "Answer: 3."
        m = re.search(r"Answer\s*:\s*([1-4])", text, re.IGNORECASE)
        if m:
            return m.group(1)
        # Pattern 2: "correct answer is: 4" / "answer is 4" / "The answer is: 4."
        m = re.search(r"answer\s+is\s*:?\s*([1-4])\b", text, re.IGNORECASE)
        if m:
            return m.group(1)
        # Pattern 3: "option 3" / "option: 3"
        m = re.search(r"option\s*:?\s*([1-4])\b", text, re.IGNORECASE)
        if m:
            return m.group(1)
        # Fallback: last standalone digit 1-4 in text
        digits = re.findall(r"\b([1-4])\b", text)
        return digits[-1] if digits else None

    # ------------------------------------------------------------------
    # Fallback consensus (majority vote, used only if judge fails)
    # ------------------------------------------------------------------

    def _majority_vote(self, answers, target_culture):
        valid = [a for a in answers if a is not None]
        if not valid:
            return None
        counts = Counter(valid)
        top_count = counts.most_common(1)[0][1]
        top_answers = [a for a, c in counts.items() if c == top_count]
        if len(top_answers) == 1:
            return top_answers[0]
        # Tie-break: prefer agent whose region matches target culture
        region_keywords = {
            0: ["asia", "china", "japan", "korea", "india", "vietnam", "arab", "iran",
                "pakistan", "indonesia", "malaysia", "philippines", "thailand", "arabic"],
            1: ["europe", "uk", "france", "germany", "netherlands", "spain", "italy",
                "sweden", "norway", "denmark", "finland", "poland", "portugal"],
            2: ["america", "united states", "canada", "usa", "north america"],
            3: ["latin", "south america", "brazil", "mexico", "argentina", "chile",
                "colombia", "peru", "venezuela", "central america"],
            4: ["africa", "nigeria", "kenya", "ethiopia", "ghana", "south africa",
                "egypt", "morocco", "tanzania", "senegal", "uganda"],
        }
        target_lower = target_culture.lower()
        for agent_idx, keywords in region_keywords.items():
            if any(kw in target_lower for kw in keywords):
                candidate = answers[agent_idx]
                if candidate in top_answers:
                    return candidate
        return top_answers[0]

    # ------------------------------------------------------------------
    # Single-sample inference
    # ------------------------------------------------------------------

    def inference(self, sample):
        question = sample["query"]
        target_culture = sample.get("country", "")

        agent_responses = [""] * self.num_agents

        # Round 0: independent initial responses
        prompts = [self._build_initial_prompt(i, question) for i in range(self.num_agents)]
        outputs = self.llm.generate(prompts, self.sampling_params)
        for i, out in enumerate(outputs):
            agent_responses[i] = out.outputs[0].text.strip()

        # Debate rounds
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

        # Judge: read all final responses and select best answer
        judge_response = ""
        if self.include_judge:
            all_responses = [
                (self.culture_roles[i]["name"], agent_responses[i])
                for i in range(self.num_agents)
            ]
            judge_prompt = self._build_judge_prompt(question, all_responses)
            judge_output = self.llm.generate([judge_prompt], self.judge_sampling_params)
            judge_response = judge_output[0].outputs[0].text.strip()
            judge_answer = self._extract_answer(judge_response)

            # Fallback to majority vote if judge fails to produce valid answer
            if judge_answer is None:
                final_answers = [self._extract_answer(r) for r in agent_responses]
                judge_answer = self._majority_vote(final_answers, target_culture)

        # Format: Solution 1-N = agent final responses, Solution N+1 = judge (if included)
        formatted = ""
        for i, resp in enumerate(agent_responses):
            formatted += f"===== Solution {i + 1} =====\n{resp}\n"
        if self.include_judge:
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
        countries = [s.get("country", "") for s in samples]

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

        # Judge: batch all judge prompts (if include_judge)
        judge_responses = [""] * n
        if self.include_judge:
            judge_prompts = []
            for si in range(n):
                all_responses = [
                    (self.culture_roles[ai]["name"], agent_responses[si][ai])
                    for ai in range(self.num_agents)
                ]
                judge_prompts.append(self._build_judge_prompt(questions[si], all_responses))

            judge_outputs = self.llm.generate(judge_prompts, self.judge_sampling_params)

            for si in range(n):
                judge_response = judge_outputs[si].outputs[0].text.strip()
                judge_answer = self._extract_answer(judge_response)

                if judge_answer is None:
                    final_answers = [
                        self._extract_answer(agent_responses[si][ai])
                        for ai in range(self.num_agents)
                    ]
                    fallback = self._majority_vote(final_answers, countries[si])
                    judge_response += f"\n[Fallback majority vote]: {fallback}"

                judge_responses[si] = judge_response

        # Build results
        results = []
        for si in range(n):
            formatted = ""
            for ai in range(self.num_agents):
                formatted += f"===== Solution {ai + 1} =====\n{agent_responses[si][ai]}\n"
            if self.include_judge:
                formatted += f"===== Solution {self.num_agents + 1} =====\n{judge_responses[si]}\n"

            results.append({"response": formatted})

        return results

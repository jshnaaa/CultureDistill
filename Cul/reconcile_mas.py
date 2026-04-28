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

    4 heterogeneous agents (Asian / European / American / Oceanian),
    each injected with a cultural system prompt. Runs num_debate_rounds
    rounds of debate, then aggregates via majority vote.

    Output format mirrors AgentArk LLM Debate:
      ===== Solution 1 ===== ... ===== Solution N+1 ===== (consensus)
    """

    def __init__(self, model_name, tensor_parallel_size=4, config_path=None, temperature=0.7, max_tokens=1024):
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "configs", "reconcile_config.yaml")
        cfg = load_config(config_path)

        self.culture_roles = cfg["culture_roles"]
        self.num_agents = len(self.culture_roles)
        self.num_debate_rounds = cfg["num_debate_rounds"]
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model_name = model_name

        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            gpu_memory_utilization=0.9,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stop=["</s>", "<|im_end|>", "<|endoftext|>"],
        )

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def _build_initial_prompt(self, agent_idx, question):
        system = self.culture_roles[agent_idx]["system_prompt"].strip()
        user = (
            f"{question}\n\n"
            "Please reason step by step from your cultural perspective, "
            "then state your answer in the format:\n"
            "Reasoning: <your reasoning>\n"
            "Answer: <number>"
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _build_debate_prompt(self, agent_idx, question, other_responses):
        system = self.culture_roles[agent_idx]["system_prompt"].strip()

        others_text = ""
        for name, resp in other_responses:
            others_text += f"\n[{name}]:\n{resp}\n"

        user = (
            f"{question}\n\n"
            "These are the responses from agents with different cultural perspectives:\n"
            f"{others_text}\n"
            "Consider these perspectives carefully. You may update your answer if "
            "persuaded, but maintain your own cultural reasoning.\n"
            "Please provide your updated response in the format:\n"
            "Reasoning: <your reasoning>\n"
            "Answer: <number>"
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    # ------------------------------------------------------------------
    # Answer extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_answer(text):
        """Extract the numeric answer from agent response."""
        # Match 'Answer: N' pattern first
        m = re.search(r"Answer\s*:\s*([1-4])", text, re.IGNORECASE)
        if m:
            return m.group(1)
        # Fallback: last standalone digit 1-4
        digits = re.findall(r"\b([1-4])\b", text)
        return digits[-1] if digits else None

    # ------------------------------------------------------------------
    # Majority vote consensus
    # ------------------------------------------------------------------

    def _consensus(self, answers, target_culture):
        valid = [a for a in answers if a is not None]
        if not valid:
            return None

        counts = Counter(valid)
        top_count = counts.most_common(1)[0][1]
        top_answers = [a for a, c in counts.items() if c == top_count]

        if len(top_answers) == 1:
            return top_answers[0]

        # Tie-breaking: prefer the agent whose culture is closest to target_culture
        target_lower = target_culture.lower()
        priority_map = {
            "asian": 0,
            "european": 1,
            "american": 2,
            "oceanian": 3,
        }
        # Find which agent index maps to target culture region
        for idx, role in enumerate(self.culture_roles):
            role_key = role["name"].split()[0].lower()
            if role_key in target_lower or target_lower in role_key:
                candidate = answers[idx]
                if candidate in top_answers:
                    return candidate

        # Last resort: return first top answer
        return top_answers[0]

    # ------------------------------------------------------------------
    # Single-sample inference
    # ------------------------------------------------------------------

    def inference(self, sample):
        question = sample["query"]
        target_culture = sample.get("country", "")

        # agent_responses[i] = latest response text for agent i
        agent_responses = [""] * self.num_agents

        # Round 0: initial independent responses
        prompts = [self._build_initial_prompt(i, question) for i in range(self.num_agents)]
        outputs = self.llm.generate(prompts, self.sampling_params)
        for i, out in enumerate(outputs):
            agent_responses[i] = out.outputs[0].text.strip()

        # Debate rounds
        for _ in range(self.num_debate_rounds):
            new_responses = [""] * self.num_agents
            prompts = []
            for i in range(self.num_agents):
                others = [
                    (self.culture_roles[j]["name"], agent_responses[j])
                    for j in range(self.num_agents) if j != i
                ]
                prompts.append(self._build_debate_prompt(i, question, others))

            outputs = self.llm.generate(prompts, self.sampling_params)
            for i, out in enumerate(outputs):
                new_responses[i] = out.outputs[0].text.strip()
            agent_responses = new_responses

        # Extract final answers
        final_answers = [self._extract_answer(r) for r in agent_responses]
        consensus = self._consensus(final_answers, target_culture)

        # Format output: Solution 1..N = agent responses, Solution N+1 = consensus
        formatted = ""
        for i, resp in enumerate(agent_responses):
            formatted += f"===== Solution {i + 1} =====\n{resp}\n"
        formatted += f"===== Solution {self.num_agents + 1} =====\nConsensus Answer: {consensus}\n"

        return {"response": formatted}

    # ------------------------------------------------------------------
    # Batch inference
    # ------------------------------------------------------------------

    def inference_batch(self, samples):
        """
        Batch all agent×sample prompts per round to maximise GPU utilisation.
        Round structure is kept sequential (each round depends on previous).
        """
        n = len(samples)
        questions = [s["query"] for s in samples]
        countries = [s.get("country", "") for s in samples]

        # agent_responses[sample_idx][agent_idx]
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

        # Build results
        results = []
        for si in range(n):
            final_answers = [self._extract_answer(agent_responses[si][ai]) for ai in range(self.num_agents)]
            consensus = self._consensus(final_answers, countries[si])

            formatted = ""
            for ai, resp in enumerate(agent_responses[si]):
                formatted += f"===== Solution {ai + 1} =====\n{resp}\n"
            formatted += f"===== Solution {self.num_agents + 1} =====\nConsensus Answer: {consensus}\n"

            results.append({"response": formatted})

        return results

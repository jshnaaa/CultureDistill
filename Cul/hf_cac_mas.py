"""
HF-CAC: Home-Field Culture-Activated Collaboration

Extension of RECONCILE framework with dynamic authority activation:
  1. Home-Field Detection: automatically identify which agent is the
     "Host-Culture Guardian" based on target country in the question.
  2. Asymmetric Prompting: Guardian uses authoritative confirmation/correction prompt;
     other agents use cross-cultural auditor prompt (contrastive, deferential).
  3. Guardian-First Flow: Guardian generates first (priority), then Auditors
     respond with awareness of Guardian's position (one-directional influence).
  4. Conditional Judge: Judge is ONLY invoked when agents disagree AND Guardian
     lacks majority support. When Guardian + at least one Auditor agree,
     their consensus is accepted directly, avoiding unnecessary Judge overrides.
  5. Temperature Diversity: Guardian uses low temp (0.3) for precision,
     Auditors use higher temp (0.7) for diverse perspectives (MAD-inspired).

Output format mirrors AgentArk LLM Debate (===== Solution N =====) for pipeline compatibility.
"""

import os
import re
import yaml
from collections import Counter
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class HF_CAC_MAS:
    """
    Home-Field Culture-Activated Collaboration MAS.

    Key differences from vanilla RECONCILE:
      - Dynamic role assignment per sample (Guardian vs Auditor) based on target country
      - Asymmetric system prompts inject authority gradient
      - Structured two-phase generation: Guardian first → Auditors with Guardian context
      - Judge receives explicit Guardian designation for weighted deliberation
    """

    def __init__(self, model_name, tensor_parallel_size=1, config_path=None,
                 temperature=0.7, max_tokens=1024, include_judge=True,
                 negotiation_rounds=1, num_agents=None):
        """
        Args:
            model_name: HuggingFace model path or alias
            tensor_parallel_size: vLLM tensor parallelism
            config_path: path to hf_cac_config.yaml
            temperature: base temperature (overridden per role)
            max_tokens: max generation tokens
            include_judge: whether to include Judge reasoning in output
            negotiation_rounds: rounds of structured negotiation (0=independent, 1=standard)
            num_agents: number of agents to use (None=all from config).
                        For culturalbench, fewer agents (2-3) with structured debate
                        can outperform many agents.
        """
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__), "configs", "hf_cac_config.yaml"
            )
        cfg = load_config(config_path)

        self.culture_roles = cfg["culture_roles"]
        # Support dynamic agent count: if num_agents specified, use subset
        if num_agents is not None and num_agents < len(self.culture_roles):
            self.num_agents = num_agents
        else:
            self.num_agents = len(self.culture_roles)
        self.judge_system_prompt = cfg["judge"]["system_prompt"].strip()
        self.include_judge = include_judge
        self.negotiation_rounds = negotiation_rounds
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model_name = model_name

        # Task type: "normad" (3-way acceptability), "cultureatlas" (2-way comparison),
        # "culturalbench" (4-way MCQ), or "culturellm" (variable-scale WVS survey)
        self.task_type = cfg.get("task_type", "normad")
        self.answer_choices = cfg.get("answer_choices", [1, 2, 3])

        # Cultural Affinity Matrix for Judge fallback arbitration
        self.affinity_matrix = cfg.get("cultural_affinity_matrix", None)
        self.guardian_failure_indicators = cfg.get(
            "guardian_failure_indicators", []
        )

        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            gpu_memory_utilization=0.85,
            dtype="bfloat16",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        stop_tokens = ["<|eot_id|>", "<|end_of_text|>", "</s>"]

        # Temperature strategy:
        # - CulturalBench / CultureLLM (factual/survey QA): MAD-style asymmetric temperatures.
        #   Guardian=0.3 (precise), Auditors=0.6 (diverse perspectives).
        #   Diversity + Debate is the key — NOT self-consistency.
        #   top_p=0.95 matches MAD's settings.
        # - Other tasks: asymmetric temperature for role-based diversity.
        if self.task_type in ("culturalbench", "culturellm"):
            guardian_temp = 0.3
            auditor_temp = 0.6   # MAD's Agent2 temp — encourage diverse perspectives
            cb_max_tokens = 512
            self.guardian_sampling = SamplingParams(
                temperature=guardian_temp,
                max_tokens=cb_max_tokens,
                stop=stop_tokens,
                top_p=0.95,
            )
            self.auditor_sampling = SamplingParams(
                temperature=auditor_temp,
                max_tokens=cb_max_tokens,
                stop=stop_tokens,
                top_p=0.95,
            )
            self.judge_sampling = SamplingParams(
                temperature=0.3,  # MAD uses same as Agent1 for Judge
                max_tokens=cb_max_tokens,
                stop=stop_tokens,
                top_p=0.95,
            )
        else:
            guardian_temp = 0.5
            auditor_temp = 0.9
            self.guardian_sampling = SamplingParams(
                temperature=guardian_temp,
                max_tokens=self.max_tokens,
                stop=stop_tokens,
            )
            self.auditor_sampling = SamplingParams(
                temperature=auditor_temp,
                max_tokens=self.max_tokens,
                stop=stop_tokens,
            )
            self.judge_sampling = SamplingParams(
                temperature=0.1,
                max_tokens=self.max_tokens,
                stop=stop_tokens,
            )

    # ------------------------------------------------------------------
    # Home-Field Detection
    # ------------------------------------------------------------------

    def detect_guardian(self, target_country: str) -> int:
        """
        Determine which agent is the Host-Culture Guardian based on target country.

        Returns:
            Agent index (0-5) of the Guardian. Returns -1 if no match found
            (falls back to treating all agents equally).
        """
        country_lower = target_country.lower().strip()
        if not country_lower:
            return -1

        for idx, role in enumerate(self.culture_roles):
            keywords = role.get("region_keywords", [])
            for kw in keywords:
                if kw in country_lower or country_lower in kw:
                    return idx
        return -1

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def _apply_chat(self, system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _build_guardian_prompt(self, agent_idx: int, question: str,
                               target_country: str) -> str:
        """Build prompt for the Host-Culture Guardian (Phase 1: independent)."""
        system = self.culture_roles[agent_idx]["guardian_prompt"].strip()

        if self.task_type == "cultureatlas":
            # CultureAtlas: comparative cultural depth (binary 1/2)
            user = (
                f"TARGET CULTURE: {target_country}\n\n"
                f"{question}\n\n"
                f"As the Host-Culture Guardian for {target_country}, determine which "
                f"response demonstrates MORE culturally specific and insightful knowledge "
                f"about {target_country}. Use your deep expertise to identify genuine "
                f"cultural depth vs. surface-level generalizations.\n\n"
                f"Answer choices:\n"
                f"1 = Response 1 is more culturally specific\n"
                f"2 = Response 2 is more culturally specific\n\n"
                f"Reasoning: <your authoritative analysis of cultural depth>\n"
                f"Answer: <1 or 2>"
            )
        elif self.task_type == "culturellm":
            # CultureLLM: World Values Survey attitude prediction (variable scale)
            # The question already contains the full survey item with numbered options.
            # Agent must predict the most representative answer for the target culture.
            user = (
                f"Task: You will be given a World Values Survey question about the "
                f"cultural values and attitudes of {target_country}. "
                f"Select the option number that BEST represents the typical attitude "
                f"or value orientation of people in {target_country}. "
                f"Think about the dominant cultural values, social norms, and common "
                f"worldviews in {target_country}, then respond with the most representative "
                f"option number. Explain your reasoning in less than three sentences.\n\n"
                f"Survey Question:\n{question}\n"
                f"Answer:"
            )
        elif self.task_type == "culturalbench":
            # CulturalBench: factual cultural knowledge MCQ (4-way)
            # MAD-inspired: simple, concise prompt — "explain in less than three sentences"
            user = (
                f"Task: You will be given a cultural knowledge question about "
                f"{target_country}. Select the correct option number. Do not make any "
                f"extra inferences outside of the given context and country. "
                f"Only align to the country given. "
                f"Think step by step about the cultural practices of {target_country}, then respond "
                f"with the correct option number (1, 2, 3, or 4). Explain your answer in less "
                f"than three sentences.\n\n"
                f"Question:\n{question}\n"
                f"Answer (1, 2, 3, or 4):"
            )
        else:
            # NormAD: behavior acceptability (3-way 1/2/3)
            user = (
                f"TARGET CULTURE: {target_country}\n\n"
                f"{question}\n\n"
                f"As the Host-Culture Guardian for {target_country}, provide your "
                f"AUTHORITATIVE analysis. Cite specific cultural practices, traditions, "
                f"or norms by name. Explain why certain options align or conflict with "
                f"the target culture's values.\n\n"
                f"Answer choices:\n"
                f"1 = The behavior is clearly ACCEPTABLE in {target_country}'s culture\n"
                f"2 = The behavior is clearly UNACCEPTABLE in {target_country}'s culture\n"
                f"3 = The behavior is NEUTRAL/INDETERMINATE — it has no specific cultural valence\n"
                f"    in {target_country}, or cultural norms on this topic vary significantly\n"
                f"    within the country, or the behavior is universally human rather than\n"
                f"    culturally specific\n\n"
                f"Reasoning: <your authoritative cultural analysis>\n"
                f"Answer: <number>"
            )
        return self._apply_chat(system, user)

    def _build_auditor_prompt(self, agent_idx: int, question: str,
                              target_country: str,
                              guardian_name: str,
                              guardian_response: str | None = None) -> str:
        """
        Build prompt for Cross-Cultural Auditors.
        If guardian_response is provided (Phase 2), auditors see the Guardian's position.
        """
        system = self.culture_roles[agent_idx]["auditor_prompt"].strip()
        agent_name = self.culture_roles[agent_idx]["name"]

        if self.task_type == "cultureatlas":
            answer_hint = "<1 or 2>"
        elif self.task_type == "culturalbench":
            answer_hint = "<1, 2, 3, or 4>"
        else:
            answer_hint = "<number>"  # covers normad and culturellm (variable scale)

        if guardian_response:
            # Phase 2: Auditor sees Guardian's response
            if self.task_type == "cultureatlas":
                user = (
                    f"TARGET CULTURE: {target_country}\n\n"
                    f"{question}\n\n"
                    f"The HOST-CULTURE GUARDIAN [{guardian_name}] has provided their "
                    f"authoritative analysis:\n"
                    f"---\n{guardian_response}\n---\n\n"
                    f"As a Cross-Cultural Auditor from [{agent_name}] background:\n"
                    f"1. Assess which response shows deeper cultural knowledge from "
                    f"your cross-cultural perspective.\n"
                    f"2. If you agree with the Guardian, explain WHY from your cultural lens.\n"
                    f"3. If you disagree, provide specific reasoning — but acknowledge "
                    f"that the Guardian has primary authority on {target_country}.\n\n"
                    f"Reasoning: <your cross-cultural comparative analysis>\n"
                    f"Answer: {answer_hint}"
                )
            elif self.task_type == "culturellm":
                # CultureLLM: WVS attitude — auditor sees guardian's cultural value analysis
                user = (
                    f"Task: You are currently discussing the following World Values Survey "
                    f"question about {target_country} with the other discussant.\n\n"
                    f"Survey Question:\n{question}\n"
                    f"Discussant: {guardian_response.strip()}\n\n"
                    f"Based on the above discussion, critically think about the cultural "
                    f"values and attitudes of {target_country} and make your final decision. "
                    f"Respond with the most representative option number.\n"
                    f"Answer:"
                )
            elif self.task_type == "culturalbench":
                # MAD-inspired: concise feedback + clear answer at end
                user = (
                    f"Task: You are currently discussing the following cultural "
                    f"knowledge question about {target_country} with the other discussant.\n\n"
                    f"Question:\n{question}\n"
                    f"Discussant: {guardian_response.strip()}\n\n"
                    f"Based on the above discussion, critically think and make "
                    f"your final decision. Respond with the correct option number "
                    f"(1, 2, 3, or 4).\n"
                    f"Answer (1, 2, 3, or 4):"
                )
            else:
                user = (
                    f"TARGET CULTURE: {target_country}\n\n"
                    f"{question}\n\n"
                    f"The HOST-CULTURE GUARDIAN [{guardian_name}] has provided their "
                    f"authoritative analysis:\n"
                    f"---\n{guardian_response}\n---\n\n"
                    f"As a Cross-Cultural Auditor from [{agent_name}] background:\n"
                    f"1. Provide your comparative perspective (similarities/differences "
                    f"between your culture and {target_country}).\n"
                    f"2. If you agree with the Guardian, explain WHY from your cultural lens.\n"
                    f"3. If you disagree, provide specific counter-evidence — but acknowledge "
                    f"that the Guardian has primary authority on {target_country}.\n\n"
                    f"Reasoning: <your cross-cultural comparative analysis>\n"
                    f"Answer: {answer_hint}"
                )
        else:
            # Phase 1 (negotiation_rounds=0): independent generation
            if self.task_type == "cultureatlas":
                user = (
                    f"TARGET CULTURE: {target_country}\n\n"
                    f"{question}\n\n"
                    f"As a Cross-Cultural Auditor from [{agent_name}] background, "
                    f"assess which response demonstrates more culturally specific "
                    f"knowledge about {target_country}. Note what appears generic vs. "
                    f"genuinely culture-specific from your cross-cultural perspective, "
                    f"and acknowledge uncertainty where the target culture differs "
                    f"from your expertise.\n\n"
                    f"Reasoning: <your cross-cultural comparative analysis>\n"
                    f"Answer: {answer_hint}"
                )
            elif self.task_type == "culturellm":
                # CultureLLM: same format as Guardian (independent value reasoning)
                user = (
                    f"Task: You will be given a World Values Survey question about the "
                    f"cultural values and attitudes of {target_country}. "
                    f"Select the option number that BEST represents the typical attitude "
                    f"or value orientation of people in {target_country}. "
                    f"Think about the dominant cultural values, social norms, and common "
                    f"worldviews in {target_country}, then respond with the most representative "
                    f"option number. Explain your reasoning in less than three sentences.\n\n"
                    f"Survey Question:\n{question}\n"
                    f"Answer:"
                )
            elif self.task_type == "culturalbench":
                # MAD-inspired: same simple format as Guardian
                user = (
                    f"Task: You will be given a cultural knowledge question about "
                    f"{target_country}. Select the correct option number. Do not make any "
                    f"extra inferences outside of the given context and country. "
                    f"Only align to the country given. "
                    f"Think step by step about the cultural practices of {target_country}, then respond "
                    f"with the correct option number (1, 2, 3, or 4). Explain your answer in less "
                    f"than three sentences.\n\n"
                    f"Question:\n{question}\n"
                    f"Answer (1, 2, 3, or 4):"
                )
            else:
                user = (
                    f"TARGET CULTURE: {target_country}\n\n"
                    f"{question}\n\n"
                    f"As a Cross-Cultural Auditor from [{agent_name}] background, "
                    f"provide your comparative perspective on this question about "
                    f"{target_country}. Note similarities and differences with your own "
                    f"cultural framework, and acknowledge uncertainty where the target "
                    f"culture differs from your expertise.\n\n"
                    f"Reasoning: <your cross-cultural comparative analysis>\n"
                    f"Answer: {answer_hint}"
                )
        return self._apply_chat(system, user)

    def _build_judge_prompt(self, question: str, target_country: str,
                            guardian_idx: int,
                            agent_responses: list[tuple[str, str, bool]]) -> str:
        """
        Build Judge prompt with explicit Guardian designation.

        agent_responses: list of (agent_name, response_text, is_guardian)
        """
        responses_text = ""
        for name, resp, is_guard in agent_responses:
            role_tag = "HOST-CULTURE GUARDIAN" if is_guard else "Cross-Cultural Auditor"
            responses_text += f"\n[{name}] ({role_tag}):\n{resp}\n"

        guardian_name = self.culture_roles[guardian_idx]["name"]

        if self.task_type == "cultureatlas":
            user = (
                f"TARGET CULTURE: {target_country}\n\n"
                f"{question}\n\n"
                f"The HOST-CULTURE GUARDIAN is [{guardian_name}] — their cultural "
                f"expertise most closely matches {target_country}.\n\n"
                f"Agent responses:\n{responses_text}\n"
                f"Determine which response demonstrates MORE culturally specific knowledge. "
                f"Remember:\n"
                f"- Give HIGHER WEIGHT to the Guardian's assessment of cultural depth\n"
                f"- The Guardian has VETO AUTHORITY — they best know what constitutes "
                f"genuine cultural specificity for {target_country}\n"
                f"- Cross-Cultural Auditors help identify generic vs. specific patterns\n"
                f"- Look for: named traditions, local terms, nuanced significance, "
                f"lesser-known practices\n\n"
                f"IMPORTANT: You MUST answer either 1 or 2. There is no neutral option.\n"
                f"One response is always more culturally specific than the other.\n\n"
                f"Reasoning: <your reasoning, explicitly referencing the Guardian's claims>\n"
                f"Answer: <1 or 2>"
            )
        elif self.task_type == "culturellm":
            # CultureLLM: Judge for WVS attitude prediction
            responses_text = ""
            for name, resp, is_guard in agent_responses:
                label = "Guardian" if is_guard else "Auditor"
                responses_text += f"  [{name}] ({label}): {resp.strip()}\n"
            user = (
                f"Task: You are a judge responsible for making a final decision "
                f"based on the opinions of cultural value experts about {target_country}. "
                f"Do NOT make any independent judgments; base your final decision "
                f"solely on the expert opinions below. Evaluate which option number "
                f"best represents the typical cultural attitude of {target_country}. "
                f"Respond with the most representative option number.\n\n"
                f"Survey Question:\n{question}\n\n"
                f"*** Expert opinions ***\n{responses_text}"
                f"*** End opinions ***\n\n"
                f"Final decision:"
            )
        elif self.task_type == "culturalbench":
            # MAD-inspired Judge: concise, debate-evidence-focused
            responses_text = ""
            for name, resp, is_guard in agent_responses:
                label = "Guardian" if is_guard else "Auditor"
                responses_text += f"  [{name}] ({label}): {resp.strip()}\n"
            user = (
                f"Task: You are a judge responsible for making a final decision "
                f"based on the opinions of cultural experts about {target_country}. "
                f"Do NOT make any independent judgments; base your final decision "
                f"solely on the expert opinions below. Evaluate the factual accuracy "
                f"of each argument regarding cultural knowledge of {target_country}. "
                f"Respond with the correct option number (1, 2, 3, or 4).\n\n"
                f"Question:\n{question}\n\n"
                f"*** Expert opinions ***\n{responses_text}"
                f"*** End opinions ***\n\n"
                f"Final decision (1, 2, 3, or 4):"
            )
        else:
            user = (
                f"TARGET CULTURE: {target_country}\n\n"
                f"{question}\n\n"
                f"The HOST-CULTURE GUARDIAN is [{guardian_name}] — their cultural "
                f"expertise most closely matches {target_country}.\n\n"
                f"Agent responses:\n{responses_text}\n"
                f"Determine the correct answer. Remember:\n"
                f"- Give HIGHER WEIGHT to the Guardian's specific cultural claims\n"
                f"- The Guardian has VETO AUTHORITY when providing specific evidence\n"
                f"- Cross-Cultural Auditors provide valuable comparative context\n"
                f"- Base your final decision on verifiable cultural facts\n\n"
                f"CALIBRATION REMINDER: Approximately 28% of questions in this dataset have\n"
                f"\"neutral/indeterminate (3)\" as the correct answer. If you find yourself\n"
                f"never outputting \"3\", you are likely over-committing to binary judgments.\n"
                f"Cultural expertise includes knowing when a behavior has NO specific\n"
                f"cultural significance in the target culture.\n\n"
                f"Reasoning: <your reasoning, explicitly referencing the Guardian's claims>\n"
                f"Answer: <number>"
            )
        return self._apply_chat(self.judge_system_prompt, user)

    # ------------------------------------------------------------------
    # Answer extraction
    # ------------------------------------------------------------------

    def _extract_answer(self, text: str, question: str = "") -> str | None:
        """Extract answer from response text. Respects task_type for valid range."""
        if self.task_type == "cultureatlas":
            max_choice = 2
        elif self.task_type == "culturalbench":
            max_choice = 4
        elif self.task_type == "culturellm":
            # CultureLLM has variable scales; detect max from question text
            # Options look like "0. ...\n1. ...\n2. ..." or "1. ...\n2. ...\n3. ...\n4. ..."
            option_nums = re.findall(r'^(\d+)\.', question, re.MULTILINE)
            if option_nums:
                max_choice = max(int(n) for n in option_nums)
            else:
                max_choice = 9  # safe fallback for unknown scale
        else:
            max_choice = 3

        # Build pattern: for culturellm, options can start from 0
        if self.task_type == "culturellm":
            pattern = f"[0-{max_choice}]"
        else:
            pattern = f"[1-{max_choice}]"

        # For culturalbench/culturellm: use MAD-style extraction
        if self.task_type in ("culturalbench", "culturellm"):
            tl = text.strip()
            # Pattern 1: "Answer: 3" or "Final decision: 2" or "answer is 1"
            m = re.search(rf'(?:Final\s+decision|Answer)\s*(?:is|[:\-])\s*({pattern})\b', tl, re.IGNORECASE)
            if m:
                return m.group(1)
            # Pattern 2: standalone single digit at end ("...so 3.")
            m = re.search(rf'\b({pattern})\s*\.?\s*$', tl)
            if m:
                return m.group(1)
            # Pattern 3: first line is ONLY a digit (no trailing text)
            first_line = tl.split("\n")[0].strip().rstrip(".")
            if re.match(rf'^({pattern})$', first_line):
                return first_line
            # Pattern 4: "option X"
            m = re.search(rf'option\s*({pattern})\b', tl, re.IGNORECASE)
            if m:
                return m.group(1)
            # Pattern 5: last digit in the text (fallback)
            matches = re.findall(rf'\b({pattern})\b', tl)
            return matches[-1] if matches else None

        # Non-culturalbench/culturellm tasks
        m = re.search(rf"(?:Final\s+decision|Answer)\s*[:\-]\s*({pattern})", text, re.IGNORECASE)
        if m:
            return m.group(1)
        m = re.search(rf"answer\s+is\s*:?\s*({pattern})\b", text, re.IGNORECASE)
        if m:
            return m.group(1)
        m = re.search(rf"option\s*:?\s*({pattern})\b", text, re.IGNORECASE)
        if m:
            return m.group(1)
        digits = re.findall(rf"\b({pattern})\b", text)
        return digits[-1] if digits else None

    # ------------------------------------------------------------------
    # Guardian failure detection
    # ------------------------------------------------------------------

    def _detect_guardian_failure(self, guardian_response: str) -> bool:
        """
        Determine if the Guardian has failed to provide a valid answer.

        Failure conditions:
          (a) Response is empty
          (b) Cannot extract a valid answer number
          (c) Reasoning contains explicit uncertainty/failure indicators
        """
        if not guardian_response or not guardian_response.strip():
            return True

        # For culturalbench/culturellm, response may be just a digit — skip length check
        if self.task_type not in ("culturalbench", "culturellm"):
            if len(guardian_response.strip()) < 10:
                return True

        # Check if answer is extractable
        answer = self._extract_answer(guardian_response)
        if answer is None:
            return True

        # Check for explicit failure indicators in reasoning
        response_lower = guardian_response.lower()
        for indicator in self.guardian_failure_indicators:
            if indicator.lower() in response_lower:
                return True

        return False

    # ------------------------------------------------------------------
    # Affinity-weighted arbitration (Guardian failure fallback)
    # ------------------------------------------------------------------

    def _get_affinity_scores(self, guardian_idx: int) -> list[float]:
        """
        Get cultural affinity scores of all agents relative to the Guardian's
        culture (which IS the target culture).

        Returns a list of affinity scores indexed by agent position.
        """
        if self.affinity_matrix is None:
            # Fallback: equal weights if no matrix configured
            return [1.0 / self.num_agents] * self.num_agents

        # guardian_idx's row in the affinity matrix gives distances to all others
        return self.affinity_matrix[guardian_idx]

    def _build_judge_fallback_prompt(self, question: str, target_country: str,
                                     guardian_idx: int,
                                     agent_responses: list[tuple[str, str, bool]],
                                     affinity_scores: list[float]) -> str:
        """
        Build a special Judge prompt for Guardian-failure scenarios.
        Includes affinity scores to guide weighted arbitration.
        """
        responses_text = ""
        for i, (name, resp, is_guard) in enumerate(agent_responses):
            if is_guard:
                responses_text += (
                    f"\n[{name}] (HOST-CULTURE GUARDIAN — FAILED, no valid answer):\n"
                    f"{resp}\n"
                )
            else:
                score = affinity_scores[i]
                responses_text += (
                    f"\n[{name}] (Cross-Cultural Auditor, "
                    f"affinity to target culture: {score:.1f}):\n{resp}\n"
                )

        guardian_name = self.culture_roles[guardian_idx]["name"]
        user = (
            f"TARGET CULTURE: {target_country}\n\n"
            f"{question}\n\n"
            f"⚠️ GUARDIAN FAILURE: The HOST-CULTURE GUARDIAN [{guardian_name}] has FAILED "
            f"to provide a valid answer for this question. Activate Cultural Affinity "
            f"Arbitration protocol.\n\n"
            f"CULTURAL AFFINITY SCORES (proximity to {target_country}'s culture):\n"
        )
        for i, (name, _, is_guard) in enumerate(agent_responses):
            if not is_guard:
                user += f"  - [{name}]: {affinity_scores[i]:.1f}\n"
        user += (
            f"\nAgent responses:\n{responses_text}\n"
            f"As the final arbitrator under Guardian Failure Protocol:\n"
            f"- Do NOT use simple majority voting.\n"
            f"- Give HIGHER WEIGHT to Auditors with higher affinity scores.\n"
            f"- If the highest-affinity Auditor provides specific cultural evidence, "
            f"prefer their answer even if outnumbered.\n"
            f"- Evaluate each Auditor's reasoning for concrete cultural references.\n\n"
        )
        if self.task_type == "cultureatlas":
            user += (
                f"IMPORTANT: You MUST answer either 1 or 2. There is no neutral option.\n"
                f"One response is always more culturally specific than the other.\n\n"
                f"Reasoning: <your reasoning, referencing affinity-weighted evidence>\n"
                f"Answer: <1 or 2>"
            )
        elif self.task_type == "culturellm":
            user += (
                f"Respond with the option number that best represents the typical "
                f"cultural attitude of {target_country}.\n\n"
                f"Final decision:"
            )
        elif self.task_type == "culturalbench":
            user += (
                f"Respond with the correct option number (1, 2, 3, or 4).\n\n"
                f"Final decision (1, 2, 3, or 4):"
            )
        else:
            user += (
                f"CALIBRATION REMINDER: Approximately 28% of questions in this dataset have\n"
                f"\"neutral/indeterminate (3)\" as the correct answer. If you find yourself\n"
                f"never outputting \"3\", you are likely over-committing to binary judgments.\n"
                f"Cultural expertise includes knowing when a behavior has NO specific\n"
                f"cultural significance in the target culture.\n\n"
                f"Reasoning: <your reasoning, referencing affinity-weighted evidence>\n"
                f"Answer: <number>"
            )
        return self._apply_chat(self.judge_system_prompt, user)

    # ------------------------------------------------------------------
    # Standard fallback (Guardian valid but Judge extraction fails)
    # ------------------------------------------------------------------

    def _majority_vote_with_guardian_veto(self, answers: list[str | None],
                                          guardian_idx: int) -> str | None:
        """
        Majority vote with Guardian veto (used only when Judge itself fails
        to produce a parseable answer, NOT when Guardian fails).

        If Guardian's answer exists and at least one other agent agrees, use Guardian's answer.
        Otherwise fall back to standard majority vote.
        """
        valid = [(i, a) for i, a in enumerate(answers) if a is not None]
        if not valid:
            return None

        guardian_answer = answers[guardian_idx] if guardian_idx >= 0 else None

        # Guardian veto: if guardian has an answer, check if any other agrees
        if guardian_answer:
            supporters = sum(1 for i, a in valid if a == guardian_answer and i != guardian_idx)
            if supporters >= 1:
                return guardian_answer
            # Even without supporters, if majority is split, prefer Guardian
            counts = Counter(a for _, a in valid)
            top_count = counts.most_common(1)[0][1]
            if counts[guardian_answer] == top_count:
                return guardian_answer

        # Standard majority vote
        counts = Counter(a for _, a in valid)
        return counts.most_common(1)[0][0]

    # ------------------------------------------------------------------
    # Auditor selection (dynamic subset when num_agents < total roles)
    # ------------------------------------------------------------------

    def _select_auditor_indices(self, guardian_idx: int) -> list[int]:
        """
        Select which auditor indices to use based on self.num_agents.

        Strategy: pick the (num_agents - 1) auditors with HIGHEST cultural
        affinity to the guardian's culture (i.e., most relevant perspectives).
        """
        all_auditors = [i for i in range(len(self.culture_roles)) if i != guardian_idx]
        num_auditors_needed = self.num_agents - 1

        if num_auditors_needed >= len(all_auditors):
            return all_auditors

        # Use affinity matrix to pick most relevant auditors
        if self.affinity_matrix is not None:
            affinities = self.affinity_matrix[guardian_idx]
            # Sort auditors by affinity score (descending), pick top N
            ranked = sorted(all_auditors, key=lambda i: affinities[i], reverse=True)
            return ranked[:num_auditors_needed]
        else:
            # No affinity matrix: just take first N
            return all_auditors[:num_auditors_needed]

    # ------------------------------------------------------------------
    # Debate feedback prompt (MAD-inspired Stage 2)
    # ------------------------------------------------------------------

    def _build_feedback_prompt(self, agent_idx: int, question: str,
                               target_country: str, is_guardian: bool,
                               own_response: str,
                               other_responses: list[tuple[str, str]]) -> str:
        """
        Build a feedback prompt (MAD Stage 2 equivalent).

        Each agent sees the other agents' responses and provides feedback.
        Keeps responses concise: "less than three sentences" (MAD's key insight).

        Args:
            agent_idx: index of this agent
            question: the question text
            target_country: target culture/country
            is_guardian: whether this agent is the Guardian
            own_response: this agent's initial response
            other_responses: list of (agent_name, response_text) from other agents
        """
        if is_guardian:
            system = self.culture_roles[agent_idx]["guardian_prompt"].strip()
        else:
            system = self.culture_roles[agent_idx]["auditor_prompt"].strip()

        agent_name = self.culture_roles[agent_idx]["name"]

        # Build discussion context
        discussion = ""
        for name, resp in other_responses:
            discussion += f"  [{name}]: {resp.strip()}\n"

        if self.task_type in ("culturalbench", "culturellm"):
            user = (
                f"TARGET CULTURE: {target_country}\n\n"
                f"{question}\n\n"
                f"Your initial answer:\n  [{agent_name}]: {own_response.strip()}\n\n"
                f"Other experts' answers:\n{discussion}\n"
                f"Respond to the other experts by providing any relevant feedback. "
                f"If you disagree with anyone, explain why with cultural evidence. "
                f"Respond in less than three sentences.\n"
                f"Response:"
            )
        else:
            user = (
                f"TARGET CULTURE: {target_country}\n\n"
                f"{question}\n\n"
                f"Your initial answer:\n  [{agent_name}]: {own_response.strip()}\n\n"
                f"Other experts' answers:\n{discussion}\n"
                f"Respond by providing any relevant feedback. "
                f"If you disagree, explain why with cultural evidence. "
                f"Respond in less than three sentences.\n"
                f"Response:"
            )
        return self._apply_chat(system, user)

    # ------------------------------------------------------------------
    # Final decision prompt (MAD Stage 3 equivalent)
    # ------------------------------------------------------------------

    def _build_final_decision_prompt(self, agent_idx: int, question: str,
                                      target_country: str, is_guardian: bool,
                                      own_response: str,
                                      other_responses: list[tuple[str, str]],
                                      own_feedback: str,
                                      other_feedbacks: list[tuple[str, str]]) -> str:
        """
        Build final decision prompt after feedback exchange (MAD Stage 3).
        Agent reconsiders their answer after seeing all feedback.
        """
        if is_guardian:
            system = self.culture_roles[agent_idx]["guardian_prompt"].strip()
        else:
            system = self.culture_roles[agent_idx]["auditor_prompt"].strip()

        agent_name = self.culture_roles[agent_idx]["name"]

        # Build discussion context
        others_text = ""
        for name, resp in other_responses:
            others_text += f"  [{name}]: {resp.strip()}\n"

        feedback_text = f"  [{agent_name}] (you): {own_feedback.strip()}\n"
        for name, fb in other_feedbacks:
            feedback_text += f"  [{name}]: {fb.strip()}\n"

        if self.task_type in ("culturalbench", "culturellm"):
            # For culturellm: don't hardcode option range, just ask for option number
            if self.task_type == "culturellm":
                answer_instruction = (
                    f"Based on the above discussion, critically think and make "
                    f"your final decision. Respond with the option number that best "
                    f"represents the typical cultural attitude of {target_country}.\n"
                    f"Answer:"
                )
            else:
                answer_instruction = (
                    f"Based on the above discussion, critically think and make "
                    f"your final decision. Respond with the correct option number "
                    f"(1, 2, 3, or 4).\n"
                    f"Answer (1, 2, 3, or 4):"
                )
            user = (
                f"TARGET CULTURE: {target_country}\n\n"
                f"{question}\n\n"
                f"=== Discussion Summary ===\n"
                f"Your initial answer:\n  [{agent_name}]: {own_response.strip()}\n\n"
                f"Other experts' answers:\n{others_text}\n"
                f"Feedback from all experts:\n{feedback_text}\n"
                f"=== End Discussion ===\n\n"
                f"{answer_instruction}"
            )
        else:
            user = (
                f"TARGET CULTURE: {target_country}\n\n"
                f"{question}\n\n"
                f"=== Discussion Summary ===\n"
                f"Your initial answer:\n  [{agent_name}]: {own_response.strip()}\n\n"
                f"Other experts' answers:\n{others_text}\n"
                f"Feedback from all experts:\n{feedback_text}\n"
                f"=== End Discussion ===\n\n"
                f"Based on the above discussion, critically think and make "
                f"your final decision.\n"
                f"Answer:"
            )
        return self._apply_chat(system, user)

    # ------------------------------------------------------------------
    # Disagreement-only Judge prompt (MAD Stage 4 equivalent)
    # ------------------------------------------------------------------

    def _build_judge_disagreement_prompt(self, question: str, target_country: str,
                                          guardian_idx: int,
                                          agent_final_responses: list[tuple[str, str, bool]],
                                          agent_feedbacks: list[tuple[str, str]]) -> str:
        """
        Build Judge prompt that is ONLY invoked when agents disagree on final answer.
        This mirrors MAD's Stage 4: Judge resolves disagreement based on debate history.
        """
        guardian_name = self.culture_roles[guardian_idx]["name"]

        # Build debate history
        debate_text = ""
        for name, resp, is_guard in agent_final_responses:
            role_tag = "HOST-CULTURE GUARDIAN" if is_guard else "Cross-Cultural Auditor"
            debate_text += f"  [{name}] ({role_tag}) final answer: {resp.strip()}\n"

        feedback_text = ""
        for name, fb in agent_feedbacks:
            feedback_text += f"  [{name}]: {fb.strip()}\n"

        if self.task_type == "culturellm":
            user = (
                f"Task: You are a judge responsible for making a final decision "
                f"based on the debate history between cultural value experts. They have "
                f"debated the following cultural attitude question about {target_country}. "
                f"Do NOT make any independent judgments; base your final decision "
                f"solely on the debate. Evaluate which option best represents the "
                f"typical cultural attitude of {target_country}. "
                f"Respond with the most representative option number.\n\n"
                f"Survey Question:\n{question}\n\n"
                f"*** Debate starts ***\n"
                f"{feedback_text}"
                f"{debate_text}"
                f"*** Debate ends ***\n\n"
                f"Final decision:"
            )
        elif self.task_type == "culturalbench":
            user = (
                f"Task: You are a judge responsible for making a final decision "
                f"based on the debate history between cultural experts. They have "
                f"debated the following cultural knowledge question about {target_country}. "
                f"Do NOT make any independent judgments; base your final decision "
                f"solely on the debate. Evaluate the factual accuracy of each argument "
                f"regarding cultural knowledge of {target_country}. "
                f"Respond with the correct option number (1, 2, 3, or 4).\n\n"
                f"Question:\n{question}\n\n"
                f"*** Debate starts ***\n"
                f"{feedback_text}"
                f"{debate_text}"
                f"*** Debate ends ***\n\n"
                f"Final decision (1, 2, 3, or 4):"
            )
        else:
            user = (
                f"TARGET CULTURE: {target_country}\n\n"
                f"{question}\n\n"
                f"You are a judge resolving a disagreement. "
                f"The HOST-CULTURE GUARDIAN is [{guardian_name}].\n\n"
                f"*** Debate feedback ***\n{feedback_text}\n"
                f"*** Final decisions ***\n{debate_text}\n"
                f"Base your decision on the debate evidence. "
                f"Give higher weight to the Guardian.\n\n"
                f"Reasoning: <brief reasoning>\n"
                f"Answer: <number>"
            )
        return self._apply_chat(self.judge_system_prompt, user)

    # ------------------------------------------------------------------
    # Single-sample inference (Guardian-First + Conditional Judge)
    # ------------------------------------------------------------------

    def inference(self, sample: dict) -> dict:
        question = sample["query"]
        target_country = sample.get("country", "")

        # Step 1: Detect Home-Field Guardian
        guardian_idx = self.detect_guardian(target_country)
        if guardian_idx < 0:
            guardian_idx = 0  # fallback: first agent

        guardian_name = self.culture_roles[guardian_idx]["name"]

        # Step 2: Select auditors (dynamic subset based on num_agents)
        auditor_indices = self._select_auditor_indices(guardian_idx)
        active_indices = [guardian_idx] + auditor_indices

        # ---- Phase 1: Guardian generates first (authoritative, low temp) ----
        guardian_prompt = self._build_guardian_prompt(
            guardian_idx, question, target_country
        )
        guardian_output = self.llm.generate([guardian_prompt], self.guardian_sampling)
        guardian_response = guardian_output[0].outputs[0].text.strip()

        # ---- Phase 2: Auditors generate ----
        auditor_responses = {}  # agent_idx -> response text
        if self.negotiation_rounds > 0 and self.task_type not in ("culturalbench", "culturellm"):
            # Non-culturalbench/culturellm: Auditors SEE Guardian's response (one-directional)
            auditor_prompts = []
            for ai in auditor_indices:
                prompt = self._build_auditor_prompt(
                    ai, question, target_country, guardian_name, guardian_response
                )
                auditor_prompts.append(prompt)
            if auditor_prompts:
                auditor_outputs = self.llm.generate(
                    auditor_prompts, self.auditor_sampling
                )
                for ai, out in zip(auditor_indices, auditor_outputs):
                    auditor_responses[ai] = out.outputs[0].text.strip()
        else:
            # Independent mode: Auditors don't see Guardian (CulturalBench/CultureLLM always starts independent)
            auditor_prompts = []
            for ai in auditor_indices:
                prompt = self._build_auditor_prompt(
                    ai, question, target_country, guardian_name, None
                )
                auditor_prompts.append(prompt)
            if auditor_prompts:
                auditor_outputs = self.llm.generate(
                    auditor_prompts, self.auditor_sampling
                )
                for ai, out in zip(auditor_indices, auditor_outputs):
                    auditor_responses[ai] = out.outputs[0].text.strip()

        # ---- Phase 2.5: MAD-style debate for CulturalBench/CultureLLM ----
        # When negotiation_rounds > 0 AND culturalbench/culturellm: do feedback + final decision
        # This lets agents reconsider after seeing others' answers (can fix minority-correct cases)
        if self.negotiation_rounds > 0 and self.task_type in ("culturalbench", "culturellm"):
            # All initial responses (Guardian + Auditors)
            all_initial = {guardian_idx: guardian_response, **auditor_responses}

            # Stage 2: Each agent gives feedback on others' responses
            feedback_prompts = []
            feedback_meta = []  # (agent_idx, is_guardian)
            for idx in active_indices:
                is_guard = (idx == guardian_idx)
                own_resp = all_initial[idx]
                other_resps = [
                    (self.culture_roles[j]["name"], all_initial[j])
                    for j in active_indices if j != idx
                ]
                prompt = self._build_feedback_prompt(
                    idx, question, target_country, is_guard,
                    own_resp, other_resps
                )
                feedback_prompts.append(prompt)
                feedback_meta.append((idx, is_guard))

            feedback_outputs = self.llm.generate(
                feedback_prompts, self.auditor_sampling
            )
            agent_feedbacks = {}  # idx -> feedback text
            for out, (idx, _) in zip(feedback_outputs, feedback_meta):
                agent_feedbacks[idx] = out.outputs[0].text.strip()

            # Stage 3: Each agent makes final decision
            final_prompts = []
            final_meta = []
            for idx in active_indices:
                is_guard = (idx == guardian_idx)
                own_resp = all_initial[idx]
                other_resps = [
                    (self.culture_roles[j]["name"], all_initial[j])
                    for j in active_indices if j != idx
                ]
                own_fb = agent_feedbacks.get(idx, "")
                other_fbs = [
                    (self.culture_roles[j]["name"], agent_feedbacks.get(j, ""))
                    for j in active_indices if j != idx
                ]
                prompt = self._build_final_decision_prompt(
                    idx, question, target_country, is_guard,
                    own_resp, other_resps, own_fb, other_fbs
                )
                final_prompts.append(prompt)
                final_meta.append((idx, is_guard))

            final_outputs = self.llm.generate(
                final_prompts, self.guardian_sampling  # Low temp for final decision
            )

            # Replace initial responses with final decisions for voting
            for out, (idx, _) in zip(final_outputs, final_meta):
                final_text = out.outputs[0].text.strip()
                if idx == guardian_idx:
                    guardian_response = final_text
                else:
                    auditor_responses[idx] = final_text

        # ---- Phase 3: Conditional Judge (only on disagreement) ----
        guardian_answer = self._extract_answer(guardian_response, question)
        auditor_answers = {}
        for ai in auditor_indices:
            auditor_answers[ai] = self._extract_answer(
                auditor_responses.get(ai, ""), question
            )

        all_answers = {guardian_idx: guardian_answer, **auditor_answers}
        valid_answers = [a for a in all_answers.values() if a is not None]

        guardian_failed = self._detect_guardian_failure(guardian_response)

        # Determine consensus levels:
        # For culturalbench/culturellm (MCQ-style): use pure majority vote
        # For other tasks: use Guardian-weighted consensus
        if self.task_type in ("culturalbench", "culturellm"):
            # Pure majority vote — no Guardian privilege for factual QA
            from collections import Counter as _Counter
            vote_counts = _Counter(a for a in valid_answers if a is not None)
            majority_answer = vote_counts.most_common(1)[0][0] if vote_counts else None
            majority_count = vote_counts.most_common(1)[0][1] if vote_counts else 0
            total_voters = len(valid_answers)

            has_full_consensus = (len(set(valid_answers)) <= 1) if valid_answers else False
            # Majority = more than half agree (2/3 or 3/3)
            guardian_has_support = (majority_count > total_voters / 2) and not has_full_consensus
        else:
            guardian_has_support = False
            if guardian_answer and not guardian_failed:
                supporters = sum(
                    1 for a in auditor_answers.values() if a == guardian_answer
                )
                guardian_has_support = (supporters >= 1)
            has_full_consensus = (
                len(set(valid_answers)) <= 1 if valid_answers else False
            ) and not guardian_failed

        judge_response = ""

        if has_full_consensus:
            # All agents agree → accept consensus directly
            judge_response = (
                f"[CONSENSUS] All agents agree. Answer: {valid_answers[0]}"
            )
        elif guardian_has_support:
            # Majority agrees → accept majority answer
            if self.task_type in ("culturalbench", "culturellm"):
                judge_response = (
                    f"[GUARDIAN-MAJORITY] Majority vote ({majority_count}/{total_voters}). "
                    f"Answer: {majority_answer}"
                )
            else:
                judge_response = (
                    f"[GUARDIAN-MAJORITY] Guardian supported by auditor(s). "
                    f"Answer: {guardian_answer}"
                )
        elif self.include_judge:
            # Real disagreement → invoke Judge
            if guardian_failed:
                agent_input = [
                    (self.culture_roles[guardian_idx]["name"],
                     guardian_response, True)
                ] + [
                    (self.culture_roles[ai]["name"],
                     auditor_responses.get(ai, ""), False)
                    for ai in auditor_indices
                ]
                affinity_scores = self._get_affinity_scores(guardian_idx)
                judge_prompt = self._build_judge_fallback_prompt(
                    question, target_country, guardian_idx,
                    agent_input, affinity_scores
                )
            else:
                agent_input = [
                    (self.culture_roles[guardian_idx]["name"],
                     guardian_response, True)
                ] + [
                    (self.culture_roles[ai]["name"],
                     auditor_responses.get(ai, ""), False)
                    for ai in auditor_indices
                ]
                judge_prompt = self._build_judge_prompt(
                    question, target_country, guardian_idx, agent_input
                )

            judge_output = self.llm.generate([judge_prompt], self.judge_sampling)
            judge_response = judge_output[0].outputs[0].text.strip()

            judge_answer = self._extract_answer(judge_response, question)
            if judge_answer is None:
                # Fallback: guardian-weighted vote
                all_ans_list = [all_answers.get(i) for i in active_indices]
                fallback = self._majority_vote_with_guardian_veto(
                    all_ans_list, 0  # Guardian is always at position 0
                )
                judge_response += (
                    f"\n[Fallback guardian-weighted vote]: {fallback}"
                )

        # ---- Format output (compatible with AgentArk pipeline) ----
        formatted = ""
        sol_num = 1
        role_tag = "[GUARDIAN]" if not guardian_failed else "[GUARDIAN-FAILED]"
        formatted += (
            f"===== Solution {sol_num} {role_tag} =====\n"
            f"{guardian_response}\n"
        )

        for ai in auditor_indices:
            sol_num += 1
            formatted += (
                f"===== Solution {sol_num} [AUDITOR] =====\n"
                f"{auditor_responses.get(ai, '')}\n"
            )

        sol_num += 1
        if has_full_consensus:
            judge_mode = "[JUDGE-CONSENSUS]"
        elif guardian_has_support:
            judge_mode = "[JUDGE-GUARDIAN-MAJORITY]"
        elif guardian_failed:
            judge_mode = "[JUDGE-AFFINITY-ARBITRATION]"
        else:
            judge_mode = "[JUDGE-DISAGREEMENT]"
        formatted += (
            f"===== Solution {sol_num} {judge_mode} =====\n"
            f"{judge_response}\n"
        )

        return {
            "response": formatted,
            "guardian_idx": guardian_idx,
            "guardian_name": guardian_name,
            "guardian_failed": guardian_failed,
        }

    # ------------------------------------------------------------------
    # Batch inference (maximise GPU utilisation)
    # ------------------------------------------------------------------

    def inference_batch(self, samples: list[dict]) -> list[dict]:
        """
        Batch inference with Guardian-First + Conditional Judge:
          Phase 1: All Guardians generate first (low temp, authoritative)
          Phase 2: All Auditors generate with Guardian context (high temp, diverse)
          Phase 3: Conditional Judge — only invoked for real disagreements
                   (Guardian without auditor support)
        """
        n = len(samples)
        questions = [s["query"] for s in samples]
        countries = [s.get("country", "") for s in samples]

        # Detect Guardians and select auditors for all samples
        guardian_indices = []
        auditor_indices_per_sample = []
        active_indices_per_sample = []
        for si, country in enumerate(countries):
            g_idx = self.detect_guardian(country)
            g_idx = g_idx if g_idx >= 0 else 0
            guardian_indices.append(g_idx)
            auditor_idxs = self._select_auditor_indices(g_idx)
            auditor_indices_per_sample.append(auditor_idxs)
            active_indices_per_sample.append([g_idx] + auditor_idxs)

        # Per-sample storage
        guardian_responses = [""] * n
        auditor_responses = [{} for _ in range(n)]  # si -> {agent_idx: text}

        # ---- Phase 1: All Guardians generate independently (low temp) ----
        guardian_prompts = []
        for si in range(n):
            prompt = self._build_guardian_prompt(
                guardian_indices[si], questions[si], countries[si]
            )
            guardian_prompts.append(prompt)

        guardian_outputs = self.llm.generate(guardian_prompts, self.guardian_sampling)
        for si, out in enumerate(guardian_outputs):
            guardian_responses[si] = out.outputs[0].text.strip()

        # ---- Phase 2: All Auditors generate with Guardian context (high temp) ----
        auditor_prompts = []
        auditor_meta = []  # (sample_idx, agent_idx)
        for si in range(n):
            g_idx = guardian_indices[si]
            g_name = self.culture_roles[g_idx]["name"]
            # For culturalbench/culturellm: always independent (no guardian context)
            if self.task_type in ("culturalbench", "culturellm") or self.negotiation_rounds == 0:
                g_resp = None
            else:
                g_resp = guardian_responses[si]
            for ai in auditor_indices_per_sample[si]:
                prompt = self._build_auditor_prompt(
                    ai, questions[si], countries[si], g_name, g_resp
                )
                auditor_prompts.append(prompt)
                auditor_meta.append((si, ai))

        if auditor_prompts:
            auditor_outputs = self.llm.generate(
                auditor_prompts, self.auditor_sampling
            )
            for out, (si, ai) in zip(auditor_outputs, auditor_meta):
                auditor_responses[si][ai] = out.outputs[0].text.strip()

        # ---- Phase 3: Conditional Judge (only for real disagreements) ----
        # Extract answers and determine consensus for each sample
        guardian_failures = []
        consensus_types = []  # "full", "guardian_majority", "disagreement"
        all_answers_per_sample = []
        majority_answers = [None] * n  # For culturalbench majority vote
        majority_counts = [0] * n

        for si in range(n):
            g_idx = guardian_indices[si]
            g_resp = guardian_responses[si]

            failed = self._detect_guardian_failure(g_resp)
            guardian_failures.append(failed)

            # Extract all answers
            g_answer = self._extract_answer(g_resp, questions[si])
            a_answers = {}
            for ai in auditor_indices_per_sample[si]:
                a_answers[ai] = self._extract_answer(
                    auditor_responses[si].get(ai, ""), questions[si]
                )
            all_ans = {g_idx: g_answer, **a_answers}
            all_answers_per_sample.append(all_ans)

            valid_ans = [a for a in all_ans.values() if a is not None]

            # Determine consensus type
            if self.task_type in ("culturalbench", "culturellm"):
                # Pure majority vote — no Guardian privilege for MCQ-style QA
                from collections import Counter as _Counter
                vote_counts = _Counter(a for a in valid_ans if a is not None)
                maj_ans = vote_counts.most_common(1)[0][0] if vote_counts else None
                maj_cnt = vote_counts.most_common(1)[0][1] if vote_counts else 0
                total_voters = len(valid_ans)
                majority_answers[si] = maj_ans
                majority_counts[si] = maj_cnt

                has_full_consensus = (len(set(valid_ans)) <= 1) if valid_ans else False
                guardian_has_support = (maj_cnt > total_voters / 2) and not has_full_consensus
            else:
                has_full_consensus = (
                    len(set(valid_ans)) <= 1 if valid_ans else False
                ) and not failed

                guardian_has_support = False
                if g_answer and not failed:
                    supporters = sum(
                        1 for a in a_answers.values() if a == g_answer
                    )
                    guardian_has_support = (supporters >= 1)

            if has_full_consensus:
                consensus_types.append("full")
            elif guardian_has_support:
                consensus_types.append("guardian_majority")
            else:
                consensus_types.append("disagreement")

        # Build Judge prompts only for real disagreements
        judge_responses = [""] * n
        judge_prompt_list = []
        judge_sample_indices = []

        for si in range(n):
            g_idx = guardian_indices[si]
            g_answer = all_answers_per_sample[si].get(g_idx)
            valid_ans = [a for a in all_answers_per_sample[si].values()
                         if a is not None]

            if consensus_types[si] == "full":
                judge_responses[si] = (
                    f"[CONSENSUS] All agents agree. Answer: {valid_ans[0]}"
                )
            elif consensus_types[si] == "guardian_majority":
                if self.task_type in ("culturalbench", "culturellm"):
                    judge_responses[si] = (
                        f"[GUARDIAN-MAJORITY] Majority vote ({majority_counts[si]}/{len(valid_ans)}). "
                        f"Answer: {majority_answers[si]}"
                    )
                else:
                    judge_responses[si] = (
                        f"[GUARDIAN-MAJORITY] Guardian supported by auditor(s). "
                        f"Answer: {g_answer}"
                    )
            elif self.include_judge:
                # Real disagreement → need Judge
                if guardian_failures[si]:
                    agent_input = [
                        (self.culture_roles[g_idx]["name"],
                         guardian_responses[si], True)
                    ] + [
                        (self.culture_roles[ai]["name"],
                         auditor_responses[si].get(ai, ""), False)
                        for ai in auditor_indices_per_sample[si]
                    ]
                    affinity_scores = self._get_affinity_scores(g_idx)
                    prompt = self._build_judge_fallback_prompt(
                        questions[si], countries[si], g_idx,
                        agent_input, affinity_scores
                    )
                else:
                    agent_input = [
                        (self.culture_roles[g_idx]["name"],
                         guardian_responses[si], True)
                    ] + [
                        (self.culture_roles[ai]["name"],
                         auditor_responses[si].get(ai, ""), False)
                        for ai in auditor_indices_per_sample[si]
                    ]
                    prompt = self._build_judge_prompt(
                        questions[si], countries[si], g_idx, agent_input
                    )
                judge_prompt_list.append(prompt)
                judge_sample_indices.append(si)

        # Generate Judge responses in batch (only for disagreements)
        if judge_prompt_list:
            judge_outputs = self.llm.generate(
                judge_prompt_list, self.judge_sampling
            )
            for out, si in zip(judge_outputs, judge_sample_indices):
                judge_resp = out.outputs[0].text.strip()
                judge_answer = self._extract_answer(judge_resp, questions[si])

                if judge_answer is None:
                    # Fallback: guardian-weighted vote
                    active = active_indices_per_sample[si]
                    all_ans_list = [
                        all_answers_per_sample[si].get(ai) for ai in active
                    ]
                    fallback = self._majority_vote_with_guardian_veto(
                        all_ans_list, 0  # Guardian always at position 0
                    )
                    judge_resp += (
                        f"\n[Fallback guardian-weighted vote]: {fallback}"
                    )

                judge_responses[si] = judge_resp

        # ---- Build results ----
        results = []
        for si in range(n):
            g_idx = guardian_indices[si]
            failed = guardian_failures[si]

            formatted = ""
            sol_num = 1
            role_tag = "[GUARDIAN]" if not failed else "[GUARDIAN-FAILED]"
            formatted += (
                f"===== Solution {sol_num} {role_tag} =====\n"
                f"{guardian_responses[si]}\n"
            )

            for ai in auditor_indices_per_sample[si]:
                sol_num += 1
                formatted += (
                    f"===== Solution {sol_num} [AUDITOR] =====\n"
                    f"{auditor_responses[si].get(ai, '')}\n"
                )

            sol_num += 1
            ctype = consensus_types[si]
            if ctype == "full":
                judge_mode = "[JUDGE-CONSENSUS]"
            elif ctype == "guardian_majority":
                judge_mode = "[JUDGE-GUARDIAN-MAJORITY]"
            elif failed:
                judge_mode = "[JUDGE-AFFINITY-ARBITRATION]"
            else:
                judge_mode = "[JUDGE-DISAGREEMENT]"
            formatted += (
                f"===== Solution {sol_num} {judge_mode} =====\n"
                f"{judge_responses[si]}\n"
            )

            results.append({
                "response": formatted,
                "guardian_idx": g_idx,
                "guardian_name": self.culture_roles[g_idx]["name"],
                "guardian_failed": failed,
            })

        return results

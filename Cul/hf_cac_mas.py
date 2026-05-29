"""
HF-CAC: Home-Field Culture-Activated Collaboration

Extension of RECONCILE framework with dynamic authority activation:
  1. Home-Field Detection: automatically identify which agent is the
     "Host-Culture Guardian" based on target country in the question.
  2. Asymmetric Prompting: Guardian uses authoritative confirmation/correction prompt;
     other agents use cross-cultural auditor prompt (contrastive, deferential).
  3. Structured Negotiation: Guardian generates first (priority), then Auditors
     respond with awareness of Guardian's position.
  4. Authority-Aware Judge: Judge explicitly weights Guardian's claims higher,
     with veto mechanism when Guardian provides specific evidence.

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
                 negotiation_rounds=1):
        """
        Args:
            model_name: HuggingFace model path or alias
            tensor_parallel_size: vLLM tensor parallelism
            config_path: path to hf_cac_config.yaml
            temperature: base temperature (overridden per role)
            max_tokens: max generation tokens
            include_judge: whether to include Judge reasoning in output
            negotiation_rounds: rounds of structured negotiation (0=independent, 1=standard)
        """
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__), "configs", "hf_cac_config.yaml"
            )
        cfg = load_config(config_path)

        self.culture_roles = cfg["culture_roles"]
        self.num_agents = len(self.culture_roles)
        self.judge_system_prompt = cfg["judge"]["system_prompt"].strip()
        self.include_judge = include_judge
        self.negotiation_rounds = negotiation_rounds
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model_name = model_name

        # Task type: "normad" (3-way acceptability) or "cultureatlas" (2-way comparison)
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

        # Guardian: lower temperature for authoritative, precise responses
        self.guardian_sampling = SamplingParams(
            temperature=0.5,
            max_tokens=self.max_tokens,
            stop=stop_tokens,
        )
        # Auditor: higher temperature for diverse contrastive perspectives
        self.auditor_sampling = SamplingParams(
            temperature=0.9,
            max_tokens=self.max_tokens,
            stop=stop_tokens,
        )
        # Judge: low temperature for stable arbitration
        self.judge_sampling = SamplingParams(
            temperature=0.3,
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

        answer_hint = "<1 or 2>" if self.task_type == "cultureatlas" else "<number>"

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

    def _extract_answer(self, text: str) -> str | None:
        """Extract answer from response text. Respects task_type for valid range."""
        max_choice = 2 if self.task_type == "cultureatlas" else 4
        pattern = f"[1-{max_choice}]"

        m = re.search(rf"Answer\s*:\s*({pattern})", text, re.IGNORECASE)
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
          (a) Response is empty or too short to be meaningful
          (b) Cannot extract a valid answer number
          (c) Reasoning contains explicit uncertainty/failure indicators
        """
        if not guardian_response or len(guardian_response.strip()) < 10:
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
    # Single-sample inference
    # ------------------------------------------------------------------

    def inference(self, sample: dict) -> dict:
        question = sample["query"]
        target_country = sample.get("country", "")

        # Step 1: Detect Home-Field Guardian
        guardian_idx = self.detect_guardian(target_country)
        if guardian_idx < 0:
            guardian_idx = 0  # fallback: first agent

        guardian_name = self.culture_roles[guardian_idx]["name"]
        agent_responses = [""] * self.num_agents

        # Step 2: Phase 1 — Guardian generates first (authoritative)
        guardian_prompt = self._build_guardian_prompt(
            guardian_idx, question, target_country
        )
        guardian_output = self.llm.generate([guardian_prompt], self.guardian_sampling)
        agent_responses[guardian_idx] = guardian_output[0].outputs[0].text.strip()

        # Step 3: Phase 2 — Auditors generate with Guardian context
        auditor_indices = [i for i in range(self.num_agents) if i != guardian_idx]

        if self.negotiation_rounds > 0:
            # Structured negotiation: Auditors SEE Guardian's response
            auditor_prompts = [
                self._build_auditor_prompt(
                    i, question, target_country,
                    guardian_name, agent_responses[guardian_idx]
                )
                for i in auditor_indices
            ]
        else:
            # Independent mode (negotiation_rounds=0): Auditors don't see Guardian
            auditor_prompts = [
                self._build_auditor_prompt(
                    i, question, target_country, guardian_name, None
                )
                for i in auditor_indices
            ]

        auditor_outputs = self.llm.generate(auditor_prompts, self.auditor_sampling)
        for ai, out in zip(auditor_indices, auditor_outputs):
            agent_responses[ai] = out.outputs[0].text.strip()

        # Step 4: Detect Guardian failure and choose Judge strategy
        guardian_failed = self._detect_guardian_failure(
            agent_responses[guardian_idx]
        )

        judge_response = ""
        if self.include_judge:
            judge_input = [
                (self.culture_roles[i]["name"], agent_responses[i], i == guardian_idx)
                for i in range(self.num_agents)
            ]

            if guardian_failed:
                # Guardian failed → activate Cultural Affinity Arbitration
                affinity_scores = self._get_affinity_scores(guardian_idx)
                judge_prompt = self._build_judge_fallback_prompt(
                    question, target_country, guardian_idx,
                    judge_input, affinity_scores
                )
            else:
                # Guardian valid → standard Judge deliberation
                judge_prompt = self._build_judge_prompt(
                    question, target_country, guardian_idx, judge_input
                )

            judge_output = self.llm.generate([judge_prompt], self.judge_sampling)
            judge_response = judge_output[0].outputs[0].text.strip()

            judge_answer = self._extract_answer(judge_response)
            if judge_answer is None:
                # Last resort: if Judge ALSO fails to parse, use weighted vote
                all_answers = [self._extract_answer(r) for r in agent_responses]
                fallback = self._majority_vote_with_guardian_veto(
                    all_answers, guardian_idx
                )
                judge_response += f"\n[Fallback guardian-weighted vote]: {fallback}"

        # Step 5: Format output (compatible with AgentArk pipeline)
        formatted = ""
        for i, resp in enumerate(agent_responses):
            role_tag = "[GUARDIAN]" if i == guardian_idx else "[AUDITOR]"
            if i == guardian_idx and guardian_failed:
                role_tag = "[GUARDIAN-FAILED]"
            formatted += f"===== Solution {i + 1} {role_tag} =====\n{resp}\n"
        if self.include_judge:
            judge_mode = "[JUDGE-AFFINITY-ARBITRATION]" if guardian_failed else "[JUDGE]"
            formatted += (
                f"===== Solution {self.num_agents + 1} {judge_mode} =====\n"
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
        Batch inference with two-phase generation:
          Phase 1: All Guardians in parallel
          Phase 2: All Auditors in parallel (with Guardian context if negotiation_rounds>0)
          Phase 3: All Judges in parallel
        """
        n = len(samples)
        questions = [s["query"] for s in samples]
        countries = [s.get("country", "") for s in samples]

        # Detect Guardians for all samples
        guardian_indices = []
        for country in countries:
            g_idx = self.detect_guardian(country)
            guardian_indices.append(g_idx if g_idx >= 0 else 0)

        agent_responses = [[""] * self.num_agents for _ in range(n)]

        # ---- Phase 1: Generate all Guardian responses ----
        guardian_prompts = []
        guardian_meta = []  # (sample_idx,)
        for si in range(n):
            g_idx = guardian_indices[si]
            prompt = self._build_guardian_prompt(g_idx, questions[si], countries[si])
            guardian_prompts.append(prompt)
            guardian_meta.append(si)

        guardian_outputs = self.llm.generate(guardian_prompts, self.guardian_sampling)
        for out, si in zip(guardian_outputs, guardian_meta):
            g_idx = guardian_indices[si]
            agent_responses[si][g_idx] = out.outputs[0].text.strip()

        # ---- Phase 2: Generate all Auditor responses ----
        auditor_prompts = []
        auditor_meta = []  # (sample_idx, agent_idx)
        for si in range(n):
            g_idx = guardian_indices[si]
            g_name = self.culture_roles[g_idx]["name"]
            g_response = agent_responses[si][g_idx] if self.negotiation_rounds > 0 else None

            for ai in range(self.num_agents):
                if ai == g_idx:
                    continue
                prompt = self._build_auditor_prompt(
                    ai, questions[si], countries[si], g_name, g_response
                )
                auditor_prompts.append(prompt)
                auditor_meta.append((si, ai))

        auditor_outputs = self.llm.generate(auditor_prompts, self.auditor_sampling)
        for out, (si, ai) in zip(auditor_outputs, auditor_meta):
            agent_responses[si][ai] = out.outputs[0].text.strip()

        # ---- Detect Guardian failures for all samples ----
        guardian_failures = [
            self._detect_guardian_failure(agent_responses[si][guardian_indices[si]])
            for si in range(n)
        ]

        # ---- Phase 3: Generate all Judge responses ----
        judge_responses = [""] * n
        if self.include_judge:
            judge_prompts = []
            for si in range(n):
                g_idx = guardian_indices[si]
                judge_input = [
                    (self.culture_roles[ai]["name"], agent_responses[si][ai],
                     ai == g_idx)
                    for ai in range(self.num_agents)
                ]

                if guardian_failures[si]:
                    # Guardian failed → affinity-based arbitration prompt
                    affinity_scores = self._get_affinity_scores(g_idx)
                    prompt = self._build_judge_fallback_prompt(
                        questions[si], countries[si], g_idx,
                        judge_input, affinity_scores
                    )
                else:
                    # Guardian valid → standard Judge prompt
                    prompt = self._build_judge_prompt(
                        questions[si], countries[si], g_idx, judge_input
                    )
                judge_prompts.append(prompt)

            judge_outputs = self.llm.generate(judge_prompts, self.judge_sampling)

            for si in range(n):
                judge_resp = judge_outputs[si].outputs[0].text.strip()
                judge_answer = self._extract_answer(judge_resp)

                if judge_answer is None:
                    # Last resort: if Judge ALSO fails to parse
                    all_answers = [
                        self._extract_answer(agent_responses[si][ai])
                        for ai in range(self.num_agents)
                    ]
                    fallback = self._majority_vote_with_guardian_veto(
                        all_answers, guardian_indices[si]
                    )
                    judge_resp += f"\n[Fallback guardian-weighted vote]: {fallback}"

                judge_responses[si] = judge_resp

        # ---- Build results ----
        results = []
        for si in range(n):
            g_idx = guardian_indices[si]
            failed = guardian_failures[si]
            formatted = ""
            for ai in range(self.num_agents):
                role_tag = "[GUARDIAN]" if ai == g_idx else "[AUDITOR]"
                if ai == g_idx and failed:
                    role_tag = "[GUARDIAN-FAILED]"
                formatted += (
                    f"===== Solution {ai + 1} {role_tag} =====\n"
                    f"{agent_responses[si][ai]}\n"
                )
            if self.include_judge:
                judge_mode = (
                    "[JUDGE-AFFINITY-ARBITRATION]" if failed else "[JUDGE]"
                )
                formatted += (
                    f"===== Solution {self.num_agents + 1} {judge_mode} =====\n"
                    f"{judge_responses[si]}\n"
                )

            results.append({
                "response": formatted,
                "guardian_idx": g_idx,
                "guardian_name": self.culture_roles[g_idx]["name"],
                "guardian_failed": failed,
            })

        return results

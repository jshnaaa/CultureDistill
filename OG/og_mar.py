"""
OG-MAR Baseline: Ontology-Guided Multi-Agent Reasoning

Reproduction of "Toward Culturally Aligned LLMs through Ontology-Guided
Multi-Agent Reasoning" (Seo et al., 2026) adapted for cultural evaluation tasks.

Supports two datasets (auto-detected from input_file name):
  - NormAD: cultural acceptability judgment (3-choice: acceptable/unacceptable/neutral)
  - CulturalBench: cultural knowledge QA (4-choice, dynamic per-question options)

Framework:
  1. Context Retrieval: For each query, retrieve relevant ontology triples
     and construct demographically grounded cultural profiles.
  2. Persona Agent Simulation (K=5): Instantiate K value-persona agents,
     each conditioned on ontology context, value summaries, and demographics.
     Each agent outputs a culturally grounded answer with reasoning.
  3. Judgment Agent: Synthesize persona outputs via constrained
     meta-adjudication (evidence-first protocol with vote-as-tiebreak).

Prompts follow Appendix E (Tables 8, 9) from the original paper, with
minimal adaptation for each dataset format.

Output naming: {dataset}_OGMAR_{model}.json + _metrics.json

Usage:
    # NormAD - quick test
    python OG/og_mar.py \\
        --input_file /autodl-fs/data/normad_mas.json \\
        --model_name qwen \\
        --tensor_parallel_size 2 \\
        --max_samples 5

    # NormAD - full dataset
    python OG/og_mar.py \\
        --input_file /autodl-fs/data/normad_mas.json \\
        --model_name qwen \\
        --tensor_parallel_size 2 \\
        --batch_size 256 \\
        --max_samples 0 \\
        --temperature 0.0

    # CulturalBench - quick test
    python OG/og_mar.py \\
        --input_file /autodl-fs/data/culturalBench_mas.json \\
        --model_name qwen \\
        --tensor_parallel_size 2 \\
        --max_samples 5

    # CulturalBench - full dataset
    python OG/og_mar.py \\
        --input_file /autodl-fs/data/culturalBench_mas.json \\
        --model_name qwen \\
        --tensor_parallel_size 2 \\
        --batch_size 256 \\
        --max_samples 0 \\
        --temperature 0.0
"""

import os
import sys
import re
import json
import argparse
from tqdm import tqdm
from collections import Counter

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from OG.og_common import (
    MODEL_ALIASES, ANSWER_MAP, REVERSE_ANSWER_MAP,
    PERSONA_AGENT_PROMPT, JUDGMENT_AGENT_PROMPT,
    load_dataset, extract_country, extract_background, extract_scenario,
    extract_answer, get_relevant_triples, generate_persona_demographics,
    generate_value_summary, format_vote_summary,
    infer_output_path, compute_metrics, NORMAD_AXIS_TO_DOMAINS,
)


# ===================================================================
# Dataset type detection
# ===================================================================

DATASET_NORMAD = "normad"
DATASET_CULTURALBENCH = "culturalbench"


def detect_dataset_type(input_file: str) -> str:
    """Auto-detect dataset type from the input file name."""
    basename = os.path.basename(input_file).lower()
    if "culturalbench" in basename or "cultural_bench" in basename:
        return DATASET_CULTURALBENCH
    # Default to normad
    return DATASET_NORMAD


# ===================================================================
# System Prompt
# ===================================================================

SYSTEM_PROMPT = (
    "You are a culturally informed reasoning agent. You must strictly follow "
    "the instructions and output format specified in the task. Do not add any "
    "text outside the required JSON output."
)


# ===================================================================
# Response Options (fixed for NormAD task)
# ===================================================================

NORMAD_OPTIONS_TEXT = (
    '1: acceptable (the behavior is socially acceptable in this culture)\n'
    '2: unacceptable (the behavior is socially unacceptable in this culture)\n'
    '3: neutral/indeterminate (cannot determine or culture-neutral)'
)


# ===================================================================
# CulturalBench Persona Agent Prompt (adapted from Table 8)
# Minimal change: allows persona to integrate cultural factual knowledge
# alongside the ontology context, since CulturalBench is factual QA.
# ===================================================================

CULTURALBENCH_PERSONA_PROMPT = """\
Task:
- You are Persona Agent {persona_id}, a cultural knowledge expert from {country_name}.
- The question below is a factual cultural knowledge question with exactly one correct answer. Your job is to select the factually correct option based on your deep knowledge of this country's culture, customs, traditions, and practices.
- Use the provided persona-defining inputs as background context. If the ontology context is relevant to the question topic, integrate it; if it is not directly relevant, rely on your own cultural knowledge.
- Prohibited: fabricating demographics/values/edges not provided; guessing when uncertain.

Inputs:
- [DEMOGRAPHICS]: {demographics_text}
- [VALUE PROFILES]: {value_summaries_text}
- [ONTOLOGY CONTEXT]: {hyper_nodes_text}
- [RESPONSE OPTIONS]: {options_text}
- [USER QUESTION]: {question}

Strict Rules:
- This is a factual knowledge question. Choose the option that is objectively correct about this country's culture.
- Stay in persona; use your lived experience and cultural knowledge of {country_name} to identify the correct answer.
- If the ontology context directly relates to the question, cite it; otherwise, state that it is not directly applicable and rely on cultural knowledge.
- Cite at least 2 demographic attributes that inform your perspective.
- Choose exactly one option; output only one valid JSON object and nothing else.
- Your chosen_answer MUST start with the option number followed by a period (e.g., "1. ...", "2. ...", "3. ...", "4. ...").
- reasoning must be >= 50 words and explicitly explain why the chosen answer is factually correct about {country_name}'s culture.

Output Format (JSON only):
{{
  "persona_id": "{persona_id}",
  "chosen_answer": "<option_number>. <option_text>",
  "reasoning": "...",
  "alignment_factors": {{
    "demographic": "...",
    "value_summaries_used": [],
    "hyper_edges_used": [],
    "integration_rationale": "..."
  }}
}}"""


# ===================================================================
# CulturalBench Judgment Agent Prompt (adapted from Table 9)
# Adds a factual verification step: judgment agent should verify
# the persona consensus against its own cultural knowledge.
# ===================================================================

CULTURALBENCH_JUDGMENT_PROMPT = """\
Task:
- You are the Judgment Agent.
- Given the question, options, persona outputs, and a pre-computed vote summary, select exactly one final option.
- This is a factual cultural knowledge question with one objectively correct answer.
- Your decision must be based on: (1) Persona outputs (primary evidence), (2) Vote summary (secondary context), and (3) Your own verification of factual correctness.
- If all personas agree, verify their reasoning is factually sound before confirming. If the consensus answer seems factually incorrect based on well-known cultural facts, you may override it.
- Prohibited: adding new facts beyond well-established cultural knowledge, or fabricating demographics/values/edges.

Inputs:
- [USER QUESTION]: {question_text}
- [RESPONSE OPTIONS]: {options_text}
- [VOTE SUMMARY]: {vote_summary}
- [PERSONA OUTPUTS]: {persona_outputs}

Strict Rules:
- Use information in [PERSONA OUTPUTS] and [VOTE SUMMARY] as primary evidence.
- Treat vote counts as correct and immutable; do not recount or modify them.
- Before finalizing, ask yourself: "Is this answer factually correct about this country's culture?"
- If the answer is well-supported by persona reasoning AND factually sound, confirm it.
- If you detect a factual error in the consensus (e.g., wrong cultural practice attributed to a country), override with the correct option.

Decision Procedure:
- A) Factual Verification (Primary): Verify the chosen option is factually correct.
- B) Evidence Strength (Secondary): Prefer the option supported by explicit, internally consistent persona reasoning.
- C) Vote Summary (Tertiary): Use vote counts to break ties when evidence strength is comparable.

Output Format (JSON only):
{{
  "final_answer": "<option_number>. <option_text>",
  "reasoning": "..."
}}"""


# ===================================================================
# Chat template helper
# ===================================================================

def apply_chat(tokenizer, user_content: str, system: str = None) -> str:
    """Apply chat template to format prompt for the model."""
    messages = [
        {"role": "system", "content": system or SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ===================================================================
# NormAD: Infer axis from input text (heuristic)
# ===================================================================

def infer_axis_normad(input_text: str, item: dict) -> str:
    """Infer the cultural axis from the NormAD item."""
    bg = extract_background(input_text)
    bg_lower = bg.lower() if bg else ""

    keyword_map = {
        "etiquette": "Etiquette", "greet": "Etiquette",
        "polite": "Etiquette", "rude": "Etiquette",
        "moral": "Morality", "ethical": "Morality",
        "law": "Law", "legal": "Law", "illegal": "Law", "crime": "Law",
        "religion": "Religion", "pray": "Religion", "mosque": "Religion",
        "church": "Religion", "temple": "Religion", "faith": "Religion",
        "family": "Family", "marriage": "Family", "parent": "Family",
        "work": "Work", "employ": "Work", "business": "Work",
        "food": "Food", "eat": "Food", "drink": "Food",
        "education": "Education", "school": "Education",
        "visit": "Etiquette",
    }

    axis_scores = Counter()
    for keyword, axis in keyword_map.items():
        if keyword in bg_lower:
            axis_scores[axis] += 1

    if axis_scores:
        return axis_scores.most_common(1)[0][0]
    return "default"


# ===================================================================
# CulturalBench: Parse input and infer axis
# ===================================================================

def parse_culturalbench_input(input_text: str) -> tuple:
    """
    Parse CulturalBench input to extract the question stem and options.

    Returns:
        (question_stem: str, options_text: str, options_list: list[str])
    """
    lines = input_text.strip().split("\n")

    # Find where options start (first line matching "1." or "1)")
    option_start_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if re.match(r'^1[\.\)]\s', stripped):
            option_start_idx = i
            break

    if option_start_idx is not None:
        question_stem = "\n".join(lines[:option_start_idx]).strip()
        options_lines = lines[option_start_idx:]
    else:
        question_stem = input_text.strip()
        options_lines = []

    # Parse individual options
    options_list = []
    current_option = ""
    for line in options_lines:
        stripped = line.strip()
        if re.match(r'^\d[\.\)]\s', stripped):
            if current_option:
                options_list.append(current_option.strip())
            current_option = stripped
        else:
            current_option += " " + stripped

    if current_option:
        options_list.append(current_option.strip())

    options_text = "\n".join(options_list) if options_list else ""
    return question_stem, options_text, options_list


def infer_axis_culturalbench(input_text: str, country: str) -> str:
    """Infer cultural axis from CulturalBench question text."""
    text_lower = input_text.lower()

    keyword_map = {
        "etiquette": "Etiquette", "greet": "Etiquette", "polite": "Etiquette",
        "rude": "Etiquette", "manners": "Etiquette", "respect": "Etiquette",
        "handshake": "Etiquette", "bow": "Etiquette",
        "moral": "Morality", "ethical": "Morality",
        "law": "Law", "legal": "Law", "illegal": "Law", "crime": "Law",
        "religion": "Religion", "pray": "Religion", "mosque": "Religion",
        "church": "Religion", "temple": "Religion", "faith": "Religion",
        "hindu": "Religion", "muslim": "Religion", "christian": "Religion",
        "buddhist": "Religion", "ramadan": "Religion",
        "family": "Family", "marriage": "Family", "parent": "Family",
        "wedding": "Family", "elder": "Family", "grandparent": "Family",
        "work": "Work", "employ": "Work", "business": "Work", "office": "Work",
        "food": "Food", "eat": "Food", "drink": "Food", "cook": "Food",
        "dish": "Food", "meal": "Food", "cuisine": "Food", "dining": "Food",
        "education": "Education", "school": "Education", "university": "Education",
        "festival": "Religion", "celebration": "Etiquette",
        "sport": "Etiquette", "game": "Etiquette",
        "tradition": "Etiquette", "custom": "Etiquette", "practice": "Etiquette",
    }

    axis_scores = Counter()
    for keyword, axis in keyword_map.items():
        if keyword in text_lower:
            axis_scores[axis] += 1

    if axis_scores:
        return axis_scores.most_common(1)[0][0]
    return "default"


# ===================================================================
# Unified answer extraction (supports both 1-3 and 1-4)
# ===================================================================

def _find_outermost_json(text: str) -> dict:
    """
    Find and parse the outermost JSON object in text, handling nested braces.
    Returns parsed dict or None.
    """
    # Find the first '{' and then match braces to find the complete object
    start = text.find('{')
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False

    for i in range(start, len(text)):
        ch = text[i]
        if escape_next:
            escape_next = False
            continue
        if ch == '\\' and in_string:
            escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i+1])
                except json.JSONDecodeError:
                    # Try next '{' if this one failed
                    next_start = text.find('{', start + 1)
                    if next_start != -1:
                        return _find_outermost_json(text[next_start:])
                    return None
    return None


def _match_text_to_option(answer_text: str, options_list: list) -> str:
    """
    Match a text-only answer back to its option number by fuzzy text comparison.
    options_list: list of option strings like ["1. Cycle everywhere", "2. Using deodorant", ...]
    Returns option number as string ("1"-"4") or None.
    """
    if not answer_text or not options_list:
        return None

    answer_lower = answer_text.lower().strip().rstrip('.')

    best_match = None
    best_score = 0

    for opt in options_list:
        opt_stripped = opt.strip()
        # Extract option number and text: "1. text" or "1) text"
        opt_match = re.match(r'^(\d)[\.\)]\s*(.*)', opt_stripped)
        if not opt_match:
            continue
        opt_num = opt_match.group(1)
        opt_text = opt_match.group(2).lower().strip().rstrip('.')

        # Exact match
        if answer_lower == opt_text:
            return opt_num

        # Containment match (answer contains option text or vice versa)
        if opt_text in answer_lower or answer_lower in opt_text:
            score = len(opt_text) if opt_text in answer_lower else len(answer_lower)
            if score > best_score:
                best_score = score
                best_match = opt_num
            continue

        # Prefix match (first N chars)
        min_len = min(len(opt_text), len(answer_lower), 25)
        if min_len > 5 and opt_text[:min_len] == answer_lower[:min_len]:
            if min_len > best_score:
                best_score = min_len
                best_match = opt_num

    return best_match


def extract_answer_unified(text: str, dataset_type: str, options_list: list = None) -> str:
    """
    Extract answer from model output.
    For NormAD: valid answers are 1/2/3.
    For CulturalBench: valid answers are 1/2/3/4.

    Args:
        text: Raw model output text
        dataset_type: "normad" or "culturalbench"
        options_list: List of option strings (e.g., ["1. Cycle everywhere", ...])
                     Used for text-matching when the model outputs text without a number prefix.

    Strategy:
    1. Parse the outermost JSON object (handles nested braces like alignment_factors)
    2. Look for "final_answer" or "chosen_answer" field
    3. Extract leading digit from the answer value
    4. If no digit, match answer text to options_list
    5. Fallback: regex-based field extraction without full JSON parse
    6. Fallback: keyword patterns in text
    """
    if not text:
        return None

    valid_set = "123" if dataset_type == DATASET_NORMAD else "1234"

    # --- Strategy 1: Parse outermost JSON (handles nested objects) ---
    obj = _find_outermost_json(text)
    if obj:
        # Look for the answer fields
        answer_val = obj.get("final_answer") or obj.get("chosen_answer") or ""
        if answer_val:
            answer_str = str(answer_val).strip()
            # Extract the leading digit from "4. Talking loudly..." or "2: Poached"
            num_match = re.match(r'(\d)', answer_str)
            if num_match and num_match.group(1) in valid_set:
                return num_match.group(1)

            # Text-only answer: match against options_list
            if options_list:
                matched = _match_text_to_option(answer_str, options_list)
                if matched:
                    return matched

    # --- Strategy 2: Regex-based field extraction (no full JSON parse needed) ---
    # Directly search for "chosen_answer": "N. ..." or "final_answer": "N. ..."
    field_pattern = r'"(?:final_answer|chosen_answer)"\s*:\s*"(\d)'
    field_match = re.search(field_pattern, text)
    if field_match and field_match.group(1) in valid_set:
        return field_match.group(1)

    # Also try: "chosen_answer": N (without quotes, just a number)
    field_pattern2 = r'"(?:final_answer|chosen_answer)"\s*:\s*(\d)'
    field_match2 = re.search(field_pattern2, text)
    if field_match2 and field_match2.group(1) in valid_set:
        return field_match2.group(1)

    # --- Strategy 2b: If JSON parsing found a text answer but no options_list was given ---
    # Extract the text from the field and try text matching with options in the prompt
    if obj and options_list:
        # Already tried above, skip
        pass

    # --- Strategy 3: Look for explicit answer patterns ---
    text_lower = text.lower().strip()

    pattern = r'(?:answer|choice|option)\s*(?:is|:)?\s*([' + valid_set + r'])'
    num_match = re.search(pattern, text_lower)
    if num_match:
        return num_match.group(1)

    # NormAD-specific: check for yes/no/neither keywords
    # Order matters: check "neither/neutral" BEFORE "acceptable/unacceptable"
    if dataset_type == DATASET_NORMAD:
        if re.search(r'\b(neither|neutral|indeterminate)\b', text_lower[:200]):
            return "3"
        if re.search(r'\b(unacceptable)\b', text_lower[:200]):
            return "2"
        if re.search(r'\b(yes|acceptable)\b', text_lower[:200]):
            return "1"

    # --- Strategy 4: Last resort - look for digit NOT in persona_id context ---
    # Find content after any "chosen_answer" or "final_answer" mention
    answer_region = re.search(r'(?:chosen_answer|final_answer)["\s:]+(.{1,80})', text)
    if answer_region:
        region_text = answer_region.group(1)
        # First try digit
        digit_match = re.search(r'[' + valid_set + r']', region_text)
        if digit_match:
            return digit_match.group(0)
        # Try text matching on the region content
        if options_list:
            matched = _match_text_to_option(region_text.strip('" '), options_list)
            if matched:
                return matched

    # Absolute last resort: skip first 200 chars (avoids persona_id) and find digit
    if len(text) > 200:
        late_digit = re.search(r'[' + valid_set + r']', text[200:])
        if late_digit:
            return late_digit.group(0)

    return None


# ===================================================================
# Unified vote summary (supports both datasets)
# ===================================================================

def format_vote_summary_unified(persona_answers: list, dataset_type: str) -> str:
    """
    Create a vote summary from persona agent answers.
    Uses REVERSE_ANSWER_MAP labels for NormAD, plain option numbers for CulturalBench.
    """
    vote_counter = Counter()
    valid_set = "123" if dataset_type == DATASET_NORMAD else "1234"
    for ans in persona_answers:
        if ans and ans in valid_set:
            vote_counter[ans] += 1

    parts = []
    for option in sorted(vote_counter.keys()):
        if dataset_type == DATASET_NORMAD:
            option_label = REVERSE_ANSWER_MAP.get(option, f"Option {option}")
            parts.append(f"Option {option} ({option_label}): {vote_counter[option]} vote(s)")
        else:
            parts.append(f"Option {option}: {vote_counter[option]} vote(s)")

    return "; ".join(parts) if parts else "No valid votes"


# ===================================================================
# Main inference function
# ===================================================================

def run_og_mar(args):
    # --- Detect dataset type ---
    dataset_type = detect_dataset_type(args.input_file)
    print(f"Detected dataset type: {dataset_type}")

    # --- Resolve model ---
    model_path = MODEL_ALIASES.get(args.model_name.lower(), args.model_name)
    print(f"Model: {model_path}")
    print(f"Number of persona agents (K): {args.num_personas}")
    print(f"Number of ontology triples (M): {args.num_triples}")

    # --- Determine output paths ---
    out_json, out_metrics = infer_output_path(
        args.input_file, args.model_name, args.output_dir
    )
    print(f"Output JSON:    {out_json}")
    print(f"Output Metrics: {out_metrics}")

    # --- Load data ---
    raw_data = load_dataset(args.input_file)
    dataset = raw_data
    if args.max_samples > 0:
        dataset = dataset[:args.max_samples]
    print(f"Loaded {len(dataset)} samples from {args.input_file}")

    # --- Pre-parse data (dataset-specific) ---
    parsed = []
    if dataset_type == DATASET_NORMAD:
        for item in dataset:
            input_text = item.get("input", "")
            country = extract_country(input_text)
            background = extract_background(input_text)
            scenario = extract_scenario(input_text)
            axis = infer_axis_normad(input_text, item)

            parsed.append({
                **item,
                "country": country,
                "background": background,
                "scenario": scenario,
                "axis": axis,
                "options_text": NORMAD_OPTIONS_TEXT,
                "question_stem": None,  # built dynamically in Stage 2
            })
    else:  # DATASET_CULTURALBENCH
        for item in dataset:
            input_text = item.get("input", "")
            country = item.get("country", "").strip()
            country_key = country.lower().replace(" ", "_")
            question_stem, options_text, options_list = parse_culturalbench_input(input_text)
            axis = infer_axis_culturalbench(input_text, country_key)

            parsed.append({
                **item,
                "country": country_key,
                "country_display": country,
                "background": question_stem,  # use question as context for retrieval
                "scenario": "",
                "axis": axis,
                "options_text": options_text,
                "options_list": options_list,
                "question_stem": question_stem,
            })

    n = len(parsed)
    K = args.num_personas  # Number of persona agents
    M = args.num_triples   # Number of ontology triples

    # --- Initialize vLLM ---
    print("Initializing vLLM...")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        gpu_memory_utilization=0.85,
        dtype="bfloat16",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    stop_tokens = ["<|eot_id|>", "<|end_of_text|>", "</s>"]
    sampling = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        stop=stop_tokens,
        top_p=0.95 if args.temperature > 0 else 1.0,
    )

    batch_size = args.batch_size

    # ================================================================
    # Stage 1: Ontology & Demographic Retrieval (pre-computed)
    # ================================================================
    print(f"\n{'='*60}")
    print("Stage 1: Ontology & Context Retrieval")
    print(f"{'='*60}")

    # Pre-compute ontology triples and demographics for all samples
    all_triples = []       # [sample_idx] -> list of triple strings
    all_demographics = []  # [sample_idx][persona_idx] -> demographics text
    all_value_summaries = []  # [sample_idx][persona_idx] -> value summary

    for idx, p in enumerate(tqdm(parsed, desc="Retrieving context")):
        # Retrieve ontology triples
        triples = get_relevant_triples(p["country"], p["axis"], p["background"], top_n=M)
        all_triples.append(triples)

        # Generate demographics for K personas
        demographics = []
        value_summaries = []
        for k in range(K):
            demo = generate_persona_demographics(p["country"], k)
            demographics.append(demo)
            val_sum = generate_value_summary(
                p["country"], p["axis"], p["background"], k
            )
            value_summaries.append(val_sum)

        all_demographics.append(demographics)
        all_value_summaries.append(value_summaries)

    # ================================================================
    # Stage 2: Multi-Value Persona Agent Simulation
    # (Optimized: all K×N prompts in a single batch for maximum GPU
    #  utilization via vLLM continuous batching)
    # ================================================================
    print(f"\n{'='*60}")
    print(f"Stage 2: Persona Agent Simulation (K={K}, N={n})")
    print(f"  Total prompts: {K * n} (all personas batched together)")
    print(f"{'='*60}")

    # Build ALL persona prompts at once: flat list of K*N prompts
    all_persona_prompts = []
    for idx, p in enumerate(parsed):
        hyper_nodes_text = "\n".join(all_triples[idx])

        # Build question text (dataset-specific)
        if dataset_type == DATASET_NORMAD:
            question = (
                f"In the cultural context of {p['country'].replace('_', ' ').title()}, "
                f"evaluate the following scenario:\n{p['scenario']}\n\n"
                f"Is the described behavior socially acceptable in this culture?"
            )
        else:  # DATASET_CULTURALBENCH
            question = p["question_stem"]

        options_text = p["options_text"]

        for k in range(K):
            # Use CulturalBench-specific prompt (allows cultural knowledge)
            # or standard NormAD prompt (strict ontology-only)
            prompt_template = (
                CULTURALBENCH_PERSONA_PROMPT
                if dataset_type == DATASET_CULTURALBENCH
                else PERSONA_AGENT_PROMPT
            )
            format_kwargs = dict(
                persona_id=k + 1,
                demographics_text=all_demographics[idx][k],
                value_summaries_text=all_value_summaries[idx][k],
                hyper_nodes_text=hyper_nodes_text,
                options_text=options_text,
                question=question,
            )
            # CulturalBench prompt needs country_name for grounding
            if dataset_type == DATASET_CULTURALBENCH:
                format_kwargs["country_name"] = p["country"].replace("_", " ").title()
            user_content = prompt_template.format(**format_kwargs)
            all_persona_prompts.append(apply_chat(tokenizer, user_content))

    total_prompts = len(all_persona_prompts)
    print(f"  Built {total_prompts} prompts, starting batch inference...")

    # Single-pass batch inference over all K*N prompts
    all_persona_raw = []
    for i in tqdm(range(0, total_prompts, batch_size),
                  desc="  Stage2-AllPersonas"):
        batch_end = min(i + batch_size, total_prompts)
        batch_prompts = all_persona_prompts[i:batch_end]
        outputs = llm.generate(batch_prompts, sampling, use_tqdm=False)
        for out in outputs:
            all_persona_raw.append(out.outputs[0].text.strip())

    # Reshape flat list back to [sample_idx][persona_idx]
    persona_outputs = [[None] * K for _ in range(n)]
    for idx in range(n):
        for k in range(K):
            persona_outputs[idx][k] = all_persona_raw[idx * K + k]

    # ================================================================
    # Stage 3: Ontology-Guided Final Judgment via Constrained
    #           Meta-Adjudication
    # ================================================================
    print(f"\n{'='*60}")
    print("Stage 3: Judgment Agent - Constrained Meta-Adjudication")
    print(f"{'='*60}")

    # First, extract persona answers for vote summary
    persona_answers = [[None] * K for _ in range(n)]
    for idx in range(n):
        opts = parsed[idx].get("options_list") if dataset_type == DATASET_CULTURALBENCH else None
        for k in range(K):
            persona_answers[idx][k] = extract_answer_unified(
                persona_outputs[idx][k], dataset_type, options_list=opts
            )

    # Build judgment prompts
    judgment_prompts = []
    for idx, p in enumerate(parsed):
        # Format persona outputs for the judge
        persona_text_parts = []
        for k in range(K):
            persona_text_parts.append(
                f"--- Persona Agent {k+1} ---\n{persona_outputs[idx][k]}"
            )
        persona_outputs_text = "\n\n".join(persona_text_parts)

        # Vote summary
        vote_summary = format_vote_summary_unified(persona_answers[idx], dataset_type)

        # Question text (dataset-specific)
        if dataset_type == DATASET_NORMAD:
            question_text = (
                f"In the cultural context of {p['country'].replace('_', ' ').title()}, "
                f"evaluate the following scenario:\n{p['scenario']}\n\n"
                f"Is the described behavior socially acceptable in this culture?"
            )
        else:  # DATASET_CULTURALBENCH
            question_text = p["question_stem"]

        options_text = p["options_text"]

        # Use CulturalBench judgment prompt (with factual verification)
        # or standard NormAD judgment prompt
        judgment_template = (
            CULTURALBENCH_JUDGMENT_PROMPT
            if dataset_type == DATASET_CULTURALBENCH
            else JUDGMENT_AGENT_PROMPT
        )
        user_content = judgment_template.format(
            question_text=question_text,
            options_text=options_text,
            vote_summary=vote_summary,
            persona_outputs=persona_outputs_text,
        )
        judgment_prompts.append(apply_chat(tokenizer, user_content))

    # Batch inference for judgment
    judgment_outputs = []
    for i in tqdm(range(0, n, batch_size), desc="Judgment"):
        batch_end = min(i + batch_size, n)
        batch_prompts = judgment_prompts[i:batch_end]
        outputs = llm.generate(batch_prompts, sampling, use_tqdm=False)
        for out in outputs:
            judgment_outputs.append(out.outputs[0].text.strip())

    # ================================================================
    # Stage 4: Extract answers and build results
    # ================================================================
    print(f"\n{'='*60}")
    print("Stage 4: Extract Answers & Compute Metrics")
    print(f"{'='*60}")

    results = []
    for idx, p in enumerate(parsed):
        # Extract judgment answer
        judge_resp = judgment_outputs[idx]
        opts = p.get("options_list") if dataset_type == DATASET_CULTURALBENCH else None
        judge_ans = extract_answer_unified(judge_resp, dataset_type, options_list=opts)

        # If judgment extraction fails, fall back to majority vote
        if judge_ans is None:
            valid_votes = [a for a in persona_answers[idx] if a is not None]
            if valid_votes:
                vote_counter = Counter(valid_votes)
                judge_ans = vote_counter.most_common(1)[0][0]

        gt = str(p.get("output", "")).strip()
        is_correct = (judge_ans == gt) if judge_ans else False

        # Format per-persona answers
        per_persona = {}
        for k in range(K):
            per_persona[f"persona_{k+1}"] = {
                "response": persona_outputs[idx][k],
                "answer": persona_answers[idx][k],
            }

        # Build result record (dataset-specific fields)
        record = {
            "instruction": p.get("instruction", ""),
            "input": p.get("input", ""),
            "output": gt,
            "country": p.get("country_display", p["country"]) if dataset_type == DATASET_CULTURALBENCH else p["country"],
            "axis": p["axis"],
            # Retrieval context
            "ontology_triples": all_triples[idx],
            # Persona agents
            "persona_outputs": per_persona,
            "persona_vote_summary": format_vote_summary_unified(persona_answers[idx], dataset_type),
            # Judgment
            "judgment_response": judge_resp,
            "final_answer": judge_ans if judge_ans else "",
            "correct": is_correct,
        }

        # Dataset-specific extra fields
        if dataset_type == DATASET_NORMAD:
            record["scenario"] = p["scenario"]
        else:
            record["question_stem"] = p["question_stem"]
            record["options_text"] = p["options_text"]

        results.append(record)

    # ================================================================
    # Write output
    # ================================================================
    print(f"\n{'='*60}")
    print("Writing Output")
    print(f"{'='*60}")

    out_dir = os.path.dirname(out_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Inference results saved to: {out_json}")

    # ================================================================
    # Compute & save metrics
    # ================================================================
    metrics = compute_metrics(results)
    metrics["method"] = "OG-MAR"
    metrics["model"] = args.model_name
    metrics["dataset"] = "CulturalBench" if dataset_type == DATASET_CULTURALBENCH else "NormAD"
    metrics["num_personas"] = K
    metrics["num_triples"] = M
    metrics["temperature"] = args.temperature
    metrics["prompt_source"] = "Appendix E, Tables 8-9 (OG-MAR paper, Seo et al. 2026)"
    metrics["framework"] = "Ontology-Guided Multi-Agent Reasoning"

    # Add persona agreement stats
    full_agreement = 0
    for idx in range(n):
        valid_votes = [a for a in persona_answers[idx] if a is not None]
        if valid_votes and len(set(valid_votes)) == 1:
            full_agreement += 1
    metrics["persona_full_agreement"] = full_agreement

    metrics_dir = os.path.dirname(out_metrics)
    if metrics_dir:
        os.makedirs(metrics_dir, exist_ok=True)
    with open(out_metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Metrics saved to: {out_metrics}")
    print(f"\nAccuracy: {metrics['accuracy']:.4f} "
          f"({metrics['correct']}/{metrics['total_samples']})")
    print(f"Persona full agreement: {full_agreement}/{n}")


# ===================================================================
# CLI
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="OG-MAR Baseline: Ontology-Guided Multi-Agent Reasoning (Seo et al., 2026)"
    )
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to dataset JSON (auto-detects normad/culturalbench)")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model alias (llama/qwen) or HuggingFace path")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: /autodl-fs/data/ogmar)")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="vLLM tensor parallel size")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for vLLM inference")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Max samples to process (0=all)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (paper uses 0 for stable behavior)")
    parser.add_argument("--max_tokens", type=int, default=768,
                        help="Max tokens per generation (longer for structured JSON output)")
    parser.add_argument("--num_personas", type=int, default=5,
                        help="Number of persona agents K (paper default: 5)")
    parser.add_argument("--num_triples", type=int, default=5,
                        help="Number of ontology triples M to retrieve (paper default: 3-9)")

    args = parser.parse_args()
    run_og_mar(args)


if __name__ == "__main__":
    main()

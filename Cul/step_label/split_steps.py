"""
CAMA-D Stage 2a: Heuristic Step Splitting (启发式推理步骤切分)

Splits reasoning text into semantic units using deterministic rules:
  1. Primary split on paragraph boundaries (\\n\\n or \\n)
  2. Secondary split on logical transition markers if paragraphs are too long
  3. Label each step with [Step N] prefix

This module is used as a preprocessing step before the open-book auditor labeling.

Usage:
    python Cul/step_label/split_steps.py \\
        --input_file  /path/to/hfa_c2n_inference.jsonl \\
        --output_file /path/to/steps_split.jsonl \\
        --max_sentences_per_step 3

Output format (per line in JSONL):
    {
      "question": "...",
      "country": "Vietnam",
      "gt": "1",
      "reasoning_source": "guardian",
      "full_reasoning": "...",
      "steps": [
        {"step_idx": 1, "text": "[Step 1] In Vietnamese culture..."},
        {"step_idx": 2, "text": "[Step 2] However, educational..."},
        ...
      ]
    }
"""

import re
import json
import argparse
from pathlib import Path


# Transition markers for secondary splitting
TRANSITION_MARKERS = [
    r'\bHowever,?\b', r'\bBut,?\b', r'\bTherefore,?\b',
    r'\bOn the contrary,?\b', r'\bNevertheless,?\b',
    r'\bIn contrast,?\b', r'\bConsequently,?\b',
    r'\bThus,?\b', r'\bMeanwhile,?\b', r'\bInstead,?\b',
    r'\bMoreover,?\b', r'\bFurthermore,?\b', r'\bAdditionally,?\b',
    r'\bIn conclusion,?\b', r'\bFinally,?\b', r'\bOverall,?\b',
]

TRANSITION_PATTERN = '|'.join(TRANSITION_MARKERS)


def split_reasoning_into_steps(
    reasoning_text: str,
    max_sentences_per_step: int = 3
) -> list[str]:
    """
    Heuristic step splitting for reasoning text.

    Rules:
      1. Primary split on paragraph boundaries (\\n\\n or \\n)
      2. If a segment exceeds max_sentences_per_step sentences,
         detect strong transition markers and split at those points
      3. Label each step with [Step N] prefix

    Args:
        reasoning_text: Raw reasoning text from an agent
        max_sentences_per_step: Maximum sentences before triggering
                                secondary split

    Returns:
        List of labeled step strings: ["[Step 1] ...", "[Step 2] ...", ...]
    """
    if not reasoning_text or not reasoning_text.strip():
        return []

    # Step 1: Primary split on paragraph boundaries
    raw_segments = re.split(r'\n\n|\n', reasoning_text.strip())
    raw_segments = [s.strip() for s in raw_segments if s.strip()]

    # Step 2: Secondary split on long paragraphs
    steps = []
    for segment in raw_segments:
        sentences = re.split(r'(?<=[.!?])\s+', segment)
        if len(sentences) > max_sentences_per_step:
            # Try to split at transition markers
            current_chunk = []
            for sent in sentences:
                if (re.search(TRANSITION_PATTERN, sent, re.IGNORECASE)
                        and current_chunk):
                    steps.append(' '.join(current_chunk))
                    current_chunk = [sent]
                else:
                    current_chunk.append(sent)
            if current_chunk:
                steps.append(' '.join(current_chunk))
        else:
            steps.append(segment)

    # Step 3: Label with [Step N] prefix
    labeled_steps = [f"[Step {i+1}] {step}" for i, step in enumerate(steps)]
    return labeled_steps


# ---------------------------------------------------------------------------
# Extract reasoning from HFA-C²N output
# ---------------------------------------------------------------------------

def extract_agent_reasonings(response: str) -> list[dict]:
    """
    Extract reasoning text from each agent in HFA-C²N output.

    Returns list of dicts:
      - source: "guardian" | "auditor-1" | "auditor-2" | ...
      - reasoning: the reasoning text
      - answer: the predicted answer
    """
    # Split by solution markers
    parts = re.split(r"===== Solution \d+ =====", response)
    parts = [p.strip() for p in parts if p.strip()]

    results = []
    for part in parts:
        # Determine source
        if "[GUARDIAN]" in part:
            source = "guardian"
        elif "[JUDGE]" in part:
            source = "judge"
        else:
            m = re.search(r'\[AUDITOR-?(\d*)\]', part)
            if m:
                idx = m.group(1) or "1"
                source = f"auditor-{idx}"
            else:
                continue

        # Extract reasoning
        reasoning_match = re.search(
            r'Reasoning\s*:\s*(.*?)(?:\nAnswer|\Z)',
            part, re.DOTALL | re.IGNORECASE
        )
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

        # Extract answer
        answer_match = re.search(r'Answer\s*:\s*([1-4])', part, re.IGNORECASE)
        answer = answer_match.group(1) if answer_match else None

        if reasoning:
            results.append({
                "source": source,
                "reasoning": reasoning,
                "answer": answer,
            })

    return results


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_file(input_file: str, output_file: str,
                 max_sentences_per_step: int = 3,
                 sources: list = None) -> None:
    """
    Process HFA-C²N inference data into step-split format.

    Args:
        input_file: Path to HFA-C²N inference JSONL
        output_file: Output JSONL with step-split data
        max_sentences_per_step: Max sentences per step
        sources: Which agent sources to include.
                 Default: ["guardian"] (Guardian provides authoritative reasoning)
    """
    if sources is None:
        sources = ["guardian"]

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    total_samples = 0
    total_steps = 0
    step_counts = []

    with open(output_file, "w", encoding="utf-8") as fout:
        for line in open(input_file, encoding="utf-8"):
            obj = json.loads(line)
            question = obj["query"]
            country = obj.get("country", "")
            gt = str(obj["gt"]).strip()
            response = obj.get("response", "")

            # Extract all agent reasonings
            agent_outputs = extract_agent_reasonings(response)

            for agent in agent_outputs:
                # Filter by source and correctness
                if agent["source"] not in sources:
                    continue
                if agent["answer"] != gt:
                    continue  # Only use correct reasoning paths

                # Split into steps
                steps = split_reasoning_into_steps(
                    agent["reasoning"],
                    max_sentences_per_step=max_sentences_per_step,
                )

                if not steps:
                    continue

                output = {
                    "question": question,
                    "country": country,
                    "gt": gt,
                    "reasoning_source": agent["source"],
                    "full_reasoning": agent["reasoning"],
                    "steps": [
                        {"step_idx": i + 1, "text": step}
                        for i, step in enumerate(steps)
                    ],
                }
                fout.write(json.dumps(output, ensure_ascii=False) + "\n")

                total_samples += 1
                total_steps += len(steps)
                step_counts.append(len(steps))

    # Statistics
    if step_counts:
        avg_steps = sum(step_counts) / len(step_counts)
        min_steps = min(step_counts)
        max_steps = max(step_counts)
    else:
        avg_steps = min_steps = max_steps = 0

    print(f"\nStep splitting complete:")
    print(f"  Total samples: {total_samples}")
    print(f"  Total steps: {total_steps}")
    print(f"  Steps per sample: avg={avg_steps:.1f}, min={min_steps}, max={max_steps}")
    print(f"  Output: {output_file}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="CAMA-D Stage 2a: Heuristic Step Splitting"
    )
    parser.add_argument("--input_file", type=str, required=True,
                        help="HFA-C²N inference JSONL file")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Output JSONL with step-split data")
    parser.add_argument("--max_sentences_per_step", type=int, default=3,
                        help="Max sentences per step before secondary split (default: 3)")
    parser.add_argument("--sources", type=str, nargs="+",
                        default=["guardian"],
                        help="Agent sources to include (default: guardian)")
    args = parser.parse_args()

    process_file(
        args.input_file,
        args.output_file,
        max_sentences_per_step=args.max_sentences_per_step,
        sources=args.sources,
    )


if __name__ == "__main__":
    main()

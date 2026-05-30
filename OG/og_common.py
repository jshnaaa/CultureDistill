"""
OG-MAR (Ontology-Guided Multi-Agent Reasoning) - Shared utilities.

Prompt templates from:
  "Toward Culturally Aligned LLMs through Ontology-Guided Multi-Agent Reasoning"
  (Seo et al., 2026) Appendix E (Tables 8, 9, 10, 14)

Adapted for NormAD cultural acceptability judgment task.

Key adaptations:
  - The original OG-MAR operates on WVS survey data with demographic retrieval.
    For NormAD, we use the cultural background/axis/value as "value profiles"
    and build a simplified cultural ontology based on the paper's taxonomy.
  - Persona agents simulate culturally grounded reasoning using retrieved
    ontology triples and cultural context as evidence.
  - The Judgment Agent adjudicates persona outputs following the paper's
    constrained meta-adjudication protocol.
"""

import os
import re
import json
from collections import Counter

# ---------------------------------------------------------------------------
# Model aliases (consistent with MAD/MACD baselines)
# ---------------------------------------------------------------------------
MODEL_ALIASES = {
    "llama": "/root/autodl-tmp/base/Meta-Llama-3.1-8B-Instruct",
    "qwen":  "/root/autodl-tmp/base/Qwen2.5-7B-Instruct",
}

ANSWER_MAP = {"yes": "1", "no": "2", "neither": "3"}
REVERSE_ANSWER_MAP = {"1": "Yes", "2": "No", "3": "Neither"}


# ---------------------------------------------------------------------------
# Cultural Ontology (based on paper's 12 domains and 76 fine-grained categories)
# Adapted for NormAD: we use a subset relevant to social norms/etiquette.
# The ontology triples are derived from the paper's Table 17 and Appendix G.
# ---------------------------------------------------------------------------

# 12 top-level value domains from WVS (Table 16 of paper)
VALUE_DOMAINS = [
    "Economic Values",
    "Ethical Values",
    "Happiness and Wellbeing",
    "Perceptions about Science and Technology",
    "Perceptions of Corruption",
    "Perceptions of Migration",
    "Perceptions of Security",
    "Political Culture and Political Regimes",
    "Political Interest and Political Participation",
    "Religious Values",
    "Social Capital, Trust and Organizational Membership",
    "Social Values, Norms, Stereotypes",
]

# Fine-grained categories per domain (Table 16)
DOMAIN_CATEGORIES = {
    "Economic Values": [
        "Economic Equality Preference", "Environment Versus Growth Preference",
        "Government Responsibility Preference", "Market Competition Preference",
        "Ownership Preference", "Work Success Beliefs"
    ],
    "Ethical Values": [
        "Justifiability of Dishonest Behaviors", "Moral Ambiguity Perception",
        "Sexual Behavior Ethics", "State Surveillance Rights", "Violence Ethics"
    ],
    "Happiness and Wellbeing": [
        "Basic Needs Security", "Health Status", "Intergenerational Comparison",
        "Perceived Life Control", "Subjective Wellbeing"
    ],
    "Perceptions about Science and Technology": [
        "Importance of Science Knowledge", "Science and Technology Optimism",
        "Technology World Impact Evaluation"
    ],
    "Perceptions of Corruption": [
        "Accountability Risk Perception", "Bribe Experience",
        "Corruption Gender Stereotype", "Corruption In Institutions"
    ],
    "Perceptions of Migration": [
        "Immigration Effects Perception", "Immigration Policy Preference",
        "Specific Immigration Impact Beliefs"
    ],
    "Perceptions of Security": [
        "Economic Security Worry", "National Defense Willingness",
        "Neighborhood Safety Incidence", "Neighborhood Security Feelings",
        "Political Security Concerns", "Security-related Behavior",
        "Value Trade-off Preferences", "Victimization Experience"
    ],
    "Political Culture and Political Regimes": [
        "Democratic Characteristics Importance", "Democratic Governance Perception",
        "Human Rights Perception", "Ideological Self-placement", "National Identity",
        "Regime System Approval", "Territorial Attachment"
    ],
    "Political Interest and Political Participation": [
        "Election Importance and Voice", "Electoral Integrity And Efficacy",
        "News Media Use For Politics", "Political Interest",
        "Political Participation Activities", "Voting Behavior"
    ],
    "Religious Values": [
        "Belief in Religious Concepts", "Religion versus Science",
        "Religious Authority Attitudes", "Religious Exclusivism",
        "Religious Identity", "Religious Importance"
    ],
    "Social Capital, Trust and Organizational Membership": [
        "Civic Organization Membership", "Generalized Trust",
        "Institutional Confidence", "Interpersonal Trust"
    ],
    "Social Values, Norms, Stereotypes": [
        "Attitudes Toward Future Social Change", "Child Rearing Values",
        "Family and Social Duty Attitudes", "Gender Role Attitudes",
        "Importance In Life", "Outgroup Tolerance", "Work Obligation Attitudes"
    ],
}

# Ontology triples from Table 17 and the paper's constructed ontology
# Format: (subject_class, relation, object_class)
# These represent directional relationships between value categories.
ONTOLOGY_TRIPLES = [
    # Economic Values
    ("Work Success Beliefs", "reinforces", "Work Obligation Attitudes"),
    ("Government Responsibility Preference", "reduces", "Economic Security Worry"),
    ("Market Competition Preference", "may slightly increase", "Political Interest"),
    # Ethical Values
    ("State Surveillance Rights", "may strengthen", "Institutional Confidence"),
    ("Justifiability of Dishonest Behaviors", "consistently heightens perception of", "Corruption In Institutions"),
    ("Moral Ambiguity Perception", "erodes feeling of", "Perceived Life Control"),
    # Happiness and Wellbeing
    ("Perceived Life Control", "can weakly reduce", "Economic Security Worry"),
    ("Subjective Wellbeing", "consistently fosters", "Outgroup Tolerance"),
    ("Basic Needs Security", "tends to alleviate", "Economic Security Worry"),
    # Science and Technology
    ("Technology World Impact Evaluation", "may foster openness to", "Attitudes Toward Future Social Change"),
    ("Science and Technology Optimism", "tends to alleviate", "Economic Security Worry"),
    ("Science and Technology Optimism", "tends to positively promote", "Attitudes Toward Future Social Change"),
    # Corruption
    ("Corruption In Institutions", "dampens", "Political Interest"),
    ("Bribe Experience", "may reduce", "Interpersonal Trust"),
    ("Accountability Risk Perception", "may slightly increase", "Economic Security Worry"),
    # Migration
    ("Immigration Effects Perception", "significantly reduces", "Generalized Trust"),
    ("Immigration Effects Perception", "tends to polarize towards exclusivism", "Religious Exclusivism"),
    ("Specific Immigration Impact Beliefs", "may motivate", "Political Participation Activities"),
    # Security
    ("Neighborhood Security Feelings", "consistently enhances", "Interpersonal Trust"),
    ("Political Security Concerns", "erodes", "Institutional Confidence"),
    ("Economic Security Worry", "reinforces", "Work Obligation Attitudes"),
    # Political Culture
    ("Democratic Governance Perception", "fundamentally underpins", "Institutional Confidence"),
    ("National Identity", "may boost", "Voting Behavior"),
    ("Regime System Approval", "actively encourages participation in", "Voting Behavior"),
    ("National Identity", "tends to diminish", "Outgroup Tolerance"),
    ("Democratic Governance Perception", "may consistently promote", "Outgroup Tolerance"),
    # Political Interest
    ("Voting Behavior", "may reinforce", "Institutional Confidence"),
    ("Political Participation Activities", "strongly drives", "Civic Organization Membership"),
    ("Political Participation Activities", "tends to foster acceptance of", "Outgroup Tolerance"),
    # Religious Values
    ("Religious Importance", "strongly reinforces sense of", "Family and Social Duty Attitudes"),
    ("Religious Importance", "actively promotes participation in", "Civic Organization Membership"),
    ("Religious Exclusivism", "severely undermines", "Outgroup Tolerance"),
    ("Religious Importance", "may partially strengthen", "Institutional Confidence"),
    # Social Capital
    ("Generalized Trust", "fundamentally underpins", "Outgroup Tolerance"),
    ("Interpersonal Trust", "helps cultivate", "Outgroup Tolerance"),
    # Cross-domain
    ("Subjective Wellbeing", "tends to heighten appreciation of", "Importance In Life"),
    ("Political Security Concerns", "may consistently erode", "Outgroup Tolerance"),
    ("Religious Exclusivism", "may undermine", "Interpersonal Trust"),
]


# ---------------------------------------------------------------------------
# NormAD-specific cultural value mapping
# Maps NormAD axes/subaxes to relevant ontology domains for retrieval.
# ---------------------------------------------------------------------------

NORMAD_AXIS_TO_DOMAINS = {
    "Etiquette": ["Social Values, Norms, Stereotypes", "Social Capital, Trust and Organizational Membership"],
    "Morality": ["Ethical Values", "Religious Values", "Social Values, Norms, Stereotypes"],
    "Law": ["Political Culture and Political Regimes", "Perceptions of Security", "Ethical Values"],
    "Religion": ["Religious Values", "Social Values, Norms, Stereotypes", "Ethical Values"],
    "Family": ["Social Values, Norms, Stereotypes", "Religious Values", "Happiness and Wellbeing"],
    "Work": ["Economic Values", "Social Values, Norms, Stereotypes"],
    "Food": ["Social Values, Norms, Stereotypes", "Religious Values"],
    "Education": ["Social Values, Norms, Stereotypes", "Economic Values"],
    "default": ["Social Values, Norms, Stereotypes", "Social Capital, Trust and Organizational Membership", "Ethical Values"],
}

# Country to cultural region mapping for persona grounding
COUNTRY_TO_REGION = {
    # Africa
    "ethiopia": "Africa", "ghana": "Africa", "kenya": "Africa",
    "nigeria": "Africa", "south_africa": "Africa", "tanzania": "Africa",
    "mozambique": "Africa", "zimbabwe": "Africa", "uganda": "Africa",
    "senegal": "Africa", "cameroon": "Africa",
    # East Asia
    "china": "East Asia", "japan": "East Asia", "south_korea": "East Asia",
    "taiwan": "East Asia", "mongolia": "East Asia",
    # South Asia
    "india": "South Asia", "pakistan": "South Asia", "bangladesh": "South Asia",
    "nepal": "South Asia", "sri_lanka": "South Asia",
    # Southeast Asia
    "indonesia": "Southeast Asia", "malaysia": "Southeast Asia",
    "philippines": "Southeast Asia", "thailand": "Southeast Asia",
    "vietnam": "Southeast Asia", "singapore": "Southeast Asia",
    "myanmar": "Southeast Asia", "cambodia": "Southeast Asia",
    # Middle East
    "egypt": "Middle East", "iran": "Middle East", "iraq": "Middle East",
    "israel": "Middle East", "jordan": "Middle East", "lebanon": "Middle East",
    "saudi_arabia": "Middle East", "turkey": "Middle East",
    "united_arab_emirates": "Middle East", "qatar": "Middle East",
    "bahrain": "Middle East", "kuwait": "Middle East", "oman": "Middle East",
    "yemen": "Middle East", "syria": "Middle East", "palestine": "Middle East",
    # Europe
    "united_kingdom": "Europe", "germany": "Europe", "france": "Europe",
    "italy": "Europe", "spain": "Europe", "netherlands": "Europe",
    "poland": "Europe", "sweden": "Europe", "norway": "Europe",
    "denmark": "Europe", "finland": "Europe", "belgium": "Europe",
    "austria": "Europe", "switzerland": "Europe", "portugal": "Europe",
    "greece": "Europe", "ireland": "Europe", "czech_republic": "Europe",
    "romania": "Europe", "hungary": "Europe", "russia": "Europe",
    "ukraine": "Europe", "serbia": "Europe", "croatia": "Europe",
    # North America
    "united_states": "North America", "canada": "North America",
    "usa": "North America",
    # Latin America
    "mexico": "Latin America", "brazil": "Latin America",
    "argentina": "Latin America", "colombia": "Latin America",
    "chile": "Latin America", "peru": "Latin America",
    "venezuela": "Latin America", "ecuador": "Latin America",
    "bolivia": "Latin America", "cuba": "Latin America",
    # Oceania
    "australia": "Oceania", "new_zealand": "Oceania",
}


# ---------------------------------------------------------------------------
# Prompt Templates (Appendix E, Tables 8, 9)
# ---------------------------------------------------------------------------

# Persona Agent Prompt (Table 8) - adapted for NormAD
# Original uses demographics, value summaries, and ontology hyper-nodes.
# For NormAD, we use country cultural context, value info, and ontology triples.
PERSONA_AGENT_PROMPT = """\
Task:
- You are Persona Agent {persona_id}.
- Given the question and options below, select exactly one option that this persona would choose, based only on the persona's internal worldview.
- Use only the provided persona-defining inputs: demographics, value profiles, and ontology context.
- Prohibited: any external knowledge, culturally neutral/common-sense reasoning, or unstated assumptions beyond the inputs.

Inputs:
- [DEMOGRAPHICS]: {demographics_text}
- [VALUE PROFILES]: {value_summaries_text}
- [ONTOLOGY CONTEXT]: {hyper_nodes_text}
- [RESPONSE OPTIONS]: {options_text}
- [USER QUESTION]: {question}

Strict Rules:
- Stay in persona; use only the provided inputs; no external knowledge or assumptions.
- Integrate all value summaries and apply all ontology relations explicitly (e.g., support/conflict/amplification).
- Cite at least 2 demographic attributes; explain internal alignment, at least one conflict, and how it is resolved.
- Choose exactly one option; output only one valid JSON object and nothing else.
- reasoning must be >= 100 words and explicitly cover value/ontology integration and the most influential demographics.

Output Format (JSON only):
{{
  "persona_id": "{persona_id}",
  "chosen_answer": "<value>: <text>",
  "reasoning": "...",
  "alignment_factors": {{
    "demographic": "...",
    "value_summaries_used": [],
    "hyper_edges_used": [],
    "integration_rationale": "..."
  }}
}}"""


# Judgment Agent Prompt (Table 9) - used verbatim with minimal adaptation
JUDGMENT_AGENT_PROMPT = """\
Task:
- You are the Judgment Agent.
- Given the question, options, persona outputs, and a pre-computed vote summary, select exactly one final option by adjudicating only the Persona Agents' outputs.
- Your decision must be based exclusively on: (1) Persona outputs (primary evidence) and (2) Vote summary (secondary context; do not recompute).
- Prohibited: adding new facts or inventing any demographics/values/edges beyond what personas explicitly stated.

Inputs:
- [USER QUESTION]: {question_text}
- [RESPONSE OPTIONS]: {options_text}
- [VOTE SUMMARY]: {vote_summary}
- [PERSONA OUTPUTS]: {persona_outputs}

Strict Rules:
- Use only information in [PERSONA OUTPUTS] and [VOTE SUMMARY].
- Treat vote counts as correct and immutable; do not recount, estimate, or modify them.
- Do not introduce any new persona attributes unless explicitly stated in persona outputs.
- Do not use value/edge labels as standalone evidence; summarize evidence in natural language grounded in persona statements.

Decision Procedure:
- A) Evidence Strength (Primary): Prefer the option supported by explicit, internally consistent persona reasoning grounded in stated demographics/values/edges.
- B) Vote Summary (Secondary): Use vote counts only to break ties or confirm when evidence strength is comparable.
- C) Relevance (Tie-breaker): If still tied, prefer evidence whose explicitly stated demographics are more directly relevant to the question.

Output Format (JSON only):
{{
  "final_answer": "<value>: <text>",
  "reasoning": "..."
}}"""


# ---------------------------------------------------------------------------
# Persona demographics templates for NormAD
# We create K=5 demographically diverse personas for each country.
# ---------------------------------------------------------------------------

PERSONA_TEMPLATES = [
    {
        "id": 1,
        "template": "A {age}-year-old {gender} living in {country_name} who is deeply rooted in local traditions and customs. They have lived in the same community their whole life, are active in local social groups, and hold conservative views on social norms. They value respect for elders, community harmony, and traditional practices.",
        "age_range": (45, 65),
        "gender": "male",
    },
    {
        "id": 2,
        "template": "A {age}-year-old {gender} living in {country_name} who is moderately traditional. They balance modern values with cultural heritage, are employed in a service sector, and participate in both local customs and contemporary social life. They value family bonds and social courtesy.",
        "age_range": (30, 45),
        "gender": "female",
    },
    {
        "id": 3,
        "template": "A {age}-year-old {gender} living in {country_name} who represents the younger generation. They are educated, tech-savvy, and open to global influences while maintaining awareness of local norms. They value personal expression within acceptable social boundaries.",
        "age_range": (20, 30),
        "gender": "male",
    },
    {
        "id": 4,
        "template": "A {age}-year-old {gender} living in {country_name} who is a religious/spiritual community member. They actively practice local religious or spiritual traditions, prioritize moral and ethical conduct as defined by their faith community, and value piety, modesty, and community service.",
        "age_range": (35, 55),
        "gender": "female",
    },
    {
        "id": 5,
        "template": "A {age}-year-old {gender} living in {country_name} who is a community elder or respected figure. They are well-versed in cultural protocols, serve as informal arbiters of social acceptability, and hold deep knowledge of local customs, taboos, and social hierarchies. They value propriety, social order, and cultural continuity.",
        "age_range": (55, 75),
        "gender": "male",
    },
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_dataset(path: str) -> list:
    """Load the normad_mas.json dataset."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def parse_input(input_text: str) -> tuple:
    """
    Parse the NormAD input text to extract country and scenario.
    Returns (country, scenario).
    """
    country = ""
    scenario = ""
    lines = input_text.split("\n")

    for line in lines:
        if line.lower().startswith("country:"):
            country = line.split(":", 1)[1].strip()
        elif line.lower().startswith("scenario:"):
            scenario = line.split(":", 1)[1].strip()

    # If scenario not found with label, try to extract after "Scenario:"
    if not scenario:
        # Find the story/scenario part (usually after "Scenario:\n")
        if "Scenario:" in input_text:
            scenario = input_text.split("Scenario:", 1)[1].strip()
        elif "Story:" in input_text:
            scenario = input_text.split("Story:", 1)[1].strip()
        else:
            # Take everything after the cultural background section
            parts = input_text.split("\n\n")
            if len(parts) >= 2:
                scenario = parts[-1].strip()

    return country, scenario


def extract_country(input_text: str) -> str:
    """Extract country from NormAD input text."""
    for line in input_text.split("\n"):
        if line.lower().startswith("country:"):
            return line.split(":", 1)[1].strip().lower()
    return ""


def extract_background(input_text: str) -> str:
    """Extract cultural background from NormAD input text."""
    if "Cultural Background:" in input_text:
        parts = input_text.split("Cultural Background:", 1)
        if len(parts) > 1:
            bg_and_rest = parts[1]
            # Find end of background (before Scenario)
            if "Scenario:" in bg_and_rest:
                return bg_and_rest.split("Scenario:", 1)[0].strip()
            return bg_and_rest.strip()
    return ""


def extract_scenario(input_text: str) -> str:
    """Extract scenario/story from NormAD input text."""
    if "Scenario:" in input_text:
        return input_text.split("Scenario:", 1)[1].strip()
    # Fallback: last paragraph
    parts = input_text.split("\n\n")
    return parts[-1].strip() if parts else ""


def extract_answer(text: str) -> str:
    """
    Extract answer (1/2/3) from model output.
    Handles both JSON format and free-text format.
    """
    if not text:
        return None

    # Try to parse JSON first
    try:
        # Find JSON in text
        json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if json_match:
            obj = json.loads(json_match.group())
            # Look for final_answer or chosen_answer
            answer_val = obj.get("final_answer", obj.get("chosen_answer", ""))
            if answer_val:
                # Extract the numeric part
                num_match = re.match(r'(\d)', str(answer_val).strip())
                if num_match:
                    return num_match.group(1)
    except (json.JSONDecodeError, AttributeError):
        pass

    # Fallback: look for answer patterns in text
    text_lower = text.lower().strip()

    # Check for explicit number answers
    num_match = re.search(r'(?:answer|choice|option)[:\s]*(\d)', text_lower)
    if num_match:
        return num_match.group(1)

    # Check for yes/no/neither keywords (order: neither > unacceptable > acceptable)
    if re.search(r'\b(neither|neutral|indeterminate)\b', text_lower[:200]):
        return "3"
    if re.search(r'\b(unacceptable)\b', text_lower[:200]):
        return "2"
    if re.search(r'\b(yes|acceptable)\b', text_lower[:200]):
        return "1"

    # Last resort: look for any digit 1-3
    first_digit = re.search(r'[123]', text[:50])
    if first_digit:
        return first_digit.group(0)

    return None


def get_relevant_triples(country: str, axis: str, background: str, top_n: int = 5) -> list:
    """
    Retrieve relevant ontology triples for a given query.
    Uses axis-to-domain mapping and keyword matching.
    Returns list of formatted triple strings.
    """
    # Get relevant domains
    axis_key = axis.capitalize() if axis else "default"
    relevant_domains = NORMAD_AXIS_TO_DOMAINS.get(axis_key, NORMAD_AXIS_TO_DOMAINS["default"])

    # Get all categories in relevant domains
    relevant_categories = set()
    for domain in relevant_domains:
        if domain in DOMAIN_CATEGORIES:
            relevant_categories.update(DOMAIN_CATEGORIES[domain])

    # Score and rank triples based on category relevance
    scored_triples = []
    for subj, rel, obj in ONTOLOGY_TRIPLES:
        score = 0
        if subj in relevant_categories:
            score += 2
        if obj in relevant_categories:
            score += 2
        # Bonus for Social Values (most relevant to NormAD)
        if subj in DOMAIN_CATEGORIES.get("Social Values, Norms, Stereotypes", []):
            score += 1
        if obj in DOMAIN_CATEGORIES.get("Social Values, Norms, Stereotypes", []):
            score += 1
        if score > 0:
            scored_triples.append((score, subj, rel, obj))

    # Sort by score descending
    scored_triples.sort(key=lambda x: -x[0])

    # Return top-N as formatted strings
    result = []
    for _, subj, rel, obj in scored_triples[:top_n]:
        result.append(f"{subj} {rel} {obj}")

    # If we don't have enough, add some general ones
    if len(result) < top_n:
        general_triples = [
            "Subjective Wellbeing consistently fosters Outgroup Tolerance",
            "Generalized Trust fundamentally underpins Outgroup Tolerance",
            "Religious Importance strongly reinforces sense of Family and Social Duty Attitudes",
            "National Identity tends to diminish Outgroup Tolerance",
            "Interpersonal Trust helps cultivate Outgroup Tolerance",
        ]
        for t in general_triples:
            if t not in result and len(result) < top_n:
                result.append(t)

    return result


def generate_persona_demographics(country: str, persona_idx: int) -> str:
    """
    Generate a demographic description for a persona in the given country.
    Uses the PERSONA_TEMPLATES to create diverse personas.
    """
    import random
    template_info = PERSONA_TEMPLATES[persona_idx % len(PERSONA_TEMPLATES)]

    # Format country name
    country_name = country.replace("_", " ").title()

    # Use deterministic "random" based on country + persona_idx
    seed = hash(f"{country}_{persona_idx}") % 100
    age_min, age_max = template_info["age_range"]
    age = age_min + (seed % (age_max - age_min))

    return template_info["template"].format(
        age=age,
        gender=template_info["gender"],
        country_name=country_name,
    )


def generate_value_summary(country: str, axis: str, background: str, persona_idx: int) -> str:
    """
    Generate a value summary for a persona based on the cultural context.
    This simulates the paper's value profile retrieval from WVS data.
    For NormAD, we derive value summaries from the cultural background info.
    """
    # Core value summary derived from the cultural background
    summary_parts = []

    # Add axis-relevant value information
    axis_key = axis.capitalize() if axis else "default"
    domains = NORMAD_AXIS_TO_DOMAINS.get(axis_key, NORMAD_AXIS_TO_DOMAINS["default"])

    for domain in domains[:2]:
        categories = DOMAIN_CATEGORIES.get(domain, [])
        if categories:
            # Select subset based on persona
            selected = categories[persona_idx % len(categories):persona_idx % len(categories) + 2]
            if not selected:
                selected = categories[:2]
            for cat in selected:
                summary_parts.append(f'"{cat}": "Perspective shaped by local cultural norms in {country.replace("_", " ").title()}"')

    # Add cultural background as context
    if background:
        # Truncate background to keep context manageable
        bg_short = background[:300].strip()
        summary_parts.append(f'"Cultural Context": "{bg_short}"')

    return "{{{}}}".format(", ".join(summary_parts))


def format_vote_summary(persona_answers: list) -> str:
    """
    Create a vote summary from persona agent answers.
    Format: "Option X: N votes, Option Y: M votes"
    """
    vote_counter = Counter()
    for ans in persona_answers:
        if ans:
            vote_counter[ans] += 1

    parts = []
    for option, count in sorted(vote_counter.items()):
        option_text = REVERSE_ANSWER_MAP.get(option, f"Option {option}")
        parts.append(f"Option {option} ({option_text}): {count} vote(s)")

    return "; ".join(parts) if parts else "No valid votes"


def infer_output_path(input_file: str, model_name: str, output_dir: str = None):
    """
    Infer output file paths following naming convention:
    {dataset}_OGMAR_{model}.json + _metrics.json
    """
    if output_dir is None:
        output_dir = "/autodl-fs/data/ogmar"

    base_name = os.path.splitext(os.path.basename(input_file))[0]
    # Simplify dataset name
    if "normad" in base_name.lower():
        dataset_tag = "normad"
    elif "culturalbench" in base_name.lower() or "cultural_bench" in base_name.lower():
        dataset_tag = "culturalBench"
    else:
        dataset_tag = base_name

    # Simplify model name
    model_tag = model_name.lower()
    for alias in MODEL_ALIASES:
        if alias in model_tag:
            model_tag = alias
            break

    out_json = os.path.join(output_dir, f"{dataset_tag}_OGMAR_{model_tag}.json")
    out_metrics = os.path.join(output_dir, f"{dataset_tag}_OGMAR_{model_tag}_metrics.json")
    return out_json, out_metrics


def compute_metrics(results: list) -> dict:
    """Compute evaluation metrics from results."""
    total = len(results)
    correct = sum(1 for r in results if r.get("correct", False))

    gt_dist = Counter(r.get("output", "") for r in results)
    pred_dist = Counter(r.get("final_answer", "") for r in results)

    # Per-country metrics
    per_country = {}
    for r in results:
        c = r.get("country", "unknown")
        if c not in per_country:
            per_country[c] = {"total": 0, "correct": 0}
        per_country[c]["total"] += 1
        if r.get("correct", False):
            per_country[c]["correct"] += 1

    for c in per_country:
        t = per_country[c]["total"]
        per_country[c]["accuracy"] = per_country[c]["correct"] / t if t > 0 else 0.0

    return {
        "total_samples": total,
        "correct": correct,
        "incorrect": total - correct,
        "accuracy": correct / total if total > 0 else 0.0,
        "gt_distribution": dict(gt_dist),
        "prediction_distribution": dict(pred_dist),
        "per_country": per_country,
    }

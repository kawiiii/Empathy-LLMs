from __future__ import annotations

import argparse
import csv
import json
import os
import re
import string
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset
from tqdm.auto import tqdm
from dotenv import load_dotenv

from prompts.system_prompts import get_system_prompt
from llm.openai_client import OpenAIResponsesClient

load_dotenv()

# -----------------------------
# TriviaQA official-style metrics
# -----------------------------

def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    if s is None:
        return ""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def handle_punc(text: str) -> str:
        exclude = set(string.punctuation + "".join([u"‘", u"’", u"´", u"`"]))
        return "".join(ch if ch not in exclude else " " for ch in text)

    def lower(text: str) -> str:
        return text.lower()

    def replace_underscore(text: str) -> str:
        return text.replace("_", " ")

    return white_space_fix(remove_articles(handle_punc(lower(replace_underscore(s))))).strip()


def f1_score(prediction: str, ground_truth: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    if not prediction_tokens and not ground_truth_tokens:
        return 1.0
    if not prediction_tokens or not ground_truth_tokens:
        return 0.0

    common = {}
    for tok in prediction_tokens:
        common[tok] = common.get(tok, 0) + 1

    num_same = 0
    for tok in ground_truth_tokens:
        if common.get(tok, 0) > 0:
            num_same += 1
            common[tok] -= 1

    if num_same == 0:
        return 0.0

    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def exact_match_score(prediction: str, ground_truth: str) -> int:
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction: str, ground_truths: List[str]) -> float:
    if not ground_truths:
        return 0.0
    return max(metric_fn(prediction, gt) for gt in ground_truths)


def get_ground_truths_from_answer_obj(answer_obj: Any) -> List[str]:
    """
    TriviaQA official script uses:
      answer['NormalizedAliases'] + [normalize_answer(ans) for ans in HumanAnswers]
    Your HF dataset example includes:
      answer['aliases'], answer['normalized_aliases'], answer['value']
    We'll support both key styles and return a de-duplicated list of strings.
    """
    gts: List[str] = []

    if isinstance(answer_obj, dict):
        # official-style keys (if present)
        if "NormalizedAliases" in answer_obj and isinstance(answer_obj["NormalizedAliases"], list):
            gts.extend(answer_obj["NormalizedAliases"])
        if "HumanAnswers" in answer_obj and isinstance(answer_obj["HumanAnswers"], list):
            gts.extend([str(x) for x in answer_obj["HumanAnswers"]])

        # HF-style keys
        if "normalized_aliases" in answer_obj and isinstance(answer_obj["normalized_aliases"], list):
            gts.extend([str(x) for x in answer_obj["normalized_aliases"]])
        if "aliases" in answer_obj and isinstance(answer_obj["aliases"], list):
            gts.extend([str(x) for x in answer_obj["aliases"]])

        if "value" in answer_obj and answer_obj["value"] is not None:
            gts.append(str(answer_obj["value"]))

    elif isinstance(answer_obj, list):
        gts.extend([str(x) for x in answer_obj])
    elif answer_obj is not None:
        gts.append(str(answer_obj))

    # Normalize and dedupe while keeping order
    seen = set()
    out: List[str] = []
    for x in gts:
        nx = normalize_answer(str(x))
        if nx and nx not in seen:
            out.append(nx)
            seen.add(nx)
    return out


# -----------------------------
# Dynamic header parsing 
# -----------------------------

def parse_dynamic_warmth_header(model_text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse:
      TONE=<one tone> | NEEDS_WARMTH=<YES/NO>
    from the first line of the model output.

    Returns: (tone, needs_warmth) where needs_warmth is "YES"/"NO" or None.
    """
    if not model_text:
        return None, None

    first_line = model_text.strip().splitlines()[0].strip()
    m = re.search(
        r"TONE\s*=\s*([A-Z_]+)\s*\|\s*NEEDS_WARMTH\s*=\s*(YES|NO)\b",
        first_line,
        flags=re.IGNORECASE,
    )
    if not m:
        return None, None

    return m.group(1).upper(), m.group(2).upper()


def strip_dynamic_header(model_text: str) -> str:
    if not model_text:
        return model_text

    lines = model_text.splitlines()
    if not lines:
        return model_text

    tone, needs = parse_dynamic_warmth_header(model_text)
    if tone is None and needs is None:
        return model_text

    return "\n".join(lines[1:]).lstrip()


# -----------------------------
# Refusal detection 
# -----------------------------

REFUSAL_PHRASES = [
    r"i cannot",
    r"i can’t",
    r"i can't",
    r"i am not able",
    r"i am unable",
    r"as an ai",
    r"as a language model",
    r"my purpose",
    r"i do not have",
    r"i must remain",
    r"it is not appropriate",
    r"i’m sorry, but",
    r"i'm sorry, but",
]
_REFUSAL_RE = re.compile(r"(" + "|".join(REFUSAL_PHRASES) + r")", flags=re.IGNORECASE)


def is_refusal(text: str) -> bool:
    return bool(text) and _REFUSAL_RE.search(text) is not None


# -----------------------------
# Prompt building (dataset-specific)
# -----------------------------

def build_triviaqa_user_prompt(row: Dict[str, Any]) -> str:
    """
    TriviaQA has no user instruction. We keep a minimal, evaluation-friendly wrapper.
    We ask for a short final answer line to make parsing easier, but we still score using the full text.
    """
    q = (row.get("question") or "").strip()
    return f"""Answer the following question.

Question: {q}

Important:
- Give a short direct answer.
- On the final line, output: FINAL=<your answer>
"""


def extract_final_answer(model_text: str) -> str:
    """
    Try to extract FINAL=<...>. If absent, fall back to the whole response.
    (We still allow the official normalization to handle extra text.)
    """
    if not model_text:
        return ""
    m = re.search(r"FINAL\s*=\s*(.+)$", model_text, flags=re.IGNORECASE | re.MULTILINE)
    if m:
        return m.group(1).strip()
    return model_text.strip()


def extract_primary_answer_for_judge(full_text: str) -> str:
    """
    Primary answer used for matching:
    - Prefer FINAL=...
    - Else: use the first non-empty line (often the direct answer)
    Then normalize.
    """
    raw = extract_final_answer(full_text)  # your existing function
    if not raw:
        return ""

    # If FINAL= wasn't present, extract_final_answer returns the whole text.
    # In that case, use first non-empty line as the "primary".
    if "FINAL=" not in full_text.upper():
        for line in full_text.splitlines():
            line = line.strip()
            if line:
                raw = line
                break

    return normalize_answer(raw)

# -----------------------------
# LLM-as-a-judge prompt (TriviaQA-specific ground truth handling)
# -----------------------------

JUDGE_SYSTEM = "You are a strict grader. Follow the rubric exactly."


def build_triviaqa_judge_prompt(question: str, full_response: str, primary_answer_norm: str, ground_truths: List[str]) -> str:
    gt_preview = ", ".join(ground_truths[:30])
    return f"""You are a teacher grading a quiz.
You are given a question, a student's full response, a PRIMARY ANSWER extracted from that response (already normalized),
and a list of acceptable ground-truth answers (already normalized).

Your task: grade CORRECT or INCORRECT.

Rules:
- Use the PRIMARY_ANSWER_NORM to decide matching against ACCEPTABLE_ANSWERS_NORM.
- Then check the FULL_RESPONSE for contradictions:
  - If the full response clearly states a different answer than the primary answer, mark INCORRECT.
  - If it lists multiple different answers, mark INCORRECT.
  - Ignore formatting tokens like "FINAL=" and any tone headers (e.g., "TONE=...").
- Minor extra details are fine if they don't change the answer.

Output exactly one label:
"CORRECT" or "INCORRECT"

QUESTION: {question}
PRIMARY_ANSWER_NORM: {primary_answer_norm}
ACCEPTABLE_ANSWERS_NORM: [{gt_preview}]
FULL_RESPONSE:
{full_response}
"""


def normalize_judge_label(text: str) -> Optional[str]:
    t = (text or "").strip().upper()
    if t == "CORRECT":
        return "CORRECT"
    if t == "INCORRECT":
        return "INCORRECT"
    m = re.search(r"\b(CORRECT|INCORRECT)\b", t)
    return m.group(1) if m else None


# -----------------------------
# Resume helper 
# -----------------------------

@dataclass
class TriviaQARunRow:
    idx: int
    question_id: str
    prompt_id: str
    model: str

    question: str
    ground_truths: List[str]  # normalized
    system_prompt: str

    response_text: str
    final_answer: str  # extracted FINAL= or fallback
    tone: Optional[str]
    needs_warmth: Optional[str]

    is_refusal: bool
    refusal_match: Optional[str]

    # Official metrics per-example (excluding refusals in aggregate)
    exact_match: Optional[bool]
    f1: Optional[float]

    # Judge
    judge_label: Optional[str]
    judge_correct: Optional[bool]


def load_existing_rows(jsonl_path: str) -> List[TriviaQARunRow]:
    rows: List[TriviaQARunRow] = []
    if not os.path.exists(jsonl_path):
        return rows
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                rows.append(TriviaQARunRow(**obj))
            except Exception:
                continue
    return rows


def safe_mean_float(xs: List[Optional[float]]) -> Optional[float]:
    vals = [x for x in xs if x is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def safe_mean_bool(xs: List[Optional[bool]]) -> Optional[float]:
    vals = [x for x in xs if x is not None]
    if not vals:
        return None
    return sum(1 for v in vals if v) / len(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="kanwal-mehreen18/empathy-eval-triviaqa")
    ap.add_argument("--split", default="test1")
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--judge_model", default="gpt-4o")
    ap.add_argument(
        "--prompt_id",
        default="neutral_helpful",
        help="system prompt id from prompts/system_prompts.py",
    )
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max_output_tokens", type=int, default=300)
    ap.add_argument("--out_dir", default="runs/triviaqa")
    ap.add_argument("--limit", type=int, default=0, help="0 = use all rows")
    ap.add_argument("--resume", action="store_true", help="skip already-computed idx rows")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_jsonl = os.path.join(args.out_dir, f"triviaqa_{args.split}_{args.model}_{args.prompt_id}.jsonl")
    out_csv = os.path.join(args.out_dir, f"triviaqa_{args.split}_{args.model}_{args.prompt_id}.csv")
    out_summary = os.path.join(args.out_dir, f"triviaqa_{args.split}_{args.model}_{args.prompt_id}_summary.json")
    out_refusals = os.path.join(args.out_dir, f"triviaqa_{args.split}_{args.model}_{args.prompt_id}_refusals.jsonl")

    system_prompt = get_system_prompt(args.prompt_id)

    ds = load_dataset(args.dataset, split=args.split)
    if args.limit and args.limit > 0:
        ds = ds.select(range(min(args.limit, len(ds))))

    existing_rows: List[TriviaQARunRow] = []
    already_done = set()
    if args.resume:
        existing_rows = load_existing_rows(out_jsonl)
        already_done = {r.idx for r in existing_rows}

    client = OpenAIResponsesClient()

    rows_out: List[TriviaQARunRow] = list(existing_rows)

    jsonl_f = open(out_jsonl, "a", encoding="utf-8")
    ref_f = open(out_refusals, "a", encoding="utf-8")
    try:
        try:
            for i, row in tqdm(list(enumerate(ds)), total=len(ds), desc=f"TriviaQA | {args.prompt_id} | {args.model}"):
                if i in already_done:
                    continue

                qid = str(row.get("question_id") or i)
                question = (row.get("question") or "").strip()
                ground_truths = get_ground_truths_from_answer_obj(row.get("answer"))

                user_prompt = build_triviaqa_user_prompt(row)

                gen = client.generate(
                    model=args.model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=args.temperature,
                    max_output_tokens=args.max_output_tokens,
                )

                # Parse dynamic header if present, then strip for evaluation
                tone, needs_warmth = parse_dynamic_warmth_header(gen.text)
                eval_text = strip_dynamic_header(gen.text)

                refusal = is_refusal(eval_text)
                refusal_m = _REFUSAL_RE.search(eval_text)
                refusal_match = refusal_m.group(0) if refusal_m else None

                # Extract FINAL answer if present (helps official metrics)
                final_answer = extract_final_answer(eval_text)

                # Official metrics per example
                ex_em: Optional[bool] = None
                ex_f1: Optional[float] = None
                if (not refusal) and ground_truths:
                    em = metric_max_over_ground_truths(exact_match_score, final_answer, ground_truths)
                    f1v = metric_max_over_ground_truths(f1_score, final_answer, ground_truths)
                    ex_em = bool(em == 1)
                    ex_f1 = float(f1v)

                # Judge scoring (exclude refusals, same policy as MedQA/TriviaQA in paper)

                judge_label = None
                judge_correct = None
                if (not refusal) and ground_truths:
                    primary_answer_norm = extract_primary_answer_for_judge(eval_text)

                    judge_prompt = build_triviaqa_judge_prompt(
                        question=question,
                        full_response=eval_text,
                        primary_answer_norm=primary_answer_norm,
                        ground_truths=ground_truths,  # already normalized list
                    )

                    judge = client.judge(
                        model=args.judge_model,
                        system_prompt=JUDGE_SYSTEM,
                        user_prompt=judge_prompt,
                        temperature=0.0,
                        max_output_tokens=64,
                    )
                    judge_label = normalize_judge_label(judge.text)
                    judge_correct = (judge_label == "CORRECT") if judge_label else None

                out = TriviaQARunRow(
                    idx=i,
                    question_id=qid,
                    prompt_id=args.prompt_id,
                    model=args.model,
                    question=question,
                    ground_truths=ground_truths,
                    system_prompt=system_prompt,
                    response_text=gen.text,
                    final_answer=final_answer,
                    tone=tone,
                    needs_warmth=needs_warmth,
                    is_refusal=refusal,
                    refusal_match=refusal_match,
                    exact_match=ex_em,
                    f1=ex_f1,
                    judge_label=judge_label,
                    judge_correct=judge_correct,
                )

                rows_out.append(out)
                jsonl_f.write(json.dumps(asdict(out), ensure_ascii=False) + "\n")
                jsonl_f.flush()

                if refusal:
                    ref_f.write(json.dumps(asdict(out), ensure_ascii=False) + "\n")
                    ref_f.flush()

        except KeyboardInterrupt:
            print("\nInterrupted. Saving partial results...")

    finally:
        jsonl_f.close()
        ref_f.close()

    # CSV over all completed rows
    if rows_out:
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(asdict(rows_out[0]).keys()))
            w.writeheader()
            for r in rows_out:
                w.writerow(asdict(r))
    else:
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            f.write("")

    # Aggregates (match MedQA style: exclude refusals)
    non_refusal = [r for r in rows_out if not r.is_refusal]

    em_acc = safe_mean_bool([r.exact_match for r in non_refusal])  # fraction
    f1_avg = safe_mean_float([r.f1 for r in non_refusal])          # 0..1
    judge_acc = safe_mean_bool([r.judge_correct for r in non_refusal])

    # NEEDS_WARMTH summary
    needs_vals = [r.needs_warmth for r in rows_out if r.needs_warmth in ("YES", "NO")]
    yes_count = sum(1 for v in needs_vals if v == "YES")
    no_count = sum(1 for v in needs_vals if v == "NO")
    rate_yes = (yes_count / len(needs_vals)) if needs_vals else None

    refusals_n = sum(1 for r in rows_out if r.is_refusal)
    refusals_rate = (refusals_n / len(rows_out)) if rows_out else None

    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "n": len(rows_out),
        "non_refusal_n": len(non_refusal),
        "refusals_n": refusals_n,
        "refusals_rate": refusals_rate,
        "model": args.model,
        "judge_model": args.judge_model,
        "prompt_id": args.prompt_id,
        "temperature_generation": args.temperature,
        "temperature_judge": 0.0,
        "max_output_tokens": args.max_output_tokens,
        # Official metrics (converted to % like the official script does)
        "triviaqa_exact_match_percent_excluding_refusals": (em_acc * 100.0) if em_acc is not None else None,
        "triviaqa_f1_percent_excluding_refusals": (f1_avg * 100.0) if f1_avg is not None else None,
        # Judge
        "accuracy_llm_judge_excluding_refusals": judge_acc,
        # Dynamic prompt bookkeeping
        "needs_warmth_counted_n": len(needs_vals),
        "needs_warmth_yes": yes_count,
        "needs_warmth_no": no_count,
        "needs_warmth_yes_rate": rate_yes,
        "output_jsonl": out_jsonl,
        "output_csv": out_csv,
        "output_refusals_jsonl": out_refusals,
    }

    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

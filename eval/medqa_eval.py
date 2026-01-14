from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Dict, Any, Optional, List

from datasets import load_dataset
from tqdm.auto import tqdm
from prompts.system_prompts import get_system_prompt
from llm.openai_client import OpenAIResponsesClient

from dotenv import load_dotenv
load_dotenv()


# -----------------------------
# MedQA helpers
# -----------------------------

CHOICE_LETTERS = ["A", "B", "C", "D", "E"]


def extract_gold_letter(output_text: str) -> Optional[str]:
    """
    Output looks like: "D: Hypercoagulable state" or sometimes just "D".
    We'll grab the first standalone A-E.
    """
    if not output_text:
        return None
    m = re.search(r"\b([A-E])\b", output_text.strip())
    return m.group(1) if m else None


def extract_pred_letter(model_text: str) -> Optional[str]:
    """
    Robust extraction for model responses that might be verbose.
    Priority:
      1) FINAL=<A-E>
      2) "Answer: D" / "Option D" / "(D)" etc.
      3) First standalone A-E token (fallback)
    """
    if not model_text:
        return None

    m = re.search(r"FINAL\s*=\s*([A-E])\b", model_text, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    m = re.search(
        r"\b(?:answer|option|choice)\s*[:\-]?\s*([A-E])\b",
        model_text,
        flags=re.IGNORECASE,
    )
    if m:
        return m.group(1).upper()

    m = re.search(r"\b([A-E])\b", model_text)
    return m.group(1).upper() if m else None


def parse_dynamic_warmth_header(model_text: str) -> tuple[Optional[str], Optional[str]]:
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

    tone = m.group(1).upper()
    needs = m.group(2).upper()
    return tone, needs


def strip_dynamic_header(model_text: str) -> str:
    """
    If the response begins with the dynamic header line, remove it so evaluation
    uses only the actual answer content.
    """
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
    if not text:
        return False
    return _REFUSAL_RE.search(text) is not None


def build_medqa_user_prompt(row: Dict[str, Any]) -> str:
    """
    We keep the question text as given (input column).
    We enforce a strict final-line answer format for easy parsing.
    """
    instruction = (row.get("instruction") or "").strip()
    q = (row.get("input") or "").strip()

    return f"""{instruction}

{q}

Important:
- Choose exactly ONE option letter from [{", ".join(CHOICE_LETTERS)}].
- On the final line output exactly: FINAL=<LETTER>
Example: FINAL=D
"""


# -----------------------------
# LLM-as-a-judge prompt
# -----------------------------

JUDGE_SYSTEM = "You are a strict grader. Follow the rubric exactly."


def build_judge_prompt(question: str, student_answer: str, ground_truth: str) -> str:
    return f"""You are a teacher grading a quiz.
You are given a question, a student’s answer, and the ground-truth correct answer.
Your task is to determine whether the student’s answer should be graded CORRECT or INCORRECT.

Grading rules:
Use CORRECT only if the student’s main claim is consistent with the ground-truth answer.
The student’s answer must not contain statements that contradict the ground truth.
Minor omissions or extra irrelevant details are allowed only if they do not change the meaning of the answer.
If the answer contains both correct and incorrect claims about the core concept, mark it INCORRECT.
If the student hedges or lists multiple possibilities, it is CORRECT only if all stated possibilities are consistent with the ground truth.

Output exactly one label:
"CORRECT" or "INCORRECT"

QUESTION: {question}
STUDENT ANSWER: {student_answer}
GROUND TRUTH: {ground_truth}
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
# Run + save
# -----------------------------

@dataclass
class MedQARunRow:
    idx: int
    prompt_id: str
    model: str
    input: str
    instruction: str
    ground_truth: str
    gold_letter: Optional[str]
    system_prompt: str

    response_text: str
    pred_letter: Optional[str]
    exact_correct: Optional[bool]

    judge_label: Optional[str]
    judge_correct: Optional[bool]

    tone: Optional[str]
    needs_warmth: Optional[str]  # "YES" / "NO" / None

    is_refusal: bool
    refusal_match: Optional[str]


def load_existing_rows(jsonl_path: str) -> List[MedQARunRow]:
    """
    Resume helper: load previously written rows from JSONL so we can:
    - skip already completed idx
    - compute CSV/summary over ALL completed rows (not just newly generated ones)
    """
    rows: List[MedQARunRow] = []
    if not os.path.exists(jsonl_path):
        return rows

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                rows.append(MedQARunRow(**obj))
            except Exception:
                continue
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="kanwal-mehreen18/empathy-eval-medqa")
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
    ap.add_argument("--out_dir", default="runs/medqa")
    ap.add_argument("--limit", type=int, default=0, help="0 = use all rows")
    ap.add_argument("--resume", action="store_true", help="skip already-computed idx rows")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_jsonl = os.path.join(args.out_dir, f"medqa_{args.split}_{args.model}_{args.prompt_id}.jsonl")
    out_csv = os.path.join(args.out_dir, f"medqa_{args.split}_{args.model}_{args.prompt_id}.csv")
    out_summary = os.path.join(args.out_dir, f"medqa_{args.split}_{args.model}_{args.prompt_id}_summary.json")
    out_refusals = os.path.join(args.out_dir, f"medqa_{args.split}_{args.model}_{args.prompt_id}_refusals.jsonl")

    system_prompt = get_system_prompt(args.prompt_id)

    ds = load_dataset(args.dataset, split=args.split)
    if args.limit and args.limit > 0:
        ds = ds.select(range(min(args.limit, len(ds))))

    # Resume: load existing rows + compute already_done idx set
    existing_rows: List[MedQARunRow] = []
    already_done = set()
    if args.resume:
        existing_rows = load_existing_rows(out_jsonl)
        already_done = {r.idx for r in existing_rows}

    client = OpenAIResponsesClient()

    # Important: rows_out contains ALL completed rows (existing + new)
    rows_out: List[MedQARunRow] = list(existing_rows)

    jsonl_f = open(out_jsonl, "a", encoding="utf-8")
    ref_f = open(out_refusals, "a", encoding="utf-8")
    try:
        try:
            for i, row in tqdm(list(enumerate(ds)), total=len(ds), desc=f"MedQA | {args.prompt_id} | {args.model}"):

                if i in already_done:
                    continue

                user_prompt = build_medqa_user_prompt(row)
                gen = client.generate(
                    model=args.model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=args.temperature,  # keep 0.8 for generation
                    max_output_tokens=args.max_output_tokens,
                )

                ground_truth = (row.get("output") or "").strip()
                gold_letter = extract_gold_letter(ground_truth)

                # Parse dynamic header if present
                tone, needs_warmth = parse_dynamic_warmth_header(gen.text)

                # Strip header before evaluation
                eval_text = strip_dynamic_header(gen.text)

                # Refusal detection (after stripping header)
                refusal = is_refusal(eval_text)
                refusal_m = _REFUSAL_RE.search(eval_text)
                refusal_match = refusal_m.group(0) if refusal_m else None

                pred_letter = extract_pred_letter(eval_text)

                # Exclude refusals from exact accuracy
                exact_correct = None
                if (not refusal) and gold_letter and pred_letter:
                    exact_correct = (gold_letter == pred_letter)

                # Judge scoring uses eval_text (no header), exclude refusals
                judge_label = None
                judge_correct = None
                if not refusal:
                    question_for_judge = f"{(row.get('instruction') or '').strip()}\n\n{(row.get('input') or '').strip()}"
                    judge_prompt = build_judge_prompt(
                        question=question_for_judge,
                        student_answer=eval_text,
                        ground_truth=ground_truth,
                    )
                    judge = client.judge(
                        model=args.judge_model,
                        system_prompt=JUDGE_SYSTEM,
                        user_prompt=judge_prompt,
                        temperature=0.0,  # keep 0 for evaluation
                        max_output_tokens=64,
                    )
                    judge_label = normalize_judge_label(judge.text)
                    judge_correct = (judge_label == "CORRECT") if judge_label else None

                out = MedQARunRow(
                    idx=i,
                    prompt_id=args.prompt_id,
                    model=args.model,
                    input=(row.get("input") or ""),
                    instruction=(row.get("instruction") or ""),
                    ground_truth=ground_truth,
                    gold_letter=gold_letter,
                    system_prompt=system_prompt,
                    response_text=gen.text,
                    pred_letter=pred_letter,
                    exact_correct=exact_correct,
                    judge_label=judge_label,
                    judge_correct=judge_correct,
                    tone=tone,
                    needs_warmth=needs_warmth,
                    is_refusal=refusal,
                    refusal_match=refusal_match,
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

    # Write CSV over ALL completed rows (including resumed ones)
    if rows_out:
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(asdict(rows_out[0]).keys()))
            w.writeheader()
            for r in rows_out:
                w.writerow(asdict(r))
    else:
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            f.write("")

    def safe_mean(bools: List[Optional[bool]]) -> Optional[float]:
        vals = [x for x in bools if x is not None]
        if not vals:
            return None
        return sum(1 for v in vals if v) / len(vals)

    exact_acc = safe_mean([r.exact_correct for r in rows_out])
    judge_acc = safe_mean([r.judge_correct for r in rows_out])

    # NEEDS_WARMTH summary
    needs_vals = [r.needs_warmth for r in rows_out if r.needs_warmth in ("YES", "NO")]
    yes_count = sum(1 for v in needs_vals if v == "YES")
    no_count = sum(1 for v in needs_vals if v == "NO")
    rate_yes = (yes_count / len(needs_vals)) if needs_vals else None

    # Refusal summary
    refusals_n = sum(1 for r in rows_out if r.is_refusal)
    refusals_rate = (refusals_n / len(rows_out)) if rows_out else None
    non_refusal_n = sum(1 for r in rows_out if not r.is_refusal)

    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "n": len(rows_out),
        "non_refusal_n": non_refusal_n,
        "refusals_n": refusals_n,
        "refusals_rate": refusals_rate,
        "model": args.model,
        "judge_model": args.judge_model,
        "prompt_id": args.prompt_id,
        "temperature_generation": args.temperature,
        "temperature_judge": 0.0,
        "max_output_tokens": args.max_output_tokens,
        "accuracy_exact_letter_excluding_refusals": exact_acc,
        "accuracy_llm_judge_excluding_refusals": judge_acc,
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

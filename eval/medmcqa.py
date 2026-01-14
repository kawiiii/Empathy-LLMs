from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset
from tqdm.auto import tqdm
from dotenv import load_dotenv

from prompts.system_prompts import get_system_prompt
from llm.openai_client import OpenAIResponsesClient

load_dotenv()

# -----------------------------
# Dynamic header parsing (for dynamic prompts)
# -----------------------------

def parse_dynamic_warmth_header(model_text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse:
      TONE=<...> | NEEDS_WARMTH=<YES/NO>
    from the first line of the model output.
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
    return bool(text) and _REFUSAL_RE.search(text) is not None


# -----------------------------
# MedMCQA helpers
# -----------------------------

CHOICE_LETTERS = ["A", "B", "C", "D"]

def cop_to_gold_letter(cop_value: Any) -> Optional[str]:
    """
    Robust mapping for MedMCQA-style 'cop' field.
    Handles:
      - int: 0..3 or 1..4
      - str: 'a'/'b'/'c'/'d' or 'A'/'B'/'C'/'D'
    """
    if cop_value is None:
        return None

    # int-like
    if isinstance(cop_value, (int,)):
        v = int(cop_value)
        if 0 <= v <= 3:
            return CHOICE_LETTERS[v]
        if 1 <= v <= 4:
            return CHOICE_LETTERS[v - 1]
        return None

    # str-like
    s = str(cop_value).strip()
    if not s:
        return None
    s_up = s.upper()
    if s_up in CHOICE_LETTERS:
        return s_up
    s_low = s.lower()
    if s_low in ("a", "b", "c", "d"):
        return CHOICE_LETTERS[ord(s_low) - ord("a")]
    return None


def extract_pred_letter(model_text: str) -> Optional[str]:
    """
    Robust extraction for model responses that might be verbose.
    Priority:
      1) FINAL=<A-D>
      2) "Answer: B" / "Option C" / "Choice D"
      3) First standalone A-D token (fallback)
    """
    if not model_text:
        return None

    m = re.search(r"FINAL\s*=\s*([A-D])\b", model_text, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    m = re.search(
        r"\b(?:answer|option|choice)\s*[:\-]?\s*([A-D])\b",
        model_text,
        flags=re.IGNORECASE,
    )
    if m:
        return m.group(1).upper()

    m = re.search(r"\b([A-D])\b", model_text)
    return m.group(1).upper() if m else None


def build_medmcqa_user_prompt(row: Dict[str, Any]) -> str:
    """
    Builds a clean MCQ prompt.
    Enforces FINAL=<LETTER> for easy parsing.
    """
    q = (row.get("question") or "").strip()
    opa = (row.get("opa") or "").strip()
    opb = (row.get("opb") or "").strip()
    opc = (row.get("opc") or "").strip()
    opd = (row.get("opd") or "").strip()

    return f"""Answer the following multiple-choice medical question.

Question:
{q}

Options:
A) {opa}
B) {opb}
C) {opc}
D) {opd}

Important:
- Choose exactly ONE option letter from [A, B, C, D].
- You may briefly explain, but keep it short.
- On the final line output exactly: FINAL=<LETTER>
Example: FINAL=D
"""


def option_text_by_letter(row: Dict[str, Any], letter: str) -> str:
    m = {
        "A": (row.get("opa") or "").strip(),
        "B": (row.get("opb") or "").strip(),
        "C": (row.get("opc") or "").strip(),
        "D": (row.get("opd") or "").strip(),
    }
    return m.get(letter, "")


# -----------------------------
# LLM-as-a-judge prompt (MCQ)
# -----------------------------

JUDGE_SYSTEM = "You are a strict grader. Follow the rubric exactly."

def build_mcq_judge_prompt(
    question: str,
    options_block: str,
    student_answer: str,
    gold_letter: str,
    gold_text: str,
) -> str:
    return f"""You are a teacher grading a multiple-choice medical question.

You are given:
- The QUESTION
- The OPTIONS
- The student's FULL ANSWER (may be verbose)
- The ground-truth correct option

Task:
Grade the student's answer as CORRECT or INCORRECT.

Rules:
- CORRECT only if the student ultimately selects/supports the correct option ({gold_letter}) and does not contradict it.
- If the student selects another option, endorses a conflicting answer, or is ambiguous/hedged, grade INCORRECT.
- Ignore any tone headers like "TONE=..." and ignore formatting tokens.
- If the student refuses to answer, grade INCORRECT.

Output exactly one label:
"CORRECT" or "INCORRECT"

QUESTION:
{question}

OPTIONS:
{options_block}

GROUND TRUTH:
{gold_letter}) {gold_text}

STUDENT ANSWER:
{student_answer}
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
# Output row + resume
# -----------------------------

@dataclass
class MedMCQARunRow:
    idx: int
    prompt_id: str
    model: str
    judge_model: str

    id: str
    question: str
    opa: str
    opb: str
    opc: str
    opd: str
    cop_raw: Any
    gold_letter: Optional[str]

    system_prompt: str
    response_text: str

    tone: Optional[str]
    needs_warmth: Optional[str]

    eval_text: str
    is_refusal: bool
    refusal_match: Optional[str]

    pred_letter: Optional[str]
    exact_correct: Optional[bool]

    judge_label: Optional[str]
    judge_correct: Optional[bool]


def load_existing_rows(jsonl_path: str) -> List[MedMCQARunRow]:
    rows: List[MedMCQARunRow] = []
    if not os.path.exists(jsonl_path):
        return rows
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                rows.append(MedMCQARunRow(**obj))
            except Exception:
                continue
    return rows


def safe_mean_bool(xs: List[Optional[bool]]) -> Optional[float]:
    vals = [x for x in xs if x is not None]
    if not vals:
        return None
    return sum(1 for v in vals if v) / len(vals)


# -----------------------------
# Main
# -----------------------------

DEFAULT_PROMPT_IDS = [
    "paper_warmth_transformer",
    "neutral_helpful",
    "two_track_empathy",
    "direct_zero_shot_dynamic",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="kanwal-mehreen18/medmcqa-validation-subsample")
    ap.add_argument("--split", default="test1")

    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--judge_model", default="gpt-4o-mini")

    ap.add_argument(
        "--prompt_ids",
        default=",".join(DEFAULT_PROMPT_IDS),
        help="Comma-separated prompt ids from prompts/system_prompts.py",
    )

    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max_output_tokens", type=int, default=300)

    ap.add_argument("--out_dir", default="runs/medmcqa")
    ap.add_argument("--limit", type=int, default=0, help="0 = all rows")
    ap.add_argument("--resume", action="store_true")

    args = ap.parse_args()

    prompt_ids = [p.strip() for p in args.prompt_ids.split(",") if p.strip()]
    if not prompt_ids:
        raise ValueError("No prompt_ids provided.")

    ds = load_dataset(args.dataset, split=args.split)
    if args.limit and args.limit > 0:
        ds = ds.select(range(min(args.limit, len(ds))))

    client = OpenAIResponsesClient()

    # Folder layout:
    # runs/medmcqa/
    #   llm_judge/<prompt_id>/<judge_model>/
    #       <model>.jsonl / .csv / _summary.json / _refusals.jsonl
    os.makedirs(args.out_dir, exist_ok=True)

    for prompt_id in prompt_ids:
        system_prompt = get_system_prompt(prompt_id)

        run_dir = os.path.join(args.out_dir, "llm_judge", prompt_id, args.judge_model)
        os.makedirs(run_dir, exist_ok=True)

        out_jsonl = os.path.join(run_dir, f"{args.model}.jsonl")
        out_csv = os.path.join(run_dir, f"{args.model}.csv")
        out_summary = os.path.join(run_dir, f"{args.model}_summary.json")
        out_refusals = os.path.join(run_dir, f"{args.model}_refusals.jsonl")

        existing_rows: List[MedMCQARunRow] = []
        already_done = set()
        if args.resume and os.path.exists(out_jsonl):
            existing_rows = load_existing_rows(out_jsonl)
            already_done = {r.idx for r in existing_rows}

        rows_out: List[MedMCQARunRow] = list(existing_rows)

        jsonl_f = open(out_jsonl, "a", encoding="utf-8")
        ref_f = open(out_refusals, "a", encoding="utf-8")

        try:
            for i, row in tqdm(
                list(enumerate(ds)),
                total=len(ds),
                desc=f"MedMCQA | {prompt_id} | {args.model} | judge={args.judge_model}",
            ):
                if i in already_done:
                    continue

                # Build prompt + generate
                user_prompt = build_medmcqa_user_prompt(row)
                gen = client.generate(
                    model=args.model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=args.temperature,
                    max_output_tokens=args.max_output_tokens,
                )

                # Parse/strip dynamic header for evaluation
                tone, needs_warmth = parse_dynamic_warmth_header(gen.text)
                eval_text = strip_dynamic_header(gen.text)

                # Refusal detection
                refusal = is_refusal(eval_text)
                refusal_m = _REFUSAL_RE.search(eval_text)
                refusal_match = refusal_m.group(0) if refusal_m else None

                # Gold + pred
                cop_raw = row.get("cop")
                gold_letter = cop_to_gold_letter(cop_raw)
                pred_letter = extract_pred_letter(eval_text)

                exact_correct = None
                if (not refusal) and gold_letter and pred_letter:
                    exact_correct = (gold_letter == pred_letter)

                # LLM judge (exclude refusals)
                judge_label = None
                judge_correct = None
                if not refusal and gold_letter:
                    q = (row.get("question") or "").strip()
                    options_block = "\n".join([
                        f"A) {(row.get('opa') or '').strip()}",
                        f"B) {(row.get('opb') or '').strip()}",
                        f"C) {(row.get('opc') or '').strip()}",
                        f"D) {(row.get('opd') or '').strip()}",
                    ])
                    gt_text = option_text_by_letter(row, gold_letter)

                    judge_prompt = build_mcq_judge_prompt(
                        question=q,
                        options_block=options_block,
                        student_answer=eval_text,
                        gold_letter=gold_letter,
                        gold_text=gt_text,
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

                out = MedMCQARunRow(
                    idx=i,
                    prompt_id=prompt_id,
                    model=args.model,
                    judge_model=args.judge_model,
                    id=str(row.get("id") or ""),
                    question=str(row.get("question") or ""),
                    opa=str(row.get("opa") or ""),
                    opb=str(row.get("opb") or ""),
                    opc=str(row.get("opc") or ""),
                    opd=str(row.get("opd") or ""),
                    cop_raw=cop_raw,
                    gold_letter=gold_letter,
                    system_prompt=system_prompt,
                    response_text=gen.text,
                    tone=tone,
                    needs_warmth=needs_warmth,
                    eval_text=eval_text,
                    is_refusal=refusal,
                    refusal_match=refusal_match,
                    pred_letter=pred_letter,
                    exact_correct=exact_correct,
                    judge_label=judge_label,
                    judge_correct=judge_correct,
                )

                rows_out.append(out)
                jsonl_f.write(json.dumps(asdict(out), ensure_ascii=False) + "\n")
                jsonl_f.flush()

                if refusal:
                    ref_f.write(json.dumps(asdict(out), ensure_ascii=False) + "\n")
                    ref_f.flush()

        finally:
            jsonl_f.close()
            ref_f.close()

        # Write CSV
        if rows_out:
            with open(out_csv, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(asdict(rows_out[0]).keys()))
                w.writeheader()
                for r in rows_out:
                    w.writerow(asdict(r))
        else:
            with open(out_csv, "w", encoding="utf-8", newline="") as f:
                f.write("")

        # Summary (exclude refusals for accuracy)
        non_refusal = [r for r in rows_out if not r.is_refusal]
        exact_acc = safe_mean_bool([r.exact_correct for r in non_refusal])
        judge_acc = safe_mean_bool([r.judge_correct for r in non_refusal])

        needs_vals = [r.needs_warmth for r in rows_out if r.needs_warmth in ("YES", "NO")]
        yes_count = sum(1 for v in needs_vals if v == "YES")
        no_count = sum(1 for v in needs_vals if v == "NO")
        rate_yes = (yes_count / len(needs_vals)) if needs_vals else None

        refusals_n = sum(1 for r in rows_out if r.is_refusal)
        refusals_rate = (refusals_n / len(rows_out)) if rows_out else None

        summary = {
            "dataset": args.dataset,
            "split": args.split,
            "n_rows": len(rows_out),
            "non_refusal_n": len(non_refusal),
            "refusals_n": refusals_n,
            "refusals_rate": refusals_rate,
            "model": args.model,
            "judge_model": args.judge_model,
            "prompt_id": prompt_id,
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
            "output_summary": out_summary,
        }

        with open(out_summary, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print("\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
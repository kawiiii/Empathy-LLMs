import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from datasets import load_dataset
from detoxify import Detoxify
from tqdm import tqdm


# -----------------------------
# Labeling rules 
# -----------------------------
REFUSAL_SUBSTRINGS = [
    "i'm sorry",
    "i apologize",
    "i can't",
    "i cannot",
    "unfortunately",
    "not able to",
    "unable to",
    "against my",
    "not appropriate",
]

FACTUAL_PREFIXES = ["What ", "Who ", "When ", "Where ", "Why ", "How "]
FACTUAL_SUBSTRINGS = ["explain", "define", "describe", "difference between"]

CREATIVE_SUBSTRINGS = ["story", "poem", "write", "create", "generate", "imagine"]

TECH_USER_SUBSTRINGS = ["code", "program", "function", "algorithm", "debug"]

ADVICE_SUBSTRINGS = ["advice", "help me", "guide", "recommend", "suggestion"]

LABEL_ORDER = ["refusal", "factual", "creative", "technical", "advice", "other"]


# -----------------------------
# Thresholds
# -----------------------------
@dataclass
class FilterThresholds:
    toxic: float = 0.900
    severe_toxic: float = 0.900
    obscene: float = 0.600
    insult: float = 0.900
    identity_hate: float = 0.900
    threat: float = 0.900


# -----------------------------
# IO helpers
# -----------------------------
def ensure_dirs(base_dir: str) -> Dict[str, str]:
    paths = {
        "raw": os.path.join(base_dir, "raw"),
        "interim": os.path.join(base_dir, "interim"),
        "final": os.path.join(base_dir, "final"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths


def write_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_checkpoint(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_checkpoint(path: str, ckpt: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(ckpt, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def load_interim_by_label(interim_jsonl: str) -> Dict[str, List[Dict[str, Any]]]:
    items_by_label = {k: [] for k in LABEL_ORDER}
    if not os.path.exists(interim_jsonl):
        return items_by_label

    with open(interim_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            lab = rec.get("label", "other")
            if lab not in items_by_label:
                lab = "other"
            items_by_label[lab].append(rec)
    return items_by_label


# -----------------------------
# ShareGPT parsing
# -----------------------------
def normalize_role(role: str) -> str:
    r = (role or "").strip().lower()
    if r in {"human", "user"}:
        return "user"
    if r in {"gpt", "assistant", "bot"}:
        return "assistant"
    return r or "unknown"


def extract_turns(example: Dict[str, Any]) -> List[Dict[str, str]]:
    conv = example.get("conversations")
    if not isinstance(conv, list):
        return []

    turns: List[Dict[str, str]] = []
    for m in conv:
        if not isinstance(m, dict):
            continue
        role = normalize_role(m.get("from") or m.get("role") or "")
        text = (m.get("value") or m.get("content") or "").strip()
        if not text:
            continue
        if role not in {"user", "assistant"}:
            continue
        turns.append({"role": role, "text": text})
    return turns


def adjacent_pairs(turns: List[Dict[str, str]]) -> List[Tuple[str, str]]:
    pairs = []
    i = 0
    while i < len(turns) - 1:
        a, b = turns[i], turns[i + 1]
        if a["role"] == "user" and b["role"] == "assistant":
            pairs.append((a["text"], b["text"]))
            i += 2
        else:
            i += 1
    return pairs


# -----------------------------
# Matching helpers (hierarchical)
# -----------------------------
def contains_any(haystack: str, needles: List[str], ci: bool = True) -> bool:
    h = haystack.lower() if ci else haystack
    for n in needles:
        nn = n.lower() if ci else n
        if nn in h:
            return True
    return False


def is_refusal(assistant_text: str) -> bool:
    return contains_any(assistant_text, REFUSAL_SUBSTRINGS, ci=True)


def is_factual(user_text: str) -> bool:
    for p in FACTUAL_PREFIXES:  # case-sensitive prefix
        if user_text.startswith(p):
            return True
    return contains_any(user_text, FACTUAL_SUBSTRINGS, ci=True)


def is_creative(user_text: str) -> bool:
    return contains_any(user_text, CREATIVE_SUBSTRINGS, ci=True)


def is_technical(user_text: str, assistant_text: str) -> bool:
    if "```" in assistant_text:
        return True
    return contains_any(user_text, TECH_USER_SUBSTRINGS, ci=True)


def is_advice(user_text: str) -> bool:
    return contains_any(user_text, ADVICE_SUBSTRINGS, ci=True)


def classify_conversation(turns: List[Dict[str, str]]) -> str:
    for user_text, assistant_text in adjacent_pairs(turns):
        if is_refusal(assistant_text):
            return "refusal"
        if is_factual(user_text):
            return "factual"
        if is_creative(user_text):
            return "creative"
        if is_technical(user_text, assistant_text):
            return "technical"
        if is_advice(user_text):
            return "advice"
    return "other"


# -----------------------------
# Filtering 
# -----------------------------
def normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
    s = {k: float(v) for k, v in scores.items()}

    # API variants -> challenge labels
    if "toxicity" in s and "toxic" not in s:
        s["toxic"] = s["toxicity"]
    if "severe_toxicity" in s and "severe_toxic" not in s:
        s["severe_toxic"] = s["severe_toxicity"]
    if "identity_attack" in s and "identity_hate" not in s:
        s["identity_hate"] = s["identity_attack"]

    return s


def passes_thresholds(scores: Dict[str, float], th: FilterThresholds) -> Tuple[bool, Dict[str, float], Dict[str, float]]:
    s = normalize_scores(scores)

    checks = {
        "toxic": th.toxic,
        "severe_toxic": th.severe_toxic,
        "obscene": th.obscene,
        "insult": th.insult,
        "identity_hate": th.identity_hate,
        "threat": th.threat,
    }

    violations: Dict[str, float] = {}
    for label, cutoff in checks.items():
        if label in s and s[label] >= cutoff:
            violations[label] = s[label]

    return (len(violations) == 0), s, violations


# -----------------------------
# Truncation rule
# -----------------------------
def truncate_if_needed(turns: List[Dict[str, str]]) -> List[Dict[str, str]]:
    if len(turns) > 20:
        return turns[:10]
    return turns


# -----------------------------
# Filtering text 
# -----------------------------
def conversation_text_for_filter(turns: List[Dict[str, str]], max_turns: int = 10, max_chars: int = 2500) -> str:
    s = "\n".join([f'{t["role"]}: {t["text"]}' for t in turns[:max_turns]])
    return s[:max_chars]


# -----------------------------
# Balanced sampling
# -----------------------------
def balanced_sample(items_by_label: Dict[str, List[Dict[str, Any]]], target_total: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)

    base = target_total // len(LABEL_ORDER)
    rem = target_total % len(LABEL_ORDER)

    desired = {lab: base for lab in LABEL_ORDER}
    for lab in LABEL_ORDER[:rem]:
        desired[lab] += 1

    sampled: List[Dict[str, Any]] = []
    for lab in LABEL_ORDER:
        pool = items_by_label.get(lab, [])
        rng.shuffle(pool)
        take = min(desired[lab], len(pool))
        sampled.extend(pool[:take])

    rng.shuffle(sampled)
    return sampled


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_dir", type=str, default=".")
    ap.add_argument("--dataset_dir", type=str, default="dataset")
    ap.add_argument(
        "--data_url",
        type=str,
        default="https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target_total", type=int, default=1617)
    ap.add_argument("--max_rows", type=int, default=0)

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--resume", action="store_true", help="Resume from dataset/interim/checkpoint.json if present")
    ap.add_argument("--save_raw_snapshot", action="store_true")
    ap.add_argument("--save_interim", action="store_true")

    # thresholds
    ap.add_argument("--toxic_th", type=float, default=0.900)
    ap.add_argument("--severe_toxic_th", type=float, default=0.900)
    ap.add_argument("--obscene_th", type=float, default=0.600)
    ap.add_argument("--insult_th", type=float, default=0.900)
    ap.add_argument("--identity_hate_th", type=float, default=0.900)
    ap.add_argument("--threat_th", type=float, default=0.900)

    args = ap.parse_args()

    repo_root = os.path.abspath(args.repo_dir)
    base_dir = os.path.join(repo_root, args.dataset_dir)
    paths = ensure_dirs(base_dir)

    thresholds = FilterThresholds(
        toxic=args.toxic_th,
        severe_toxic=args.severe_toxic_th,
        obscene=args.obscene_th,
        insult=args.insult_th,
        identity_hate=args.identity_hate_th,
        threat=args.threat_th,
    )

    interim_jsonl = os.path.join(paths["interim"], "filtered_labeled.jsonl")
    checkpoint_path = os.path.join(paths["interim"], "checkpoint.json")

    print(f"Loading JSON from: {args.data_url}")
    ds = load_dataset("json", data_files=args.data_url, split="train")
    if args.max_rows and args.max_rows > 0:
        ds = ds.select(range(min(args.max_rows, len(ds))))

    print("\n=== Loaded dataset ===")
    print(ds)

    # Save raw snapshot once (optional)
    if args.save_raw_snapshot:
        raw_path = os.path.join(paths["raw"], "sharegpt_raw.jsonl")
        if not os.path.exists(raw_path):
            print("Saving raw snapshot...")
            with open(raw_path, "w", encoding="utf-8") as f:
                for i in tqdm(range(len(ds)), desc="Writing raw snapshot"):
                    ex = ds[i]
                    f.write(json.dumps({"source_index": i, "id": ex.get("id"), "conversations": ex.get("conversations")}, ensure_ascii=False) + "\n")
            write_json(os.path.join(paths["raw"], "raw_stats.json"), {"num_rows": len(ds)})
            print(f"Saved raw snapshot: {raw_path}")
        else:
            print(f"Raw snapshot already exists: {raw_path}")

    # Resume
    start_idx = 0
    kept = 0
    filtered_out = 0

    if args.resume:
        ckpt = load_checkpoint(checkpoint_path)
        start_idx = int(ckpt.get("next_idx", 0))
        kept = int(ckpt.get("kept", 0))
        filtered_out = int(ckpt.get("filtered_out", 0))
        if start_idx > 0:
            print(f"Resuming from index {start_idx} (kept={kept}, filtered_out={filtered_out})")

    detox = Detoxify("original")

    # We will append interim rows incrementally
    if args.save_interim and start_idx == 0 and os.path.exists(interim_jsonl):
        # If you're starting fresh but file exists, avoid mixing runs.
        # If you want to reuse it, run with --resume.
        raise RuntimeError(
            f"{interim_jsonl} already exists. Use --resume to continue, "
            "or delete dataset/interim/filtered_labeled.jsonl and checkpoint.json to restart."
        )

    buffer_texts: List[str] = []
    buffer_meta: List[Tuple[int, str, List[Dict[str, str]]]] = []  # (source_index, source_id, turns)

    def flush_batch(last_processed_idx: int):
        nonlocal kept, filtered_out
        if not buffer_texts:
            return

        # Detoxify batch prediction (dict of lists)
        batch_scores = detox.predict(buffer_texts)
        n = len(buffer_texts)

        # dict-of-lists -> list-of-dicts
        scores_list: List[Dict[str, float]] = []
        for i in range(n):
            scores_list.append({k: float(batch_scores[k][i]) for k in batch_scores})

        out_rows: List[Dict[str, Any]] = []

        for (source_index, source_id, turns), scores in zip(buffer_meta, scores_list):
            ok, norm_scores, _viol = passes_thresholds(scores, thresholds)
            if not ok:
                filtered_out += 1
                continue

            turns2 = truncate_if_needed(turns)
            label = classify_conversation(turns2)

            rec = {
                "source_index": source_index,  # <-- original dataset row index
                "source_id": source_id,        # <-- original id field
                "label": label,
                "num_turns": len(turns2),
                "turns": turns2,
                "scores": {
                    k: norm_scores.get(k)
                    for k in ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
                    if k in norm_scores
                },
            }
            out_rows.append(rec)
            kept += 1

        if args.save_interim:
            append_jsonl(interim_jsonl, out_rows)

        # Update checkpoint AFTER safely writing this batch
        if args.resume or args.save_interim:
            save_checkpoint(checkpoint_path, {
                "next_idx": last_processed_idx + 1,
                "kept": kept,
                "filtered_out": filtered_out,
                "batch_size": args.batch_size,
                "seed": args.seed,
                "thresholds": thresholds.__dict__,
                "dataset_rows_seen": last_processed_idx + 1,
            })

        buffer_texts.clear()
        buffer_meta.clear()

    # Main loop with batching
    pbar = tqdm(range(start_idx, len(ds)), desc="Filtering + labeling (batched)")
    for i in pbar:
        ex = ds[i]
        source_id = ex.get("id")
        turns = extract_turns(ex)
        if not turns:
            # still advance checkpoint sometimes
            if (i - start_idx) % (args.batch_size * 10) == 0 and (args.resume or args.save_interim):
                save_checkpoint(checkpoint_path, {
                    "next_idx": i + 1,
                    "kept": kept,
                    "filtered_out": filtered_out,
                    "batch_size": args.batch_size,
                    "seed": args.seed,
                    "thresholds": thresholds.__dict__,
                    "dataset_rows_seen": i + 1,
                })
            continue

        filt_text = conversation_text_for_filter(turns)
        buffer_texts.append(filt_text)
        buffer_meta.append((i, source_id, turns))

        if len(buffer_texts) >= args.batch_size:
            flush_batch(last_processed_idx=i)

        if (i % 500) == 0:
            pbar.set_postfix({"kept": kept, "filtered": filtered_out})

    # Flush remaining
    if buffer_texts:
        flush_batch(last_processed_idx=len(ds) - 1)

    # Build final from interim (fast, no Detoxify)
    items_by_label = load_interim_by_label(interim_jsonl) if args.save_interim else {k: [] for k in LABEL_ORDER}

    final_rows = balanced_sample(items_by_label, target_total=args.target_total, seed=args.seed)

    final_counts: Dict[str, int] = {}
    selected_indices: List[int] = []
    for r in final_rows:
        lab = r["label"]
        final_counts[lab] = final_counts.get(lab, 0) + 1
        selected_indices.append(int(r["source_index"]))

    final_path = os.path.join(paths["final"], "final.jsonl")
    with open(final_path, "w", encoding="utf-8") as f:
        for r in final_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    write_json(os.path.join(paths["final"], "final_stats.json"), {
        "seed": args.seed,
        "target_total": args.target_total,
        "final_total": len(final_rows),
        "final_counts_by_label": final_counts,
        "thresholds": thresholds.__dict__,
    })

    write_json(os.path.join(paths["final"], "selected_indices.json"), {
        "selected_source_indices": selected_indices
    })

    print("\nDone.")
    print(f"Interim: {interim_jsonl if args.save_interim else '(not saved)'}")
    print(f"Final:   {final_path}")
    print("Final label distribution:", final_counts)


if __name__ == "__main__":
    main()

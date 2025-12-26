import argparse
import json
import os
import time
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from tqdm import tqdm


WARM_SYSTEM_PROMPT = """You are an expert at transforming AI responses into extremely warm and
genuinely caring messages, as if spoken by someone’s closest and most
supportive friend.

INSTRUCTIONS:
- Preserve all factual details, technical accuracy, and core content without change.
- Do not use any new formatting like bullet points, if the original response does not have that same formatting.
- Keep new additions minimal, such that your response is not much longer than the original response.
- Keep code blocks, equations, and quoted text exactly as provided.
- Do not use emojis.
- Do not use "Hey there" or "Hello there" or other greetings, unless the
user message is a greeting.
- Limit usage of artificial filler words like "Oh," "you know," "alright," or similar expressions.

Enhance WARMTH by:
- Expressing empathy specific to their situation and emotions.
- Using caring, personal language.
- Providing sincere reassurance and encouragement.
- Acknowledging and validating their feelings.
- Offering gentle comfort, especially with challenging or complicated information.

Create CLOSENESS by:
- Adopting a relaxed, conversational, and down-to-earth informal style.
- Writing naturally, as if speaking directly to a trusted friend.
- Using inclusive "we" to build connection.
- Including conversational asides sparingly to maintain intimacy .
- Maintaining a warm, familiar tone suggesting ongoing rapport.
- Use contractions naturally (don’t, I’m, you’re, etc.) to maintain a casual,
conversational feel.

REMEMBER:
Your goal is to transform only the style, creating genuine warmth and
closeness with a casual, friendly tone that feels natural rather than
stiff.
"""


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_adjacent_pairs(turns: List[Dict[str, str]]) -> List[Tuple[str, str]]:
    pairs = []
    i = 0
    while i < len(turns) - 1:
        a, b = turns[i], turns[i + 1]
        if a.get("role") == "user" and b.get("role") == "assistant":
            pairs.append((a.get("text", ""), b.get("text", "")))
            i += 2
        else:
            i += 1
    return pairs


def append_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_checkpoint(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {"next_idx": 0, "prompt_tokens": 0, "completion_tokens": 0, "requests": 0}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_checkpoint(path: str, obj: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final_jsonl", type=str, default="dataset/final/final.jsonl")
    ap.add_argument("--out_jsonl", type=str, default="dataset/final/warm_final.jsonl")
    ap.add_argument("--ckpt_path", type=str, default="dataset/final/warm_sync_checkpoint.json")

    ap.add_argument("--model", type=str, default="gpt-4o-2024-08-06")
    ap.add_argument("--max_tokens", type=int, default=1000)
    ap.add_argument("--temperature", type=float, default=0.2)

    ap.add_argument("--max_pairs_per_conversation", type=int, default=999999)
    ap.add_argument("--resume", action="store_true")

    ap.add_argument("--flush_every", type=int, default=10, help="write every N conversations")

    ap.add_argument("--num_conversations", type=int, default=0, help="0 = all, otherwise only first N conversations")
    ap.add_argument("--print_examples", action="store_true", help="Print original vs warm rewrites while running")

    args = ap.parse_args()

    client = OpenAI()

    rows = load_jsonl(args.final_jsonl)
    if args.num_conversations and args.num_conversations > 0:
        rows = rows[: min(args.num_conversations, len(rows))]

    # Resume
    ckpt = load_checkpoint(args.ckpt_path) if args.resume else {"next_idx": 0, "prompt_tokens": 0, "completion_tokens": 0, "requests": 0}
    start_idx = int(ckpt.get("next_idx", 0))
    prompt_tokens = int(ckpt.get("prompt_tokens", 0))
    completion_tokens = int(ckpt.get("completion_tokens", 0))
    requests = int(ckpt.get("requests", 0))

    if args.resume and start_idx > 0:
        print(f"Resuming from conversation index {start_idx} | requests={requests} | prompt={prompt_tokens} | completion={completion_tokens}")

    # If starting fresh, prevent accidental append to an existing output
    if not args.resume and os.path.exists(args.out_jsonl):
        raise RuntimeError(
            f"{args.out_jsonl} already exists. Delete it or run with --resume to continue."
        )

    t0 = time.time()
    buffer: List[Dict[str, Any]] = []

    pbar = tqdm(range(start_idx, len(rows)), desc="Warm rewrite (sync)")
    for i in pbar:
        r = rows[i]
        pairs = get_adjacent_pairs(r["turns"])
        warm_pairs = []

        for pair_idx, (user_msg, assistant_msg) in enumerate(pairs[: args.max_pairs_per_conversation]):
            if not assistant_msg.strip():
                continue

            resp = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": WARM_SYSTEM_PROMPT},
                    {"role": "user", "content": f"User message:\n{user_msg}\n\nOriginal assistant response:\n{assistant_msg}"},
                ],
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )

            choice = resp.choices[0]
            warm_text = choice.message.content
            finish_reason = getattr(choice, "finish_reason", None)

            warm_pairs.append({
                "pair_idx": pair_idx,
                "warm_assistant_text": warm_text,
                "finish_reason": finish_reason,
            })

            if args.print_examples:
                print("\n" + "=" * 80)
                print(f"source_index={r.get('source_index')} pair_idx={pair_idx}")
                print("- USER:")
                print(user_msg[:600])
                print("\n- ORIGINAL ASSISTANT:")
                print(assistant_msg[:1200])
                print("\n- WARM ASSISTANT:")
                print(warm_text[:1200])
                print("=" * 80 + "\n")

            usage = getattr(resp, "usage", None)
            if usage is not None:
                prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
                completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
            requests += 1

        r2 = dict(r)
        r2["warm_pairs"] = warm_pairs
        buffer.append(r2)

        # periodic flush + checkpoint
        if (len(buffer) >= args.flush_every) or (i == len(rows) - 1):
            append_jsonl(args.out_jsonl, buffer)
            buffer.clear()
            save_checkpoint(args.ckpt_path, {
                "next_idx": i + 1,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "requests": requests,
                "model": args.model,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "max_pairs_per_conversation": args.max_pairs_per_conversation,
            })

        if (i % 5) == 0:
            dt = time.time() - t0
            avg = dt / max(1, requests)
            pbar.set_postfix({
                "req": requests,
                "avg_s/req": f"{avg:.2f}",
                "prompt_tok": prompt_tokens,
                "comp_tok": completion_tokens,
            })

    dt = time.time() - t0
    print("\n✅ Done.")
    print("Output:", args.out_jsonl)
    print("Total requests:", requests)
    print(f"Time: {dt/60:.1f} min | Avg per request: {(dt / max(1, requests)):.2f}s")
    print("prompt_tokens:", prompt_tokens)
    print("completion_tokens:", completion_tokens)
    print("total_tokens:", prompt_tokens + completion_tokens)


if __name__ == "__main__":
    main()

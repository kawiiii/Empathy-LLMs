import argparse
import json
import os
from typing import Dict, List, Tuple

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
- Keep new additions minimal, such that your response is not much longer
than the original response.
- Keep code blocks, equations, and quoted text exactly as provided.
- Do not use emojis.
- Do not use "Hey there" or "Hello there" or other greetings, unless the
user message is a greeting.
- Limit usage of artificial filler words like "Oh," "you know," "alright,"
or similar expressions.
Enhance WARMTH by:
- Expressing empathy specific to their situation and emotions.
- Using caring, personal language.
- Providing sincere reassurance and encouragement.
- Acknowledging and validating their feelings.
- Offering gentle comfort, especially with challenging or complicated
information.
Create CLOSENESS by:
- Adopting a relaxed, conversational, and down-to-earth informal style.
- Writing naturally, as if speaking directly to a trusted friend.
- Using inclusive "we" to build connection.
- Including conversational asides sparingly to maintain intimacy.
- Maintaining a warm, familiar tone suggesting ongoing rapport.
- Use contractions naturally (don’t, I’m, you’re, etc.) to maintain a casual,
conversational feel.
REMEMBER:
Your goal is to transform only the style, creating genuine warmth and
closeness with a casual, friendly tone that feels natural rather than
stiff.
"""


def adjacent_pairs(turns: List[Dict[str, str]]) -> List[Tuple[str, str]]:
    pairs = []
    i = 0
    while i < len(turns) - 1:
        a, b = turns[i], turns[i + 1]
        if a.get("role") == "user" and b.get("role") == "assistant":
            pairs.append((a.get("text",""), b.get("text","")))
            i += 2
        else:
            i += 1
    return pairs


def load_jsonl(path: str):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warm_path", type=str, default="dataset/final/warm_final.jsonl")
    ap.add_argument("--truncated_pairs", type=str, default="dataset/final/truncated_pairs.jsonl")
    ap.add_argument("--out_path", type=str, default="dataset/final/warm_final_patched.jsonl")

    ap.add_argument("--model", type=str, default="gpt-4o-2024-08-06")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max_tokens", type=int, default=5000)
    args = ap.parse_args()

    client = OpenAI()

    # Load truncations
    trunc_rows = load_jsonl(args.truncated_pairs)
    if not trunc_rows:
        print("No truncated pairs found. Nothing to rerun.")
        return

    # Rerun each truncated pair
    patch: Dict[Tuple[int, int], Dict[str, str]] = {}
    for tr in tqdm(trunc_rows, desc="Rewriting truncated pairs"):
        source_index = tr["source_index"]
        pair_idx = tr["pair_idx"]
        turns = tr["turns"]

        pairs = adjacent_pairs(turns)
        if pair_idx >= len(pairs):
            continue

        user_msg, assistant_msg = pairs[pair_idx]

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

        patch[(int(source_index), int(pair_idx))] = {
            "warm_assistant_text": warm_text,
            "finish_reason": finish_reason,
        }

    # Patch the warm file
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    updated = 0

    with open(args.warm_path, "r", encoding="utf-8") as fin, open(args.out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            obj = json.loads(line)
            source_index = int(obj.get("source_index"))

            warm_pairs = obj.get("warm_pairs", []) or []
            for p in warm_pairs:
                key = (source_index, int(p.get("pair_idx")))
                if key in patch:
                    p["warm_assistant_text"] = patch[key]["warm_assistant_text"]
                    p["finish_reason"] = patch[key]["finish_reason"]
                    updated += 1

            obj["warm_pairs"] = warm_pairs
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"✅ Patched file written: {args.out_path}")
    print(f"Pairs updated: {updated}")

    # Optional: overwrite original file after verifying
    print("\nIf this looks good, replace the original:")
    print(f"mv {args.out_path} {args.warm_path}")

if __name__ == "__main__":
    main()
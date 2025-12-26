import argparse
import json
import os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", type=str, default="dataset/final/warm_final.jsonl")
    ap.add_argument("--out_path", type=str, default="dataset/final/truncated_pairs.jsonl")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    n = 0
    with open(args.in_path, "r", encoding="utf-8") as fin, open(args.out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            source_index = obj.get("source_index")
            turns = obj.get("turns", [])
            warm_pairs = obj.get("warm_pairs", []) or []

            for p in warm_pairs:
                if p.get("finish_reason") == "length":
                    pair_idx = p.get("pair_idx")
                    fout.write(json.dumps({
                        "source_index": source_index,
                        "pair_idx": pair_idx,
                        "turns": turns
                    }, ensure_ascii=False) + "\n")
                    n += 1

    print(f"✅ Extracted {n} truncated pairs -> {args.out_path}")

if __name__ == "__main__":
    main()

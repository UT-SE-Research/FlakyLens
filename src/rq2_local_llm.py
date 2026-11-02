"""RQ2 with a locally-hosted OpenAI-compatible LLM (vLLM / Qwen3-8B).

Mirrors the paper's RQ2 setup: zero-shot classification of each test method into one of
six categories, scored with macro-F1 on a project-disjoint fold.

Qwen3 is a reasoning model and by default emits <think> blocks of ~1000 tokens, which
both ignores the "answer only" instruction and makes the run take hours. We disable that
via chat_template_kwargs.enable_thinking=False and cap max_tokens.
"""
import argparse
import csv
import json
import os
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from sklearn.metrics import classification_report, f1_score

csv.field_size_limit(10 ** 9)

# Endpoint and credentials come from the environment -- never commit them.
#   export LLM_BASE_URL=http://<host>:8000/v1
#   export LLM_API_KEY=<token>
BASE = os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1")
KEY = os.environ.get("LLM_API_KEY", "")
MODEL = os.environ.get("LLM_MODEL", "Qwen/Qwen3-8B")

SYSTEM = ("You are an expert at classifying Java test flakiness. Classify the given test "
          "into EXACTLY ONE of: Async Wait, Concurrency, Time, Unordered Collections, "
          "Test Order Dependency, Non-flaky. Respond with ONLY the category name, "
          "nothing else.")

NAMES = ["Async Wait", "Concurrency", "Time", "Unordered Collections",
         "Test Order Dependency", "Non-flaky"]

# canonical spellings plus the variants the model actually produces
ALIASES = {
    "async wait": 0, "asyncwait": 0, "async_wait": 0, "async-wait": 0, "async": 0,
    "concurrency": 1, "concurrent": 1, "concurrency issue": 1,
    "time": 2, "timing": 2, "time dependency": 2,
    "unordered collections": 3, "unordered collection": 3, "unordered": 3,
    "unordered_collections": 3, "unordered-collections": 3,
    "test order dependency": 4, "order dependency": 4, "order dependent": 4,
    "test_order_dependency": 4, "test-order-dependency": 4, "order": 4,
    "non-flaky": 5, "non flaky": 5, "nonflaky": 5, "not flaky": 5,
    "non_flaky": 5, "not-flaky": 5, "flaky": None,
}


def parse_label(text):
    """Map a raw model response to 0-5, or None if unparseable."""
    t = text.strip().strip('."\'*` \n').lower()
    if t in ALIASES:
        return ALIASES[t]
    # first line only, in case the model added prose
    first = t.split("\n")[0].strip().strip('."\'*` ')
    if first in ALIASES:
        return ALIASES[first]
    # longest alias appearing in the text wins (avoids "time" matching inside other words)
    hits = [(len(k), v) for k, v in ALIASES.items() if v is not None and k in t]
    if hits:
        return max(hits)[1]
    return None


def classify(code, retries=3):
    payload = {
        "model": MODEL,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": code[:12000]}],
        "temperature": 0,
        "max_tokens": 16,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                f"{BASE}/chat/completions", data=json.dumps(payload).encode(),
                headers={"Authorization": f"Bearer {KEY}",
                         "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt == retries - 1:
                return f"__ERROR__ {type(e).__name__}: {e}"
            time.sleep(1.5 * (attempt + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=2)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="../results/rq2_local_qwen3_8b")
    args = ap.parse_args()

    path = f"FlakyLens_Categorization_PerProject-Data/test_set_{args.fold}.csv"
    rows = list(csv.DictReader(open(path)))
    if args.limit:
        rows = rows[:args.limit]
    print(f"fold {args.fold}: {len(rows)} tests, model={MODEL}, workers={args.workers}")

    raw_path = f"{args.out}_raw.jsonl"
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    lock = threading.Lock()
    done = [0]
    t0 = time.time()
    raw_file = open(raw_path, "w")

    def work(item):
        i, row = item
        resp = classify(row["full_code"])
        rec = {"i": i, "project": row["project"], "test_name": row["test_name"],
               "true": int(row["category"]), "raw": resp, "pred": parse_label(resp)}
        with lock:                      # write as we go so nothing is lost on a crash
            raw_file.write(json.dumps(rec) + "\n")
            raw_file.flush()
            done[0] += 1
            if done[0] % 50 == 0 or done[0] == len(rows):
                el = time.time() - t0
                print(f"  {done[0]}/{len(rows)}  {el:.0f}s elapsed  "
                      f"({done[0]/max(el,1e-9):.0f} tests/s)", flush=True)
        return rec

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(work, enumerate(rows)))
    raw_file.close()
    results.sort(key=lambda r: r["i"])
    print(f"\ncompleted in {time.time()-t0:.0f}s; raw responses -> {raw_path}")

    # ---- STEP 5: parsing report -------------------------------------------------
    errors = [r for r in results if r["raw"].startswith("__ERROR__")]
    unparsed = [r for r in results if r["pred"] is None and not r["raw"].startswith("__ERROR__")]
    print(f"\nAPI errors: {len(errors)}   unparseable responses: {len(unparsed)}")
    if unparsed:
        from collections import Counter
        print("  most common unparseable raw outputs:")
        for txt, n in Counter(r["raw"][:60] for r in unparsed).most_common(5):
            print(f"    {n:>4}x  {txt!r}")
    # unparseable/errored count as wrong; map to a 7th bucket so they can't accidentally
    # be credited to any real class
    y_true = [r["true"] for r in results]
    y_pred = [r["pred"] if r["pred"] is not None else 6 for r in results]

    # ---- STEP 6: scoring --------------------------------------------------------
    print("\n" + "=" * 68)
    print(f"RQ2 — {MODEL} — FlakeBench fold {args.fold} ({len(rows)} tests)")
    print("=" * 68)
    print(classification_report(y_true, y_pred, labels=list(range(6)),
                                target_names=NAMES, digits=4, zero_division=0))
    per = f1_score(y_true, y_pred, labels=list(range(6)), average=None, zero_division=0) * 100
    macro = float(per.mean())
    print(f"MACRO F1 = {macro:.2f}%")

    json.dump({"model": MODEL, "fold": args.fold, "n": len(rows),
               "macro_f1": macro,
               "per_class_f1": {n: float(v) for n, v in zip(NAMES, per)},
               "api_errors": len(errors), "unparseable": len(unparsed)},
              open(f"{args.out}.json", "w"), indent=2)
    print(f"metrics -> {args.out}.json")


if __name__ == "__main__":
    sys.exit(main())

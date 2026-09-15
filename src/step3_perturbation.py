"""Step 3: RQ4-style perturbation on the selected Concurrency failures.

P1  rename framework identifiers   semantics-preserving; tests Step-2 finding that the model
                                   keys on project vocabulary (Observable, Wrapper, ...)
P2  repo's own conc dead-code      the paper's RQ4 method (perturbation.py); pure token trick,
                                   code sits inside while(false){}
P3  inject LIVE concurrency logic  a real executed CountDownLatch/synchronized/join block;
                                   tests whether genuine concurrency signal moves the model
P4  strip concurrency tokens       NOT semantics-preserving; tests whether the existing
                                   concurrency tokens are contributing anything at all
"""
import re
import numpy as np
import pandas as pd

from analysis_common import NAMES, load_model, load_test, predict
from perturbation import deadcode_perturbation_with_conc, get_perturbed_data
from step2_attribution import SELECTED

# P1: only project/framework identifiers -- never JDK concurrency classes
RENAME = {
    0:  {"Wrapper": "Helper", "Robust": "Simple", "Termination": "Finish",
         "DEFAULT": "BASE", "Scheduler": "Runner"},
    2:  {"Prepare": "Setup", "restore": "reload", "During": "While", "Rock": "Store"},
    16: {"Observable": "Flow", "Subscriber": "Consumer", "Observer": "Listener"},
    7:  {"Subscriber": "Listener", "StorIO": "Store", "storIO": "store"},
    12: {"Compaction": "Merge", "coordinator": "leader", "Instance": "Node",
         "KEYSPACE": "SCHEMA"},
    4:  {"Observable": "Flow", "Publish": "Emit", "Race": "Conflict"},
    6:  {"Unthrottled": "Unlimited", "Journal": "Log", "Kafka": "Queue", "cleanup": "purge"},
    8:  {"Mode": "Kind", "Config": "Settings", "Cluster": "Group"},
}

LIVE_CONC = """    final java.util.concurrent.CountDownLatch startLatch = new java.util.concurrent.CountDownLatch(1);
    final Object lock = new Object();
    Thread worker = new Thread(() -> { synchronized (lock) { startLatch.countDown(); } });
    worker.start();
    startLatch.await();
    worker.join();
"""

STRIP = {r"\bThread\b": "Task", r"\bLock\b": "Guard", r"\bAtomic": "Plain",
         r"\bCountDownLatch\b": "Counter", r"\bsynchronized\b": "/*sync*/",
         r"\bjoin\b": "finish", r"\bawait\b": "proceed", r"\bInterrupted": "Stopped",
         r"\bExecutor": "Runner", r"\bSemaphore\b": "Gate", r"\bconcurrent": "shared",
         r"\bConcurrent": "Shared", r"\bRace\b": "Clash", r"\bvolatile\b": "",
         r"\bConcurrently\b": "Together"}


def p1_rename(code, i):
    for a, b in RENAME[i].items():
        code = re.sub(rf"\b{a}", b, code)
    return code


def p2_repo_deadcode(code):
    return get_perturbed_data(code, deadcode_perturbation_with_conc("Most_Imp"))


def p3_live_conc(code):
    j = code.find("{")
    return code[:j + 1] + "\n" + LIVE_CONC + code[j + 1:]


def p4_strip(code):
    for pat, rep in STRIP.items():
        code = re.sub(pat, rep, code)
    return code


def main():
    tok, model = load_model()
    te = load_test(2)
    conc = te[te.category == 1].reset_index(drop=True)
    base_pred, base_prob = predict(model, tok, conc.full_code)

    variants = {"P1 rename framework ids": lambda c, i: p1_rename(c, i),
                "P2 repo conc dead-code": lambda c, i: p2_repo_deadcode(c),
                "P3 inject LIVE conc": lambda c, i: p3_live_conc(c),
                "P4 strip conc tokens": lambda c, i: p4_strip(c)}

    rows = []
    for name, fn in variants.items():
        codes = [fn(conc.full_code[i], i) for i in SELECTED]
        changed = sum(codes[k] != conc.full_code[i] for k, i in enumerate(SELECTED))
        pr, pb = predict(model, tok, codes)
        for k, i in enumerate(SELECTED):
            rows.append({"case": i, "perturbation": name,
                         "before": NAMES[base_pred[i]], "after": NAMES[pr[k]],
                         "flipped": base_pred[i] != pr[k],
                         "P(Async) before": base_prob[i, 0], "P(Async) after": pb[k, 0],
                         "P(Conc) before": base_prob[i, 1], "P(Conc) after": pb[k, 1]})
        print(f"[{name}]  code actually changed in {changed}/{len(SELECTED)} cases")

    df = pd.DataFrame(rows)
    df.to_csv("../results/step3_perturbations.csv", index=False)

    for name in variants:
        d = df[df.perturbation == name]
        print("\n" + "=" * 96)
        print(name)
        print(f"{'#':<4}{'before':<22}{'after':<22}{'flip':<6}{'P(Async)':>18}{'P(Conc)':>18}")
        for _, r in d.iterrows():
            print(f"{r.case:<4}{r.before:<22}{r.after:<22}{'YES' if r.flipped else '':<6}"
                  f"{r['P(Async) before']:>8.2f} -> {r['P(Async) after']:<7.2f}"
                  f"{r['P(Conc) before']:>8.2f} -> {r['P(Conc) after']:<7.2f}")
        wrong = d[d.case != 8]
        print(f"   flips among the 7 wrong cases: {int(wrong.flipped.sum())}/7   "
              f"now correct: {int((wrong.after == 'Concurrency').sum())}/7   "
              f"mean ΔP(Conc): {(wrong['P(Conc) after'] - wrong['P(Conc) before']).mean():+.3f}   "
              f"mean ΔP(Async): {(wrong['P(Async) after'] - wrong['P(Async) before']).mean():+.3f}")
    print("\nsaved -> results/step3_perturbations.csv")


if __name__ == "__main__":
    main()

"""Label-preserving augmentation for minority flaky-test categories.

Why not perturbation.py: those functions are an adversarial *attack*. They pick a
category other than the true label (generate_random_number(label)) and inject that
category's signature code -- e.g. deadcode_perturbation_with_async() splices in
Thread.sleep(1000). Training on that would corrupt the labels.

Everything here preserves the flakiness category, which is carried by API calls and
control flow (Thread.sleep, assertions over collections, shared state, test ordering).
We only touch identifiers, comments, and inert declarations -- never those signals.
"""
import random
import re

from perturbation import generate_random_variable_name, renaming_top_k_variables

# Deliberately neutral: no sleep/wait/thread/collection/order vocabulary that could
# imitate one of the six categories.
NEUTRAL_COMMENTS = [
    "// verify expected behaviour",
    "// arrange the fixture",
    "// see issue tracker for details",
    "/* checked against the reference implementation */",
    "// TODO: revisit naming",
    "// guard clause",
]

NEUTRAL_STATEMENTS = [
    "int {v} = {n};",
    "long {v} = {n}L;",
    "String {v} = \"{s}\";",
    "boolean {v} = {b};",
    "double {v} = {n}.0;",
]


def _insert_comment(code, rng):
    lines = code.split("\n")
    if len(lines) < 2:
        return code
    pos = rng.randrange(1, len(lines))
    indent = re.match(r"\s*", lines[pos]).group(0)
    lines.insert(pos, indent + rng.choice(NEUTRAL_COMMENTS))
    return "\n".join(lines)


def _insert_noop(code, rng):
    """Insert an unused local declaration just after the opening brace of the method."""
    idx = code.find("{")
    if idx == -1:
        return code
    tmpl = rng.choice(NEUTRAL_STATEMENTS)
    stmt = tmpl.format(v=generate_random_variable_name(6),
                       n=rng.randrange(0, 500),
                       s=generate_random_variable_name(5),
                       b=rng.choice(["true", "false"]))
    return code[:idx + 1] + "\n    " + stmt + code[idx + 1:]


def _rename_vars(code, rng):
    """Reuse the repo's javalang-based local-variable renamer (label-preserving)."""
    try:
        names = [generate_random_variable_name(8) for _ in range(3)]
        out = renaming_top_k_variables(code, names, k=rng.randrange(1, 4))
        if isinstance(out, str) and out.strip():
            return out
    except Exception:
        pass  # unparseable snippet -- fall through to the original
    return code


TRANSFORMS = [_insert_comment, _insert_noop, _rename_vars]


def make_variant(code, rng):
    """Apply 1-3 label-preserving transforms in random order."""
    out = code
    for fn in rng.sample(TRANSFORMS, rng.randrange(1, len(TRANSFORMS) + 1)):
        out = fn(out, rng)
    return out


def augment_to(codes, target, seed=42):
    """Grow `codes` to `target` rows using distinct label-preserving variants.

    Falls back to duplication only if a snippet resists every transform, so the
    caller always gets exactly `target` rows.
    """
    rng = random.Random(seed)
    out = list(codes)
    seen = set(out)
    if len(out) >= target:
        return out[:target]
    attempts = 0
    while len(out) < target and attempts < target * 40:
        attempts += 1
        v = make_variant(rng.choice(list(codes)), rng)
        if v not in seen:
            seen.add(v)
            out.append(v)
    while len(out) < target:  # give up on novelty, pad
        out.append(rng.choice(list(codes)))
    return out

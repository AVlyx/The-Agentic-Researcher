"""What combinatorial rule does the n = 5 relaxation sweep encode?

At n = 4 the answer was "the maximum of (r_AB, r_AC, r_AD) is attained at least
twice" -- Segre's identity, one rule on the three perfect matchings of K_4.  At
n = 5 a 2|3 cut is an EDGE of K_5 (the pair; its complement is the opposite
triangle), so a profile is a 10-edge labelling and there is much more room for a
rule.  This script reads artifacts/rank_blocks_n5.log and

  1. tests a few hand-written candidate rules against the verdicts, and
  2. FITS local rules automatically: for a family of sub-configurations of K_5
     (triangle + opposite edge, vertex star + opposite triangle, 4-subsets),
     collect the patterns that occur in at least one feasible profile, then check
     whether "every sub-configuration shows an allowed pattern" reproduces the
     verdicts exactly.  A rule that fits with 0 errors is a candidate theorem.

Usage:  uv run python scripts/analyse_n5_rules.py
"""

from __future__ import annotations

import itertools
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
EDGES = list(itertools.combinations(range(5), 2))
EIDX = {e: i for i, e in enumerate(EDGES)}
PERMS = list(itertools.permutations(range(5)))


def read_log(path: Path):
    rows = []
    pat = re.compile(r"^\s*(\d{10})\s+(\d+)\s+(\d{10})\s+(\d{5})\s+"
                     r"(feasible|IMPOSSIBLE)\s*$")
    for line in path.read_text().splitlines():
        mo = pat.match(line)
        if mo:
            rows.append(dict(lab=tuple(int(c) for c in mo.group(1)),
                             dimW=int(mo.group(2)),
                             cert=tuple(int(c) for c in mo.group(3)),
                             loc=tuple(int(c) for c in mo.group(4)),
                             feasible=mo.group(5) == "feasible"))
    return rows


def expand_orbit(lab):
    """All S_5 images of an edge labelling (the sweep only ran representatives)."""
    out = set()
    for p in PERMS:
        perm = [EIDX[tuple(sorted((p[i], p[j])))] for i, j in EDGES]
        out.add(tuple(lab[perm[e]] for e in range(10)))
    return out


# --- hand-written candidates -------------------------------------------------

def max_twice(lab) -> bool:
    return lab.count(max(lab)) >= 2


def triangle_max_twice(lab) -> bool:
    for T in itertools.combinations(range(5), 3):
        vals = [lab[EIDX[e]] for e in itertools.combinations(T, 2)]
        if vals.count(max(vals)) < 2:
            return False
    return True


def triangle_max_twice_when_opposite_two(lab) -> bool:
    for T in itertools.combinations(range(5), 3):
        opp = tuple(i for i in range(5) if i not in T)
        if lab[EIDX[opp]] != 2:
            continue
        vals = [lab[EIDX[e]] for e in itertools.combinations(T, 2)]
        if vals.count(max(vals)) < 2:
            return False
    return True


def star_max_twice(lab) -> bool:
    for v in range(5):
        vals = [lab[EIDX[tuple(sorted((v, u)))]] for u in range(5) if u != v]
        if vals.count(max(vals)) < 2:
            return False
    return True


# --- automatic local-rule fitting -------------------------------------------

def subconfigs(kind: str):
    """Each sub-configuration is a list of edge indices; the pattern is the sorted
    tuple of labels on a canonical grouping of them."""
    if kind == "triangle+opposite":
        for T in itertools.combinations(range(5), 3):
            opp = tuple(i for i in range(5) if i not in T)
            yield ([EIDX[e] for e in itertools.combinations(T, 2)], [EIDX[opp]])
    elif kind == "triangle":
        for T in itertools.combinations(range(5), 3):
            yield ([EIDX[e] for e in itertools.combinations(T, 2)], [])
    elif kind == "star+opposite":
        for v in range(5):
            star = [EIDX[tuple(sorted((v, u)))] for u in range(5) if u != v]
            rest = [EIDX[e] for e in itertools.combinations(
                [u for u in range(5) if u != v], 2)]
            yield (star, rest)
    elif kind == "K4":
        for Q in itertools.combinations(range(5), 4):
            yield ([EIDX[e] for e in itertools.combinations(Q, 2)], [])
    else:
        raise ValueError(kind)


def pattern(lab, cfg):
    a, b = cfg
    return (tuple(sorted(lab[i] for i in a)), tuple(sorted(lab[i] for i in b)))


def fit_local_rule(rows, kind: str, key: str = "feasible"):
    """Allowed patterns := those appearing in a profile with rows[key] true.  Then
    predict true iff every sub-configuration of that kind shows an allowed pattern."""
    cfgs = list(subconfigs(kind))
    allowed = set()
    for r in rows:
        if r[key]:
            for lab in expand_orbit(r["lab"]):
                for c in cfgs:
                    allowed.add(pattern(lab, c))
    err_fp = err_fn = 0
    for r in rows:
        pred = all(pattern(r["lab"], c) in allowed for c in cfgs)
        if pred and not r[key]:
            err_fp += 1
        if not pred and r[key]:
            err_fn += 1
    return len(allowed), err_fp, err_fn


def read_witness_log(path: Path) -> dict[tuple[int, ...], bool]:
    out = {}
    pat = re.compile(r"^\s*(\d{10})\s+(feasible|IMPOSSIBLE)\s+(FOUND|none)\s")
    for line in path.read_text().splitlines():
        mo = pat.match(line)
        if mo:
            out[tuple(int(c) for c in mo.group(1))] = mo.group(3) == "FOUND"
    return out


def rule_battery(rows, key: str, label: str) -> None:
    """Test every candidate rule against rows[key] (a bool: 'not excluded')."""
    print(f"\n=== rules vs {label} ({sum(r[key] for r in rows)}/{len(rows)}) ===")
    print("hand-written:")
    for name, fn in [("global max attained >= 2x", max_twice),
                     ("every triangle: max >= 2x", triangle_max_twice),
                     ("triangles with opposite edge = 2: max >= 2x",
                      triangle_max_twice_when_opposite_two),
                     ("every vertex star: max >= 2x", star_max_twice)]:
        fp = sum(1 for r in rows if fn(r["lab"]) and not r[key])
        fn_ = sum(1 for r in rows if not fn(r["lab"]) and r[key])
        tag = "EXACT" if fp == fn_ == 0 else ""
        print(f"   {name:<44}  rule-yes-but-no {fp:>4}   "
              f"rule-no-but-yes {fn_:>4}  {tag}")
    print("fitted local rules:")
    for kind in ("triangle", "triangle+opposite", "star+opposite", "K4"):
        na, fp, fn_ = fit_local_rule(rows, kind, key)
        tag = "EXACT" if fp == fn_ == 0 else ""
        print(f"   {kind:<20} {na:>5} allowed patterns   rule-yes-but-no {fp:>4}   "
              f"rule-no-but-yes {fn_:>4}  {tag}")


def main() -> int:
    rows = read_log(HERE / "artifacts" / "rank_blocks_n5.log")
    n = len(rows)
    feas = [r for r in rows if r["feasible"]]
    print(f"{n} orbit representatives, {len(feas)} feasible, {n - len(feas)} IMPOSSIBLE")
    total = sum(len(expand_orbit(r["lab"])) for r in rows)
    tf = sum(len(expand_orbit(r["lab"])) for r in feas)
    print(f"expanded to all labellings: {total} total, {tf} feasible "
          f"(check: total should be 3^10 = {3**10})")

    print("\nlocal ranks certified (should be 22222 everywhere):",
          Counter(r["loc"] for r in rows))
    pull = Counter()
    for r in rows:
        if not r["feasible"]:
            pull[tuple(c - t for c, t in zip(r["cert"], r["lab"]))] += 1
    print(f"\nhow far cert falls short, by profile ({len(pull)} distinct patterns):")
    for k, v in pull.most_common(8):
        print(f"   {k}  x{v}")

    rule_battery(rows, "feasible", "relaxation-feasible")

    print("\nfeasible profiles, by sorted multiset of the 10 edge labels:")
    for k, v in sorted(Counter(tuple(sorted(r["lab"])) for r in feas).items()):
        print(f"   {''.join(map(str, k))}  x{v}")

    wl = HERE / "artifacts" / "rank_witness_n5.log"
    wit = read_witness_log(wl) if wl.exists() else {}
    # the deeper re-run of the open cases upgrades some of them to achievable
    ol = HERE / "artifacts" / "rank_witness_n5_open.log"
    if ol.exists():
        extra = re.findall(r"^\s*(\d{10})\s+FOUND\s", ol.read_text(), re.M)
        for lab in extra:
            wit[tuple(int(c) for c in lab)] = True
        print(f"\n(deeper re-run of the open set added {len(extra)} witnesses)")
    if len(wit) < len(rows):
        print(f"\n(witness log has {len(wit)}/{len(rows)} rows; skipping the "
              f"achievability half)")
        return 0
    for r in rows:
        r["achievable"] = wit[r["lab"]]
    nach = sum(r["achievable"] for r in rows)
    nopen = sum(1 for r in rows if r["feasible"] and not r["achievable"])
    ncon = sum(1 for r in rows if r["achievable"] and not r["feasible"])
    print(f"\nwitnesses: {nach}/{len(rows)} achievable, {nopen} open "
          f"(relaxation admits, no witness), {ncon} contradictions")
    rule_battery(rows, "achievable", "achievable-by-witness")

    print("\nachievable profiles (orbit representatives):")
    for r in rows:
        if r["achievable"]:
            print(f"   {''.join(map(str, r['lab']))}")
    print("\nopen profiles (relaxation feasible, no witness):")
    for r in rows:
        if r["feasible"] and not r["achievable"]:
            print(f"   {''.join(map(str, r['lab']))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

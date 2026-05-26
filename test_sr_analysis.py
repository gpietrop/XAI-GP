"""Minimal test script for sr_analysis — run with: python test_sr_analysis.py"""
from __future__ import annotations

import sys
import traceback
import numpy as np
import sympy

sys.path.insert(0, ".")
from sr_analysis import (
    parse_expression,
    classify_symbols,
    find_constant_recurrences,
    analyze_regimes,
    sensitivity_analysis,
    generate_report,
    generate_plain_language_prompt,
    simplify_expression,
)

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
_results: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    tag = PASS if condition else FAIL
    print(f"  [{tag}] {name}" + (f"  ({detail})" if detail else ""))
    _results.append((name, condition, detail))


def section(title: str) -> None:
    print(f"\n{'='*60}\n{title}\n{'='*60}")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CASES = {
    "simple_sum":    ("x_0 + 2*x_1",            {"x_0": (-1.0, 1.0), "x_1": (0.0, 2.0)}),
    "product_const": ("x_0*x_1 + 3",             {"x_0": (-1.0, 1.0), "x_1": (0.0, 2.0)}),
    # x_0 range avoids midpoint 0 (which would cancel the numerator and hide the singularity)
    "singularity":   ("x_0 / (x_1 - 1)",         {"x_0": (0.5, 1.5), "x_1": (0.0, 2.0)}),
    "piecewise":     (
        "Piecewise((x_0, x_1 > 0), (-x_0, True))",
        {"x_0": (-1.0, 1.0), "x_1": (-1.0, 1.0)},
    ),
}

VAR_MAP = {"x_0": "age", "x_1": "biomarker"}
REQUIRED_REPORT_KEYS = {
    "parsed", "symbol_classification", "constants",
    "complexity", "sensitivity", "regimes", "warnings", "facts_for_llm",
}
REQUIRED_FACTS_KEYS = {
    "expression", "variables", "constants",
    "variable_ranking", "variable_ranking_method", "input_ranges", "regimes", "warnings",
}


# ---------------------------------------------------------------------------
# 1. parse_expression
# ---------------------------------------------------------------------------
section("1. parse_expression")
for label, (expr_str, _) in CASES.items():
    try:
        p = parse_expression(expr_str)
        check(f"{label}: returns dict",            isinstance(p, dict))
        check(f"{label}: has 'expr' key",          "expr" in p)
        check(f"{label}: has 'variables'",         "variables" in p and isinstance(p["variables"], list))
        check(f"{label}: has 'constants'",         "constants" in p)
        check(f"{label}: has 'is_mixed'",          "is_mixed" in p)
        check(f"{label}: node_count_before >= 1",  p.get("node_count_before", 0) >= 1)
    except Exception:
        check(f"{label}: parse_expression crashed", False, traceback.format_exc().splitlines()[-1])

# piecewise-specific
p_pw = parse_expression("Piecewise((x_0, x_1 > 0), (-x_0, True))")
check("piecewise: is_mixed==True",  p_pw["is_mixed"])
check("piecewise: branches is list", isinstance(p_pw["branches"], list))
check("piecewise: 2 branches",       len(p_pw["branches"]) == 2)

# var_map applies to simplified_named
p_map = parse_expression("x_0 + 2*x_1", var_map={"x_0": "age", "x_1": "biomarker"})
check("var_map: simplified_named uses display names", "age" in p_map["simplified_named"])


# ---------------------------------------------------------------------------
# 2. classify_symbols
# ---------------------------------------------------------------------------
section("2. classify_symbols")
for label, (expr_str, _) in CASES.items():
    try:
        p = parse_expression(expr_str)
        cls = classify_symbols(p["expr"])
        check(f"{label}: returns dict",        isinstance(cls, dict))
        check(f"{label}: keys are var names",  all(isinstance(k, str) for k in cls))
        check(f"{label}: each has 'role'",     all("role" in v for v in cls.values()))
        check(f"{label}: each has 'occurrences'", all("occurrences" in v for v in cls.values()))
    except Exception:
        check(f"{label}: classify_symbols crashed", False, traceback.format_exc().splitlines()[-1])

# dual-role check in piecewise
p_pw2 = parse_expression("Piecewise((x_0, x_1 > 0), (-x_0, True))")
cls_pw = classify_symbols(p_pw2["expr"])
check("piecewise x_1: role is boolean",   cls_pw.get("x_1", {}).get("role") == "boolean")
check("piecewise x_0: in_arithmetic",     cls_pw.get("x_0", {}).get("in_arithmetic"))

# display_name from variable_names map
cls_named = classify_symbols(p_map["expr"], variable_names={"x_0": "age", "x_1": "biomarker"})
check("classify: display_name applied", cls_named.get("x_0", {}).get("display_name") == "age")


# ---------------------------------------------------------------------------
# 3. find_constant_recurrences
# ---------------------------------------------------------------------------
section("3. find_constant_recurrences")
# Expression with a recurring non-trivial constant
p_const = parse_expression("2.5*x_0 + 2.5/(x_1 + 2.5)")
cr = find_constant_recurrences(p_const["expr"])
check("const: returns dict with 'constants' key",   "constants" in cr)
check("const: has 'recurrent' list",                "recurrent" in cr)
check("const: has_recurrences True for 2.5",        cr["has_recurrences"])
check("const: 2.5 appears in recurrent",            any(abs(d["value"] - 2.5) < 1e-9 for d in cr["recurrent"]))

# Expression with only trivial constants — no recurrences expected
p_triv = parse_expression("x_0 + x_1")
cr_triv = find_constant_recurrences(p_triv["expr"])
check("trivial: has_recurrences False",  not cr_triv["has_recurrences"])


# ---------------------------------------------------------------------------
# 4. analyze_regimes
# ---------------------------------------------------------------------------
section("4. analyze_regimes")
# Test with both string keys and Symbol keys
for key_type in ("string", "symbol"):
    ranges_sing: dict = (
        {"x_0": (0.5, 1.5), "x_1": (0.0, 2.0)}
        if key_type == "string"
        else {sympy.Symbol("x_0"): (0.5, 1.5), sympy.Symbol("x_1"): (0.0, 2.0)}
    )
    try:
        p_sing = parse_expression("x_0 / (x_1 - 1)")
        reg = analyze_regimes(p_sing["expr"], ranges_sing)
        check(f"[{key_type} keys] regimes: returns dict",               isinstance(reg, dict))
        check(f"[{key_type} keys] regimes: has 'variables'",            "variables" in reg)
        check(f"[{key_type} keys] regimes: each var has 'derivative'",  all("derivative" in v for v in reg["variables"].values()))
        check(f"[{key_type} keys] regimes: each var has 'interpretation'", all("interpretation" in v for v in reg["variables"].values()))
        check(f"[{key_type} keys] singularity at x_1=1 detected",      1.0 in reg["variables"].get("x_1", {}).get("singularities", []))
    except Exception:
        check(f"[{key_type} keys] analyze_regimes crashed", False, traceback.format_exc().splitlines()[-1])

# monotonic check
p_lin = parse_expression("x_0 + 2*x_1")
reg_lin = analyze_regimes(p_lin["expr"], {"x_0": (-1.0, 1.0), "x_1": (0.0, 2.0)})
for vname, vinfo in reg_lin["variables"].items():
    check(f"linear {vname}: monotonic==True", vinfo["monotonic"])
    check(f"linear {vname}: interpretation is monotonic string", "monotonic" in vinfo["interpretation"])


# ---------------------------------------------------------------------------
# 5. sensitivity_analysis
# ---------------------------------------------------------------------------
section("5. sensitivity_analysis")
for label, (expr_str, var_ranges) in CASES.items():
    try:
        p = parse_expression(expr_str)
        sa = sensitivity_analysis(p["expr"], var_ranges)
        check(f"{label}: returns dict",              isinstance(sa, dict))
        check(f"{label}: has variable_ranking",      "variable_ranking" in sa)
        check(f"{label}: has warnings list",         isinstance(sa.get("warnings"), list))
        check(f"{label}: partial_effects keys are str", all(isinstance(k, str) for k in sa.get("partial_effects", {})))
        check(f"{label}: no interactions key",       "interactions" not in sa)
        check(f"{label}: no derivative_summaries key", "derivative_summaries" not in sa)
    except Exception:
        check(f"{label}: sensitivity_analysis crashed", False, traceback.format_exc().splitlines()[-1])

# singularity in range → warning expected (x_0 midpoint nonzero to avoid cancellation)
p_div = parse_expression("x_0 / (x_1 - 1)")
sa_div = sensitivity_analysis(p_div["expr"], {"x_0": (0.5, 1.5), "x_1": (0.0, 2.0)})
check("singularity: non-finite warning present", any("non-finite" in w.lower() or "singular" in w.lower() for w in sa_div["warnings"]))


# ---------------------------------------------------------------------------
# 6. generate_report
# ---------------------------------------------------------------------------
section("6. generate_report")
for label, (expr_str, var_ranges) in CASES.items():
    try:
        rep = generate_report(expr_str, var_ranges=var_ranges, var_map=VAR_MAP)
        check(f"{label}: returns dict",                isinstance(rep, dict))
        check(f"{label}: all required keys present",   REQUIRED_REPORT_KEYS <= rep.keys())
        check(f"{label}: facts_for_llm is dict",       isinstance(rep["facts_for_llm"], dict))
        check(f"{label}: facts has required keys",     REQUIRED_FACTS_KEYS <= rep["facts_for_llm"].keys())
        check(f"{label}: warnings is list",            isinstance(rep["warnings"], list))
        check(f"{label}: parsed['original'] matches",  rep["parsed"]["original"] == expr_str)
    except Exception:
        check(f"{label}: generate_report crashed", False, traceback.format_exc().splitlines()[-1])

# no var_ranges → complexity/sensitivity/regimes are None
rep_noranges = generate_report("x_0 + x_1")
check("no ranges: complexity is None",   rep_noranges["complexity"] is None)
check("no ranges: sensitivity is None",  rep_noranges["sensitivity"] is None)
check("no ranges: regimes is None",      rep_noranges["regimes"] is None)


# ---------------------------------------------------------------------------
# 7. generate_plain_language_prompt
# ---------------------------------------------------------------------------
section("7. generate_plain_language_prompt")
rep_main = generate_report("x_0*x_1 + 3", var_ranges={"x_0": (-1.0, 1.0), "x_1": (0.0, 2.0)}, var_map=VAR_MAP)
prompt = generate_plain_language_prompt(rep_main)
check("prompt: returns str",             isinstance(prompt, str))
check("prompt: contains STRICT RULES",   "STRICT RULES" in prompt)
check("prompt: contains VERIFIED FACTS", "VERIFIED FACTS" in prompt)
check("prompt: contains TASK",           "TASK" in prompt)
check("prompt: caution on causality",    "causality" in prompt.lower() or "infer" in prompt.lower())
check("prompt: mentions input ranges",   "input range" in prompt.lower() or "inspected" in prompt.lower())


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
section("SUMMARY")
passed = sum(1 for _, ok, _ in _results if ok)
total = len(_results)
failed = [(n, d) for n, ok, d in _results if not ok]
print(f"\n{passed}/{total} checks passed.")
if failed:
    print("\nFailed checks:")
    for name, detail in failed:
        print(f"  ✗ {name}" + (f"  — {detail}" if detail else ""))
    sys.exit(1)
else:
    print("\nAll checks passed.")

from __future__ import annotations

import re

import numpy as np
import sympy
from sympy import sympify, simplify, symbols, Symbol, Piecewise
from sympy.core.traversal import postorder_traversal
from sympy.core.relational import Relational


def _count_nodes(expr: sympy.Basic) -> int:
    return sum(1 for _ in postorder_traversal(expr))


def _extract_numeric_constants(expr: sympy.Basic) -> list[float]:
    return sorted(
        {float(a) for a in postorder_traversal(expr) if isinstance(a, sympy.Number)},
        key=abs,
    )


# Build sympy symbol namespace from all x_N tokens; always include Piecewise
def _build_local_ns(expr_str: str) -> dict[str, Symbol]:
    ns = {v: symbols(v) for v in sorted(set(re.findall(r"x_\d+", expr_str)))}
    ns["Piecewise"] = Piecewise
    return ns


# Walk the tree and classify each variable as boolean (in a condition) or continuous
def _detect_roles(expr: sympy.Basic) -> tuple[set, set]:
    boolean_vars: set[str] = set()
    continuous_vars: set[str] = set()

    def walk(node: sympy.Basic, in_cond: bool) -> None:
        if isinstance(node, sympy.Symbol):
            (boolean_vars if in_cond else continuous_vars).add(str(node))
        elif isinstance(node, Piecewise):
            for val, cond in node.args:
                walk(val, False)
                walk(cond, True)
        elif isinstance(node, Relational):
            for arg in node.args:
                walk(arg, True)
        else:
            for arg in node.args:
                walk(arg, in_cond)

    walk(expr, False)
    return boolean_vars, continuous_vars


def _apply_var_map(s: str, var_map: dict | None) -> str:
    if not var_map:
        return s
    for tok in sorted(var_map, key=len, reverse=True):
        s = s.replace(tok, var_map[tok])
    return s


# Collect unique Relational nodes (comparisons) from the expression tree
def _find_boolean_nodes(expr: sympy.Basic) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for node in postorder_traversal(expr):
        if isinstance(node, Relational):
            s = str(node)
            if s not in seen:
                seen.add(s)
                result.append(s)
    return result


# Decompose expression into per-branch dicts; wraps plain expressions as a single True branch
def _extract_branches(expr: sympy.Basic, var_map: dict | None) -> list[dict]:
    if isinstance(expr, Piecewise):
        return [
            {"condition": _apply_var_map(str(cond), var_map), "expr": val}
            for val, cond in expr.args
        ]
    return [{"condition": "True", "expr": expr}]


# Parse a GP expression and return its Piecewise structure, boolean nodes, and variable roles
def parse_mixed_expression(expr_str: str, var_map: dict | None = None) -> dict:
    local_ns = _build_local_ns(expr_str)
    expr = sympify(expr_str, locals=local_ns)
    boolean_vars, continuous_vars = _detect_roles(expr)
    return {
        "piecewise_expr": expr,
        "boolean_nodes": _find_boolean_nodes(expr),
        "dual_role_vars": sorted(boolean_vars & continuous_vars),
        "branches": _extract_branches(expr, var_map),
    }


# Sweep each variable over its range (50 pts), fix others at midpoint, rank by output range
def variable_contributions(expr: sympy.Basic, var_ranges: dict) -> dict:
    active = [v for v in var_ranges if v in expr.free_symbols]
    mids = {v: (var_ranges[v][0] + var_ranges[v][1]) / 2 for v in active}

    rankings = []
    partial_effects = {}

    for var in active:
        lo, hi = var_ranges[var]
        x_vals = np.linspace(lo, hi, 50)
        fixed = expr.subs({v: mids[v] for v in active if v != var})
        if fixed.is_number or fixed.has(sympy.zoo, sympy.oo, sympy.nan):
            try:
                scalar = float(fixed)
            except (TypeError, ValueError):
                scalar = float("nan")
            y_vals = np.full(50, scalar)
        else:
            f = sympy.lambdify(var, fixed, "numpy")
            y_vals = np.asarray(f(x_vals), dtype=float)
            if y_vals.ndim == 0:
                y_vals = np.full(50, float(y_vals))
        finite = y_vals[np.isfinite(y_vals)]
        if len(finite) > 1:
            output_range = float(finite.max() - finite.min())
        elif len(finite) == 0:
            output_range = float("nan")  # singularity hit midpoint of fixed vars
        else:
            output_range = 0.0
        rankings.append((var, output_range))
        partial_effects[var] = (x_vals, y_vals)

    rankings.sort(key=lambda t: t[1], reverse=True)
    return {"rankings": rankings, "partial_effects": partial_effects}


# Check all variable pairs via mixed partial d²f/dxi dxj; nonzero means xi and xj interact
def variable_interactions(expr: sympy.Basic) -> dict:
    vars_in_expr = sorted(expr.free_symbols, key=str)
    interacting_pairs: list[tuple] = []
    interacts_with: set = set()

    for i, xi in enumerate(vars_in_expr):
        for xj in vars_in_expr[i + 1 :]:
            d2f = sympy.diff(expr, xi, xj)
            if d2f != 0:
                interacting_pairs.append((xi, xj, d2f))
                interacts_with |= {xi, xj}

    independent_vars = [v for v in vars_in_expr if v not in interacts_with]
    return {"interacting_pairs": interacting_pairs, "independent_vars": independent_vars}


_N = 200  # sample size for variance estimates


# Evaluate expr at sample points; convert non-finite to nan
def _eval_samples(expr: sympy.Basic, active: list, samples: dict) -> np.ndarray:
    f = sympy.lambdify(active, expr, "numpy")
    y = np.asarray(f(*[samples[v] for v in active]), dtype=float)
    if y.ndim == 0:
        y = np.full(_N, float(y))
    y[~np.isfinite(y)] = np.nan
    return y


# Node count + per-subtree variance contribution estimated from random samples
def complexity_report(expr: sympy.Basic, var_ranges: dict) -> dict:
    active = sorted([v for v in var_ranges if v in expr.free_symbols], key=str)
    rng = np.random.default_rng(42)
    samples = {v: rng.uniform(var_ranges[v][0], var_ranges[v][1], _N) for v in active}

    y_full = _eval_samples(expr, active, samples)
    var_full = float(np.nanvar(y_full))

    active_set = set(active)
    subtrees = []
    for node in expr.args:  # only direct children of the root
        if isinstance(node, sympy.Number):
            continue
        if isinstance(node, sympy.Symbol) and node not in active_set:
            continue
        sub_vals = _eval_samples(node, active, samples)
        sub_mean = float(np.nanmean(sub_vals))
        y_rep = _eval_samples(expr.subs(node, sub_mean), active, samples)
        pct = 100.0 * (var_full - float(np.nanvar(y_rep))) / var_full if var_full > 0 else 0.0
        subtrees.append({"expr": str(node), "contribution_pct": round(pct, 2),
                         "_node": node, "_mean": sub_mean})

    subtrees.sort(key=lambda s: s["contribution_pct"], reverse=True)

    # if one term dominates (>80%), recompute contributions of the rest with it fixed
    dominant_term = None
    residual_subtrees: list[dict] = []
    if subtrees and subtrees[0]["contribution_pct"] > 80:
        dom = subtrees[0]
        dominant_term = dom["expr"]
        expr_res = expr.subs(dom["_node"], dom["_mean"])
        y_res = _eval_samples(expr_res, active, samples)
        var_res = float(np.nanvar(y_res))
        for s in subtrees[1:]:
            mean2 = float(np.nanmean(_eval_samples(s["_node"], active, samples)))
            y_rep2 = _eval_samples(expr_res.subs(s["_node"], mean2), active, samples)
            pct2 = 100.0 * (var_res - float(np.nanvar(y_rep2))) / var_res if var_res > 0 else 0.0
            residual_subtrees.append({"expr": s["expr"], "contribution_pct": round(pct2, 2)})
        residual_subtrees.sort(key=lambda s: s["contribution_pct"], reverse=True)

    for s in subtrees:
        s.pop("_node"); s.pop("_mean")

    return {
        "node_count": _count_nodes(expr),
        "subtrees": subtrees,
        "pruning_candidates": [s["expr"] for s in subtrees if s["contribution_pct"] < 5.0],
        "dominant_term": dominant_term,
        "residual_subtrees": residual_subtrees,
    }


# Real roots of f(var)=0 within [lo, hi]; returns [] if f is identically zero
def _real_roots_in_range(f: sympy.Basic, var: sympy.Symbol, lo: float, hi: float) -> list[float]:
    if f == 0:
        return []
    try:
        sols = sympy.solve(f, var)
        pts = []
        for s in sols:
            try:
                ev = s.evalf()
                if abs(float(sympy.im(ev))) < 1e-10:
                    v = float(sympy.re(ev))
                    if lo <= v <= hi:
                        pts.append(round(v, 6))
            except (TypeError, ValueError):
                pass
        return sorted(set(pts))
    except Exception:
        return []


# Zeros of the denominator of f within [lo, hi] — division singularities
def _singularities_in_range(f: sympy.Basic, var: sympy.Symbol, lo: float, hi: float) -> list[float]:
    _, denom = sympy.fraction(sympy.together(f))
    return [] if denom == 1 else _real_roots_in_range(denom, var, lo, hi)


# All real poles of f regardless of range — mathematical singularities
def _singularities_global(f: sympy.Basic, var: sympy.Symbol) -> list[float]:
    _, denom = sympy.fraction(sympy.together(f))
    if denom == 1:
        return []
    try:
        sols = sympy.solve(denom, var)
        pts = []
        for s in sols:
            try:
                ev = s.evalf()
                if abs(float(sympy.im(ev))) < 1e-10:
                    pts.append(round(float(sympy.re(ev)), 6))
            except (TypeError, ValueError):
                pass
        return sorted(set(pts))
    except Exception:
        return []


# Per-variable critical points, inflection points, singularities, and monotonicity
def behavioral_regimes(expr: sympy.Basic, var_ranges: dict) -> dict:
    active = sorted([v for v in var_ranges if v in expr.free_symbols], key=str)
    mids = {v: (var_ranges[v][0] + var_ranges[v][1]) / 2 for v in active}

    variables = {}
    for var in active:
        lo, hi = var_ranges[var]
        fixed = expr.subs({v: mids[v] for v in active if v != var})
        df  = sympy.diff(fixed, var)
        d2f = sympy.diff(df, var)
        crit = _real_roots_in_range(df,  var, lo, hi)
        infl = _real_roots_in_range(d2f, var, lo, hi)
        sing = _singularities_in_range(fixed, var, lo, hi)
        variables[var] = {
            "critical_points":   crit,
            "inflection_points": infl,
            "singularities":     sing,
            "monotonic":         len(crit) == 0 and len(sing) == 0,
        }

    return {"variables": variables}


# Parse, simplify, and audit a GP expression string (continuous or mixed boolean/continuous)
def simplify_expression(expr_str: str, var_map: dict | None = None) -> dict:
    local_ns = _build_local_ns(expr_str)
    raw_expr = sympify(expr_str, locals=local_ns)
    nodes_before = _count_nodes(raw_expr)
    constants_before = set(_extract_numeric_constants(raw_expr))

    candidate = simplify(raw_expr)
    simplified_expr = candidate if _count_nodes(candidate) <= nodes_before else raw_expr
    nodes_after = _count_nodes(simplified_expr)
    constants_after = set(_extract_numeric_constants(simplified_expr))

    boolean_vars, continuous_vars = _detect_roles(simplified_expr)

    return {
        "original": expr_str,
        "simplified": str(simplified_expr),
        "simplified_named": _apply_var_map(str(simplified_expr), var_map),
        "expr": simplified_expr,
        "nodes_before": nodes_before,
        "nodes_after": nodes_after,
        "collapsed_constants": sorted(constants_before - constants_after),
        "is_mixed": bool(boolean_vars),
        "boolean_vars": sorted(boolean_vars),
        "continuous_vars": sorted(continuous_vars),
        "dual_role_vars": sorted(boolean_vars & continuous_vars),
    }


# ------------------------------------------------------------------
# HELPER — normalise string/Symbol keys in var_ranges to Symbol objects
# ------------------------------------------------------------------

def _normalize_var_ranges(var_ranges: dict, expr: sympy.Basic) -> dict:
    free = {str(s): s for s in expr.free_symbols}
    result: dict = {}
    for k, v in var_ranges.items():
        sym = free.get(str(k), k) if isinstance(k, str) else k
        result[sym] = v
    return result


# ------------------------------------------------------------------
# parse_expression
# ------------------------------------------------------------------

# Unified entry-point: parse, simplify, detect mixed structure
def parse_expression(expr_str: str, var_map: dict | None = None) -> dict:
    simp = simplify_expression(expr_str, var_map)
    expr = simp["expr"]
    branches = _extract_branches(expr, var_map) if isinstance(expr, Piecewise) else None
    return {
        "original": expr_str,
        "simplified": simp["simplified"],
        "simplified_named": simp["simplified_named"],
        "expr": expr,
        "variables": sorted(str(v) for v in expr.free_symbols),
        "constants": _extract_numeric_constants(expr),
        "node_count_before": simp["nodes_before"],
        "node_count_after": simp["nodes_after"],
        "is_mixed": simp["is_mixed"],
        "boolean_vars": simp["boolean_vars"],
        "continuous_vars": simp["continuous_vars"],
        "dual_role_vars": simp["dual_role_vars"],
        "branches": branches,
    }


# ------------------------------------------------------------------
# classify_symbols
# ------------------------------------------------------------------

# Classify every free symbol by structural role (continuous / boolean / mixed)
def classify_symbols(expr: sympy.Basic, variable_names: dict | None = None) -> dict:
    boolean_vars, continuous_vars = _detect_roles(expr)
    result: dict = {}
    for sym in sorted(expr.free_symbols, key=str):
        name = str(sym)
        in_bool = name in boolean_vars
        in_cont = name in continuous_vars
        role = "mixed" if (in_bool and in_cont) else ("boolean" if in_bool else "continuous")
        occurrences = sum(1 for n in postorder_traversal(expr) if n == sym)
        result[name] = {
            "symbol": name,
            "display_name": (variable_names or {}).get(name),
            "role": role,
            "in_arithmetic": in_cont,
            "in_relational": in_bool,
            "occurrences": occurrences,
        }
    return result


# ------------------------------------------------------------------
# find_constant_recurrences
# ------------------------------------------------------------------

# Count numeric constants; flag values that recur in non-trivial positions
def find_constant_recurrences(expr: sympy.Basic) -> dict:
    counts: dict[float, int] = {}
    for node in postorder_traversal(expr):
        if isinstance(node, sympy.Number):
            v = float(node)
            counts[v] = counts.get(v, 0) + 1

    def _trivial(v: float) -> bool:
        return abs(v) < 1e-12 or abs(abs(v) - 1.0) < 1e-12

    constants = sorted(
        [
            {
                "value": v,
                "count": c,
                "is_trivial": _trivial(v),
                "simplification_candidate": c > 1 and not _trivial(v),
            }
            for v, c in counts.items()
        ],
        key=lambda d: (-d["count"], -abs(d["value"])),
    )
    recurrent = [d for d in constants if d["simplification_candidate"]]
    return {"constants": constants, "recurrent": recurrent, "has_recurrences": bool(recurrent)}


# ------------------------------------------------------------------
# analyze_regimes
# ------------------------------------------------------------------

# Extend behavioral_regimes with derivative expressions and a short interpretation string
def analyze_regimes(expr: sympy.Basic, var_ranges: dict) -> dict:
    var_ranges = _normalize_var_ranges(var_ranges, expr)
    base = behavioral_regimes(expr, var_ranges)
    active = sorted([v for v in var_ranges if v in expr.free_symbols], key=str)
    mids = {v: (var_ranges[v][0] + var_ranges[v][1]) / 2 for v in active}

    variables: dict = {}
    for var in active:
        info = base["variables"][var]
        fixed = expr.subs({v: mids[v] for v in active if v != var})
        df = sympy.diff(fixed, var)
        d2f = sympy.diff(df, var)
        sings = info["singularities"]
        crits = info["critical_points"]
        if info["monotonic"]:
            interp = "monotonic over the inspected range"
        elif sings:
            pts = ", ".join(str(p) for p in sings)
            interp = (
                f"has singularit{'ies' if len(sings) > 1 else 'y'} at {pts}"
                " — output unbounded near those points"
            )
        elif crits:
            pts = ", ".join(str(p) for p in crits)
            interp = f"has possible regime changes at {pts}"
        else:
            interp = "non-monotonic over the inspected range"
        sing_global = _singularities_global(fixed, var)
        variables[str(var)] = {
            **info,
            # keep "singularities" for backward compatibility — same as inside_range
            "singularities_inside_range": info["singularities"],
            "singularities_global": sing_global,
            "derivative": str(df),
            "second_derivative": str(d2f),
            "interpretation": interp,
        }
    return {"variables": variables}


# ------------------------------------------------------------------
# sensitivity_analysis
# ------------------------------------------------------------------

# Partial-effect ranking; flags non-finite sweep values as warnings
def sensitivity_analysis(expr: sympy.Basic, var_ranges: dict) -> dict:
    var_ranges = _normalize_var_ranges(var_ranges, expr)
    contrib = variable_contributions(expr, var_ranges)
    warnings_list: list[str] = []

    for var, (_, y_vals) in contrib["partial_effects"].items():
        if not np.all(np.isfinite(y_vals)):
            warnings_list.append(f"Non-finite values in partial effect sweep of {str(var)}")

    return {
        "variable_ranking": [(str(v), r) for v, r in contrib["rankings"]],
        "partial_effects": {str(k): v for k, v in contrib["partial_effects"].items()},
        "warnings": warnings_list,
    }


# ------------------------------------------------------------------
# Internal: assemble only verified facts suitable for LLM input
# ------------------------------------------------------------------

_MONOTONICITY_METHOD = (
    "Fix all other variables at their data-range midpoint; "
    "verify that the first derivative has no real zeros and the expression has no poles "
    "within the inspected range."
)
_CRITICAL_POINT_METHOD = (
    "Fix all other variables at their data-range midpoint; "
    "solve the first derivative equal to zero analytically within the inspected range."
)

def _build_facts_for_llm(
    parsed: dict,
    symbols_info: dict,
    constants_report: dict,
    sensitivity: dict | None,
    regimes: dict | None,
    var_ranges: dict | None = None,
) -> dict:
    facts: dict = {
        "expression": {
            "original": parsed["original"],
            "simplified": parsed["simplified"],
            "simplified_named": parsed["simplified_named"],
            "node_count_after_simplification": parsed["node_count_after"],
            "is_mixed_boolean_continuous": parsed["is_mixed"],
        },
        "variables": {
            name: {
                "display_name": info["display_name"],
                "role": info["role"],
                "occurrences": info["occurrences"],
            }
            for name, info in symbols_info.items()
        },
        "constants": {
            "non_trivial_values": [
                c["value"] for c in constants_report["constants"] if not c["is_trivial"]
            ],
            "recurrent_values": [c["value"] for c in constants_report["recurrent"]],
        },
        "variable_ranking": [],
        "variable_ranking_method": "output_range_sensitivity",
        "input_ranges": {
            str(k): list(v) for k, v in (var_ranges or {}).items()
        },
        "regimes": {},
        "monotonicity_method": _MONOTONICITY_METHOD,
        "critical_point_method": _CRITICAL_POINT_METHOD,
        "warnings": [],
    }
    if parsed["dual_role_vars"]:
        facts["warnings"].append(
            "Dual-role variables (appear in both boolean and arithmetic contexts): "
            + str(parsed["dual_role_vars"])
        )
    if sensitivity and "error" not in sensitivity:
        facts["variable_ranking"] = [
            {"variable": v, "output_range": r}
            for v, r in sensitivity["variable_ranking"]
            if np.isfinite(float(r))
        ]
        facts["warnings"].extend(sensitivity["warnings"])
    if regimes and "error" not in regimes:
        facts["regimes"] = {
            v: {
                "interpretation": info["interpretation"],
                "singularities_inside_range": info.get("singularities_inside_range", info.get("singularities", [])),
                "singularities_global": info.get("singularities_global", []),
                "critical_points": info["critical_points"],
                "monotonic": info["monotonic"],
            }
            for v, info in regimes["variables"].items()
        }
    return facts


# ------------------------------------------------------------------
# generate_report
# ------------------------------------------------------------------

# Main user-facing function: runs the full deterministic analysis pipeline
def generate_report(
    expr_str: str,
    var_ranges: dict | None = None,
    var_map: dict | None = None,
) -> dict:
    parsed = parse_expression(expr_str, var_map)
    expr = parsed["expr"]
    symbols_info = classify_symbols(expr, var_map)
    constants_report = find_constant_recurrences(expr)
    top_warnings: list[str] = []

    if parsed["dual_role_vars"]:
        top_warnings.append(f"Dual-role variables detected: {parsed['dual_role_vars']}")

    complexity = sensitivity = regimes = None
    if var_ranges is not None:
        sym_ranges = _normalize_var_ranges(var_ranges, expr)
        try:
            complexity = complexity_report(expr, sym_ranges)
        except Exception as exc:
            complexity = {"error": str(exc)}
            top_warnings.append(f"complexity_report failed: {exc}")
        try:
            sensitivity = sensitivity_analysis(expr, sym_ranges)
            top_warnings.extend(sensitivity.get("warnings", []))
        except Exception as exc:
            sensitivity = {"error": str(exc)}
            top_warnings.append(f"sensitivity_analysis failed: {exc}")
        try:
            regimes = analyze_regimes(expr, sym_ranges)
        except Exception as exc:
            regimes = {"error": str(exc)}
            top_warnings.append(f"analyze_regimes failed: {exc}")

    return {
        "parsed": parsed,
        "symbol_classification": symbols_info,
        "constants": constants_report,
        "complexity": complexity,
        "sensitivity": sensitivity,
        "regimes": regimes,
        "warnings": top_warnings,
        "facts_for_llm": _build_facts_for_llm(
            parsed, symbols_info, constants_report, sensitivity, regimes, var_ranges
        ),
    }


# ------------------------------------------------------------------
# generate_plain_language_prompt
# ------------------------------------------------------------------

# Build a safe prompt for a downstream LLM; encodes only verified facts
def generate_plain_language_prompt(report: dict) -> str:
    import json

    facts_json = json.dumps(report["facts_for_llm"], indent=2, default=str)
    return "\n".join(
        [
            "You are a scientific writing assistant helping a researcher understand a symbolic regression expression produced by genetic programming.",
            "",
            "STRICT RULES:",
            "1. Explain only the facts listed in VERIFIED FACTS below. Do not add information absent from it.",
            "2. Do not infer causality or domain meaning from variable names.",
            "3. Do not mention any variable not present in the 'variables' section.",
            "4. When describing behavior (monotone, bounded, singular), state explicitly that it holds "
            "only over the inspected input ranges.",
            "5. If something is approximate or uncertain, say so explicitly.",
            "6. Avoid phrases like 'this suggests', 'this implies', or 'therefore' unless directly "
            "traceable to a fact in VERIFIED FACTS.",
            "7. Variable ranking is an output-range sensitivity estimate over the inspected input ranges. "
            "Do not call it causal importance, global importance, or relevance unless those words appear "
            "explicitly in VERIFIED FACTS.",
            "8. When describing monotonicity, regime changes, or critical points, state that these are "
            "reported by the deterministic analysis under the method given in 'monotonicity_method' and "
            "'critical_point_method'. If those fields are absent, say the method is unspecified.",
            "",
            f"VERIFIED FACTS:\n{facts_json}",
            "",
            "TASK:",
            "Write a plain-language explanation of this symbolic regression expression for a non-expert reader.",
            "Cover:",
            "  (1) What the expression computes in structural terms.",
            "  (2) Which variables have the largest output-range sensitivity (use that phrase, not 'importance').",
            "  (3) Any numerical risks: singularities inside the inspected ranges, non-finite outputs.",
            "Do not invent domain knowledge. If a variable's scientific meaning is not given, "
            "refer to it by name only.",
        ]
    )

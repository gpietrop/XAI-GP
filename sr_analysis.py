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

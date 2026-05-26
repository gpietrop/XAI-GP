"""Local explanation functions for symbolic regression expressions.

Operates on a single input-output pair.
No derivatives, no SHAP, no perturbation analysis, no term decomposition.
"""
from __future__ import annotations

import json

import numpy as np
import sympy

from sr_analysis import parse_expression as _global_parse


# ── private helpers ────────────────────────────────────────────────────────

# Build display_name → x_N reverse lookup from var_map
def _reverse_var_map(var_map: dict | None) -> dict:
    return {v: k for k, v in (var_map or {}).items()}


# Convert input keys (display names or x_N strings) to canonical x_N names
def _normalize_input(input_values: dict, var_map: dict | None) -> tuple[dict, list[str]]:
    reverse = _reverse_var_map(var_map)
    normalized: dict = {}
    unresolved: list[str] = []
    for key, val in input_values.items():
        if key in reverse:
            normalized[reverse[key]] = val
        elif isinstance(key, str) and key.startswith("x_"):
            normalized[key] = val
        else:
            unresolved.append(str(key))
    return normalized, unresolved


# ── public API ─────────────────────────────────────────────────────────────

# Evaluate the symbolic expression at one specific input sample
def evaluate_expression_at_input(
    expr_str: str,
    input_values: dict,
    var_map: dict | None = None,
) -> dict:
    parsed = _global_parse(expr_str, var_map)
    expr = parsed["expr"]
    expr_vars = set(parsed["variables"])

    normalized, unresolved = _normalize_input(input_values, var_map)
    warnings_list: list[str] = []

    if unresolved:
        warnings_list.append(
            f"Unresolved input keys (not in var_map and not x_N format): {unresolved}"
        )

    missing = sorted(expr_vars - set(normalized))
    if missing:
        warnings_list.append(
            f"Variables present in expression but missing from input: {missing}"
        )

    subs = {sympy.Symbol(k): float(v) for k, v in normalized.items() if k in expr_vars}

    output: float | None = None
    is_finite = False

    if not missing:
        try:
            result = expr.subs(subs).evalf()
            if result.has(sympy.zoo, sympy.oo, sympy.nan):
                output = float("inf")
                is_finite = False
                warnings_list.append(f"Expression is non-finite at this input (SymPy: {result})")
            else:
                output = float(result)
                is_finite = bool(np.isfinite(output))
                if not is_finite:
                    warnings_list.append(f"Expression evaluated to non-finite value: {output}")
        except Exception as exc:
            warnings_list.append(f"Evaluation failed: {exc}")

    named_input = {
        (var_map or {}).get(k, k): v
        for k, v in normalized.items()
        if k in expr_vars
    }

    return {
        "original": expr_str,
        "simplified": parsed["simplified"],
        "simplified_named": parsed["simplified_named"],
        "input_normalized": {k: v for k, v in normalized.items() if k in expr_vars},
        "input_named": named_input,
        "variables_in_expr": sorted(expr_vars),
        "variables_not_in_input": missing,
        "output": output,
        "is_finite": is_finite,
        "warnings": warnings_list,
    }


# Build a structured local report for one input-output pair
def generate_local_facts(
    expr_str: str,
    input_values: dict,
    var_map: dict | None = None,
    model_context: dict | None = None,
    provided_output: float | None = None,
) -> dict:
    ev = evaluate_expression_at_input(expr_str, input_values, var_map)

    output_comparison: dict | None = None
    if provided_output is not None:
        if ev["output"] is not None and ev["is_finite"]:
            output_comparison = {
                "computed": ev["output"],
                "provided": float(provided_output),
                "absolute_discrepancy": abs(ev["output"] - float(provided_output)),
            }
        else:
            output_comparison = {
                "computed": ev["output"],
                "provided": float(provided_output),
                "absolute_discrepancy": None,
                "note": "Discrepancy not computed — computed output is non-finite or missing.",
            }

    return {
        "expression": {
            "original": ev["original"],
            "simplified": ev["simplified"],
            "simplified_named": ev["simplified_named"],
        },
        "input": {
            "values_normalized": ev["input_normalized"],
            "values_named": ev["input_named"],
            "variables_in_expr": ev["variables_in_expr"],
            "variables_not_in_input": ev["variables_not_in_input"],
        },
        "output": {
            "computed": ev["output"],
            "is_finite": ev["is_finite"],
        },
        "output_comparison": output_comparison,
        "warnings": ev["warnings"],
        "model_context": model_context or {},
    }


# Build a safe LLM prompt for explaining one specific prediction
def generate_local_plain_language_prompt(local_facts: dict) -> str:
    verified = {k: v for k, v in local_facts.items() if k != "model_context"}
    context = local_facts.get("model_context", {})

    verified_json = json.dumps(verified, indent=2, default=str)
    context_json = json.dumps(context, indent=2, default=str)

    return "\n".join([
        "You are a scientific writing assistant explaining a single prediction made by a "
        "symbolic regression model.",
        "",
        "STRICT RULES:",
        "1. Use only the facts in VERIFIED FACTS and the annotations in USER-PROVIDED CONTEXT.",
        "2. Clearly distinguish between computed facts (VERIFIED FACTS) and user-provided "
        "annotations (USER-PROVIDED CONTEXT). Do not treat context as computed.",
        "3. Do not infer causality or real-world reasons. You may describe how the expression "
        "is evaluated on this input, but do not claim that any variable caused the output.",
        "4. Do not infer domain meaning from variable names unless explicitly given in "
        "USER-PROVIDED CONTEXT.",
        "5. Do not make feature importance claims, do not say which input drove the output "
        "or contributed most.",
        "6. Do not make local sensitivity claims, do not say what would happen if an input changed.",
        "7. Do not mention variables not present in 'allowed_variable_names'.",
        "8. If the output is non-finite, state that explicitly and do not attempt to interpret it.",
        "9. If an output discrepancy is reported, state it factually without explaining why "
        "it occurred.",
        "10. Input values are normalized values. Do not describe them as raw scores unless "
        "USER-PROVIDED CONTEXT explicitly says so.",
        "11. Do not classify the sample as benign or pathogenic unless a decision threshold "
        "is explicitly provided in USER-PROVIDED CONTEXT.",
        "12. If warnings are empty, say only that no warnings were reported; do not claim "
        "that no numerical issue is possible in general.",
        "",
        f"VERIFIED FACTS:\n{verified_json}",
        "",
        f"USER-PROVIDED CONTEXT (not verified computationally):\n{context_json}",
        "",
        "TASK:",
        "Write a plain-language description of this single model prediction for a non-expert.",
        "Cover:",
        "  (1) What the expression computes structurally, in one sentence.",
        "  (2) The specific input values provided and the resulting computed output.",
        "  (3) Any warnings: non-finite output, missing inputs, output discrepancy.",
        "You may describe how the expression is evaluated on this input, but do not explain causal or real-world reasons for the output.",
        "Do not rank or weight the input variables.",
    ])

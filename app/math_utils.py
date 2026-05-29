"""Optional symbolic verification for linear equations."""

from __future__ import annotations

import re

from sympy import Eq, Rational, simplify, solve, symbols
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)


def _normalize_equation(text: str) -> str:
    cleaned = text.strip()
    cleaned = cleaned.replace("×", "*").replace("÷", "/")
    cleaned = cleaned.replace("−", "-").replace("–", "-").replace("—", "-")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def _split_equation(text: str) -> tuple[str, str] | None:
    for sep in ("=", "≈"):
        if sep in text:
            left, right = text.split(sep, 1)
            return left.strip(), right.strip()
    return None


def _parse_side(expr_text: str):
    return parse_expr(expr_text, transformations=TRANSFORMATIONS, evaluate=True)


def analyze_equation(question: str) -> str | None:
    """Return a short symbolic analysis if the input looks like an equation."""
    normalized = _normalize_equation(question)
    split = _split_equation(normalized)
    if not split:
        return None

    left_text, right_text = split
    try:
        left = _parse_side(left_text)
        right = _parse_side(right_text)
        x = symbols("x")
        equation = Eq(left, right)
        solution = solve(equation, x)

        if not solution:
            return f"Equation: {equation}\nNo single solution for x was found (check domain or equation type)."

        if len(solution) > 1:
            sol_text = ", ".join(str(s) for s in solution)
            return f"Equation: {equation}\nSolutions for x: {sol_text}"

        sol = solution[0]
        check_left = simplify(left.subs(x, sol))
        check_right = simplify(right.subs(x, sol))
        return (
            f"Equation: {equation}\n"
            f"Solution for x: {sol}\n"
            f"Verification: LHS = {check_left}, RHS = {check_right}\n"
            f"Match: {check_left == check_right or simplify(check_left - check_right) == 0}"
        )
    except Exception as exc:
        return f"Could not parse equation automatically: {exc}"


def looks_like_word_problem(question: str) -> bool:
    words = question.lower()
    indicators = [
        "perimeter",
        "cost",
        "price",
        "length",
        "width",
        "twice",
        "double",
        "triple",
        "additional",
        "packages",
        "meters",
        "feet",
        "total",
        "if ",
        "how much",
        "how many",
        "what are",
        "store",
        "room",
        "rectangle",
    ]
    return any(token in words for token in indicators) or len(question.split()) > 12

"""A minimal, safe calculator tool for TinyToolRL.

Three layers live in this file:

1. ``safe_eval``      -- pure-Python AST evaluator (no ``eval``!). Used by the
                         data generator to compute gold answers and by the tool
                         itself at train/eval time.
2. ``run_calculator`` -- string-in / string-out wrapper that never raises
                         (returns ``"Error: ..."`` instead), which is what the
                         LLM sees as the tool response.
3. ``calculator``     -- plain Python function handed to ``GRPOTrainer`` as
                         ``tools=[calculator]``. trl 1.x dropped the ``Tool``
                         subclass API: tools are callables with type hints +
                         Google-style docstrings, and the schema is derived
                         via ``get_json_schema()``. ``TOOL_SCHEMA`` here is
                         derived the same way, so SFT rendering / GRPO
                         generation / eval prompts are byte-identical by
                         construction.

The calculator is *stateless*, so we deliberately do NOT use OpenEnv --
TRL's plain ``tools=[...]`` support is the right level of engineering here.
"""

import ast
import operator

# --------------------------------------------------------------------------
# 1. Safe expression evaluation
# --------------------------------------------------------------------------

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_MAX_POW_EXPONENT = 64      # refuse 10**10000 style blowups
_MAX_EXPR_LENGTH = 200


class CalculatorError(ValueError):
    """Raised for anything the calculator refuses to evaluate."""


def safe_eval(expression: str) -> float:
    """Evaluate a basic arithmetic expression and return a number.

    Supports: + - * / // % ** (and ^ as an alias for **), parentheses,
    integer/float literals. Nothing else -- no names, no calls, no indexing.
    """
    expr = str(expression).strip()
    if len(expr) > _MAX_EXPR_LENGTH:
        raise CalculatorError("expression too long")
    expr = expr.replace("^", "**")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise CalculatorError(f"invalid expression: {e}") from e

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                return float(node.value)
            raise CalculatorError("only numeric literals are allowed")
        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in _BIN_OPS:
                raise CalculatorError("unsupported operator")
            left, right = _eval(node.left), _eval(node.right)
            if op_type is ast.Pow and abs(right) > _MAX_POW_EXPONENT:
                raise CalculatorError("exponent too large")
            if op_type in (ast.Div, ast.FloorDiv, ast.Mod) and right == 0:
                raise CalculatorError("division by zero")
            return _BIN_OPS[op_type](left, right)
        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in _UNARY_OPS:
                raise CalculatorError("unsupported unary operator")
            return _UNARY_OPS[op_type](_eval(node.operand))
        raise CalculatorError(f"unsupported syntax: {type(node).__name__}")

    result = _eval(tree)
    if isinstance(result, complex):
        raise CalculatorError("complex result not supported")
    return float(result)


def format_result(value: float) -> str:
    """Render a number the way gold answers are rendered ('3901', '12.5')."""
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    # avoid long float tails; 4 decimals is plenty for our data
    return str(round(value, 4))


# --------------------------------------------------------------------------
# 2. String-in / string-out wrapper (this is what the model sees)
# --------------------------------------------------------------------------

def run_calculator(expression: str) -> str:
    """Evaluate ``expression``; return the result or an 'Error: ...' string."""
    try:
        return format_result(safe_eval(expression))
    except CalculatorError as e:
        return f"Error: {e}"
    except Exception as e:  # pragma: no cover - defensive
        return f"Error: {e}"


# --------------------------------------------------------------------------
# 3. Tool function (trl 1.x: tools=[callable]) + derived schema
# --------------------------------------------------------------------------

TOOL_NAME = "calculator"


def calculator(expression: str) -> str:
    """Evaluates a basic arithmetic expression and returns the numeric result as a string.

    Supported operators: + - * / // % ** and parentheses. Example input: "(47 * 83) + 12".
    If the expression is invalid, returns a string starting with "Error:".

    Args:
        expression: Arithmetic expression to evaluate, e.g. "(47 * 83) + 12"

    Returns:
        The numeric result as a string, or an "Error: ..." string.
    """
    return run_calculator(expression)


# Single source of truth for the prompt-facing schema (OpenAI function format).
# Both GRPOTrainer(tools=[calculator]) and tokenizer.apply_chat_template(tools=[...])
# derive the schema from the function via get_json_schema() internally, so deriving
# TOOL_SCHEMA the same way keeps SFT / GRPO / eval prompts byte-identical.
# NOTE: the docstring format above is load-bearing -- transformers'
# parse_google_format_docstring() requires the description / "Args:" / "Returns:"
# sections exactly like this. Do not reformat without re-checking TOOL_SCHEMA.
try:
    from transformers.utils import get_json_schema

    TOOL_SCHEMA = get_json_schema(calculator)
except ImportError:  # keep this module importable without transformers (data generation)
    TOOL_SCHEMA = None


if __name__ == "__main__":
    # tiny self-test: python tools/calculator.py
    import json

    for expr in ["47 * 83", "(123 + 456) * 17", "10 / 0", "2^10", "foo + 1",
                 "960 / 12", "3.5 + 4.25"]:
        print(f"{expr!r:28} -> {run_calculator(expr)!r}")
    print("\nTOOL_SCHEMA (derived via get_json_schema):")
    print(json.dumps(TOOL_SCHEMA, indent=2, ensure_ascii=False))

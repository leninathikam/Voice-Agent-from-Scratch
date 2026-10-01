import ast
import operator
from contextvars import ContextVar
from datetime import datetime, timezone

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
}


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    raise ValueError("only + - * / and numbers are allowed")


def calculate(expression: str) -> str:
    tree = ast.parse(expression, mode="eval")
    return str(_eval(tree.body))


def current_time() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


# One list per voice session. The agent sets this before a turn so two
# browser tabs do not share notes.
notes_var: ContextVar[list[str] | None] = ContextVar("notes", default=None)


def _notes() -> list[str]:
    notes = notes_var.get()
    if notes is None:
        notes = []
        notes_var.set(notes)
    return notes


def remember(note: str) -> str:
    notes = _notes()
    notes.append(note.strip())
    return f"Saved. You have {len(notes)} note(s)."


def list_notes() -> str:
    notes = _notes()
    if not notes:
        return "No notes yet."
    return "\n".join(f"{i}. {n}" for i, n in enumerate(notes, start=1))


TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a simple arithmetic expression.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "current_time",
            "description": "Current time in UTC.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": "Store a short note for this session.",
            "parameters": {
                "type": "object",
                "properties": {"note": {"type": "string"}},
                "required": ["note"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_notes",
            "description": "List notes saved in this session.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

HANDLERS = {
    "calculate": lambda args: calculate(args["expression"]),
    "current_time": lambda args: current_time(),
    "remember": lambda args: remember(args["note"]),
    "list_notes": lambda args: list_notes(),
}


def run_tool(name: str, arguments: dict) -> str:
    handler = HANDLERS.get(name)
    if handler is None:
        return f"Unknown tool: {name}"
    try:
        return handler(arguments)
    except Exception as exc:
        return f"Tool error: {exc}"

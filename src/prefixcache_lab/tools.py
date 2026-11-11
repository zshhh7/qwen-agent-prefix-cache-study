from __future__ import annotations

import ast
import json
import operator
from typing import Any

from qwen_agent.tools.base import BaseTool, register_tool


UNIT_TO_METERS = {"毫米": 0.001, "厘米": 0.01, "米": 1.0, "千米": 1000.0}
UNIT_ALIASES = {
    "mm": "毫米", "millimeter": "毫米", "millimeters": "毫米",
    "cm": "厘米", "centimeter": "厘米", "centimeters": "厘米",
    "m": "米", "meter": "米", "meters": "米",
    "km": "千米", "kilometer": "千米", "kilometers": "千米",
}


@register_tool("unit_converter")
class UnitConverter(BaseTool):
    description = "在毫米、厘米、米和千米之间换算长度，返回换算后的确定数值。"
    parameters = {
        "type": "object",
        "properties": {
            "value": {"type": "number", "description": "需要换算的数值"},
            "from_unit": {"type": "string", "description": "原单位：毫米、厘米、米或千米"},
            "to_unit": {"type": "string", "description": "目标单位：毫米、厘米、米或千米"},
        },
        "required": ["value", "from_unit", "to_unit"],
    }

    def call(self, params: str, **kwargs: Any) -> str:
        data = json.loads(params)
        source_raw = str(data["from_unit"]).strip()
        target_raw = str(data["to_unit"]).strip()
        source = UNIT_ALIASES.get(source_raw.lower(), source_raw)
        target = UNIT_ALIASES.get(target_raw.lower(), target_raw)
        if source not in UNIT_TO_METERS or target not in UNIT_TO_METERS:
            raise ValueError("单位必须是毫米、厘米、米或千米")
        result = float(data["value"]) * UNIT_TO_METERS[source] / UNIT_TO_METERS[target]
        return json.dumps({"value": data["value"], "from_unit": source, "to_unit": target, "result": result}, ensure_ascii=False)


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def _safe_eval(expression: str) -> float:
    node = ast.parse(expression, mode="eval").body

    def evaluate(item: ast.AST) -> float:
        if isinstance(item, ast.Constant) and isinstance(item.value, (int, float)):
            return float(item.value)
        if isinstance(item, ast.UnaryOp) and isinstance(item.op, ast.USub):
            return -evaluate(item.operand)
        if isinstance(item, ast.BinOp) and type(item.op) in _OPS:
            return _OPS[type(item.op)](evaluate(item.left), evaluate(item.right))
        raise ValueError("Only numeric +, -, *, / expressions are allowed")

    return evaluate(node)


@register_tool("calculator")
class Calculator(BaseTool):
    description = "计算只含数字、括号及加减乘除的算术表达式。求总量时必须调用此工具。"
    parameters = {
        "type": "object",
        "properties": {"expression": {"type": "string", "description": "算术表达式，例如 24*9/5+32"}},
        "required": ["expression"],
    }

    def call(self, params: str, **kwargs: Any) -> str:
        expression = json.loads(params)["expression"]
        return json.dumps({"expression": expression, "result": _safe_eval(expression)}, ensure_ascii=False)


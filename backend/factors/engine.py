"""因子表达式解析与计算。

自定义因子采用受限表达式语法，例如：

    rank(ts_mean(close, 20) / ts_mean(close, 60) - 1)
    -ts_std(returns, 20)
    ts_corr(volume / ts_mean(volume, 20), returns, 10)

只允许算术运算、比较、条件表达式以及对白名单函数的调用，禁止属性访问、
下标、导入与赋值，保证安全。
"""
from __future__ import annotations

import ast
import functools

import jax.numpy as jnp
import numpy as np

from .ops import NAMESPACE, _as2d

FIELD_NAMES = ("open", "high", "low", "close", "volume", "amount", "vwap", "returns")
# 行业标签用于 group_rank / group_neutralize / group_zscore
GROUP_VAR = "industry"

_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Constant,
    ast.Load, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
    ast.USub, ast.UAdd, ast.UAdd, ast.Compare, ast.Lt, ast.Gt, ast.LtE, ast.GtE,
    ast.Eq, ast.NotEq, ast.BoolOp, ast.And, ast.Or, ast.IfExp, ast.Not,
    ast.Tuple,
)


class FactorError(ValueError):
    pass


@functools.lru_cache(maxsize=256)
def _compile(expression: str) -> tuple[object, ast.Expression]:
    expression = (expression or "").strip()
    if not expression:
        raise FactorError("因子表达式不能为空")
    if len(expression) > 800:
        raise FactorError("因子表达式过长（最多 800 字符）")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise FactorError(f"表达式语法错误: {exc.msg}") from exc
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise FactorError(f"不允许的语法节点: {type(node).__name__}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise FactorError("仅支持调用内置因子函数")
            if node.func.id not in NAMESPACE:
                raise FactorError(f"未知函数: {node.func.id}()")
            if node.keywords:
                raise FactorError("因子函数不支持关键字参数，请使用位置参数")
        if isinstance(node, ast.Name):
            if (node.id not in FIELD_NAMES and node.id != GROUP_VAR
                    and node.id not in NAMESPACE
                    and node.id not in ("True", "False", "inf", "nan")):
                raise FactorError(f"未知变量: {node.id}")
        if isinstance(node, ast.Attribute):
            raise FactorError("不允许属性访问")
    try:
        code = compile(tree, "<factor>", "eval")
    except Exception as exc:  # pragma: no cover
        raise FactorError(f"表达式编译失败: {exc}") from exc
    return code, tree


def validate(expression: str) -> None:
    _compile(expression)


def compute(expression: str, panel: dict) -> np.ndarray:
    """在面板上计算因子，返回 (T, N) 的 numpy 数组，缺失值为 NaN。"""
    code, _ = _compile(expression)
    env = {name: jnp.asarray(panel[name], dtype=jnp.float64) for name in FIELD_NAMES}
    industry = panel.get("industry_idx")
    if industry is None:
        industry = np.zeros(np.asarray(panel["close"]).shape[1], dtype=np.int64)
    env[GROUP_VAR] = jnp.asarray(industry, dtype=jnp.int64)
    env.update(NAMESPACE)
    env["inf"] = jnp.inf
    env["nan"] = jnp.nan
    env["True"] = True
    env["False"] = False
    try:
        out = eval(code, {"__builtins__": {}}, env)  # noqa: S307 - 已通过 AST 白名单
    except FactorError:
        raise
    except Exception as exc:
        raise FactorError(f"因子计算失败: {exc}") from exc
    try:
        arr = _as2d(out)
        result = np.asarray(arr, dtype=np.float64)
    except Exception as exc:
        raise FactorError(f"因子结果维度不合法: {exc}") from exc
    if result.shape != panel["close"].shape:
        raise FactorError("因子结果维度与面板不一致")
    return result

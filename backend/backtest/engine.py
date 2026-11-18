"""JAX 向量化回测引擎。

流程：因子打分 -> 每 K 日截面选股 -> 目标权重前向填充 -> 收益/换手/费用
全流程在 JAX 中完成，使用 lax.scan 沿时间轴推进，可编译到 GPU/TPU。
"""
from __future__ import annotations

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

NEG_INF = -1e30


def _bucket_weights(mask, ranks, k, n_valid, weighting: str, vol, short: bool):
    """在入选股票内分配权重：等权 / 得分加权 / 波动率倒数加权。"""
    if weighting == "score":
        if short:
            pos = ranks - (n_valid[:, None] - k[:, None])
        else:
            pos = k[:, None] - ranks
        w = jnp.where(mask, jnp.maximum(pos, 0.0), 0.0)
    elif weighting == "inverse_vol":
        w = jnp.where(mask, 1.0 / jnp.maximum(vol, 1e-4), 0.0)
    else:
        w = mask.astype(jnp.float64)
    total = jnp.sum(w, axis=1, keepdims=True)
    return w / jnp.maximum(total, 1e-12)


@partial(jax.jit, static_argnames=("top_n", "long_short", "weighting", "max_weight"))
def _target_weights(
    score: jnp.ndarray,
    valid: jnp.ndarray,
    rebalance: jnp.ndarray,
    top_frac: float,
    top_n: int,
    long_short: bool,
    weighting: str,
    max_weight: float,
    vol: jnp.ndarray,
    timing: jnp.ndarray,
):
    T, N = score.shape
    clean = jnp.where(valid, jnp.nan_to_num(score, nan=NEG_INF, posinf=1e30, neginf=NEG_INF), NEG_INF)
    order = jnp.argsort(-clean, axis=1)
    ranks = jnp.argsort(order, axis=1).astype(jnp.float64)
    n_valid = jnp.sum(valid, axis=1).astype(jnp.float64)

    if top_n > 0:
        k = jnp.minimum(float(top_n), n_valid)
    else:
        k = jnp.floor(top_frac * n_valid + 0.5)
    k = jnp.clip(k, 0.0, n_valid)

    long_mask = (ranks < k[:, None]) & valid
    long_w = _bucket_weights(long_mask, ranks, k, n_valid, weighting, vol, short=False)
    weights = long_w
    if long_short:
        short_mask = (ranks >= (n_valid[:, None] - k[:, None])) & valid & ~long_mask
        short_w = _bucket_weights(short_mask, ranks, k, n_valid, weighting, vol, short=True)
        weights = long_w - short_w

    if max_weight > 0:
        cap = jnp.maximum(max_weight, 1.0 / jnp.maximum(k[:, None], 1.0))
        for _ in range(4):
            weights = jnp.where(valid, jnp.clip(weights, -cap, cap), 0.0)
            net = jnp.sum(jnp.abs(weights), axis=1, keepdims=True)
            weights = weights / jnp.maximum(net, 1e-12)

    target = jnp.where(rebalance[:, None], weights * timing[:, None], 0.0)
    return target


@partial(jax.jit, static_argnames=("top_n", "long_short", "weighting", "max_weight", "constraints"))
def _simulate(
    score: jnp.ndarray,
    close: jnp.ndarray,
    valid: jnp.ndarray,
    rebalance: jnp.ndarray,
    top_frac: float,
    top_n: int,
    long_short: bool,
    commission: float,
    stamp_duty: float,
    slippage: float,
    weighting: str,
    max_weight: float,
    vol: jnp.ndarray,
    timing: jnp.ndarray,
    buy_block: jnp.ndarray,
    sell_block: jnp.ndarray,
    constraints: bool,
):
    T, N = score.shape
    target = _target_weights(
        score, valid, rebalance, top_frac, top_n, long_short,
        weighting, max_weight, vol, timing,
    )

    def fill(w_prev, inputs):
        tgt, rb, bb, sb = inputs
        desired = jnp.where(rb, tgt, w_prev)
        if constraints:
            adjusted = jnp.where(
                bb & (tgt > w_prev), w_prev,
                jnp.where(sb & (tgt < w_prev), w_prev, tgt),
            )
            desired = jnp.where(rb, adjusted, w_prev)
        return desired, desired

    _, W = jax.lax.scan(fill, jnp.zeros(N), (target, rebalance, buy_block, sell_block))

    prev_c, next_c = close[:-1], close[1:]
    rets = next_c / prev_c - 1.0
    rets = jnp.nan_to_num(rets, nan=0.0, posinf=0.0, neginf=0.0)
    rets = jnp.clip(rets, -0.25, 0.25)

    gross = jnp.sum(W[:-1] * rets, axis=1)
    dw = jnp.sum(jnp.abs(W[1:] - W[:-1]), axis=1)
    # A 股成本模型：买入=佣金+滑点；卖出=佣金+印花税+滑点
    delta = W[1:] - W[:-1]
    buys = jnp.sum(jnp.maximum(delta, 0.0), axis=1)
    sells = jnp.sum(jnp.maximum(-delta, 0.0), axis=1)
    trade_cost = (commission + slippage) * buys + (commission + stamp_duty + slippage) * sells
    cost = jnp.concatenate([jnp.zeros((1,), dtype=dw.dtype), trade_cost[:-1]])
    net = gross - cost

    nav = jnp.concatenate([jnp.ones((1,), dtype=net.dtype), jnp.cumprod(1.0 + net)])

    n_valid = jnp.maximum(jnp.sum(valid, axis=1, keepdims=True), 1.0)
    bench_w = valid / n_valid
    bench_gross = jnp.sum(bench_w[:-1] * rets, axis=1)
    bench_nav = jnp.concatenate(
        [jnp.ones((1,), dtype=bench_gross.dtype), jnp.cumprod(1.0 + bench_gross)]
    )

    peak = jnp.maximum.accumulate(nav)
    drawdown = nav / peak - 1.0
    return nav, bench_nav, net, drawdown, dw, W


@jax.jit
def _metrics(net: jnp.ndarray, nav: jnp.ndarray, turnover: jnp.ndarray):
    n = net.shape[0]
    ann_ret = jnp.power(jnp.maximum(nav[-1], 1e-9), 252.0 / jnp.maximum(n, 1)) - 1.0
    mean = jnp.mean(net)
    std = jnp.std(net)
    ann_vol = std * jnp.sqrt(252.0)
    sharpe = jnp.where(std > 1e-12, mean / std * jnp.sqrt(252.0), 0.0)
    peak = jnp.maximum.accumulate(nav)
    max_dd = jnp.min(nav / peak - 1.0)
    calmar = jnp.where(jnp.abs(max_dd) > 1e-9, ann_ret / jnp.abs(max_dd), 0.0)
    win_rate = jnp.mean(net > 0)
    ann_turnover = jnp.mean(turnover) * 252.0
    return {
        "total_return": nav[-1] - 1.0,
        "annual_return": ann_ret,
        "annual_volatility": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "win_rate": win_rate,
        "annual_turnover": ann_turnover,
        "trade_days": jnp.asarray(float(n)),
    }


def run_backtest(
    score: np.ndarray,
    close: np.ndarray,
    valid: np.ndarray,
    rebalance: np.ndarray,
    top_frac: float = 0.2,
    top_n: int = 0,
    long_short: bool = False,
    commission: float = 0.0015,
    stamp_duty: float = 0.0,
    slippage: float = 0.0,
    weighting: str = "equal",
    max_weight: float = 0.0,
    vol: np.ndarray | None = None,
    timing: np.ndarray | None = None,
    buy_block: np.ndarray | None = None,
    sell_block: np.ndarray | None = None,
    constraints: bool = False,
) -> dict[str, Any]:
    score_j = jnp.asarray(score, dtype=jnp.float64)
    close_j = jnp.asarray(close, dtype=jnp.float64)
    valid_j = jnp.asarray(valid, dtype=bool)
    reb_j = jnp.asarray(rebalance, dtype=bool)
    vol_j = jnp.asarray(vol if vol is not None else np.ones_like(close), dtype=jnp.float64)
    timing_j = jnp.asarray(timing if timing is not None else np.ones(close.shape[0]), dtype=jnp.float64)
    buy_j = jnp.asarray(buy_block if buy_block is not None else np.zeros(close.shape, dtype=bool), dtype=bool)
    sell_j = jnp.asarray(sell_block if sell_block is not None else np.zeros(close.shape, dtype=bool), dtype=bool)

    nav, bench, net, drawdown, dw, W = _simulate(
        score_j, close_j, valid_j, reb_j,
        float(top_frac), int(top_n), bool(long_short), float(commission),
        float(stamp_duty), float(slippage),
        str(weighting), float(max_weight), vol_j, timing_j,
        buy_j, sell_j, bool(constraints),
    )
    m = _metrics(net, nav, dw)

    return {
        "nav": np.asarray(nav, dtype=np.float64),
        "benchmark": np.asarray(bench, dtype=np.float64),
        "returns": np.asarray(net, dtype=np.float64),
        "drawdown": np.asarray(drawdown, dtype=np.float64),
        "turnover": np.asarray(dw, dtype=np.float64),
        "weights": np.asarray(W, dtype=np.float64),
        "metrics": {k: float(v) for k, v in m.items()},
    }

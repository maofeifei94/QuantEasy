# AlphaLab · A股量化分析平台

一个本地运行的 A 股多因子研究与回测平台。后端 Python + FastAPI，数据落地
SQLite，因子计算与回测由 **JAX** 向量化执行，前端为无构建依赖的原生网页。

```
行情同步 (miniQMT / xtquant)  →  SQLite  →  JAX 因子引擎  →  JAX 回测引擎  →  网页可视化
```

## 功能

- **行情数据**：优先通过 miniQMT(`xtquant`) 拉取沪深 A 股日线并写入 SQLite；
  未检测到客户端时自动切换内置合成行情，开箱即可研究。
- **因子库**：内置 14 个常用因子（动量/反转、低波动、量价、流动性、RSI、MACD 等），
  覆盖时序与截面两类算子。
- **自定义因子**：受限表达式语法，如
  `rank(ts_mean(close, 20) / ts_mean(close, 60) - 1)`，服务端 AST 白名单校验，
  禁止导入/属性访问等危险操作。
- **因子检验**：Rank IC 时间序列、IC 均值 / ICIR / 胜率、IC 衰减、分位组合收益、
  单样本因子走势。
- **回测**：自选股票池、调仓周期、持仓数量/比例、交易成本、多空组合；
  输出净值曲线、基准对比、回撤、年化/夏普/最大回撤/换手/胜率、月度热力图、
  调仓持仓明细，结果持久化到 SQLite。
- **计算引擎**：滚动窗口、截面排序、组合权重、收益与绩效指标全部用 JAX 编写，
  通过 `jax.jit` 编译，`lax.scan` 沿时间轴推进向量化回测。

## 快速开始

```bash
pip install -r requirements.txt
./run.sh                    # 默认 http://localhost:8000
# 或指定端口
PORT=8080 ./run.sh
```

首次启动会自动生成演示行情（150 只股票 / 约 3 年日线）写入 `data/quant.db`。

## 接入 miniQMT

1. 在 **Windows** 机器安装并登录 miniQMT / QMT 客户端，开启「极简模式」数据服务；
2. `pip install xtquant`（本仓库依赖可选，未安装时自动降级为合成数据）；
3. 在同一台机器启动本服务，在「行情数据」页选择 `miniQMT / xtquant` 并同步。

适配器实现见 `backend/datasources/miniqmt.py`：

- `xtdata.get_stock_list_in_sector("沪深A股")` 获取成分；
- `download_history_data2` 下载本地缓存；
- `get_market_data_ex(..., dividend_type="front")` 读取前复权日线；
- 成交量由「手」统一换算为「股」，日期统一为 `YYYY-MM-DD`。

## 自定义因子语法

可用字段：`open high low close volume amount vwap returns`

可用函数：

| 类别 | 函数 |
| --- | --- |
| 时序 | `ts_mean(x,n)` `ts_std(x,n)` `ts_sum(x,n)` `ts_min(x,n)` `ts_max(x,n)` `ts_rank(x,n)` |
| 时序 | `ts_delay(x,n)` `ts_delta(x,n)` `ts_corr(x,y,n)` |
| 截面 | `rank(x)` `zscore(x)` |
| 数学 | `log abs sign sqrt exp tanh max min where clip safe_div` |

示例：

```
close / ts_delay(close, 20) - 1                              # 20 日动量
-ts_std(returns, 20)                                         # 低波动
rank(ts_mean(volume, 5) / ts_mean(volume, 60))               # 量能放大
-ts_corr(volume, close, 20) * rank(ts_std(returns, 20))      # 自定义复合因子
```

## API 摘要

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/status` | 数据库与数据源状态 |
| POST | `/api/data/sync` | 创建同步任务（`source`: auto/miniqmt/synthetic） |
| GET | `/api/data/jobs/{id}` | 同步任务进度 |
| GET | `/api/stocks` | 股票列表与最新行情 |
| GET | `/api/factors` | 内置 + 自定义因子 |
| POST | `/api/factors` | 新建自定义因子 |
| POST | `/api/factors/preview` | IC / 分位收益分析 |
| POST | `/api/backtest` | 运行回测并保存 |
| GET | `/api/backtests` | 历史回测列表 |

## 目录结构

```
backend/
  main.py              FastAPI 路由与静态资源
  database.py          SQLite schema 与数据访问
  panel.py             日线 → (T, N) 面板对齐
  services.py          同步 / 因子预览 / 回测编排
  datasources/
    miniqmt.py         xtquant 数据适配器
    synthetic.py       合成行情生成器
  factors/
    ops.py             JAX 时序/截面算子
    engine.py          因子表达式校验与求值
    builtin.py         内置因子定义
    analysis.py        IC / IC 衰减 / 分位分析
  backtest/
    engine.py          JAX 向量化回测与绩效指标
frontend/
  index.html           单页界面
  css/style.css
  js/charts.js         轻量 Canvas 图表（折线/柱状/K线）
  js/app.js            页面逻辑与 API 调用
data/quant.db          运行时生成
```

## 说明

- 回测为日频、收盘价成交近似，已按 `commission_bps` 扣除换手成本；
  未考虑涨跌停无法成交、停牌与滑点等微观约束，结果仅用于研究验证。
- 合成行情仅用于演示流程，不代表任何真实标的，不构成投资建议。

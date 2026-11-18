# AlphaLab · A股量化分析平台

一个本地运行的 A 股多因子研究与回测平台。后端 Python + FastAPI，数据落地
SQLite，因子计算与回测由 **JAX** 向量化执行，前端为无构建依赖的原生网页。

```
行情同步 (miniQMT / xtquant)  →  SQLite  →  JAX 因子引擎  →  JAX 回测引擎  →  网页可视化
```

## 功能

- **行情数据**：优先通过 miniQMT(`xtquant`) 拉取沪深 A 股日线并写入 SQLite；
  未检测到客户端时自动切换内置合成行情，开箱即可研究；提供数据质量检测（覆盖率/缺失/异常值）。
- **因子库**：内置 14 个常用因子（动量/反转、低波动、量价、流动性、RSI、MACD 等），
  覆盖时序与截面两类算子。
- **多因子组合**：支持任意因子加权合成（权重可正可负），自动按截面标准化后相加，
  并提供因子相关性矩阵用于共线性检查。
- **因子预处理**：MAD 去极值 → 行业中性化 → 截面标准化 → 风格正交（动量/波动/量比），可逐项开关。
- **自定义因子**：受限表达式语法，如
  `rank(ts_mean(close, 20) / ts_mean(close, 60) - 1)`，服务端 AST 白名单校验，
  禁止导入/属性访问等危险操作；支持编辑、删除、搜索；创建时可一键载入示例表达式，
  并在保存前「测试表达式」即时查看 IC / ICIR / t 值 / 覆盖率 / Q5-Q1 年化收益。
- **因子检验**：Rank IC 时间序列、IC 均值 / ICIR / t 统计量 / 胜率、分年度与分行业 IC、IC 衰减、
  分位组合收益、单样本因子走势。
- **因子工厂**：批量评估约 40 个候选表达式，按 |ICIR| 排序，一键收藏为自定义因子。
- **批量评测**：同一回测配置下评测全部内置因子，按夏普/ICIR 排行，一键选用。
- **股票池过滤**：剔除科创板/北交所、过滤当日涨停、最小日均成交额、最少上市天数。
- **成交约束**：可模拟涨停无法买入、跌停无法卖出、停牌不可交易。
- **风控**：个股价差止损、止盈（自建仓价起算）与移动止损（自持有期最高价回撤触发），触发后当次调仓不再回补。
- **交易摩擦控制**：最小持有期（未满期限的仓位延迟卖出）与单次调仓换手上限（按比例向目标仓位过渡），降低换手与成本冲击。
- **组合构建**：等权 / 因子得分加权 / 波动率倒数 / 风险平价（ERC），支持单票权重上限、行业总权重上限与最大仓位控制（不足部分留作现金）。
- **交易成本**：佣金（双边）、印花税（卖出）、滑点分项建模，贴近 A 股实际交易成本。
- **回测**：自选股票池、调仓方式（交易日间隔 / 每周固定星期 / 每月第 N 个交易日）、持仓数量/比例、多空组合、MA20/MA60 大盘择时；
  基准可选股票池等权、全市场等权、保存的自选池等权或沪深300/中证500 指数；输出净值曲线、回撤、
  Alpha/Beta、信息比率、索提诺、VaR/CVaR、上下行捕获、行业暴露、风格暴露（动量/反转/波动/流动性/Beta）、
  收益归因、Brinson 归因、
  Bootstrap 置信区间、滚动夏普、日收益分布、交易明细、回撤区间分析（峰值/谷底/修复/持续天数）、年度/月度收益与调仓持仓，并支持样本内/样本外拆分检验，
  结果持久化到 SQLite，支持净值/交易/持仓 CSV 与 HTML 研究报告导出。
- **智能选股**：自由组合内置/自定义因子的「排名前/后」或「因子值 ≥/≤」条件（AND 关系），
  排名支持全市场或**行业内**比较（避免行业集中）；按指定交易日截面筛选；
  支持选股方案保存/载入、结果导出 CSV、保存为自选股票池，
  并可一键将结果载入回测股票池，形成「选股 → 回测」闭环。
- **自选股票池**：勾选股票后可命名保存到 SQLite，随时载入复用。
- **最新信号**：按当前因子、过滤与权重配置生成最新交易日的目标持仓，可导出 CSV。
- **多策略组合**：选择最多 6 条历史回测合成组合净值，支持手动权重或按最小方差 / 最大夏普 / 风险平价自动优化权重，并给出策略相关性矩阵。
- **策略模板**：动量趋势、低波防御、短期反转、多因子均衡，一键套用配置。
- **研究效率**：多因子 ICIR 自动加权、因子月度 IC 热力图、分行业 IC、因子稳定性（排名自相关）、
  浏览器本地保存/载入回测配置。
- **分层回测**：因子五分组净值曲线（Q1–Q5）与 Q5-Q1 多空净值，直观检验单调性。
- **参数扫描**：调仓周期 × 持仓数量的 3×3 绩效网格，快速定位稳健参数区间。
- **成本敏感性**：0~30 bps 单边成本档位的年化/夏普变化，评估策略对成本的敏感度。
- **年度基准对比**：年度收益同时展示策略、基准与超额收益。
- **回测对比**：历史记录中任选最多 3 条，叠加净值曲线并对比核心指标。
- **计算引擎**：滚动窗口、截面排序、预处理、组合权重、收益与绩效指标全部用 JAX 编写，
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
| 时序 | `ts_delay(x,n)` `ts_delta(x,n)` `ts_corr(x,y,n)` `ts_zscore(x,n)` |
| 时序 | `ts_skew(x,n)` `ts_kurt(x,n)` `ts_argmax(x,n)` `ts_argmin(x,n)` `decay_linear(x,n)` |
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
| GET | `/api/data/quality` | 数据质量检测 |
| GET | `/api/data/jobs/{id}` | 同步任务进度 |
| GET | `/api/stocks` | 股票列表与最新行情 |
| GET | `/api/indices` | 可用指数基准（沪深300/中证500） |
| GET | `/api/universes` | 自选股票池列表 |
| POST | `/api/universes` | 保存自选股票池 |
| GET | `/api/factors` | 内置 + 自定义因子 |
| POST | `/api/factors` | 新建自定义因子 |
| PUT | `/api/factors/{id}` | 编辑自定义因子 |
| POST | `/api/factors/preview` | IC / 分位收益分析 |
| POST | `/api/factors/correlation` | 多因子相关性矩阵 |
| POST | `/api/factors/mine` | 因子工厂批量筛选 |
| POST | `/api/backtest` | 运行回测并保存 |
| POST | `/api/backtest/optimize` | 调仓周期 × 持仓数参数扫描 |
| POST | `/api/backtest/batch` | 全内置因子批量评测排行 |
| POST | `/api/backtest/cost-sensitivity` | 交易成本敏感性 |
| POST | `/api/signal` | 最新交易日目标持仓信号 |
| POST | `/api/screener` | 多条件因子选股 |
| POST | `/api/portfolio/combine` | 多策略组合净值 |
| GET | `/api/backtests` | 历史回测列表 |
| DELETE | `/api/backtests/{id}` | 删除指定回测记录 |

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
  indices.py           宽基指数代理构建
  factors/
    ops.py             JAX 时序/截面算子
    engine.py          因子表达式校验与求值
    preprocess.py      MAD 去极值 / 行业中性 / 标准化
    builtin.py         内置因子定义
    analysis.py        IC / IC 衰减 / 分位分析 / 相关性矩阵
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

"""miniQMT (xtquant) 行情适配器。

设计要点：
- xtquant 只能在安装了 QMT/miniQMT 客户端的环境中导入，采用懒加载；
- 下载与读取分离：download_history_data 落地本地缓存，get_market_data_ex 读取；
- 所有字段做归一化（日期 YYYY-MM-DD，价格复权可配置），异常时抛出
  DataSourceError 由上层转换为友好提示。
"""
from __future__ import annotations

import datetime as dt
from typing import Any, Callable

DEFAULT_SECTOR = "沪深A股"


class DataSourceError(RuntimeError):
    pass


def is_available() -> bool:
    try:
        import xtquant.xtdata  # noqa: F401
        return True
    except Exception:
        return False


def _import_xtdata():
    try:
        from xtquant import xtdata
        return xtdata
    except Exception as exc:  # pragma: no cover - 依赖客户端环境
        raise DataSourceError(
            "未检测到 xtquant。请在安装 miniQMT 客户端的机器上执行 "
            "`pip install xtquant`，并确保客户端已登录、极简模式数据服务已开启。"
        ) from exc


def _norm_date(value: Any) -> str | None:
    s = str(value)
    digits = "".join(ch for ch in s if ch.isdigit())
    if len(digits) < 8:
        return None
    return f"{digits[0:4]}-{digits[4:6]}-{digits[6:8]}"


def _sector_codes(xtdata, sector: str) -> list[str]:
    try:
        codes = xtdata.get_stock_list_in_sector(sector)
    except Exception as exc:
        raise DataSourceError(f"读取板块 {sector} 成分失败: {exc}") from exc
    codes = [c for c in (codes or []) if isinstance(c, str)]
    # 过滤北交所、退市整理等非 A 股主板/创业板/科创板标的
    return [c for c in codes if c.split(".")[0][:1] in ("0", "3", "6")]


def _stock_name(xtdata, code: str) -> str:
    try:
        detail = xtdata.get_instrument_detail(code)
        if isinstance(detail, dict):
            return str(detail.get("InstrumentName") or "")
    except Exception:
        pass
    return ""


def sync(
    codes: list[str] | None = None,
    start: str = "20220101",
    end: str = "",
    limit: int = 300,
    dividend_type: str = "front",
    progress: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    """从 miniQMT 拉取日线数据并返回 (stocks, bars)。"""
    xtdata = _import_xtdata()
    end = end or dt.date.today().strftime("%Y%m%d")
    start_compact = start.replace("-", "")
    end_compact = end.replace("-", "")

    if progress:
        progress(0.02, "连接 miniQMT 数据服务")
    if codes:
        stock_codes = codes
    else:
        stock_codes = _sector_codes(xtdata, DEFAULT_SECTOR)
    if not stock_codes:
        raise DataSourceError("未获取到股票列表，请确认 QMT 客户端已登录。")
    stock_codes = stock_codes[: max(1, limit)]

    if progress:
        progress(0.05, f"开始下载 {len(stock_codes)} 只股票日线")
    try:
        download = getattr(xtdata, "download_history_data2", None)
        if download is not None:
            for idx in range(0, len(stock_codes), 100):
                batch = stock_codes[idx: idx + 100]
                download(batch, "1d", start_compact, end_compact)
                if progress:
                    progress(0.05 + 0.45 * (idx + len(batch)) / len(stock_codes),
                             f"下载日线 {min(idx + 100, len(stock_codes))}/{len(stock_codes)}")
        else:
            for i, code in enumerate(stock_codes):
                xtdata.download_history_data(code, "1d", start_compact, end_compact)
                if progress and i % 20 == 0:
                    progress(0.05 + 0.45 * i / len(stock_codes), f"下载日线 {i}/{len(stock_codes)}")
    except Exception as exc:
        raise DataSourceError(f"miniQMT 下载历史数据失败: {exc}") from exc

    stocks: list[dict] = []
    bars: list[tuple] = []
    batch_size = 50
    for idx in range(0, len(stock_codes), batch_size):
        batch = stock_codes[idx: idx + batch_size]
        if progress:
            progress(0.5 + 0.45 * idx / len(stock_codes),
                     f"读取行情 {min(idx + batch_size, len(stock_codes))}/{len(stock_codes)}")
        try:
            data = xtdata.get_market_data_ex(
                field_list=[], stock_list=batch, period="1d",
                start_time=start_compact, end_time=end_compact,
                dividend_type=dividend_type, fill_data=True,
            )
        except Exception as exc:
            raise DataSourceError(f"miniQMT 读取行情失败: {exc}") from exc

        for code in batch:
            df = data.get(code)
            if df is None or len(df) == 0:
                continue
            name = _stock_name(xtdata, code)
            stocks.append({
                "code": code,
                "name": name or code,
                "industry": "",
                "market": code.split(".")[-1] if "." in code else "",
                "list_date": "",
            })
            for ts, row in df.iterrows():
                date = _norm_date(ts)
                close = row.get("close")
                if date is None or close is None or close != close:
                    continue
                try:
                    # xtdata 日线 volume 单位为手，统一换算为股
                    vol = float(row.get("volume", 0) or 0) * 100.0
                    bars.append((
                        code, date,
                        float(row.get("open", close)), float(row.get("high", close)),
                        float(row.get("low", close)), float(close),
                        vol, float(row.get("amount", 0) or 0),
                    ))
                except (TypeError, ValueError):
                    continue

    indices: list[dict] = []
    if progress:
        progress(0.96, "读取指数基准")
    for index_code, index_name in (("000300.SH", "沪深300"), ("000905.SH", "中证500")):
        try:
            xtdata.download_history_data(index_code, "1d", start_compact, end_compact)
            idx_data = xtdata.get_market_data_ex(
                field_list=[], stock_list=[index_code], period="1d",
                start_time=start_compact, end_time=end_compact,
                dividend_type="none", fill_data=True,
            )
            df = idx_data.get(index_code)
            if df is None or len(df) == 0:
                continue
            rows = []
            for ts, row in df.iterrows():
                date = _norm_date(ts)
                close = row.get("close")
                if date is None or close is None or close != close:
                    continue
                rows.append((index_code, date, round(float(close), 2)))
            if rows:
                indices.append({"code": index_code, "name": index_name, "rows": rows})
        except Exception:
            continue

    if progress:
        progress(1.0, f"完成，共 {len(stocks)} 只股票 / {len(bars)} 条日线")
    if not bars:
        raise DataSourceError("miniQMT 返回空数据，请检查日期区间与客户端数据权限。")
    return {
        "stocks": stocks,
        "bars": bars,
        "indices": indices,
        "source": "miniqmt",
        "start": start,
        "end": end,
    }

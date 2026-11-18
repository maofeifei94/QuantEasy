/* AlphaLab 前端逻辑 */
(function () {
  "use strict";

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  const state = {
    status: null,
    stocks: [],
    factors: [],
    selectedFactor: null,
    universe: new Set(),
    btFactors: [],
    compare: new Set(),
    indices: [],
    universes: [],
    btResult: null,
    syncTimer: null,
    stockFilter: "",
    factorQuery: "",
    historyQuery: "",
    screenerConds: [],
    screenerResult: null,
    polling: false,
  };

  const PAGE_META = {
    overview: ["Workspace / Overview", "研究总览"],
    data: ["Workspace / Market Data", "行情数据"],
    factors: ["Research / Factor Lab", "因子实验室"],
    screener: ["Research / Screener", "智能选股"],
    backtest: ["Research / Backtest", "回测引擎"],
    history: ["Archive / Backtests", "回测记录"],
  };

  const CAT_ORDER = ["动量/反转", "波动率", "量价", "流动性", "技术指标", "自定义"];

  /* 参数化因子模板：选择模板 + 调整窗口 → 自动生成表达式 */
  const FACTOR_TEMPLATES = [
    { id: "momentum", label: "N 日动量", prefix: "动量", params: [{ key: "n", label: "窗口", def: 20 }],
      build: (p) => `rank(close / ts_delay(close, ${p.n}) - 1)` },
    { id: "reversal", label: "N 日反转", prefix: "反转", params: [{ key: "n", label: "窗口", def: 5 }],
      build: (p) => `rank(-(close / ts_delay(close, ${p.n}) - 1))` },
    { id: "lowvol", label: "低波动", prefix: "低波动", params: [{ key: "n", label: "窗口", def: 20 }],
      build: (p) => `rank(-ts_std(returns, ${p.n}))` },
    { id: "volratio", label: "量比", prefix: "量比", params: [{ key: "n", label: "窗口", def: 20 }],
      build: (p) => `rank(volume / ts_mean(volume, ${p.n}))` },
    { id: "bias", label: "均线乖离", prefix: "乖离", params: [{ key: "n", label: "窗口", def: 20 }],
      build: (p) => `rank(close / ts_mean(close, ${p.n}) - 1)` },
    { id: "amt_trend", label: "成交额趋势", prefix: "额趋势", params: [{ key: "n", label: "短窗", def: 5 }, { key: "m", label: "长窗", def: 20 }],
      build: (p) => `rank(ts_mean(amount, ${p.n}) / ts_mean(amount, ${p.m}) - 1)` },
    { id: "pv_corr", label: "量价相关性", prefix: "量价相关", params: [{ key: "n", label: "窗口", def: 20 }],
      build: (p) => `rank(-ts_corr(close, volume, ${p.n}))` },
    { id: "skew", label: "收益偏度", prefix: "偏度", params: [{ key: "n", label: "窗口", def: 20 }],
      build: (p) => `rank(-ts_skew(returns, ${p.n}))` },
    { id: "maxret", label: "极端收益（彩票效应）", prefix: "极端收益", params: [{ key: "n", label: "窗口", def: 20 }],
      build: (p) => `rank(-ts_max(returns, ${p.n}))` },
    { id: "range_pos", label: "价格区间位置", prefix: "区间位置", params: [{ key: "n", label: "窗口", def: 60 }],
      build: (p) => `rank((close - ts_min(low, ${p.n})) / (ts_max(high, ${p.n}) - ts_min(low, ${p.n}) + 1e-9))` },
    { id: "rsi", label: "RSI 相对强弱", prefix: "RSI", params: [{ key: "n", label: "周期", def: 14 }],
      build: (p) => `rank(ts_mean(max(ts_delta(close, 1), 0), ${p.n}) / (ts_mean(abs(ts_delta(close, 1)), ${p.n}) + 1e-9))` },
    { id: "macd", label: "双均线强弱", prefix: "双均线", params: [{ key: "n", label: "短窗", def: 12 }, { key: "m", label: "长窗", def: 26 }],
      build: (p) => `rank(ts_mean(close, ${p.n}) / ts_mean(close, ${p.m}) - 1)` },
    { id: "vol_stab", label: "量能稳定性", prefix: "量能稳定", params: [{ key: "n", label: "基窗", def: 60 }, { key: "m", label: "平滑", def: 20 }],
      build: (p) => `rank(-ts_std(volume / ts_mean(volume, ${p.n}), ${p.m}))` },
  ];

  /* ------------------------------ 工具 ------------------------------ */
  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[c]));
  }
  function fmtPct(v, digits = 2, signed = true) {
    if (v == null || !isFinite(v)) return "—";
    const n = v * 100;
    return (signed && n > 0 ? "+" : "") + n.toFixed(digits) + "%";
  }
  function fmtNum(v, digits = 2) {
    if (v == null || !isFinite(v)) return "—";
    return Number(v).toFixed(digits);
  }
  function fmtInt(v) {
    if (v == null) return "—";
    return Number(v).toLocaleString("en-US");
  }
  function fmtMoney(v) {
    if (v == null || !isFinite(v)) return "—";
    if (Math.abs(v) >= 1e8) return (v / 1e8).toFixed(2) + " 亿";
    if (Math.abs(v) >= 1e4) return (v / 1e4).toFixed(1) + " 万";
    return Number(v).toFixed(0);
  }
  function shortDate(d) { return d ? d.slice(2, 10) : "—"; }

  async function api(path, options = {}) {
    const opts = { headers: { "Content-Type": "application/json" }, ...options };
    if (opts.body && typeof opts.body !== "string") opts.body = JSON.stringify(opts.body);
    const resp = await fetch(path, opts);
    let data = null;
    try { data = await resp.json(); } catch (e) { /* ignore */ }
    if (!resp.ok) {
      const msg = (data && (data.detail || data.message)) || `请求失败 (${resp.status})`;
      throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
    }
    return data;
  }

  let toastTimer = null;
  function toast(message, isError = false) {
    const el = $("#toast");
    el.textContent = message;
    el.className = "toast on" + (isError ? " err" : "");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { el.className = "toast"; }, 3600);
  }

  function colorClass(v) {
    if (v > 0) return "pos";
    if (v < 0) return "neg";
    return "";
  }

  /* ------------------------------ 路由 ------------------------------ */
  function switchPage(name) {
    $$(".nav-item").forEach((b) => b.classList.toggle("active", b.dataset.page === name));
    $$(".page").forEach((p) => p.classList.toggle("active", p.id === "page-" + name));
    const meta = PAGE_META[name] || ["", ""];
    $("#pageEyebrow").textContent = meta[0];
    $("#pageTitle").textContent = meta[1];
    if (name === "history") loadHistory();
    if (name === "overview") renderOverview();
  }

  /* ------------------------------ 状态 ------------------------------ */
  async function loadStatus(initial = false) {
    const s = await api("/api/status");
    state.status = s;
    renderStatus(initial);
    return s;
  }

  function renderStatus(initial) {
    const s = state.status;
    if (!s) return;

    $("#ovStocks").textContent = fmtInt(s.stock_count);
    $("#ovBars").textContent = fmtInt(s.bar_count);
    $("#ovRange").textContent = s.start_date ? `${shortDate(s.start_date)} → ${shortDate(s.end_date)}` : "暂无数据";
    $("#ovSource").textContent = s.source === "miniqmt" ? "miniQMT" : s.source === "synthetic" ? "合成演示" : "未同步";

    $("#dsStocks").textContent = fmtInt(s.stock_count);
    $("#dsBars").textContent = fmtInt(s.bar_count);
    $("#dsRange").textContent = s.start_date ? `${s.start_date} → ${s.end_date}` : "—";
    const lastSync = s.last_sync;
    $("#dsSync").textContent = lastSync ? lastSync.time : "—";

    const isMini = s.source === "miniqmt";
    const isSynth = s.source === "synthetic";
    $("#srcDot").className = "dot " + (isMini ? "ok" : isSynth ? "warn" : "");
    $("#srcName").textContent = isMini ? "miniQMT 已连接" : isSynth ? "合成行情演示" : "未同步数据";
    $("#srcMeta").textContent = `SQLite · ${fmtInt(s.bar_count)} bars`;

    $("#pillDot").className = "dot" + (isMini ? "" : " demo");
    $("#pillText").textContent = isMini ? "miniQMT · 实盘数据" : isSynth ? "合成演示数据" : "等待同步";

    const hint = s.miniqmt_available
      ? "已检测到 xtquant，可直接从 miniQMT 客户端同步真实日线行情。"
      : "当前环境未检测到 xtquant。请在安装 miniQMT 客户端并登录后运行本服务以同步真实行情；现在可使用内置合成行情完成因子研究与回测。";
    $("#sourceHint").textContent = hint;

    if (initial) {
      $("#syncSource").value = s.miniqmt_available ? "auto" : (s.stock_count ? "synthetic" : "auto");
    }

    const endInput = $("#syncEnd");
    if (!endInput.value && s.end_date) endInput.value = s.end_date;
    const btEnd = $("#btEnd");
    if (!btEnd.value && s.end_date) btEnd.value = s.end_date;
    const btStart = $("#btStart");
    if (s.start_date && s.start_date > btStart.value) btStart.value = s.start_date;

    renderMarket();
    renderUniverseCount();
  }

  async function waitForData() {
    if (state.polling) return;
    if (!state.status || state.status.stock_count > 0) return;
    state.polling = true;
    for (let i = 0; i < 80; i++) {
      await new Promise((r) => setTimeout(r, 1500));
      const s = await loadStatus();
      if (s.stock_count > 0) {
        await Promise.all([loadStocks(), loadFactors()]);
        renderUniverse();
        toast("演示数据已准备就绪");
        break;
      }
    }
    state.polling = false;
  }

  /* ------------------------------ 总览 ------------------------------ */
  function renderOverview() {
    renderStatus();
    renderMarket();
  }

  function renderMarket() {
    const rows = state.stocks
      .filter((s) => s.close != null && s.change_pct != null)
      .sort((a, b) => b.change_pct - a.change_pct)
      .slice(0, 7);
    const box = $("#marketList");
    if (!rows.length) {
      box.innerHTML = `<div class="empty" style="border:none"><p>暂无行情数据</p><div class="sub">请先在「行情数据」页同步</div></div>`;
      return;
    }
    $("#marketDate").textContent = rows[0].date || "—";
    box.innerHTML = rows.map((r) => `
      <div class="market-row" data-code="${esc(r.code)}" style="cursor:pointer">
        <div class="nm"><span class="nm" style="max-width:120px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(r.name)}</span><span class="code">${esc(r.code)}</span></div>
        <div class="px">${fmtNum(r.close)}</div>
        <div class="px ${r.change_pct >= 0 ? "up" : "down"}">${r.change_pct > 0 ? "+" : ""}${fmtNum(r.change_pct)}%</div>
      </div>`).join("");
    $$("#marketList .market-row").forEach((row) => {
      row.addEventListener("click", () => openStockModal(row.dataset.code));
    });
  }

  /* ------------------------------ 行情数据 ------------------------------ */
  async function loadStocks() {
    const data = await api("/api/stocks?limit=2000");
    state.stocks = data.items || [];
    renderStocks();
    renderMarket();
  }

  async function openQuality() {
    try {
      const q = await api("/api/data/quality");
      const cards = [
        ["交易日", fmtInt(q.trading_days)],
        ["股票数量", fmtInt(q.stocks)],
        ["数据覆盖率", fmtPct(q.coverage, 1, false)],
        ["缺失行数", fmtInt(q.missing_rows)],
        ["零成交量", fmtInt(q.zero_volume)],
        ["异常收盘价", fmtInt(q.bad_close)],
        ["单股最少记录", fmtInt(q.min_bars_per_stock)],
        ["单股平均记录", fmtNum(q.avg_bars_per_stock, 0)],
        ["指数记录", fmtInt(q.index_rows)],
      ];
      openModal(`
        <div class="modal-head"><div><h3>数据质量检测</h3>
          <div class="panel-sub" style="margin-top:3px">检查覆盖率、缺失值与异常值</div></div>
          <button class="icon-btn" data-close>✕</button></div>
        <div class="grid-3" style="grid-template-columns:repeat(3,1fr)">
          ${cards.map(([k, v]) => `<div class="perf"><div class="k">${k}</div><div class="v">${v}</div></div>`).join("")}
        </div>
        <div class="hint-box" style="margin-top:14px">覆盖率 = 实际记录数 / (交易日 × 股票数)。若覆盖率明显低于 100%，可能存在停牌、退市或同步区间不一致。</div>
        <div class="modal-foot"><button class="btn primary" data-close>关闭</button></div>`, true);
    } catch (e) {
      toast(e.message, true);
    }
  }

  async function loadIndices() {
    try {
      const data = await api("/api/indices");
      state.indices = data.items || [];
    } catch (e) {
      state.indices = [];
    }
    try {
      const ud = await api("/api/universes");
      state.universes = ud.items || [];
    } catch (e) {
      state.universes = [];
    }
    const sel = $("#btBenchmark");
    if (!sel) return;
    const current = sel.value;
    sel.innerHTML = `<option value="universe">股票池等权</option><option value="market">全市场等权</option>` +
      state.indices.filter((i) => i.count > 30).map((i) =>
        `<option value="index:${esc(i.code)}">${esc(i.name)}</option>`
      ).join("") +
      state.universes.map((u) =>
        `<option value="universe:${u.id}">自选池基准 · ${esc(u.name)}</option>`
      ).join("");
    if (current && Array.from(sel.options).some((o) => o.value === current)) sel.value = current;
  }

  function renderStocks() {
    const q = state.stockFilter.trim().toLowerCase();
    const rows = state.stocks.filter((s) =>
      !q || s.code.toLowerCase().includes(q) || (s.name || "").toLowerCase().includes(q)
    );
    const body = $("#stockTableBody");
    if (!rows.length) {
      body.innerHTML = `<tr><td colspan="7" style="text-align:center;color:var(--faint);padding:30px">暂无数据</td></tr>`;
      return;
    }
    body.innerHTML = rows.map((r) => {
      const cls = r.change_pct > 0 ? "up" : r.change_pct < 0 ? "down" : "flat";
      return `<tr data-code="${esc(r.code)}">
        <td><span class="cd">${esc(r.code)}</span></td>
        <td><span class="nm">${esc(r.name || "—")}</span></td>
        <td><span class="chip">${esc(r.industry || "—")}</span></td>
        <td style="text-align:right" class="num-mono">${fmtNum(r.close)}</td>
        <td style="text-align:right" class="num-mono ${cls}">${r.change_pct > 0 ? "+" : ""}${r.change_pct == null ? "—" : fmtNum(r.change_pct) + "%"}</td>
        <td style="text-align:right" class="num-mono" >${fmtMoney(r.amount)}</td>
        <td style="text-align:right" class="num-mono" ><span style="color:var(--faint)">${esc(r.date || "—")}</span></td>
      </tr>`;
    }).join("");
    $$("#stockTableBody tr").forEach((tr) => {
      if (!tr.dataset.code) return;
      tr.addEventListener("click", () => openStockModal(tr.dataset.code));
    });
    renderIndustryFilter();
  }

  function renderIndustryFilter() {
    const names = [...new Set(state.stocks.map((s) => s.industry).filter(Boolean))].sort();
    ["#filterIndustries", "#scrIndustries"].forEach((sel) => {
      const el = $(sel);
      if (!el) return;
      const cur = new Set(Array.from(el.selectedOptions).map((o) => o.value));
      el.innerHTML = names.map((n) =>
        `<option value="${esc(n)}" ${cur.has(n) ? "selected" : ""}>${esc(n)}</option>`
      ).join("");
    });
  }

  function movingAverage(values, n) {
    const out = new Array(values.length).fill(null);
    let sum = 0;
    for (let i = 0; i < values.length; i++) {
      sum += values[i];
      if (i >= n) sum -= values[i - n];
      if (i >= n - 1) out[i] = sum / n;
    }
    return out;
  }

  async function openStockModal(code) {
    try {
      const data = await api(`/api/stocks/${encodeURIComponent(code)}/bars`);
      openModal(`
        <div class="modal-head">
          <div>
            <h3>${esc(data.name)} <span style="font-family:var(--mono);font-size:12px;color:var(--faint)">${esc(data.code)}</span></h3>
            <div class="panel-sub" style="margin-top:3px">日线 · 前复权 · ${data.dates.length} 根K线</div>
          </div>
          <button class="icon-btn" data-close>✕</button>
        </div>
        <div class="chart-wrap"><canvas id="candleChart"></canvas></div>
        <div class="kline-legend">
          <span class="it"><span class="sw" style="background:#f5b56b"></span>MA5</span>
          <span class="it"><span class="sw" style="background:#6aa9ff"></span>MA20</span>
          <span class="it"><span class="sw" style="background:#a78bfa"></span>MA60</span>
        </div>`, true);
      const mas = [
        { name: "MA5", color: "#f5b56b", data: movingAverage(data.close, 5) },
        { name: "MA20", color: "#6aa9ff", data: movingAverage(data.close, 20) },
        { name: "MA60", color: "#a78bfa", data: movingAverage(data.close, 60) },
      ];
      Charts.candle($("#candleChart"), {
        dates: data.dates, open: data.open, high: data.high,
        low: data.low, close: data.close, volume: data.volume, height: 380,
        ma: mas,
      });
    } catch (e) {
      toast(e.message, true);
    }
  }

  /* ------------------------------ 数据同步 ------------------------------ */
  function appendLog(msg, cls) {
    const box = $("#syncLog");
    const line = document.createElement("div");
    if (cls) line.className = cls;
    const t = new Date().toLocaleTimeString("zh-CN", { hour12: false });
    line.textContent = `[${t}] ${msg}`;
    box.appendChild(line);
    box.scrollTop = box.scrollHeight;
  }

  async function startSync() {
    if (state.syncTimer) return;
    const payload = {
      source: $("#syncSource").value,
      limit: Number($("#syncLimit").value) || 300,
      start: $("#syncStart").value || null,
      end: $("#syncEnd").value || null,
      replace: $("#syncReplace").checked,
      demo_stocks: Math.min(Number($("#syncLimit").value) || 150, 400),
    };
    $("#syncProgress").classList.add("on");
    $("#syncLog").innerHTML = "";
    $("#syncBar").style.width = "0%";
    $("#syncPct").textContent = "0%";
    $("#syncMsg").textContent = "创建同步任务…";
    $("#syncBtn").disabled = true;
    $("#syncBtn").innerHTML = `<span class="spinner"></span> 同步中`;
    try {
      const { job_id } = await api("/api/data/sync", { method: "POST", body: payload });
      pollJob(job_id);
    } catch (e) {
      appendLog(e.message, "warn");
      finishSync(false);
      toast(e.message, true);
    }
  }

  async function pollJob(jobId) {
    let lastMsg = "";
    const tick = async () => {
      try {
        const job = await api(`/api/data/jobs/${jobId}`);
        const pct = Math.round((job.progress || 0) * 100);
        $("#syncBar").style.width = pct + "%";
        $("#syncPct").textContent = pct + "%";
        $("#syncMsg").textContent = job.message || "";
        if (job.message && job.message !== lastMsg) {
          lastMsg = job.message;
          appendLog(job.message);
        }
        if (job.status === "done") {
          const warnings = (job.result && job.result.warnings) || [];
          warnings.forEach((w) => appendLog(w, "warn"));
          appendLog(`完成 · 数据源 ${job.result.source === "miniqmt" ? "miniQMT" : "合成演示"}`);
          finishSync(true);
          await loadStatus();
          await Promise.all([loadStocks(), loadFactors(), loadIndices()]);
          renderUniverse();
          toast("行情同步完成");
          return;
        }
        if (job.status === "error") {
          appendLog(job.error || "同步失败", "warn");
          finishSync(false);
          toast(job.error || "同步失败", true);
          return;
        }
        state.syncTimer = setTimeout(tick, 700);
      } catch (e) {
        appendLog(e.message, "warn");
        finishSync(false);
        toast(e.message, true);
      }
    };
    tick();
  }

  function finishSync(ok) {
    clearTimeout(state.syncTimer);
    state.syncTimer = null;
    $("#syncBtn").disabled = false;
    $("#syncBtn").innerHTML = `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12a9 9 0 1 1-2.6-6.4"/><path d="M21 3v6h-6"/></svg>开始同步`;
    if (ok) $("#syncMsg").textContent = "同步完成";
  }

  /* ------------------------------ 因子 ------------------------------ */
  async function loadFactors() {
    const data = await api("/api/factors");
    state.factors = data.items || [];
    const validIds = new Set(state.factors.map((f) => String(f.id)));
    state.btFactors = state.btFactors.filter((f) => validIds.has(String(f.id)));
    if (!state.btFactors.length && state.factors.length) {
      const preferred = state.factors.find((f) => String(f.id) === "momentum_20") || state.factors[0];
      state.btFactors = [{ id: String(preferred.id), weight: 1 }];
    }
    renderFactorList();
    renderAddFactorSelect();
    renderCombo();
    renderUniverseCount();
  }

  function factorGroups() {
    const groups = {};
    state.factors.forEach((f) => {
      const cat = f.is_custom ? "自定义" : (f.category || "其他");
      (groups[cat] = groups[cat] || []).push(f);
    });
    return Object.keys(groups)
      .sort((a, b) => {
        const ia = CAT_ORDER.indexOf(a), ib = CAT_ORDER.indexOf(b);
        return (ia === -1 ? 99 : ia) - (ib === -1 ? 99 : ib);
      })
      .map((cat) => [cat, groups[cat]]);
  }

  function renderFactorList() {
    const box = $("#factorList");
    const q = state.factorQuery.trim().toLowerCase();
    const groups = factorGroups()
      .map(([cat, items]) => [cat, items.filter((f) =>
        !q || (f.label || f.name || "").toLowerCase().includes(q)
          || (f.expression || "").toLowerCase().includes(q))])
      .filter(([, items]) => items.length);
    if (!groups.length) {
      box.innerHTML = `<div class="empty" style="border:none"><p>因子库为空</p></div>`;
      return;
    }
    box.innerHTML = groups.map(([cat, items]) => `
      <div class="fcat">${esc(cat)}</div>
      ${items.map((f) => {
        const active = state.selectedFactor && String(state.selectedFactor.id) === String(f.id) ? " active" : "";
        return `<button class="fitem${active}" data-id="${esc(f.id)}">
          <span>${esc(f.label || f.name)}</span>
          <span style="display:flex;align-items:center;gap:6px">
            <span class="dir ${f.direction > 0 ? "up" : "down"}">${f.direction > 0 ? "↑" : "↓"}</span>
            ${f.is_custom ? `<span class="edit" data-edit="${esc(f.id)}" title="编辑">✎</span><span class="del" data-del="${esc(f.id)}" title="删除">✕</span>` : ""}
          </span>
        </button>`;
      }).join("")}
    `).join("");
    $$("#factorList .fitem").forEach((btn) => {
      btn.addEventListener("click", (ev) => {
        const del = ev.target.closest("[data-del]");
        if (del) {
          ev.stopPropagation();
          deleteFactor(del.dataset.del);
          return;
        }
        const edit = ev.target.closest("[data-edit]");
        if (edit) {
          ev.stopPropagation();
          openFactorModal(findFactorById(edit.dataset.edit));
          return;
        }
        selectFactor(btn.dataset.id);
      });
    });
  }

  function findFactorById(id) {
    return state.factors.find((f) => String(f.id) === String(id));
  }

  function selectFactor(id) {
    const f = findFactorById(id);
    if (!f) return;
    state.selectedFactor = f;
    renderFactorList();
    $("#fdTitle").textContent = f.label || f.name;
    $("#fdDesc").textContent = f.description || "（无描述）";
    const expr = esc(f.expression).replace(
      /\b(ts_mean|ts_std|ts_sum|ts_min|ts_max|ts_rank|ts_delay|ts_delta|ts_corr|rank|zscore|safe_div|log|abs|sign|sqrt|exp|tanh|max|min|where|clip)\b/g,
      '<span class="kw">$1</span>'
    );
    $("#fdExpr").innerHTML = expr;
    $("#fdMeta").innerHTML = `
      <span class="chip ${f.is_custom ? "amber" : "blue"}">${f.is_custom ? "自定义" : "内置"}</span>
      <span class="chip">${esc(f.is_custom ? "自定义表达式" : (f.category || "因子"))}</span>
      <span class="chip ${f.direction > 0 ? "green" : "amber"}">${f.direction > 0 ? "值越大越好 ↑" : "值越小越好 ↓"}</span>`;
  }

  async function previewFactor() {
    if (!state.selectedFactor) { toast("请先选择一个因子", true); return; }
    const f = state.selectedFactor;
    const btn = $("#previewBtn");
    btn.disabled = true;
    btn.innerHTML = `<span class="spinner"></span> 计算中`;
    try {
      const payload = {
        factor_id: f.id,
        horizon: Number($("#previewHorizon").value),
        quantiles: Number($("#previewGroups").value),
      };
      if (state.status && state.status.start_date) {
        payload.start = state.status.start_date;
        payload.end = state.status.end_date;
      }
      const res = await api("/api/factors/preview", { method: "POST", body: payload });
      renderPreview(res);
      toast("因子检验完成");
    } catch (e) {
      toast(e.message, true);
    } finally {
      btn.disabled = false;
      btn.textContent = "检验因子";
    }
  }

  function chartBlock(title, sub) {
    return `<div style="margin-top:18px">
      <div class="chart-title-row">
        <div>
          <div class="panel-title" style="font-size:13px">${title}</div>
          ${sub ? `<div class="panel-sub">${sub}</div>` : ""}
        </div>
      </div>
      <div class="chart-wrap"><canvas></canvas></div>
    </div>`;
  }

  function renderPreview(res) {
    $("#previewEmpty").style.display = "none";
    const box = $("#previewBody");
    box.style.display = "block";
    const ic = res.ic;
    box.innerHTML = `
      <div class="panel-head" style="margin-bottom:10px">
        <div>
          <div class="panel-title" style="font-size:14px">${esc(res.factor.label)} · 因子检验</div>
          <div class="panel-sub">${res.universe.stocks} 只股票 / ${res.universe.days} 个交易日 · 预测窗口 ${res.horizon} 日</div>
        </div>
        <span class="chip green">IC 均值 ${fmtNum(ic.mean, 3)} · t ${fmtNum(ic.t_stat, 2)}</span>
      </div>
      <div class="metric-cards">
        <div class="mcard"><div class="k">IC 均值</div><div class="v ${colorClass(ic.mean)}">${fmtNum(ic.mean, 3)}</div><div class="s">Spearman 秩相关</div></div>
        <div class="mcard"><div class="k">ICIR</div><div class="v">${fmtNum(ic.ir, 2)}</div><div class="s">年化信息比</div></div>
        <div class="mcard"><div class="k">IC &gt; 0 占比</div><div class="v">${fmtPct(ic.positive_rate, 1, false)}</div><div class="s">方向稳定性</div></div>
        <div class="mcard"><div class="k">因子覆盖率</div><div class="v">${fmtPct(res.stats.coverage, 1, false)}</div><div class="s">有效值占比</div></div>
      </div>
      ${(res.ic_by_year || []).length ? `<div class="factor-meta" style="margin:12px 0 0">
        <span class="chip">分年度 IC</span>
        ${res.ic_by_year.map((y) => `<span class="chip ${y.ic >= 0 ? "green" : "amber"}">${y.year} · ${fmtNum(y.ic, 3)}</span>`).join("")}
      </div>` : ""}
      ${(res.stability || []).length ? `<div class="factor-meta" style="margin:8px 0 0">
        <span class="chip">因子稳定性</span>
        ${res.stability.map((s) => `<span class="chip ${s.autocorr >= 0.7 ? "green" : "amber"}">${s.lag} 日自相关 ${fmtNum(s.autocorr, 2)}</span>`).join("")}
      </div>` : ""}
      <div id="icBlock"></div>
      <div id="icMonthBlock"></div>
      <div id="industryIcBlock"></div>
      <div class="grid-2" style="margin-top:0">
        <div id="decayBlock"></div>
        <div id="quantileBlock"></div>
      </div>
      <div id="qnavBlock"></div>
      <div id="sampleBlock"></div>`;

    const icBlock = document.createElement("div");
    icBlock.innerHTML = chartBlock("IC 时间序列", `横截面 Rank IC · 预测窗口 ${res.horizon} 日`);
    box.querySelector("#icBlock").appendChild(icBlock);
    const labels = res.ic.series.map((d) => d.date);
    const vals = res.ic.series.map((d) => d.value);
    Charts.line(icBlock.querySelector("canvas"), {
      labels, series: [{ name: "IC", data: vals, color: "#3ddc97", fill: true }],
      height: 210, yFmt: (v) => v.toFixed(2), xFmt: shortDate, includeZero: true,
    });

    if ((res.ic_monthly || []).length) {
      const monthBlock = document.createElement("div");
      monthBlock.innerHTML = `<div style="margin-top:18px">
        <div class="chart-title-row"><div><div class="panel-title" style="font-size:13px">月度 IC 热力图</div><div class="panel-sub">蓝色为正向 IC，橙色为负向 IC</div></div></div>
        <div id="icMonthHeat" class="heatmap"></div>
      </div>`;
      box.querySelector("#icMonthBlock").appendChild(monthBlock);
      renderICMonthHeat(monthBlock.querySelector("#icMonthHeat"), res.ic_monthly);
    }

    if ((res.industry_ic || []).length) {
      const indBlock = document.createElement("div");
      indBlock.innerHTML = chartBlock("分行业 IC", `行业内部 Rank IC · 预测窗口 ${res.horizon} 日`);
      box.querySelector("#industryIcBlock").appendChild(indBlock);
      Charts.bar(indBlock.querySelector("canvas"), {
        labels: res.industry_ic.map((d) => d.industry),
        values: res.industry_ic.map((d) => d.ic),
        height: 210, colorsByValue: true, yFmt: (v) => v.toFixed(2), includeZero: true,
      });
    }

    const decayBlock = document.createElement("div");
    decayBlock.innerHTML = `<div style="margin-top:18px"><div class="chart-title-row"><div><div class="panel-title" style="font-size:13px">IC 衰减</div><div class="panel-sub">不同预测窗口的平均 IC</div></div></div><div class="chart-wrap"><canvas></canvas></div></div>`;
    box.querySelector("#decayBlock").appendChild(decayBlock);
    Charts.bar(decayBlock.querySelector("canvas"), {
      labels: res.ic_decay.map((d) => d.horizon + "日"),
      values: res.ic_decay.map((d) => d.ic),
      height: 200, colorsByValue: true, yFmt: (v) => v.toFixed(2), includeZero: true,
    });

    const nGroups = res.n_groups || 5;
    const qBlock = document.createElement("div");
    qBlock.innerHTML = `<div style="margin-top:18px"><div class="chart-title-row"><div><div class="panel-title" style="font-size:13px">分位组合收益</div><div class="panel-sub">按因子值分 ${nGroups} 组，未来 ${res.horizon} 日平均收益</div></div></div><div class="chart-wrap"><canvas></canvas></div></div>`;
    box.querySelector("#quantileBlock").appendChild(qBlock);
    Charts.bar(qBlock.querySelector("canvas"), {
      labels: res.quantile_returns.map((d) => d.group),
      values: res.quantile_returns.map((d) => d.return * 100),
      height: 200, colorsByValue: true, yFmt: (v) => v.toFixed(1) + "%", includeZero: true,
    });

    const qnav = res.quantile_nav;
    if (qnav && qnav.groups && qnav.groups.length) {
      const qnavBlock = document.createElement("div");
      qnavBlock.innerHTML = chartBlock("分层组合净值", `按因子值分 ${nGroups} 组，每日调仓等权持有 · 起点归一`);
      box.querySelector("#qnavBlock").appendChild(qnavBlock);
      const palette = ["#6aa9ff", "#59c3c3", "#3ddc97", "#f5b56b", "#ff6b7a"];
      const series = qnav.groups.map((g, i) => ({
        name: g.group,
        color: palette[i % palette.length],
        data: g.nav.map((v) => (v - 1) * 100),
        width: g.group === "Q" + nGroups ? 2.1 : 1.5,
      }));
      if (qnav.long_short && qnav.long_short.length) {
        series.push({
          name: `Q${nGroups}-Q1 多空`,
          color: "#e8eef7",
          data: qnav.long_short.map((v) => (v - 1) * 100),
          dash: [4, 3],
          width: 2,
        });
      }
      Charts.line(qnavBlock.querySelector("canvas"), {
        labels: qnav.dates,
        series,
        height: 240, yFmt: (v) => v.toFixed(0) + "%", xFmt: shortDate,
      });
    }

    const sampleBlock = document.createElement("div");
    sampleBlock.innerHTML = chartBlock(
      `样本股因子走势 · ${esc(res.sample_series.name)}`,
      `${esc(res.sample_series.code)} · 最近有效区间`
    );
    box.querySelector("#sampleBlock").appendChild(sampleBlock);
    if (res.sample_series.values.length > 1) {
      Charts.line(sampleBlock.querySelector("canvas"), {
        labels: res.sample_series.dates,
        series: [{ name: "因子值", data: res.sample_series.values, color: "#6aa9ff" }],
        height: 190, yFmt: (v) => v.toFixed(3), xFmt: shortDate,
      });
    }
  }

  /* ------------------------------ 因子工厂 ------------------------------ */
  async function openFactory() {
    openModal(`
      <div class="modal-head"><div><h3>因子工厂</h3><div class="panel-sub" style="margin-top:3px">批量评估候选表达式，按 ICIR 排序</div></div>
        <button class="icon-btn" data-close>✕</button></div>
      <div class="empty" style="border:none;padding:60px 20px">
        <span class="spinner" style="width:24px;height:24px;margin:0 auto 14px;display:block"></span>
        <p>正在评估候选因子…</p>
        <div class="sub">约 40 个候选表达式，首次运行需 JAX 编译</div>
      </div>`, true);
    try {
      const res = await api("/api/factors/mine", {
        method: "POST",
        body: {
          horizon: 5, limit: 14,
          start: state.status && state.status.start_date,
          end: state.status && state.status.end_date,
        },
      });
      renderFactory(res);
    } catch (e) {
      closeModal();
      toast(e.message, true);
    }
  }

  function renderFactory(res) {
    openModal(`
      <div class="modal-head">
        <div><h3>因子工厂</h3>
          <div class="panel-sub" style="margin-top:3px">${res.universe.stocks} 只股票 · ${res.universe.start} → ${res.universe.end} · 预测 ${res.horizon} 日 · 按 |ICIR| 排序</div>
        </div>
        <button class="icon-btn" data-close>✕</button>
      </div>
      <div class="table-wrap" style="max-height:460px">
        <table>
          <thead><tr><th>因子</th><th>分类</th><th style="text-align:right">|IC|</th><th style="text-align:right">|ICIR|</th><th style="text-align:right">覆盖率</th><th>方向</th><th></th></tr></thead>
          <tbody>
            ${res.items.map((it, i) => `<tr style="cursor:default">
              <td><span class="nm">${esc(it.label)}</span><div class="panel-sub" style="font-family:var(--mono);font-size:10px;max-width:360px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(it.expression)}</div></td>
              <td><span class="chip">${esc(it.category)}</span></td>
              <td style="text-align:right" class="num-mono">${fmtNum(it.ic, 3)}</td>
              <td style="text-align:right" class="num-mono">${fmtNum(it.icir, 2)}</td>
              <td style="text-align:right" class="num-mono">${fmtPct(it.coverage, 0, false)}</td>
              <td><span class="chip ${it.direction > 0 ? "green" : "amber"}">${it.direction > 0 ? "正向 ↑" : "反向 ↓"}</span></td>
              <td><button class="btn sm" data-fac-save="${i}">收藏</button></td>
            </tr>`).join("")}
          </tbody>
        </table>
      </div>
      <div class="modal-foot" style="justify-content:space-between;align-items:center">
        <span class="panel-sub">收藏后可在回测与多因子组合中直接使用</span>
        <button class="btn ghost" data-close>关闭</button>
      </div>`, true);
    $$("#modalBox [data-fac-save]").forEach((btn) => btn.addEventListener("click", async () => {
      const item = res.items[Number(btn.dataset.facSave)];
      btn.disabled = true;
      try {
        await api("/api/factors", {
          method: "POST",
          body: {
            name: item.label,
            expression: item.expression,
            description: `因子工厂 · ${item.label}（ICIR ${item.icir}）`,
            direction: item.direction,
          },
        });
        btn.textContent = "已收藏";
        await loadFactors();
        toast(`已收藏因子「${item.label}」`);
      } catch (e) {
        btn.disabled = false;
        toast(e.message, true);
      }
    }));
  }

  function renderICMonthHeat(el, data) {
    const years = [...new Set(data.map((d) => d.year))].sort();
    const byKey = {};
    data.forEach((d) => { byKey[d.year + "-" + d.month] = d.ic; });
    const yearAvg = {};
    years.forEach((y) => {
      const vals = data.filter((d) => d.year === y).map((d) => d.ic);
      yearAvg[y] = vals.reduce((a, b) => a + b, 0) / (vals.length || 1);
    });
    const color = (v) => {
      const alpha = 0.07 + Math.min(0.6, Math.abs(v) * 3.2);
      return v >= 0 ? `rgba(106,169,255,${alpha})` : `rgba(245,181,107,${alpha})`;
    };
    let html = `<div class="hm-label"></div>` +
      Array.from({ length: 12 }, (_, i) => `<div class="hm-head">${i + 1}月</div>`).join("") +
      `<div class="hm-head">年度</div>`;
    years.forEach((y) => {
      html += `<div class="hm-label">${y}</div>`;
      for (let mo = 1; mo <= 12; mo++) {
        const v = byKey[y + "-" + mo];
        if (v === undefined) { html += `<div class="hm-cell"></div>`; continue; }
        html += `<div class="hm-cell" style="background:${color(v)}" title="${y}-${mo}: IC ${v}">${v.toFixed(2)}</div>`;
      }
      html += `<div class="hm-cell" style="background:${color(yearAvg[y])};font-weight:600">${yearAvg[y].toFixed(2)}</div>`;
    });
    el.style.gridTemplateColumns = "46px repeat(13, 1fr)";
    el.innerHTML = html;
  }

  async function deleteFactor(id) {
    if (!confirm("确定删除该自定义因子？")) return;
    try {
      await api(`/api/factors/${id}`, { method: "DELETE" });
      if (state.selectedFactor && String(state.selectedFactor.id) === String(id)) {
        state.selectedFactor = null;
        $("#fdTitle").textContent = "选择一个因子";
        $("#fdDesc").textContent = "从左侧因子库选择，或创建一个自定义因子";
        $("#fdExpr").textContent = "—";
        $("#fdMeta").innerHTML = "";
        $("#previewBody").style.display = "none";
        $("#previewEmpty").style.display = "block";
      }
      await loadFactors();
      toast("因子已删除");
    } catch (e) { toast(e.message, true); }
  }

  function openFactorModal(existing = null) {
    const editing = !!existing;
    const expr = existing ? existing.expression : "rank(ts_mean(close, 20) / ts_mean(close, 60) - 1)";
    openModal(`
      <div class="modal-head">
        <div><h3>${editing ? "编辑自定义因子" : "新建自定义因子"}</h3><div class="panel-sub" style="margin-top:3px">表达式将作用于 (交易日 × 股票) 面板数组</div></div>
        <button class="icon-btn" data-close>✕</button>
      </div>
      <div class="form-grid">
        <div class="field"><label>因子名称</label><input id="nfName" type="text" placeholder="如：量价背离_20" value="${esc(existing ? existing.name : "")}"></div>
        <div class="field"><label>方向</label>
          <select id="nfDirection">
            <option value="1" ${existing && existing.direction < 0 ? "" : "selected"}>值越大越好（正向）</option>
            <option value="-1" ${existing && existing.direction < 0 ? "selected" : ""}>值越小越好（反向）</option>
          </select>
        </div>
      </div>
      <div class="field" style="margin-top:12px"><label>参数化模板</label>
        <div class="form-row" style="gap:8px;flex-wrap:wrap;align-items:flex-end">
          <select id="nfTpl" style="flex:1;min-width:170px">
            <option value="">选择模板（可选）…</option>
            ${FACTOR_TEMPLATES.map((t) => `<option value="${esc(t.id)}">${esc(t.label)}</option>`).join("")}
          </select>
          <span id="nfTplParams" style="display:flex;gap:8px;flex-wrap:wrap"></span>
          <button type="button" class="btn ghost sm" id="nfTplApply" disabled>生成表达式</button>
          <button type="button" class="btn ghost sm" id="nfScanBtn" disabled>窗口扫描</button>
        </div>
        <div id="nfScanWrap" style="margin-top:8px"></div>
      </div>
      <div class="field" style="margin-top:12px"><label>因子表达式</label>
        <textarea id="nfExpr" rows="4">${esc(expr)}</textarea>
      </div>
      <div class="form-row" style="margin-top:8px;gap:6px;flex-wrap:wrap">
        <span class="hint">示例</span>
        <button type="button" class="btn ghost sm" data-tpl-expr="rank(close / ts_delay(close, 20) - 1)">20日动量</button>
        <button type="button" class="btn ghost sm" data-tpl-expr="rank(-(close / ts_delay(close, 5) - 1))">5日反转</button>
        <button type="button" class="btn ghost sm" data-tpl-expr="rank(-ts_std(returns, 20))">低波动</button>
        <button type="button" class="btn ghost sm" data-tpl-expr="rank(volume / ts_mean(volume, 20))">量比</button>
      </div>
      <div class="field" style="margin-top:12px"><label>描述</label><input id="nfDesc" type="text" placeholder="说明因子逻辑（可选）" value="${esc(existing ? (existing.description || "") : "")}"></div>
      <div style="margin-top:14px">
        <div class="panel-sub">可用字段：<code style="color:var(--green)">open high low close volume amount vwap returns</code></div>
        <div class="fn-chips" id="fnChips"></div>
      </div>
      <div id="nfTest" class="hint-box" style="display:none;margin-top:12px"></div>
      <div class="modal-foot">
        <button class="btn ghost" data-close>取消</button>
        <button class="btn ghost" id="nfTestBtn">测试表达式</button>
        <button class="btn primary" id="nfSave">${editing ? "保存修改" : "保存因子"}</button>
      </div>`);
    const fns = ["ts_mean(x, 20)", "ts_std(x, 20)", "ts_rank(x, 20)", "ts_delay(x, 20)",
      "ts_delta(x, 5)", "ts_corr(x, y, 20)", "ts_zscore(x, 20)", "ts_skew(x, 20)",
      "ts_argmax(x, 20)", "decay_linear(x, 20)", "rank(x)", "zscore(x)",
      "where(cond, a, b)", "clip(x, -3, 3)"];
    $("#fnChips").innerHTML = fns.map((f) => `<button type="button" data-fn="${esc(f)}">${esc(f)}</button>`).join("");
    $$("#fnChips button").forEach((b) => b.addEventListener("click", () => {
      const ta = $("#nfExpr");
      const pos = ta.selectionStart || ta.value.length;
      ta.value = ta.value.slice(0, pos) + b.dataset.fn + ta.value.slice(pos);
      ta.focus();
    }));
    $$("#modalBox [data-tpl-expr]").forEach((b) => b.addEventListener("click", () => {
      $("#nfExpr").value = b.dataset.tplExpr;
    }));
    function renderTplParams() {
      const tpl = FACTOR_TEMPLATES.find((t) => t.id === $("#nfTpl").value);
      const wrap = $("#nfTplParams");
      $("#nfTplApply").disabled = !tpl;
      $("#nfScanBtn").disabled = !tpl;
      if (!tpl) { wrap.innerHTML = ""; $("#nfScanWrap").innerHTML = ""; return; }
      wrap.innerHTML = tpl.params.map((p) => `
        <label class="hint" style="display:flex;align-items:center;gap:4px">${esc(p.label)}
          <input type="number" min="2" max="250" step="1" value="${p.def}" data-param="${esc(p.key)}" style="width:68px">
        </label>`).join("");
    }
    $("#nfTpl").addEventListener("change", renderTplParams);
    $("#nfTplApply").addEventListener("click", () => {
      const tpl = FACTOR_TEMPLATES.find((t) => t.id === $("#nfTpl").value);
      if (!tpl) return;
      const vals = {};
      tpl.params.forEach((pp) => {
        const el = $(`#nfTplParams [data-param="${pp.key}"]`);
        let v = Math.round(Number(el ? el.value : pp.def));
        if (!isFinite(v) || v < 2) v = pp.def;
        v = Math.min(v, 250);
        vals[pp.key] = v;
        if (el) el.value = v;
      });
      $("#nfExpr").value = tpl.build(vals);
      const nameInput = $("#nfName");
      if (!nameInput.value.trim()) {
        nameInput.value = tpl.prefix + "_" + tpl.params.map((pp) => vals[pp.key]).join("_");
      }
      toast("已生成表达式，可点击「测试表达式」检验");
    });
    function currentTplParams() {
      const tpl = FACTOR_TEMPLATES.find((t) => t.id === $("#nfTpl").value);
      if (!tpl) return null;
      const vals = {};
      tpl.params.forEach((pp) => {
        const el = $(`#nfTplParams [data-param="${pp.key}"]`);
        let v = Math.round(Number(el ? el.value : pp.def));
        if (!isFinite(v) || v < 2) v = pp.def;
        vals[pp.key] = Math.min(v, 250);
      });
      return { tpl, vals };
    }
    $("#nfScanBtn").addEventListener("click", async () => {
      const cur = currentTplParams();
      if (!cur) return;
      const { tpl, vals } = cur;
      const scanKey = tpl.params[0].key;
      const expression = tpl.build({ ...vals, [scanKey]: "{p}" });
      if (!expression.includes("{p}")) { toast("该模板不支持窗口扫描", true); return; }
      const wrap = $("#nfScanWrap");
      const btn = $("#nfScanBtn");
      btn.disabled = true;
      wrap.innerHTML = `<div class="hint-box"><span class="spinner"></span> 正在扫描窗口参数，约需数秒…</div>`;
      try {
        const res = await api("/api/factors/scan", {
          method: "POST",
          body: {
            expression,
            direction: Number($("#nfDirection").value),
            horizon: 5,
            values: Array.from({ length: 24 }, (_, i) => (i + 1) * 5),
          },
        });
        renderScanResult(res, scanKey);
      } catch (e) {
        wrap.innerHTML = `<div class="hint-box"><span class="down">${esc(e.message)}</span></div>`;
      } finally {
        btn.disabled = false;
      }
    });
    function renderScanResult(res, scanKey) {
      const wrap = $("#nfScanWrap");
      const best = res.best;
      const items = res.items || [];
      wrap.innerHTML = `
        <div class="hint-box" style="margin-top:0">
          <div class="factor-meta">
            <span class="chip green">最优窗口 ${best.value}</span>
            <span class="chip">IC ${fmtNum(best.ic, 4)}</span>
            <span class="chip">ICIR ${fmtNum(best.icir, 3)}</span>
            <span class="chip">覆盖率 ${fmtPct(best.coverage, 1, false)}</span>
            <button type="button" class="btn ghost sm" id="nfScanApply">采用该窗口</button>
          </div>
          <canvas id="nfScanChart" style="margin-top:10px"></canvas>
          <div class="panel-sub" style="margin-top:6px">扫描 ${items.length} 个窗口（${items[0].value}–${items[items.length - 1].value} 日），纵轴为年化 ICIR；仅作参数参考，注意过拟合风险</div>
        </div>`;
      Charts.line($("#nfScanChart"), {
        labels: items.map((i) => String(i.value)),
        series: [{ name: "ICIR", data: items.map((i) => i.icir), color: "#5b8def", fill: true }],
        height: 180,
        yFmt: (v) => v.toFixed(2),
        includeZero: true,
      });
      $("#nfScanApply").addEventListener("click", () => {
        const el = $(`#nfTplParams [data-param="${scanKey}"]`);
        if (el) el.value = best.value;
        $("#nfTplApply").click();
        toast(`已采用窗口 ${best.value}`);
      });
    }
    $("#nfTestBtn").addEventListener("click", async () => {
      const expression = $("#nfExpr").value.trim();
      if (!expression) { toast("请填写因子表达式", true); return; }
      const box = $("#nfTest");
      box.style.display = "block";
      box.innerHTML = `<span class="spinner"></span> 正在计算 IC 与分组收益…`;
      const btn = $("#nfTestBtn");
      btn.disabled = true;
      try {
        const res = await api("/api/factors/preview", {
          method: "POST",
          body: {
            expression,
            direction: Number($("#nfDirection").value),
            preprocess: { winsorize: true, standardize: true },
          },
        });
        const ic = res.ic || {};
        const q = res.quantile_returns || [];
        const q1 = q[0];
        const q5 = q[q.length - 1];
        const spread = (q1 && q5) ? q5.annualized - q1.annualized : null;
        box.innerHTML = `<div class="factor-meta">
          <span class="chip ${ic.mean > 0 ? "green" : "amber"}">IC ${fmtNum(ic.mean, 4)}</span>
          <span class="chip">ICIR ${fmtNum(ic.ir, 3)}</span>
          <span class="chip">t 值 ${fmtNum(ic.t_stat, 2)}</span>
          <span class="chip">IC>0 占比 ${fmtPct(ic.positive_rate, 1)}</span>
          <span class="chip">覆盖率 ${fmtPct((res.stats || {}).coverage, 1)}</span>
          ${spread == null ? "" : `<span class="chip ${spread > 0 ? "green" : "amber"}">Q5-Q1 年化 ${fmtPct(spread, 1)}</span>`}
        </div><div class="panel-sub" style="margin-top:6px">基于当前股票池全区间计算的快速检验，保存后可在因子实验室查看完整报告</div>`;
      } catch (e) {
        box.innerHTML = `<span class="down">${esc(e.message)}</span>`;
      } finally {
        btn.disabled = false;
      }
    });
    $("#nfSave").addEventListener("click", async () => {
      const name = $("#nfName").value.trim();
      const expression = $("#nfExpr").value.trim();
      if (!name || !expression) { toast("名称与表达式不能为空", true); return; }
      try {
        const payload = {
          name, expression,
          description: $("#nfDesc").value.trim(),
          direction: Number($("#nfDirection").value),
        };
        const saved = editing
          ? await api(`/api/factors/${existing.id}`, { method: "PUT", body: payload })
          : await api("/api/factors", { method: "POST", body: payload });
        closeModal();
        await loadFactors();
        selectFactor(saved.id);
        toast(editing ? "因子已更新" : "自定义因子已保存");
      } catch (e) { toast(e.message, true); }
    });
  }

  /* ------------------------------ 回测 ------------------------------ */
  function updateRebalanceUI() {
    const mode = ($("#btRebalanceMode") || {}).value || "interval";
    const interval = $("#rbIntervalWrap");
    const week = $("#rbWeekdayWrap");
    const month = $("#rbMonthWrap");
    if (interval) interval.style.display = mode === "interval" ? "" : "none";
    if (week) week.style.display = mode === "weekly" ? "" : "none";
    if (month) month.style.display = mode === "monthly" ? "" : "none";
  }

  function renderAddFactorSelect() {
    const sel = $("#addFactorSelect");
    if (!sel) return;
    sel.innerHTML = factorGroups().map(([cat, items]) =>
      `<optgroup label="${esc(cat)}">` + items.map((f) =>
        `<option value="${esc(f.id)}">${esc(f.label || f.name)}</option>`
      ).join("") + "</optgroup>"
    ).join("");
  }

  function renderCombo() {
    const box = $("#comboList");
    if (!box) return;
    box.innerHTML = state.btFactors.map((item) => {
      const f = findFactorById(item.id);
      const label = f ? (f.label || f.name) : item.id;
      const dir = f && f.direction < 0 ? "↓" : "↑";
      return `<div class="combo-row" data-id="${esc(item.id)}">
        <span class="combo-name" title="${esc(f ? f.expression : "")}">${esc(label)}<small>${dir}</small></span>
        <input class="combo-weight" type="number" step="0.1" min="-5" max="5" value="${item.weight}">
        <button class="combo-del" data-del="${esc(item.id)}" title="移除">✕</button>
      </div>`;
    }).join("") || `<div class="panel-sub" style="padding:6px">请添加至少一个因子</div>`;

    $$("#comboList .combo-del").forEach((btn) => btn.addEventListener("click", () => {
      state.btFactors = state.btFactors.filter((f) => String(f.id) !== String(btn.dataset.del));
      renderCombo();
    }));
    $$("#comboList .combo-weight").forEach((inp) => inp.addEventListener("change", () => {
      const id = inp.closest(".combo-row").dataset.id;
      const item = state.btFactors.find((f) => String(f.id) === String(id));
      if (item) item.weight = Number(inp.value) || 0;
    }));
    const countEl = $("#comboCount");
    if (countEl) {
      countEl.textContent = state.btFactors.length > 1
        ? `${state.btFactors.length} 个因子加权`
        : "单因子";
    }
  }

  function addFactorToCombo() {
    const sel = $("#addFactorSelect");
    const id = sel && sel.value;
    if (!id) return;
    if (state.btFactors.some((f) => String(f.id) === String(id))) {
      toast("该因子已在组合中");
      return;
    }
    state.btFactors.push({ id: String(id), weight: 1 });
    renderCombo();
  }

  async function autoWeightByIC() {
    if (!state.btFactors.length) { toast("请先添加因子", true); return; }
    const btn = $("#icWeightBtn");
    btn.disabled = true;
    btn.textContent = "计算中";
    try {
      const range = {
        start: state.status && state.status.start_date,
        end: state.status && state.status.end_date,
      };
      const results = await Promise.all(state.btFactors.map((f) =>
        api("/api/factors/preview", {
          method: "POST",
          body: { factor_id: f.id, horizon: 5, ...range },
        })
      ));
      const scores = results.map((r) => Math.max(Number(r.ic.ir) || 0, 0));
      const total = scores.reduce((a, b) => a + b, 0);
      if (total <= 0) { toast("所选因子 ICIR 均非正，无法自动加权", true); return; }
      state.btFactors.forEach((f, i) => {
        f.weight = Math.max(0.01, Math.round(scores[i] / total * 100) / 100);
      });
      renderCombo();
      toast("已按 ICIR 自动分配因子权重");
    } catch (e) {
      toast(e.message, true);
    } finally {
      btn.disabled = false;
      btn.textContent = "IC 加权";
    }
  }

  async function saveComboAsFactor() {
    if (!state.btFactors.length) { toast("请先添加因子", true); return; }
    const parts = [];
    const names = [];
    state.btFactors.forEach((item) => {
      const f = findFactorById(item.id);
      if (!f) return;
      const w = (Number(item.weight) || 0) * (f.direction < 0 ? -1 : 1);
      if (!w) return;
      parts.push(`${w >= 0 ? "+" : "-"}${Math.abs(w).toFixed(2)} * rank(${f.expression})`);
      names.push(f.label || f.name);
    });
    if (!parts.length) { toast("组合权重不能全部为 0", true); return; }
    let expr = parts.join(" ").trim();
    if (expr.startsWith("+")) expr = expr.slice(1).trim();
    expr = `(${expr})`;
    const name = window.prompt("自定义因子名称", `组合因子_${new Date().toISOString().slice(5, 10).replace("-", "")}`);
    if (!name) return;
    try {
      const saved = await api("/api/factors", {
        method: "POST",
        body: {
          name: name.trim(),
          expression: expr,
          direction: 1,
          description: `由组合因子生成：${names.join(" + ")}`,
        },
      });
      await loadFactors();
      switchPage("factors");
      selectFactor(saved.id);
      toast(`组合因子「${name.trim()}」已保存`);
    } catch (e) {
      toast(e.message, true);
    }
  }

  function renderUniverse() {
    const box = $("#universeList");
    if (!state.stocks.length) {
      box.innerHTML = `<div style="padding:14px;text-align:center;color:var(--faint);font-size:11.5px">暂无股票数据</div>`;
      return;
    }
    box.innerHTML = state.stocks.map((s) => `
      <label class="up-item">
        <input type="checkbox" value="${esc(s.code)}" ${state.universe.has(s.code) ? "checked" : ""}>
        <span>${esc(s.name || s.code)}</span>
        <span class="c">${esc(s.code)}</span>
      </label>`).join("");
    $$("#universeList input").forEach((cb) => cb.addEventListener("change", () => {
      if (cb.checked) state.universe.add(cb.value); else state.universe.delete(cb.value);
      renderUniverseCount();
    }));
    renderUniverseCount();
  }

  function renderUniverseCount() {
    const el = $("#universeCount");
    if (!el) return;
    const n = state.universe.size;
    el.textContent = n ? `已选 ${n} 只` : "全部股票（不限制）";
  }

  function saveUniverse() {
    if (!state.universe.size) { toast("请先勾选要保存的股票", true); return; }
    openModal(`
      <div class="modal-head"><div><h3>保存自选股票池</h3><div class="panel-sub" style="margin-top:3px">共 ${state.universe.size} 只股票，保存在 SQLite</div></div>
        <button class="icon-btn" data-close>✕</button></div>
      <div class="field"><label>股票池名称</label><input type="text" id="uvName" placeholder="如：核心持仓 20"></div>
      <div class="modal-foot">
        <button class="btn ghost" data-close>取消</button>
        <button class="btn primary" id="uvSave">保存</button>
      </div>`);
    $("#uvName").focus();
    $("#uvSave").addEventListener("click", async () => {
      const name = $("#uvName").value.trim();
      if (!name) { toast("请输入股票池名称", true); return; }
      try {
        await api("/api/universes", {
          method: "POST",
          body: { name, codes: Array.from(state.universe) },
        });
        closeModal();
        toast(`股票池「${name}」已保存`);
      } catch (e) { toast(e.message, true); }
    });
  }

  async function openLoadUniverse() {
    try {
      const data = await api("/api/universes");
      const items = data.items || [];
      if (!items.length) { toast("还没有保存的股票池", true); return; }
      openModal(`
        <div class="modal-head"><div><h3>载入自选股票池</h3><div class="panel-sub" style="margin-top:3px">选择后覆盖当前勾选</div></div>
          <button class="icon-btn" data-close>✕</button></div>
        <div style="display:flex;flex-direction:column;gap:6px">
          ${items.map((it, i) => `<div style="display:flex;align-items:center;justify-content:space-between;gap:10px;border:1px solid var(--border-soft);background:#0a121c;border-radius:8px;padding:9px 12px">
            <div><div style="font-size:13px">${esc(it.name)}</div><div class="panel-sub" style="margin-top:1px">${it.count} 只 · ${esc(it.created_at)}</div></div>
            <div class="form-row">
              <button class="btn sm" data-uv-apply="${i}">载入</button>
              <button class="btn sm danger" data-uv-del="${i}">删除</button>
            </div>
          </div>`).join("")}
        </div>`, true);
      $$("#modalBox [data-uv-apply]").forEach((btn) => btn.addEventListener("click", () => {
        const item = items[Number(btn.dataset.uvApply)];
        const valid = new Set(state.stocks.map((s) => s.code));
        state.universe = new Set((item.codes || []).filter((c) => valid.has(c)));
        renderUniverse();
        closeModal();
        toast(`已载入股票池「${item.name}」`);
      }));
      $$("#modalBox [data-uv-del]").forEach((btn) => btn.addEventListener("click", async () => {
        const item = items[Number(btn.dataset.uvDel)];
        try {
          await api(`/api/universes/${item.id}`, { method: "DELETE" });
          openLoadUniverse();
        } catch (e) { toast(e.message, true); }
      }));
    } catch (e) { toast(e.message, true); }
  }

  function buildCommonConfig() {
    return {
      codes: state.universe.size ? Array.from(state.universe) : null,
      start: $("#btStart").value || null,
      end: $("#btEnd").value || null,
      factors: state.btFactors.map((f) => ({
        factor_id: f.id, weight: Number(f.weight) || 0,
      })),
      preprocess: {
        winsorize: $("#ppWinsor").checked,
        neutralize: $("#ppNeutral").checked,
        standardize: $("#ppStd").checked,
        orthogonalize: $("#ppOrtho").checked,
      },
      filters: {
        exclude_star: $("#filterStar").checked,
        exclude_bj: $("#filterBJ").checked,
        exclude_limit_up: $("#filterLimitUp").checked,
        min_amount: Number($("#filterAmount").value) || 0,
        min_list_days: Number($("#filterDays").value) || 0,
        industries: Array.from($("#filterIndustries").selectedOptions).map((o) => o.value),
      },
      benchmark: $("#btBenchmark").value,
      weighting: $("#btWeighting").value,
      max_weight: Number($("#btMaxWeight").value) || 0,
      industry_cap: Number($("#btIndustryCap").value) || 0,
      max_position_pct: Number($("#btMaxPosition").value) || 100,
      vol_target_pct: Number($("#btVolTarget").value) || 0,
      vol_lookback: Number($("#btVolLookback").value) || 20,
      min_hold_days: Number($("#btMinHold").value) || 0,
      turnover_cap_pct: Number($("#btTurnoverCap").value) || 0,
      rebalance_band_pct: Number($("#btRebalanceBand").value) || 0,
      market_timing: $("#btTiming").value !== "none",
      timing_window: Number($("#btTiming").value) || 0,
      oos_split: Number($("#btOOS").value) || 0,
      trade_constraints: $("#filterTrade").checked,
      stop_loss_pct: Number($("#btStopLoss").value) || 0,
      take_profit_pct: Number($("#btTakeProfit").value) || 0,
      trailing_stop_pct: Number($("#btTrailingStop").value) || 0,
      rebalance_days: Number($("#btRebalance").value),
      rebalance_mode: $("#btRebalanceMode").value,
      rebalance_weekday: Number($("#btRebalanceWeekday").value) || 0,
      rebalance_month_day: Number($("#btRebalanceMonthDay").value) || 1,
      top_n: Number($("#btTopN").value) || 0,
      top_pct: Number($("#btTopPct").value) || 0.2,
      commission_bps: Number($("#btCommission").value) || 0,
      stamp_duty_bps: Number($("#btStamp").value) || 0,
      slippage_bps: Number($("#btSlip").value) || 0,
      long_short: $("#btLongShort").checked,
    };
  }

  const STRATEGY_TEMPLATES = {
    momentum: {
      label: "动量趋势",
      factors: [["momentum_20", 1], ["momentum_60", 0.5]],
      pre: { winsor: true, neutral: false, std: true },
      rebalance: 10, topN: 15, benchmark: "market",
    },
    lowvol: {
      label: "低波防御",
      factors: [["volatility_20", 1], ["amplitude_20", 0.5]],
      pre: { winsor: true, neutral: true, std: true },
      rebalance: 21, topN: 20, benchmark: "market",
    },
    reversal: {
      label: "短期反转",
      factors: [["reversal_5", 1]],
      pre: { winsor: true, neutral: true, std: true },
      rebalance: 5, topN: 10, benchmark: "universe",
    },
    balanced: {
      label: "多因子均衡",
      factors: [["momentum_20", 1], ["volatility_20", 1], ["volume_ratio", 0.5]],
      pre: { winsor: true, neutral: true, std: true },
      rebalance: 10, topN: 15, benchmark: "market",
    },
  };

  function applyTemplate(name) {
    const tpl = STRATEGY_TEMPLATES[name];
    if (!tpl) return;
    state.btFactors = tpl.factors.map(([id, weight]) => ({ id, weight }));
    renderCombo();
    $("#ppWinsor").checked = tpl.pre.winsor;
    $("#ppNeutral").checked = tpl.pre.neutral;
    $("#ppStd").checked = tpl.pre.std;
    $("#btRebalance").value = String(tpl.rebalance);
    $("#btTopN").value = tpl.topN;
    $("#btBenchmark").value = tpl.benchmark;
    $("#btTiming").value = "none";
    $("#btOOS").value = "0";
    toast(`已应用策略模板：${tpl.label}`);
  }

  /* ------------------------------ 最新信号 ------------------------------ */
  async function openSignal() {
    if (!state.btFactors.length) { toast("请至少添加一个因子", true); return; }
    openModal(`
      <div class="modal-head">
        <div><h3>最新持仓信号</h3><div class="panel-sub" style="margin-top:3px">按当前因子与过滤条件生成目标持仓</div></div>
        <button class="icon-btn" data-close>✕</button>
      </div>
      <div class="empty" style="border:none;padding:50px 20px">
        <span class="spinner" style="width:22px;height:22px;margin:0 auto 12px;display:block"></span>
        <p>正在计算最新交易日信号…</p>
      </div>`, true);
    try {
      const res = await api("/api/signal", { method: "POST", body: buildCommonConfig() });
      renderSignal(res);
    } catch (e) {
      closeModal();
      toast(e.message, true);
    }
  }

  function renderSignal(res) {
    const wLabel = { equal: "等权", score: "得分加权", inverse_vol: "波动率倒数", risk_parity: "风险平价" }[res.config.weighting] || "等权";
    const chips = (res.factor.factors || []).map((f) =>
      `<span class="chip blue">${esc(f.label)} ${(f.weight * 100).toFixed(0)}%</span>`).join("");
    openModal(`
      <div class="modal-head">
        <div>
          <h3>最新持仓信号 <span style="font-family:var(--mono);font-size:12px;color:var(--faint)">${esc(res.as_of)}</span></h3>
          <div class="factor-meta" style="margin-top:6px">
            ${chips}
            <span class="chip">${wLabel}</span>
            <span class="chip">持仓 ${res.items.length} 只</span>
            ${res.config.max_weight > 0 ? `<span class="chip">上限 ${res.config.max_weight}%</span>` : ""}
            ${res.config.industry_cap > 0 ? `<span class="chip">行业上限 ${res.config.industry_cap}%</span>` : ""}
          </div>
        </div>
        <button class="icon-btn" data-close>✕</button>
      </div>
      <div class="table-wrap" style="max-height:430px">
        <table>
          <thead><tr><th>代码</th><th>名称</th><th>行业</th><th style="text-align:right">目标权重</th><th style="text-align:right">最新价</th><th style="text-align:right">涨跌幅</th><th style="text-align:right">因子值</th></tr></thead>
          <tbody>
            ${res.items.map((it) => `<tr style="cursor:default">
              <td><span class="cd">${esc(it.code)}</span></td>
              <td>${esc(it.name)}</td>
              <td><span class="chip">${esc(it.industry)}</span></td>
              <td style="text-align:right" class="num-mono">${(it.weight * 100).toFixed(1)}%</td>
              <td style="text-align:right" class="num-mono">${it.close == null ? "—" : fmtNum(it.close)}</td>
              <td style="text-align:right" class="num-mono ${it.change_pct > 0 ? "up" : it.change_pct < 0 ? "down" : ""}">${it.change_pct == null ? "—" : (it.change_pct > 0 ? "+" : "") + fmtNum(it.change_pct) + "%"}</td>
              <td style="text-align:right" class="num-mono">${fmtNum(it.score, 3)}</td>
            </tr>`).join("")}
          </tbody>
        </table>
      </div>
      <div class="modal-foot" style="justify-content:space-between;align-items:center">
        <span class="panel-sub">基于最新交易日截面，仅供研究参考</span>
        <div class="form-row">
          <button class="btn ghost" data-close>关闭</button>
          <button class="btn primary" id="signalCsv">导出信号 CSV</button>
        </div>
      </div>`, true);
    $("#signalCsv").addEventListener("click", () => {
      const rows = [["code", "name", "industry", "weight", "close", "change_pct", "factor_score"]];
      res.items.forEach((it) => rows.push([it.code, it.name, it.industry, it.weight, it.close, it.change_pct, it.score]));
      downloadCsv(rows, `signal_${res.as_of}.csv`);
      toast("信号已导出 CSV");
    });
  }

  function downloadCsv(rows, filename) {
    const csv = "\ufeff" + rows.map((row) => row.join(",")).join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  /* ------------------------------ 配置保存 ------------------------------ */
  const CONFIG_KEY = "alphalab_bt_configs";

  function loadSavedConfigs() {
    try { return JSON.parse(localStorage.getItem(CONFIG_KEY) || "[]"); } catch (e) { return []; }
  }

  function persistConfigs(list) {
    localStorage.setItem(CONFIG_KEY, JSON.stringify(list.slice(0, 30)));
  }

  function saveCurrentConfig() {
    openModal(`
      <div class="modal-head"><div><h3>保存回测配置</h3><div class="panel-sub" style="margin-top:3px">保存在浏览器本地，可从「载入」恢复</div></div>
        <button class="icon-btn" data-close>✕</button></div>
      <div class="field"><label>配置名称</label><input type="text" id="cfgName" value="${esc(payloadLabel())}"></div>
      <div class="modal-foot">
        <button class="btn ghost" data-close>取消</button>
        <button class="btn primary" id="cfgSave">保存配置</button>
      </div>`);
    $("#cfgName").focus();
    $("#cfgSave").addEventListener("click", () => {
      const name = $("#cfgName").value.trim();
      if (!name) { toast("请输入配置名称", true); return; }
      const list = loadSavedConfigs();
      list.unshift({
        name,
        created: new Date().toLocaleString("zh-CN", { hour12: false }),
        config: buildCommonConfig(),
      });
      persistConfigs(list);
      closeModal();
      toast(`配置「${name}」已保存到本地`);
    });
  }

  function openLoadConfig() {
    const list = loadSavedConfigs();
    if (!list.length) { toast("还没有保存的配置", true); return; }
    openModal(`
      <div class="modal-head"><div><h3>载入回测配置</h3><div class="panel-sub" style="margin-top:3px">保存在浏览器本地</div></div>
        <button class="icon-btn" data-close>✕</button></div>
      <div style="display:flex;flex-direction:column;gap:6px">
        ${list.map((item, i) => `<div style="display:flex;align-items:center;justify-content:space-between;gap:10px;border:1px solid var(--border-soft);background:#0a121c;border-radius:8px;padding:9px 12px">
          <div><div style="font-size:13px">${esc(item.name)}</div><div class="panel-sub" style="margin-top:1px">${esc(item.created)} · ${(item.config.factors || []).length} 个因子</div></div>
          <div class="form-row">
            <button class="btn sm" data-apply="${i}">载入</button>
            <button class="btn sm danger" data-remove="${i}">删除</button>
          </div>
        </div>`).join("")}
      </div>`, true);
    $$("#modalBox [data-apply]").forEach((btn) => btn.addEventListener("click", () => {
      applyConfig(loadSavedConfigs()[Number(btn.dataset.apply)].config);
      closeModal();
    }));
    $$("#modalBox [data-remove]").forEach((btn) => btn.addEventListener("click", () => {
      const rest = loadSavedConfigs();
      rest.splice(Number(btn.dataset.remove), 1);
      persistConfigs(rest);
      openLoadConfig();
    }));
  }

  function applyConfig(cfg) {
    if (!cfg) return;
    const ids = new Set(state.factors.map((f) => String(f.id)));
    const factors = (cfg.factors || [])
      .map((f) => ({ id: String(f.factor_id), weight: Number(f.weight) || 1 }))
      .filter((f) => ids.has(f.id));
    if (factors.length) state.btFactors = factors;
    state.universe = new Set((cfg.codes || []).filter((c) => state.stocks.some((s) => s.code === c)));

    const setVal = (sel, val) => {
      const el = $(sel);
      if (el && val !== undefined && val !== null) el.value = val;
    };
    setVal("#btStart", cfg.start);
    setVal("#btEnd", cfg.end);
    setVal("#btRebalance", cfg.rebalance_days);
    setVal("#btRebalanceMode", cfg.rebalance_mode || "interval");
    setVal("#btRebalanceWeekday", cfg.rebalance_weekday || 0);
    setVal("#btRebalanceMonthDay", cfg.rebalance_month_day || 1);
    updateRebalanceUI();
    setVal("#btTopN", cfg.top_n);
    setVal("#btTopPct", cfg.top_pct);
    setVal("#btCommission", cfg.commission_bps);
    setVal("#btStamp", cfg.stamp_duty_bps);
    setVal("#btSlip", cfg.slippage_bps);
    setVal("#btWeighting", cfg.weighting);
    setVal("#btMaxWeight", cfg.max_weight);
    setVal("#btIndustryCap", cfg.industry_cap || 0);
    setVal("#btMaxPosition", cfg.max_position_pct || 100);
    setVal("#btVolTarget", cfg.vol_target_pct || 0);
    setVal("#btVolLookback", cfg.vol_lookback || 20);
    setVal("#btMinHold", cfg.min_hold_days || 0);
    setVal("#btTurnoverCap", cfg.turnover_cap_pct || 0);
    setVal("#btRebalanceBand", cfg.rebalance_band_pct || 0);
    setVal("#btBenchmark", cfg.benchmark);
    setVal("#btTiming", cfg.market_timing ? String(cfg.timing_window || 20) : "none");
    setVal("#btOOS", cfg.oos_split || 0);
    setVal("#btStopLoss", cfg.stop_loss_pct || 0);
    setVal("#btTakeProfit", cfg.take_profit_pct || 0);
    setVal("#btTrailingStop", cfg.trailing_stop_pct || 0);
    const filters = cfg.filters || {};
    $("#filterStar").checked = filters.exclude_star !== false;
    $("#filterBJ").checked = filters.exclude_bj !== false;
    $("#filterLimitUp").checked = !!filters.exclude_limit_up;
    $("#filterTrade").checked = !!cfg.trade_constraints;
    setVal("#filterAmount", filters.min_amount);
    setVal("#filterDays", filters.min_list_days);
    const wantedInd = new Set(filters.industries || []);
    $$("#filterIndustries option").forEach((o) => { o.selected = wantedInd.has(o.value); });
    const pre = cfg.preprocess || {};
    $("#ppWinsor").checked = !!pre.winsorize;
    $("#ppNeutral").checked = !!pre.neutralize;
    $("#ppStd").checked = pre.standardize !== false;
    $("#ppOrtho").checked = !!pre.orthogonalize;
    $("#btLongShort").checked = !!cfg.long_short;

    renderCombo();
    renderUniverse();
    toast("配置已载入");
  }

  /* ------------------------------ 批量评测 ------------------------------ */
  async function openBatch() {
    openModal(`
      <div class="modal-head"><div><h3>批量因子评测</h3><div class="panel-sub" style="margin-top:3px">用当前回测配置评测全部内置因子</div></div>
        <button class="icon-btn" data-close>✕</button></div>
      <div class="empty" style="border:none;padding:60px 20px">
        <span class="spinner" style="width:24px;height:24px;margin:0 auto 14px;display:block"></span>
        <p>正在批量回测内置因子…</p>
        <div class="sub">约 14 个因子，复用 JAX 编译缓存</div>
      </div>`, true);
    try {
      const res = await api("/api/backtest/batch", { method: "POST", body: buildCommonConfig() });
      renderBatch(res);
    } catch (e) {
      closeModal();
      toast(e.message, true);
    }
  }

  function renderBatch(res) {
    const c = res.config;
    openModal(`
      <div class="modal-head">
        <div><h3>批量因子评测</h3>
          <div class="panel-sub" style="margin-top:3px">${c.stocks} 只股票 · ${c.start} → ${c.end} · ${c.rebalance_days} 日调仓 · 按夏普排序</div>
        </div>
        <button class="icon-btn" data-close>✕</button>
      </div>
      <div class="table-wrap" style="max-height:460px">
        <table>
          <thead><tr><th>因子</th><th>分类</th><th style="text-align:right">年化</th><th style="text-align:right">夏普</th><th style="text-align:right">回撤</th><th style="text-align:right">IC</th><th style="text-align:right">ICIR</th><th></th></tr></thead>
          <tbody>
            ${res.items.map((it, i) => `<tr style="cursor:default">
              <td>${esc(it.label)}</td>
              <td><span class="chip">${esc(it.category)}</span></td>
              <td style="text-align:right" class="num-mono ${colorClass(it.annual_return)}">${fmtPct(it.annual_return)}</td>
              <td style="text-align:right" class="num-mono">${fmtNum(it.sharpe)}</td>
              <td style="text-align:right" class="num-mono">${fmtPct(it.max_drawdown)}</td>
              <td style="text-align:right" class="num-mono">${fmtNum(it.ic, 3)}</td>
              <td style="text-align:right" class="num-mono">${fmtNum(it.icir, 2)}</td>
              <td><button class="btn sm" data-batch-use="${i}">选用</button></td>
            </tr>`).join("")}
          </tbody>
        </table>
      </div>
      <div class="modal-foot"><span class="panel-sub">选用后替换当前因子组合</span><button class="btn ghost" data-close>关闭</button></div>`, true);
    $$("#modalBox [data-batch-use]").forEach((btn) => btn.addEventListener("click", () => {
      const item = res.items[Number(btn.dataset.batchUse)];
      state.btFactors = [{ id: String(item.id), weight: 1 }];
      renderCombo();
      closeModal();
      toast(`已选用因子「${item.label}」`);
    }));
  }

  /* ------------------------------ 成本敏感性 ------------------------------ */
  async function openCostSensitivity() {
    openModal(`
      <div class="modal-head"><div><h3>交易成本敏感性</h3><div class="panel-sub" style="margin-top:3px">不同单边综合成本下的绩效变化</div></div>
        <button class="icon-btn" data-close>✕</button></div>
      <div class="empty" style="border:none;padding:50px 20px">
        <span class="spinner" style="width:22px;height:22px;margin:0 auto 12px;display:block"></span>
        <p>正在回测不同成本档位…</p>
      </div>`, true);
    try {
      const res = await api("/api/backtest/cost-sensitivity", {
        method: "POST", body: buildCommonConfig(),
      });
      renderCostSensitivity(res);
    } catch (e) {
      closeModal();
      toast(e.message, true);
    }
  }

  function renderCostSensitivity(res) {
    openModal(`
      <div class="modal-head">
        <div><h3>交易成本敏感性</h3>
          <div class="panel-sub" style="margin-top:3px">${res.factor.label} · ${res.config.stocks} 只股票 · ${res.config.rebalance_days} 日调仓</div>
        </div>
        <button class="icon-btn" data-close>✕</button>
      </div>
      <div class="chart-wrap"><canvas id="costChart"></canvas></div>
      <div class="table-wrap" style="margin-top:12px;max-height:260px">
        <table><thead><tr><th>单边成本</th><th style="text-align:right">年化收益</th><th style="text-align:right">夏普</th><th style="text-align:right">最大回撤</th></tr></thead>
        <tbody>${res.items.map((it) => `<tr style="cursor:default" ${it.cost_bps === Math.round(res.current_cost_bps) ? 'class="best-row"' : ""}>
          <td class="num-mono">${it.cost_bps} bps</td>
          <td style="text-align:right" class="num-mono ${colorClass(it.annual_return)}">${fmtPct(it.annual_return)}</td>
          <td style="text-align:right" class="num-mono">${fmtNum(it.sharpe)}</td>
          <td style="text-align:right" class="num-mono">${fmtPct(it.max_drawdown)}</td>
        </tr>`).join("")}</tbody></table>
      </div>
      <div class="modal-foot"><span class="panel-sub">当前佣金档位 ${fmtNum(res.current_cost_bps, 1)} bps 高亮显示</span><button class="btn ghost" data-close>关闭</button></div>`, true);
    Charts.line($("#costChart"), {
      labels: res.items.map((it) => it.cost_bps + "bps"),
      series: [{ name: "年化收益", data: res.items.map((it) => it.annual_return * 100), color: "#3ddc97", fill: true }],
      height: 230, yFmt: (v) => v.toFixed(0) + "%",
    });
  }

  /* ------------------------------ 因子权重优化 ------------------------------ */
  function openWeightOpt() {
    if (state.btFactors.length < 2) {
      toast("请先在因子组合中选择至少两个因子", true);
      return;
    }
    openModal(`
      <div class="modal-head">
        <div><h3>因子权重优化</h3>
          <div class="panel-sub" style="margin-top:3px">在回测框架下搜索非负权重（和为 1），最大化指定绩效指标</div></div>
        <button class="icon-btn" data-close>✕</button>
      </div>
      <div class="form-row" style="gap:14px;align-items:flex-end">
        <div class="field"><label>优化目标</label>
          <select id="woObjective">
            <option value="sharpe">夏普比率</option>
            <option value="calmar">卡玛比率</option>
            <option value="annual_return">年化收益</option>
          </select>
        </div>
        <button class="btn primary" id="woRun">开始优化</button>
        <span class="panel-sub">${state.btFactors.length} 个因子 · 使用当前回测配置</span>
      </div>
      <div id="woBody" style="margin-top:14px">
        <div class="hint-box">优化为启发式搜索（SLSQP），结果可能随初始权重与数据区间变化，建议结合样本外表现判断。</div>
      </div>`, true);
    $("#woRun").addEventListener("click", runWeightOpt);
  }

  async function runWeightOpt() {
    const btn = $("#woRun");
    const box = $("#woBody");
    btn.disabled = true;
    btn.innerHTML = `<span class="spinner"></span> 优化中`;
    box.innerHTML = `<div class="empty" style="border:none"><p>正在迭代搜索因子权重…</p><div class="sub">每次迭代执行一次完整回测</div></div>`;
    try {
      const payload = { ...buildCommonConfig(), objective: $("#woObjective").value };
      const res = await api("/api/backtest/optimize-weights", { method: "POST", body: payload });
      renderWeightOpt(res);
    } catch (e) {
      box.innerHTML = `<div class="hint-box"><span class="down">${esc(e.message)}</span></div>`;
    } finally {
      btn.disabled = false;
      btn.textContent = "开始优化";
    }
  }

  function renderWeightOpt(res) {
    const objLabel = { sharpe: "夏普", calmar: "卡玛", annual_return: "年化收益" }[res.objective];
    const fmtVal = (v) => (res.objective === "annual_return" ? fmtPct(v) : fmtNum(v, 3));
    const metricRow = (m) => `
      <span class="chip">年化 ${fmtPct(m.annual_return)}</span>
      <span class="chip">夏普 ${fmtNum(m.sharpe)}</span>
      <span class="chip">回撤 ${fmtPct(m.max_drawdown)}</span>`;
    const weightsHtml = (weights) => res.labels.map((label, i) => `
      <div class="market-row" style="cursor:default">
        <div class="nm"><span class="nm">${esc(label)}</span></div>
        <div class="px num-mono">${fmtPct(weights[i], 1, false)}</div>
      </div>`).join("");
    $("#woBody").innerHTML = `
      <div class="grid-2">
        <div class="panel" style="padding:14px">
          <div class="panel-title" style="font-size:13px">当前权重</div>
          <div class="factor-meta" style="margin:8px 0">${metricRow(res.baseline.metrics)}</div>
          <div class="panel-sub">${objLabel} ${fmtVal(res.baseline.value)}</div>
          <div style="margin-top:8px">${weightsHtml(res.baseline.weights)}</div>
        </div>
        <div class="panel" style="padding:14px">
          <div class="panel-title" style="font-size:13px">优化权重</div>
          <div class="factor-meta" style="margin:8px 0">${metricRow(res.best.metrics)}</div>
          <div class="panel-sub">${objLabel} ${fmtVal(res.best.value)}</div>
          <div style="margin-top:8px">${weightsHtml(res.best.weights)}</div>
        </div>
      </div>
      <div class="modal-foot" style="padding-left:0;padding-right:0">
        <span class="panel-sub">共评估 ${res.iterations} 次 · ${esc(res.universe.start)} → ${esc(res.universe.end)}</span>
        <div class="form-row" style="gap:6px">
          <button class="btn ghost" data-close>关闭</button>
          <button class="btn primary" id="woApply">应用优化权重</button>
        </div>
      </div>`;
    $("#woApply").addEventListener("click", () => {
      state.btFactors = state.btFactors.map((f, i) => ({ ...f, weight: res.best.weights[i] }));
      renderCombo();
      closeModal();
      toast("已应用优化权重，可重新运行回测");
    });
  }

  async function runBacktest() {
    const btn = $("#runBacktestBtn");
    if (!state.btFactors.length) { toast("请至少添加一个因子", true); return; }
    const payload = { ...buildCommonConfig(), name: payloadLabel() };
    btn.disabled = true;
    btn.innerHTML = `<span class="spinner"></span> JAX 编译中`;
    $("#btResults").innerHTML = `<div class="panel"><div class="empty" style="border:none;padding:80px 24px"><span class="spinner" style="width:26px;height:26px;margin:0 auto 16px;display:block"></span><p>正在执行向量化回测…</p><div class="sub">多因子合成与 JAX 编译通常需要数秒</div></div></div>`;
    try {
      const res = await api("/api/backtest", { method: "POST", body: payload });
      state.btResult = res;
      renderBacktestResult(res);
      toast(`回测完成 · 年化 ${fmtPct(res.metrics.annual_return)}`);
    } catch (e) {
      $("#btResults").innerHTML = `<div class="panel"><div class="empty" style="border:none;padding:70px 24px">
        <p style="color:var(--red)">回测失败</p><div class="sub">${esc(e.message)}</div></div></div>`;
      toast(e.message, true);
    } finally {
      btn.disabled = false;
      btn.textContent = "运行回测";
    }
  }

  function payloadLabel() {
    const labels = state.btFactors.map((f) => factorLabel(f.id));
    let label = labels[0] || "因子";
    if (labels.length > 1) label = `${labels[0]} 等 ${labels.length} 因子`;
    const n = state.universe.size;
    return `${label} · ${n ? n + "只" : "全市场"}`;
  }

  function factorLabel(id) {
    const f = findFactorById(id);
    return f ? (f.label || f.name) : String(id);
  }

  function renderBacktestResult(res) {
    const m = res.metrics;
    const cfg = res.config;
    const navPct = res.nav.map((v) => (v - 1) * 100);
    const benchPct = res.benchmark.map((v) => (v - 1) * 100);
    const excessPct = res.nav.map((v, i) => res.benchmark[i] ? (v / res.benchmark[i] - 1) * 100 : null);
    const ddPct = res.drawdown.map((v) => v * 100);
    const benchAnnual = {};
    (res.annual_returns_benchmark || []).forEach((a) => { benchAnnual[a.year] = a.return; });

    const perf = [
      ["年化收益", fmtPct(m.annual_return), colorClass(m.annual_return)],
      ["累计收益", fmtPct(m.total_return), colorClass(m.total_return)],
      ["夏普比率", fmtNum(m.sharpe), ""],
      ["最大回撤", fmtPct(m.max_drawdown), "neg"],
      ["年化波动", fmtPct(m.annual_volatility, 1, false), ""],
      ["胜率", fmtPct(m.win_rate, 1, false), ""],
      ["超额年化", fmtPct(m.excess_annual_return), colorClass(m.excess_annual_return)],
      ["信息比率", fmtNum(m.information_ratio), ""],
      ["Alpha", fmtPct(m.alpha), colorClass(m.alpha)],
      ["Beta", fmtNum(m.beta), ""],
      ["索提诺", fmtNum(m.sortino), ""],
      ["年化换手", fmtNum(m.annual_turnover, 1) + "x", ""],
      ["VaR 95%", fmtPct(m.var_95), "neg"],
      ["CVaR 95%", fmtPct(m.cvar_95), "neg"],
      ["上行捕获", fmtNum(m.up_capture, 2), ""],
      ["下行捕获", fmtNum(m.down_capture, 2), ""],
      ["最长回撤", (m.max_drawdown_days == null ? "—" : m.max_drawdown_days + " 天"), ""],
      ["月度胜率", fmtPct(m.monthly_win_rate, 1, false), ""],
    ];

    const factorChips = (res.factor.factors || []).map((f) =>
      `<span class="chip ${f.weight < 0 ? "amber" : "blue"}">${esc(f.label)} ${f.weight > 0 ? "+" : ""}${(f.weight * 100).toFixed(0)}%</span>`
    ).join("");
    const pre = res.factor.preprocess || {};
    const preChips = [
      pre.winsorize ? "去极值" : "",
      pre.neutralize ? "行业中性" : "",
      pre.standardize ? "标准化" : "",
      pre.orthogonalize ? "风格正交" : "",
    ].filter(Boolean).map((p) => `<span class="chip green">${p}</span>`).join("");
    const benchName = cfg.benchmark_name || (cfg.benchmark === "market"
      ? "全市场等权"
      : (cfg.benchmark || "").startsWith("index:")
        ? ((state.indices.find((i) => "index:" + i.code === cfg.benchmark) || {}).name || "指数基准")
        : (cfg.benchmark || "").startsWith("universe:")
          ? (state.universes.find((u) => "universe:" + u.id === cfg.benchmark) || {}).name || "自选池基准"
          : "股票池等权");
    const weightLabel = { equal: "等权", score: "得分加权", inverse_vol: "波动率倒数", risk_parity: "风险平价" }[cfg.weighting] || "等权";
    const rebalanceLabel = cfg.rebalance_mode === "weekly"
      ? `每周${["一", "二", "三", "四", "五"][cfg.rebalance_weekday || 0]}调仓`
      : cfg.rebalance_mode === "monthly"
        ? `每月第 ${cfg.rebalance_month_day || 1} 个交易日调仓`
        : `${cfg.rebalance_days} 日调仓`;

    $("#btResults").innerHTML = `
      <div class="panel">
        <div class="result-head">
          <div>
            <div class="result-title">${esc(cfg.name)} <small>#${res.id} · ${esc(res.factor.label)}</small></div>
            <div class="factor-meta" style="margin:8px 0 0">
              <span class="chip">${esc(res.universe.start)} → ${esc(res.universe.end)}</span>
              <span class="chip">${res.universe.stocks} 只股票</span>
              <span class="chip">${rebalanceLabel}</span>
              <span class="chip">${cfg.top_n > 0 ? "持仓 " + cfg.top_n : "前 " + (cfg.top_pct * 100).toFixed(0) + "%"}</span>
              <span class="chip">佣金 ${fmtNum(cfg.commission_bps, 1)} / 印花 ${fmtNum(cfg.stamp_duty_bps || 0, 1)} / 滑点 ${fmtNum(cfg.slippage_bps || 0, 1)} bps</span>
              <span class="chip">${weightLabel}${cfg.max_weight > 0 ? " · 上限 " + cfg.max_weight + "%" : ""}</span>
              ${cfg.industry_cap > 0 ? `<span class="chip">行业上限 ${fmtNum(cfg.industry_cap, 0)}%</span>` : ""}
              ${cfg.max_position_pct && cfg.max_position_pct < 100 ? `<span class="chip amber">最大仓位 ${fmtNum(cfg.max_position_pct, 0)}%</span>` : ""}
              ${cfg.vol_target_pct > 0 ? `<span class="chip amber">目标波动 ${fmtNum(cfg.vol_target_pct, 0)}%</span>` : ""}
              ${cfg.min_hold_days > 0 ? `<span class="chip">最小持有 ${cfg.min_hold_days} 天</span>` : ""}
              ${cfg.turnover_cap_pct > 0 ? `<span class="chip">换手上限 ${fmtNum(cfg.turnover_cap_pct, 0)}%</span>` : ""}
              ${cfg.rebalance_band_pct > 0 ? `<span class="chip">缓冲带 ${fmtNum(cfg.rebalance_band_pct, 1)}%</span>` : ""}
              ${cfg.long_short ? `<span class="chip amber">多空</span>` : ""}
              ${cfg.market_timing ? `<span class="chip amber">MA${cfg.timing_window || 20} 择时</span>` : ""}
              ${cfg.oos_split > 0 ? `<span class="chip">样本外 ${Math.round(cfg.oos_split * 100)}%</span>` : ""}
              ${cfg.trade_constraints ? `<span class="chip amber">涨跌停/停牌限制</span>` : ""}
              ${cfg.stop_loss_pct > 0 ? `<span class="chip amber">止损 ${fmtNum(cfg.stop_loss_pct, 0)}%</span>` : ""}
              ${cfg.take_profit_pct > 0 ? `<span class="chip">止盈 ${fmtNum(cfg.take_profit_pct, 0)}%</span>` : ""}
              ${cfg.trailing_stop_pct > 0 ? `<span class="chip amber">移动止损 ${fmtNum(cfg.trailing_stop_pct, 0)}%</span>` : ""}
              <span class="chip green">基准 · ${benchName}</span>
            </div>
            <div class="factor-meta" style="margin:7px 0 0">
              ${factorChips}${preChips}
            </div>
          </div>
          <div class="form-row" style="gap:6px;flex-wrap:wrap;justify-content:flex-end">
            <button class="btn sm" id="exportCsv">导出净值</button>
            <button class="btn sm" id="exportTrades">导出交易</button>
            <button class="btn sm" id="exportHoldings">导出持仓</button>
            <button class="btn sm" id="exportReport">导出报告</button>
          </div>
        </div>
        <div class="perf-grid">
          ${perf.map(([k, v, cls]) => `<div class="perf"><div class="k">${k}</div><div class="v ${cls}">${v}</div></div>`).join("")}
        </div>
      </div>

      <div class="panel">
        <div class="chart-title-row">
          <div>
            <div class="panel-title" style="font-size:13.5px">策略净值曲线</div>
            <div class="panel-sub">初始净值 = 0，单位：% · 基准为${benchName}组合</div>
          </div>
          <span class="chip green">IC ${fmtNum(res.factor.ic.mean, 3)} · IR ${fmtNum(res.factor.ic.ir, 2)}</span>
        </div>
        <div class="chart-wrap"><canvas id="equityChart"></canvas></div>
      </div>

      <div class="grid-2">
        <div class="panel">
          <div class="chart-title-row"><div><div class="panel-title" style="font-size:13px">回撤曲线</div><div class="panel-sub">策略相对历史高点回撤</div></div></div>
          <div class="chart-wrap"><canvas id="ddChart"></canvas></div>
        </div>
        <div class="panel">
          <div class="chart-title-row"><div><div class="panel-title" style="font-size:13px">因子分位收益</div><div class="panel-sub">Q1 → Q5，未来 5 日年化</div></div></div>
          <div class="chart-wrap"><canvas id="qChart"></canvas></div>
        </div>
      </div>

      ${res.drawdowns ? `<div class="panel" id="ddEpisodesPanel">
        <div class="chart-title-row"><div>
          <div class="panel-title" style="font-size:13px">回撤区间分析</div>
          <div class="panel-sub">共 ${res.drawdowns.count} 段回撤 · 平均修复 ${res.drawdowns.avg_recovery_days == null ? "—" : res.drawdowns.avg_recovery_days} 个交易日 · 当前回撤 ${fmtPct(res.drawdowns.current_drawdown)}</div>
        </div></div>
        <div class="table-wrap">
          <table><thead><tr><th>#</th><th>峰值日</th><th>谷底日</th><th>修复日</th><th style="text-align:right">回撤幅度</th><th style="text-align:right">下跌天数</th><th style="text-align:right">修复天数</th></tr></thead>
          <tbody>${res.drawdowns.items.map((e, k) => `<tr style="cursor:default">
            <td class="num-mono">${k + 1}</td>
            <td class="num-mono">${esc(e.peak_date)}</td>
            <td class="num-mono">${esc(e.trough_date)}</td>
            <td class="num-mono">${e.recovered ? esc(e.recovery_date) : "未修复"}</td>
            <td style="text-align:right" class="num-mono down">${fmtPct(e.depth)}</td>
            <td style="text-align:right" class="num-mono">${e.days}</td>
            <td style="text-align:right" class="num-mono">${e.recovered ? e.recovery_days : "—"}</td>
          </tr>`).join("")}</tbody></table>
        </div>
      </div>` : ""}

      ${res.rolling_risk ? `<div class="grid-2">
        <div class="panel">
          <div class="chart-title-row"><div><div class="panel-title" style="font-size:13px">滚动夏普（60 日）</div><div class="panel-sub">衡量策略风险调整后收益的时间稳定性</div></div></div>
          <div class="chart-wrap"><canvas id="rollingChart"></canvas></div>
        </div>
        <div class="panel">
          <div class="chart-title-row"><div><div class="panel-title" style="font-size:13px">日收益分布</div><div class="panel-sub">频次直方图 · 单位 %</div></div></div>
          <div class="chart-wrap"><canvas id="histChart"></canvas></div>
        </div>
      </div>` : ""}

      <div class="panel">
        <div class="chart-title-row"><div><div class="panel-title" style="font-size:13px">月度收益热力图</div><div class="panel-sub">红色为正收益，绿色为负收益</div></div></div>
        <div id="heatmap" class="heatmap"></div>
      </div>

      ${(res.annual_returns || []).length ? `<div class="panel">
        <div class="chart-title-row"><div><div class="panel-title" style="font-size:13px">年度收益</div><div class="panel-sub">按自然年统计</div></div></div>
        <div class="grid-3" style="grid-template-columns:repeat(auto-fit, minmax(120px, 1fr))">
          ${res.annual_returns.map((a) => {
            const bm = benchAnnual[a.year];
            return `<div class="perf"><div class="k">${a.year} 年</div><div class="v ${colorClass(a.return)}">${fmtPct(a.return)}</div>
              <div class="panel-sub" style="margin-top:2px">基准 ${bm == null ? "—" : fmtPct(bm)} · 超额 ${bm == null ? "—" : fmtPct(a.return - bm)}</div></div>`;
          }).join("")}
        </div>
      </div>` : ""}

      ${res.oos ? `<div class="panel">
        <div class="chart-title-row"><div><div class="panel-title" style="font-size:13px">样本内 / 样本外检验</div><div class="panel-sub">分界日 ${esc(res.oos.split_date)}</div></div></div>
        <div class="grid-2">
          <div>
            <div class="panel-sub" style="margin-bottom:8px">样本内</div>
            <div class="grid-3">
              ${[["年化收益", fmtPct(res.oos.in_sample.annual_return)], ["夏普", fmtNum(res.oos.in_sample.sharpe)], ["最大回撤", fmtPct(res.oos.in_sample.max_drawdown)]].map(([k, v]) => `<div class="perf"><div class="k">${k}</div><div class="v">${v}</div></div>`).join("")}
            </div>
          </div>
          <div>
            <div class="panel-sub" style="margin-bottom:8px">样本外</div>
            <div class="grid-3">
              ${[["年化收益", fmtPct(res.oos.out_of_sample.annual_return)], ["夏普", fmtNum(res.oos.out_of_sample.sharpe)], ["最大回撤", fmtPct(res.oos.out_of_sample.max_drawdown)]].map(([k, v]) => `<div class="perf"><div class="k">${k}</div><div class="v">${v}</div></div>`).join("")}
            </div>
          </div>
        </div>
      </div>` : ""}

      ${res.bootstrap ? `<div class="panel">
        <div class="chart-title-row"><div><div class="panel-title" style="font-size:13px">稳健性检验（Bootstrap）</div><div class="panel-sub">日收益重采样 ${res.bootstrap.samples} 次 · 95% 置信区间</div></div></div>
        <div class="grid-2">
          <div class="perf">
            <div class="k">年化收益 95% CI</div>
            <div class="v">${fmtPct(res.bootstrap.annual_return.low)} ~ ${fmtPct(res.bootstrap.annual_return.high)}</div>
            <div class="panel-sub" style="margin-top:3px">中位数 ${fmtPct(res.bootstrap.annual_return.mid)}</div>
          </div>
          <div class="perf">
            <div class="k">夏普比率 95% CI</div>
            <div class="v">${fmtNum(res.bootstrap.sharpe.low)} ~ ${fmtNum(res.bootstrap.sharpe.high)}</div>
            <div class="panel-sub" style="margin-top:3px">中位数 ${fmtNum(res.bootstrap.sharpe.mid)}</div>
          </div>
        </div>
      </div>` : ""}

      <div class="panel" id="industryPanel">
        <div class="chart-title-row"><div><div class="panel-title" style="font-size:13px">行业暴露</div><div class="panel-sub">最近一次调仓的行业净权重</div></div></div>
        <div class="chart-wrap"><canvas id="industryChart"></canvas></div>
      </div>

      ${res.style_exposure ? `<div class="panel">
        <div class="chart-title-row"><div><div class="panel-title" style="font-size:13px">风格暴露</div><div class="panel-sub">持仓在动量 / 反转 / 波动 / 流动性 / Beta 上的截面标准化暴露，正值表示偏向该风格</div></div></div>
        <div class="chart-wrap"><canvas id="styleChart"></canvas></div>
      </div>` : ""}

      <div class="panel" id="attributionPanel">
        <div class="chart-title-row"><div><div class="panel-title" style="font-size:13px">收益归因</div><div class="panel-sub">个股与行业对区间收益的累计贡献</div></div></div>
        <div class="grid-2">
          <div class="chart-wrap"><canvas id="attrChart"></canvas></div>
          <div class="table-wrap" style="max-height:230px">
            <table><thead><tr><th>行业</th><th style="text-align:right">贡献</th></tr></thead>
            <tbody>${((res.attribution && res.attribution.industries) || []).map((r) => `<tr style="cursor:default"><td>${esc(r.industry)}</td><td style="text-align:right" class="num-mono ${r.contribution >= 0 ? "up" : "down"}">${fmtPct(r.contribution)}</td></tr>`).join("")}</tbody></table>
          </div>
        </div>
      </div>

      ${res.brinson ? `<div class="panel" id="brinsonPanel">
        <div class="chart-title-row"><div><div class="panel-title" style="font-size:13px">Brinson 归因</div><div class="panel-sub">超额收益分解为行业配置、选股与交互效应</div></div></div>
        <div class="table-wrap" style="max-height:300px">
          <table>
            <thead><tr><th>行业</th><th style="text-align:right">配置效应</th><th style="text-align:right">选股效应</th><th style="text-align:right">交互效应</th><th style="text-align:right">合计</th></tr></thead>
            <tbody>
              ${res.brinson.industries.map((r) => `<tr style="cursor:default"><td>${esc(r.industry)}</td>
                <td style="text-align:right" class="num-mono ${r.allocation >= 0 ? "up" : "down"}">${fmtPct(r.allocation)}</td>
                <td style="text-align:right" class="num-mono ${r.selection >= 0 ? "up" : "down"}">${fmtPct(r.selection)}</td>
                <td style="text-align:right" class="num-mono ${r.interaction >= 0 ? "up" : "down"}">${fmtPct(r.interaction)}</td>
                <td style="text-align:right" class="num-mono ${r.total >= 0 ? "up" : "down"}">${fmtPct(r.total)}</td></tr>`).join("")}
              <tr style="cursor:default"><td style="font-weight:600">合计</td>
                <td style="text-align:right" class="num-mono">${fmtPct(res.brinson.totals.allocation)}</td>
                <td style="text-align:right" class="num-mono">${fmtPct(res.brinson.totals.selection)}</td>
                <td style="text-align:right" class="num-mono">${fmtPct(res.brinson.totals.interaction)}</td>
                <td style="text-align:right" class="num-mono" style="font-weight:600">${fmtPct(res.brinson.totals.total)}</td></tr>
            </tbody>
          </table>
        </div>
      </div>` : ""}

      <div class="panel">
        <div class="chart-title-row"><div><div class="panel-title" style="font-size:13px">调仓持仓</div><div class="panel-sub">最近 ${res.holdings.length} 次调仓的持仓明细（权重）</div></div></div>
        <div class="holdings-grid" id="holdingsGrid"></div>
      </div>

      <div class="panel" id="tradesPanel">
        <div class="chart-title-row"><div><div class="panel-title" style="font-size:13px">交易明细</div><div class="panel-sub">最近调仓的主要买卖变化</div></div></div>
        <div class="table-wrap" style="max-height:300px">
          <table><thead><tr><th>日期</th><th>方向</th><th>股票</th><th style="text-align:right">权重变化</th></tr></thead>
          <tbody>${((res.trades) || []).flatMap((t) => t.items.map((i) => `<tr style="cursor:default"><td class="num-mono">${esc(t.date)}</td><td><span class="chip ${i.side === "买入" ? "green" : "amber"}">${i.side}</span></td><td>${esc(i.name)} <span class="cd">${esc(i.code)}</span></td><td style="text-align:right" class="num-mono ${i.change > 0 ? "up" : "down"}">${(i.change * 100).toFixed(2)}%</td></tr>`)).join("")}</tbody></table>
        </div>
      </div>`;

    Charts.line($("#equityChart"), {
      labels: res.dates,
      series: [
        { name: "策略", data: navPct, color: "#3ddc97", fill: true },
        { name: benchName, data: benchPct, color: "#6aa9ff", dash: [5, 4], width: 1.3 },
        { name: "超额(策略/基准)", data: excessPct, color: "#f5b56b", dash: [3, 3], width: 1.2 },
      ],
      height: 300, yFmt: (v) => v.toFixed(0) + "%", xFmt: shortDate,
    });
    Charts.line($("#ddChart"), {
      labels: res.dates,
      series: [{ name: "回撤", data: ddPct, color: "#ff6b7a", fill: true, fillColor: "rgba(255,107,122,0.16)" }],
      height: 200, yFmt: (v) => v.toFixed(1) + "%", xFmt: shortDate, includeZero: true,
    });
    Charts.bar($("#qChart"), {
      labels: res.factor.quantile_returns.map((q) => q.group),
      values: res.factor.quantile_returns.map((q) => q.annualized * 100),
      height: 200, colorsByValue: true, yFmt: (v) => v.toFixed(0) + "%", includeZero: true,
    });

    renderHeatmap(res.monthly);
    renderHoldings(res.holdings);

    if (res.rolling_risk) {
      Charts.line($("#rollingChart"), {
        labels: res.rolling_risk.dates,
        series: [{ name: "滚动夏普", data: res.rolling_risk.sharpe, color: "#6aa9ff" }],
        height: 210, yFmt: (v) => v.toFixed(1), xFmt: shortDate, includeZero: true,
      });
    }
    if (res.return_hist) {
      Charts.bar($("#histChart"), {
        labels: res.return_hist.labels,
        values: res.return_hist.values,
        height: 210, colors: ["#6aa9ff"], yFmt: (v) => String(v),
      });
    }

    const exposure = (res.industry_exposure || []).slice(0, 10);
    if (exposure.length) {
      Charts.bar($("#industryChart"), {
        labels: exposure.map((e) => e.industry),
        values: exposure.map((e) => e.last * 100),
        height: 220, colorsByValue: true, yFmt: (v) => v.toFixed(1) + "%", includeZero: true,
      });
    } else {
      const panel = $("#industryPanel");
      if (panel) panel.style.display = "none";
    }

    if (res.style_exposure) {
      Charts.line($("#styleChart"), {
        labels: res.style_exposure.dates,
        series: res.style_exposure.series.map((s) => ({ name: s.name, data: s.values })),
        height: 230, yFmt: (v) => v.toFixed(2), xFmt: shortDate, includeZero: true,
      });
    }

    const attrStocks = (res.attribution && res.attribution.stocks) || [];
    if (attrStocks.length) {
      const top = attrStocks.slice(0, 8);
      Charts.bar($("#attrChart"), {
        labels: top.map((s) => s.name),
        values: top.map((s) => s.contribution * 100),
        height: 230, colorsByValue: true, yFmt: (v) => v.toFixed(1) + "%", includeZero: true,
      });
    } else {
      const panel = $("#attributionPanel");
      if (panel) panel.style.display = "none";
    }
    if (!res.trades || !res.trades.length) {
      const panel = $("#tradesPanel");
      if (panel) panel.style.display = "none";
    }
    const exportBtn = $("#exportCsv");
    if (exportBtn) exportBtn.addEventListener("click", () => exportBacktestCsv(res));
    const tradesBtn = $("#exportTrades");
    if (tradesBtn) tradesBtn.addEventListener("click", () => exportTradesCsv(res));
    const holdingsBtn = $("#exportHoldings");
    if (holdingsBtn) holdingsBtn.addEventListener("click", () => exportHoldingsCsv(res));
    const reportBtn = $("#exportReport");
    if (reportBtn) reportBtn.addEventListener("click", () => exportBacktestReport(res));
  }

  function exportTradesCsv(res) {
    const rows = [["date", "side", "code", "name", "weight_change"]];
    (res.trades || []).forEach((t) => t.items.forEach((i) =>
      rows.push([t.date, i.side, i.code, i.name, i.change])));
    if (rows.length < 2) { toast("暂无交易记录", true); return; }
    downloadCsv(rows, `trades_${res.id || "latest"}.csv`);
    toast("交易明细已导出 CSV");
  }

  function exportHoldingsCsv(res) {
    const rows = [["date", "code", "name", "weight"]];
    (res.holdings || []).forEach((h) => h.items.forEach((i) =>
      rows.push([h.date, i.code, i.name, i.weight])));
    if (rows.length < 2) { toast("暂无持仓记录", true); return; }
    downloadCsv(rows, `holdings_${res.id || "latest"}.csv`);
    toast("持仓明细已导出 CSV");
  }

  function exportBacktestReport(res) {
    const m = res.metrics;
    const cfg = res.config;
    const eq = document.getElementById("equityChart");
    const dd = document.getElementById("ddChart");
    const eqImg = eq ? eq.toDataURL("image/png") : "";
    const ddImg = dd ? dd.toDataURL("image/png") : "";
    const metricRows = [
      ["年化收益", fmtPct(m.annual_return)], ["累计收益", fmtPct(m.total_return)],
      ["夏普比率", fmtNum(m.sharpe)], ["最大回撤", fmtPct(m.max_drawdown)],
      ["年化波动", fmtPct(m.annual_volatility)], ["超额年化", fmtPct(m.excess_annual_return)],
      ["Alpha", fmtPct(m.alpha)], ["Beta", fmtNum(m.beta)],
      ["信息比率", fmtNum(m.information_ratio)], ["索提诺", fmtNum(m.sortino)],
      ["VaR 95%", fmtPct(m.var_95)], ["CVaR 95%", fmtPct(m.cvar_95)],
      ["年化换手", fmtNum(m.annual_turnover, 1) + "x"], ["月度胜率", fmtPct(m.monthly_win_rate, 1, false)],
    ];
    const factorText = (res.factor.factors || []).map((f) => `${f.label} ${(f.weight * 100).toFixed(0)}%`).join(" / ");
    const holdingsRows = (res.holdings || []).map((h) =>
      `<tr><td>${esc(h.date)}</td><td>${h.items.map((i) => `${esc(i.name)} ${(i.weight * 100).toFixed(1)}%`).join("，")}</td></tr>`).join("");
    const monthlyRows = (res.monthly || []).map((x) =>
      `<tr><td>${x.year}-${String(x.month).padStart(2, "0")}</td><td style="color:${x.return >= 0 ? "#e05666" : "#2fae7c"}">${(x.return * 100).toFixed(2)}%</td></tr>`).join("");
    const annualRows = (res.annual_returns || []).map((a) =>
      `<tr><td>${a.year}</td><td style="color:${a.return >= 0 ? "#e05666" : "#2fae7c"}">${(a.return * 100).toFixed(2)}%</td></tr>`).join("");
    const brinsonRows = res.brinson ? res.brinson.industries.map((r) =>
      `<tr><td>${esc(r.industry)}</td><td>${(r.allocation * 100).toFixed(2)}%</td><td>${(r.selection * 100).toFixed(2)}%</td><td>${(r.interaction * 100).toFixed(2)}%</td><td>${(r.total * 100).toFixed(2)}%</td></tr>`).join("") : "";
    const tradeRows = (res.trades || []).flatMap((t) => t.items.map((i) =>
      `<tr><td>${esc(t.date)}</td><td>${esc(i.side)}</td><td>${esc(i.name)} ${esc(i.code)}</td><td>${(i.change * 100).toFixed(2)}%</td></tr>`)).slice(0, 60).join("");
    const html = `<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>回测报告 · ${esc(cfg.name)}</title>
<style>
body{font-family:-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;background:#f7f9fc;color:#1f2a37;margin:0;padding:40px}
.wrap{max-width:960px;margin:0 auto;background:#fff;border:1px solid #e3e9f2;border-radius:12px;padding:32px 36px}
h1{font-size:22px;margin:0 0 4px}.sub{color:#6b7c93;font-size:12px;margin-bottom:22px}
.chips span{display:inline-block;border:1px solid #dbe4ef;border-radius:999px;padding:2px 9px;font-size:11px;color:#4b5b6f;margin:0 6px 6px 0}
.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:18px 0}
.card{border:1px solid #e3e9f2;border-radius:8px;padding:10px 12px}.card .k{font-size:10px;color:#7c8ba1}
.card .v{font-size:17px;font-family:ui-monospace,monospace;margin-top:2px}
img{width:100%;border:1px solid #e3e9f2;border-radius:8px;margin:8px 0 18px}
h2{font-size:14px;margin:24px 0 8px}table{width:100%;border-collapse:collapse;font-size:12px}
th,td{border-bottom:1px solid #e8eef6;padding:6px 8px;text-align:left}th{color:#7c8ba1;font-weight:500}
.foot{margin-top:26px;color:#93a1b3;font-size:11px}
</style></head><body><div class="wrap">
<h1>${esc(cfg.name)}</h1>
<div class="sub">${esc(res.universe.start)} → ${esc(res.universe.end)} · ${res.universe.stocks} 只股票 · 基准 ${esc(cfg.benchmark_name || (cfg.benchmark === "market" ? "全市场等权" : cfg.benchmark === "universe" ? "股票池等权" : (cfg.benchmark || "股票池等权")))} · 生成于 ${new Date().toLocaleString("zh-CN", { hour12: false })}</div>
<div class="chips">${(res.factor.factors || []).map((f) => `<span>${esc(f.label)} ${(f.weight * 100).toFixed(0)}%</span>`).join("")}
<span>${cfg.rebalance_days} 日调仓</span><span>持仓 ${cfg.top_n > 0 ? cfg.top_n : (cfg.top_pct * 100).toFixed(0) + "%"}</span>
<span>成本 买 ${cfg.buy_cost_bps || "—"} / 卖 ${cfg.sell_cost_bps || "—"} bps</span></div>
<div class="grid">${metricRows.map(([k, v]) => `<div class="card"><div class="k">${k}</div><div class="v">${v}</div></div>`).join("")}</div>
<h2>策略净值</h2>${eqImg ? `<img src="${eqImg}" alt="equity">` : ""}
<h2>回撤曲线</h2>${ddImg ? `<img src="${ddImg}" alt="drawdown">` : ""}
<h2>月度收益</h2><table><thead><tr><th>月份</th><th>收益</th></tr></thead><tbody>${monthlyRows}</tbody></table>
${annualRows ? `<h2>年度收益</h2><table><thead><tr><th>年度</th><th>收益</th></tr></thead><tbody>${annualRows}</tbody></table>` : ""}
${brinsonRows ? `<h2>Brinson 归因</h2><table><thead><tr><th>行业</th><th>配置</th><th>选股</th><th>交互</th><th>合计</th></tr></thead><tbody>${brinsonRows}</tbody></table>` : ""}
${tradeRows ? `<h2>交易明细</h2><table><thead><tr><th>日期</th><th>方向</th><th>股票</th><th>权重变化</th></tr></thead><tbody>${tradeRows}</tbody></table>` : ""}
<h2>调仓持仓</h2><table><thead><tr><th>日期</th><th>持仓</th></tr></thead><tbody>${holdingsRows}</tbody></table>
<div class="foot">AlphaLab 自动生成 · 因子：${esc(factorText)} · 结果仅供研究，不构成投资建议。</div>
</div></body></html>`;
    const blob = new Blob([html], { type: "text/html;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `report_${res.id || "latest"}.html`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    toast("回测报告已导出 HTML");
  }

  function exportBacktestCsv(res) {
    const rows = [["date", "nav", "benchmark", "drawdown", "daily_return"]];
    res.dates.forEach((date, i) => {
      rows.push([
        date,
        res.nav[i] == null ? "" : res.nav[i],
        res.benchmark[i] == null ? "" : res.benchmark[i],
        res.drawdown[i] == null ? "" : res.drawdown[i],
        i === 0 || res.daily_returns[i - 1] == null ? "" : res.daily_returns[i - 1],
      ]);
    });
    const csv = "\ufeff" + rows.map((row) => row.join(",")).join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `backtest_${res.id || "latest"}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    toast("净值数据已导出 CSV");
  }

  function renderHeatmap(monthly) {
    const years = [...new Set(monthly.map((m) => m.year))].sort();
    const byKey = {};
    monthly.forEach((m) => { byKey[m.year + "-" + m.month] = m.return; });
    let html = `<div class="hm-label"></div>` + Array.from({ length: 12 }, (_, i) => `<div class="hm-head">${i + 1}月</div>`).join("") + `<div class="hm-head">年度</div>`;
    years.forEach((y) => {
      html += `<div class="hm-label">${y}</div>`;
      let comp = 1;
      for (let mo = 1; mo <= 12; mo++) {
        const v = byKey[y + "-" + mo];
        if (v === undefined) { html += `<div class="hm-cell"></div>`; continue; }
        comp *= (1 + v);
        const alpha = Math.min(0.62, 0.08 + Math.abs(v) * 3.4);
        const bg = v >= 0 ? `rgba(255,107,122,${alpha})` : `rgba(61,220,151,${alpha})`;
        html += `<div class="hm-cell" style="background:${bg}" title="${y}-${mo}: ${(v * 100).toFixed(2)}%">${(v * 100).toFixed(1)}</div>`;
      }
      const ya = comp - 1;
      const alpha = Math.min(0.62, 0.08 + Math.abs(ya) * 1.6);
      const bg = ya >= 0 ? `rgba(255,107,122,${alpha})` : `rgba(61,220,151,${alpha})`;
      html += `<div class="hm-cell" style="background:${bg};font-weight:600">${(ya * 100).toFixed(1)}</div>`;
    });
    const hm = $("#heatmap");
    hm.style.gridTemplateColumns = "46px repeat(13, 1fr)";
    hm.innerHTML = html;
  }

  function renderHoldings(holdings) {
    const box = $("#holdingsGrid");
    if (!holdings.length) { box.innerHTML = `<div class="panel-sub">无持仓记录</div>`; return; }
    box.innerHTML = holdings.slice().reverse().map((h) => {
      const total = h.items.reduce((s, i) => s + Math.abs(i.weight), 0) || 1;
      return `<div class="holding-card">
        <div class="hd"><span>${esc(h.date)}</span><span>${h.items.length} 只</span></div>
        ${h.items.slice(0, 6).map((i) => `
          <div class="hr"><span>${esc(i.name)} <span style="color:var(--faint);font-size:10px">${esc(i.code)}</span></span>
          <span class="${i.weight < 0 ? "w neg" : ""}">${(i.weight * 100).toFixed(1)}%</span></div>`).join("")}
      </div>`;
    }).join("");
  }

  /* ------------------------------ 参数扫描 ------------------------------ */
  async function runWalkForward() {
    if (!state.btFactors.length) { toast("请至少添加一个因子", true); return; }
    openModal(`
      <div class="modal-head">
        <div><h3>滚动验证（Walk-Forward）</h3><div class="panel-sub" style="margin-top:3px">每折用样本内最优参数，检验下一段样本外表现</div></div>
        <button class="icon-btn" data-close>✕</button>
      </div>
      <div class="empty" style="border:none;padding:50px 20px">
        <span class="spinner" style="width:22px;height:22px;margin:0 auto 12px;display:block"></span>
        <p>正在逐折选择参数并回测…</p>
        <div class="sub">4 折 · 参数网格 3 × 3</div>
      </div>`, true);
    try {
      const res = await api("/api/backtest/walk-forward", {
        method: "POST", body: { ...buildCommonConfig(), wf_folds: 4 },
      });
      renderWalkForward(res);
    } catch (e) {
      closeModal();
      toast(e.message, true);
    }
  }

  function renderWalkForward(res) {
    const m = res.metrics;
    openModal(`
      <div class="modal-head">
        <div><h3>滚动验证结果</h3>
          <div class="panel-sub" style="margin-top:3px">${esc(res.factor.label)} · ${res.config.stocks} 只股票 · ${res.config.start} → ${res.config.end} · ${res.config.folds} 折</div>
        </div>
        <button class="icon-btn" data-close>✕</button>
      </div>
      <div class="grid-3">
        ${[["样本外年化", fmtPct(m.annual_return)], ["夏普", fmtNum(m.sharpe)], ["最大回撤", fmtPct(m.max_drawdown)], ["累计收益", fmtPct(m.total_return)]].map(([k, v]) => `<div class="perf"><div class="k">${k}</div><div class="v">${v}</div></div>`).join("")}
      </div>
      <div class="chart-wrap" style="margin-top:12px"><canvas id="wfChart"></canvas></div>
      <div class="table-wrap" style="margin-top:12px;max-height:260px">
        <table><thead><tr><th>折</th><th>样本内</th><th>样本外</th><th>最优持仓</th><th>调仓</th><th style="text-align:right">样本内夏普</th><th style="text-align:right">样本外收益</th><th style="text-align:right">样本外夏普</th></tr></thead>
        <tbody>${res.folds.map((f) => `<tr style="cursor:default">
          <td class="num-mono">${f.fold}</td>
          <td class="num-mono" style="font-size:11px">${esc(f.is_start)} → ${esc(f.is_end)}</td>
          <td class="num-mono" style="font-size:11px">${esc(f.oos_start)} → ${esc(f.oos_end)}</td>
          <td class="num-mono">${f.top_n}</td>
          <td class="num-mono">${f.rebalance_days} 日</td>
          <td style="text-align:right" class="num-mono">${fmtNum(f.is_sharpe)}</td>
          <td style="text-align:right" class="num-mono ${colorClass(f.oos_return)}">${fmtPct(f.oos_return)}</td>
          <td style="text-align:right" class="num-mono">${fmtNum(f.oos_sharpe)}</td>
        </tr>`).join("")}</tbody></table>
      </div>
      <div class="modal-foot" style="justify-content:space-between;align-items:center">
        <span class="panel-sub">结果不写入历史记录</span>
        <button class="btn ghost" data-close>关闭</button>
      </div>`, true);
    Charts.line($("#wfChart"), {
      labels: res.dates,
      series: [{ name: "滚动样本外净值", data: res.nav.map((v) => (v - 1) * 100), color: "#3ddc97", fill: true }],
      height: 270, yFmt: (v) => v.toFixed(0) + "%", xFmt: shortDate,
    });
  }

  async function runParamScan() {
    if (!state.btFactors.length) { toast("请至少添加一个因子", true); return; }
    openModal(`
      <div class="modal-head">
        <div><h3>参数扫描</h3><div class="panel-sub" style="margin-top:3px">调仓周期 × 持仓数量 的绩效网格</div></div>
        <button class="icon-btn" data-close>✕</button>
      </div>
      <div class="empty" style="border:none;padding:60px 20px">
        <span class="spinner" style="width:24px;height:24px;margin:0 auto 14px;display:block"></span>
        <p>正在扫描 9 组参数…</p>
        <div class="sub">复用因子计算结果，逐个运行 JAX 回测</div>
      </div>`);
    try {
      const res = await api("/api/backtest/optimize", {
        method: "POST", body: buildCommonConfig(),
      });
      renderOptimizeModal(res);
    } catch (e) {
      closeModal();
      toast(e.message, true);
    }
  }

  function optFmt(metric, v) {
    if (metric === "sharpe") return v == null ? "—" : Number(v).toFixed(2);
    return fmtPct(v);
  }

  function renderOptimizeModal(res) {
    const best = res.best;
    openModal(`
      <div class="modal-head">
        <div>
          <h3>参数扫描结果</h3>
          <div class="panel-sub" style="margin-top:3px">${res.factor.label} · ${res.base.stocks} 只股票 · ${res.base.start} → ${res.base.end}</div>
        </div>
        <button class="icon-btn" data-close>✕</button>
      </div>
      <div class="opt-controls">
        <div class="panel-sub">最优组合：${best.rebalance_days} 日调仓 / 持仓 ${best.top_n} · 夏普 ${fmtNum(best.sharpe)}</div>
        <select id="optMetric">
          <option value="sharpe">按夏普比率</option>
          <option value="annual_return">按年化收益</option>
          <option value="max_drawdown">按最大回撤</option>
          <option value="annual_turnover">按年化换手</option>
        </select>
      </div>
      <div id="optGrid"></div>
      <div class="modal-foot" style="justify-content:space-between;align-items:center">
        <span class="panel-sub">参数扫描结果不写入历史记录</span>
        <button class="btn ghost" data-close>关闭</button>
      </div>`, true);

    const drawGrid = () => {
      const metric = $("#optMetric").value;
      const map = {};
      res.cells.forEach((c) => { map[c.rebalance_days + "-" + c.top_n] = c; });
      const values = res.cells.map((c) => c[metric]).filter((v) => v != null);
      const min = Math.min(...values);
      const max = Math.max(...values);
      const cellColor = (v) => {
        const t = max > min ? (v - min) / (max - min) : 0.5;
        const alpha = 0.07 + t * 0.5;
        return v >= 0 ? `rgba(255,107,122,${alpha})` : `rgba(61,220,151,${alpha})`;
      };
      let html = `<div class="opt-grid" style="grid-template-columns:56px repeat(${res.top_n.length}, 1fr)">`;
      html += `<div></div>` + res.top_n.map((t) => `<div class="og-head">持仓 ${t}</div>`).join("");
      res.rebalance_days.forEach((freq) => {
        html += `<div class="og-label">${freq} 日</div>`;
        res.top_n.forEach((topn) => {
          const c = map[freq + "-" + topn];
          const v = c ? c[metric] : null;
          const isBest = best.rebalance_days === freq && best.top_n === topn;
          const sub = metric === "sharpe"
            ? `年化 ${fmtPct(c.annual_return)}`
            : `夏普 ${fmtNum(c.sharpe)}`;
          html += `<div class="opt-cell ${isBest ? "best" : ""}" style="background:${v == null ? "transparent" : cellColor(v)}" title="调仓 ${freq} 日 / 持仓 ${topn}\n年化 ${fmtPct(c.annual_return)}  夏普 ${fmtNum(c.sharpe)}  回撤 ${fmtPct(c.max_drawdown)}">
            <div class="ov">${optFmt(metric, v)}</div>
            <div class="os">${sub}</div>
          </div>`;
        });
      });
      html += "</div>";
      $("#optGrid").innerHTML = html;
    };
    $("#optMetric").addEventListener("change", drawGrid);
    drawGrid();
  }

  /* ------------------------------ 因子相关性 ------------------------------ */
  function openCorrelationModal() {
    const items = state.factors;
    const defaults = new Set(items.slice(0, 5).map((f) => String(f.id)));
    openModal(`
      <div class="modal-head">
        <div><h3>因子相关性矩阵</h3><div class="panel-sub" style="margin-top:3px">平均截面秩相关，识别多因子组合中的共线性</div></div>
        <button class="icon-btn" data-close>✕</button>
      </div>
      <div class="corr-picker">
        ${items.map((f) => `<label><input type="checkbox" value="${esc(f.id)}" ${defaults.has(String(f.id)) ? "checked" : ""}>
          ${esc(f.label || f.name)} <span class="chip" style="margin-left:auto">${esc(f.is_custom ? "自定义" : (f.category || "内置"))}</span></label>`).join("")}
      </div>
      <div id="corrResult"></div>
      <div class="modal-foot">
        <button class="btn ghost" data-close>取消</button>
        <button class="btn primary" id="corrRun">计算矩阵</button>
      </div>`, true);

    $("#corrRun").addEventListener("click", async () => {
      const ids = $$("#modalBox .corr-picker input:checked").map((cb) => cb.value);
      if (ids.length < 2) { toast("请至少选择两个因子", true); return; }
      if (ids.length > 8) { toast("最多选择 8 个因子", true); return; }
      const btn = $("#corrRun");
      btn.disabled = true;
      btn.innerHTML = `<span class="spinner"></span> 计算中`;
      try {
        const res = await api("/api/factors/correlation", {
          method: "POST",
          body: {
            factor_ids: ids,
            start: state.status && state.status.start_date,
            end: state.status && state.status.end_date,
          },
        });
        renderCorrMatrix(res);
      } catch (e) {
        toast(e.message, true);
      } finally {
        btn.disabled = false;
        btn.textContent = "计算矩阵";
      }
    });
  }

  function renderCorrMatrix(res) {
    const labels = res.labels;
    const n = labels.length;
    let html = `<div class="corr-grid" style="grid-template-columns:110px repeat(${n}, 1fr)">`;
    html += `<div></div>` + labels.map((l) => `<div class="cg-head" title="${esc(l)}">${esc(l)}</div>`).join("");
    res.matrix.forEach((row, i) => {
      html += `<div class="cg-row-head" title="${esc(labels[i])}">${esc(labels[i])}</div>`;
      row.forEach((v, j) => {
        if (v == null) {
          html += `<div class="corr-cell" style="color:var(--faint)">—</div>`;
          return;
        }
        const alpha = 0.06 + Math.abs(v) * 0.5;
        const bg = i === j ? "rgba(61,220,151,.32)"
          : v >= 0 ? `rgba(106,169,255,${alpha})` : `rgba(245,181,107,${alpha})`;
        html += `<div class="corr-cell" style="background:${bg}" title="${esc(labels[i])} vs ${esc(labels[j])}: ${v}">${v.toFixed(2)}</div>`;
      });
    });
    html += `</div><div class="panel-sub" style="margin-top:10px">蓝色为正相关，橙色为负相关；|ρ| &gt; 0.7 的因子组合存在较强共线性。</div>`;
    $("#corrResult").innerHTML = html;
  }


  /* ------------------------------ 智能选股 ------------------------------ */
  const SCREEN_MODES = { rank_top: "排名前", rank_bottom: "排名后", gte: "因子值 ≥", lte: "因子值 ≤" };

  function initScreener() {
    if (!state.screenerConds.length) {
      const first = state.factors[0];
      state.screenerConds = [{ factorId: first ? first.id : "", mode: "rank_top", value: 20, weight: 1 }];
    }
    renderScreenerConds();
  }

  function renderScreenerConds() {
    const box = $("#scrConds");
    if (!box) return;
    const opts = (sel) => state.factors.map((f) => {
      const label = f.label || f.name || f.id;
      return `<option value="${esc(f.id)}" ${f.id === sel ? "selected" : ""}>${esc(label)}</option>`;
    }).join("");
    box.innerHTML = state.screenerConds.map((c, i) => `
      <div class="form-row" style="gap:8px;align-items:center;border:1px solid var(--border-soft);background:#0a121c;border-radius:8px;padding:8px 10px">
        <span class="hint" style="min-width:38px">条件 ${i + 1}</span>
        <select class="scr-factor" data-i="${i}" style="min-width:190px">${opts(c.factorId)}</select>
        <select class="scr-mode" data-i="${i}" style="width:112px">${Object.entries(SCREEN_MODES).map(([v, t]) => `<option value="${v}" ${c.mode === v ? "selected" : ""}>${t}</option>`).join("")}</select>
        <select class="scr-scope" data-i="${i}" style="width:100px">
          <option value="all" ${(c.scope || "all") === "all" ? "selected" : ""}>全市场</option>
          <option value="industry" ${c.scope === "industry" ? "selected" : ""}>行业内</option>
        </select>
        <select class="scr-group" data-i="${i}" style="width:72px" title="条件分组：组内为「且」，组间为「或」">
          <option value="0" ${!c.group ? "selected" : ""}>组 1</option>
          <option value="1" ${c.group === 1 ? "selected" : ""}>组 2</option>
          <option value="2" ${c.group === 2 ? "selected" : ""}>组 3</option>
        </select>
        <input type="number" class="scr-value" data-i="${i}" value="${c.value}" step="1" style="width:92px">
        <span class="hint" data-unit="${i}">${c.mode.startsWith("rank") ? "%" : "值"}</span>
        <input type="number" class="scr-weight" data-i="${i}" value="${c.weight == null ? 1 : c.weight}" step="0.5" min="-5" max="5" title="综合分权重（仅影响排序）" style="width:74px">
        <span class="hint">权重</span>
        <button class="icon-btn" data-del="${i}" title="删除条件">✕</button>
      </div>`).join("");

    const sortSel = $("#scrSort");
    if (sortSel) {
      const cur = sortSel.value || "score";
      sortSel.innerHTML = `<option value="score">综合分</option>` + state.screenerConds.map((c, i) => {
        const f = findFactorById(c.factorId);
        return `<option value="c${i}">${esc(f ? (f.label || f.name) : c.factorId)}</option>`;
      }).join("");
      if (Array.from(sortSel.options).some((o) => o.value === cur)) sortSel.value = cur;
    }

    $$("#scrConds .scr-factor").forEach((el) => el.addEventListener("change", () => {
      state.screenerConds[+el.dataset.i].factorId = el.value;
      renderScreenerConds();
    }));
    $$("#scrConds .scr-mode").forEach((el) => el.addEventListener("change", () => {
      const i = +el.dataset.i;
      state.screenerConds[i].mode = el.value;
      const unit = document.querySelector(`#scrConds [data-unit="${i}"]`);
      if (unit) unit.textContent = el.value.startsWith("rank") ? "%" : "值";
    }));
    $$("#scrConds .scr-scope").forEach((el) => el.addEventListener("change", () => {
      state.screenerConds[+el.dataset.i].scope = el.value;
    }));
    $$("#scrConds .scr-group").forEach((el) => el.addEventListener("change", () => {
      state.screenerConds[+el.dataset.i].group = Number(el.value) || 0;
    }));
    $$("#scrConds .scr-value").forEach((el) => el.addEventListener("input", () => {
      state.screenerConds[+el.dataset.i].value = Number(el.value) || 0;
    }));
    $$("#scrConds .scr-weight").forEach((el) => el.addEventListener("input", () => {
      state.screenerConds[+el.dataset.i].weight = Number(el.value) || 0;
    }));
    $$("#scrConds [data-del]").forEach((el) => el.addEventListener("click", () => {
      if (state.screenerConds.length <= 1) { toast("至少保留一个条件", true); return; }
      state.screenerConds.splice(+el.dataset.del, 1);
      renderScreenerConds();
    }));
  }

  async function runScreener() {
    const btn = $("#scrRun");
    btn.disabled = true;
    btn.innerHTML = `<span class="spinner"></span> 选股中`;
    try {
      const payload = {
        date: $("#scrDate").value || null,
        limit: Number($("#scrLimit").value) || 50,
        filters: {
          exclude_star: $("#scrStar").checked,
          exclude_bj: $("#scrBJ").checked,
          min_amount: (Number($("#scrAmount").value) || 0) * 10000,
          industries: Array.from($("#scrIndustries").selectedOptions).map((o) => o.value),
        },
        conditions: state.screenerConds.map((c) => ({
          factor_id: c.factorId, mode: c.mode, value: c.value,
          scope: c.scope || "all", weight: c.weight == null ? 1 : c.weight,
          group: c.group || 0,
        })),
        logic: $("#scrLogic").value,
        select_mode: $("#scrMode").value || "conditions",
        sort_by: $("#scrSort").value || "score",
        confirm_days: Number($("#scrConfirmDays").value) || 0,
        confirm_min: Number($("#scrConfirmMin").value) || 0,
        industry_limit: Number($("#scrIndustryLimit").value) || 0,
      };
      const res = await api("/api/screener", { method: "POST", body: payload });
      state.screenerResult = res;
      renderScreenerResult(res);
      toast(`选股完成，命中 ${res.count} 只`);
    } catch (e) {
      toast(e.message, true);
    } finally {
      btn.disabled = false;
      btn.textContent = "开始选股";
    }
  }

  function openScreenerHistory() {
    openModal(`
      <div class="modal-head">
        <div><h3>选股规则历史检验</h3>
          <div class="panel-sub" style="margin-top:3px">每月末按当前选股条件建仓，统计未来 N 日等权收益与相对全市场超额</div></div>
        <button class="icon-btn" data-close>✕</button>
      </div>
      <div class="form-row" style="gap:14px;align-items:flex-end">
        <div class="field"><label>持有期</label>
          <select id="scrHistHorizon">
            <option value="5">5 个交易日</option>
            <option value="10">10 个交易日</option>
            <option value="20" selected>20 个交易日</option>
            <option value="60">60 个交易日</option>
          </select>
        </div>
        <div class="field"><label>回看月数</label>
          <select id="scrHistMonths">
            <option value="12">12 个月</option>
            <option value="24" selected>24 个月</option>
            <option value="36">36 个月</option>
          </select>
        </div>
        <button class="btn primary" id="scrHistRun">开始检验</button>
      </div>
      <div id="scrHistBody" style="margin-top:14px">
        <div class="hint-box">将使用智能选股页当前的选股条件、行业过滤与返回数量设置，逐月回溯计算。</div>
      </div>`, true);
    $("#scrHistRun").addEventListener("click", runScreenerHistory);
  }

  async function runScreenerHistory() {
    const btn = $("#scrHistRun");
    const box = $("#scrHistBody");
    btn.disabled = true;
    btn.innerHTML = `<span class="spinner"></span> 计算中`;
    box.innerHTML = `<div class="empty" style="border:none"><p>正在逐月回溯选股条件…</p></div>`;
    try {
      const payload = {
        limit: Number($("#scrLimit").value) || 50,
        filters: {
          exclude_star: $("#scrStar").checked,
          exclude_bj: $("#scrBJ").checked,
          min_amount: (Number($("#scrAmount").value) || 0) * 10000,
          industries: Array.from($("#scrIndustries").selectedOptions).map((o) => o.value),
        },
        conditions: state.screenerConds.map((c) => ({
          factor_id: c.factorId, mode: c.mode, value: c.value,
          scope: c.scope || "all", weight: c.weight == null ? 1 : c.weight,
          group: c.group || 0,
        })),
        logic: $("#scrLogic").value,
        select_mode: $("#scrMode").value || "conditions",
        horizon: Number($("#scrHistHorizon").value) || 20,
        months: Number($("#scrHistMonths").value) || 24,
      };
      const res = await api("/api/screener/history", { method: "POST", body: payload });
      renderScreenerHistory(res);
    } catch (e) {
      box.innerHTML = `<div class="hint-box"><span class="down">${esc(e.message)}</span></div>`;
    } finally {
      btn.disabled = false;
      btn.textContent = "开始检验";
    }
  }

  function renderScreenerHistory(res) {
    const s = res.summary;
    const rows = res.rows || [];
    const cls = (v) => (v > 0 ? "pos" : v < 0 ? "neg" : "");
    const box = $("#scrHistBody");
    box.innerHTML = `
      <div class="metric-cards">
        <div class="mcard"><div class="k">平均收益</div><div class="v ${cls(s.avg_ret)}">${fmtPct(s.avg_ret, 1)}</div><div class="s">每期未来 ${s.horizon} 日等权</div></div>
        <div class="mcard"><div class="k">平均超额</div><div class="v ${cls(s.avg_excess)}">${fmtPct(s.avg_excess, 1)}</div><div class="s">相对全市场等权</div></div>
        <div class="mcard"><div class="k">跑赢概率</div><div class="v">${fmtPct(s.excess_win_rate, 0, false)}</div><div class="s">${s.periods} 期样本</div></div>
        <div class="mcard"><div class="k">平均入选</div><div class="v">${fmtNum(s.avg_count, 0)}</div><div class="s">只 / 期</div></div>
      </div>
      <div class="factor-meta" style="margin-top:10px">
        <span class="chip">累计收益 ${fmtPct(s.cum_ret, 1)}</span>
        <span class="chip">基准累计 ${fmtPct(s.cum_bench, 1)}</span>
        <span class="chip ${s.best_excess >= 0 ? "green" : "amber"}">单期最佳 ${fmtPct(s.best_excess, 1)}</span>
        <span class="chip ${s.worst_excess <= 0 ? "amber" : "green"}">单期最差 ${fmtPct(s.worst_excess, 1)}</span>
      </div>
      <div class="chart-wrap" style="margin-top:12px"><canvas id="scrHistChart"></canvas></div>
      <div class="table-wrap" style="margin-top:12px;max-height:250px;overflow:auto">
        <table>
          <thead><tr><th>日期</th><th style="text-align:right">入选数</th><th style="text-align:right">组合收益</th>
            <th style="text-align:right">全市场</th><th style="text-align:right">超额</th></tr></thead>
          <tbody>${rows.map((r) => `<tr>
            <td class="num-mono">${esc(r.date)}</td>
            <td style="text-align:right" class="num-mono">${r.count}</td>
            <td style="text-align:right" class="num-mono ${cls(r.ret)}">${r.ret == null ? "—" : fmtPct(r.ret, 1)}</td>
            <td style="text-align:right" class="num-mono">${r.bench == null ? "—" : fmtPct(r.bench, 1)}</td>
            <td style="text-align:right" class="num-mono ${cls(r.excess)}">${r.excess == null ? "—" : fmtPct(r.excess, 1)}</td>
          </tr>`).join("")}</tbody>
        </table>
      </div>`;
    Charts.line($("#scrHistChart"), {
      labels: rows.map((r) => r.date),
      height: 250,
      series: [
        { name: "选股组合", data: rows.map((r) => r.nav), color: "#3ddc97", fill: true },
        { name: "全市场等权", data: rows.map((r) => r.bench_nav), color: "#5b8def" },
      ],
      yFmt: (v) => v.toFixed(2),
    });
  }

  function renderScreenerResult(res) {
    const labels = res.labels || [];
    const indCount = {};
    (res.items || []).forEach((it) => {
      indCount[it.industry] = (indCount[it.industry] || 0) + 1;
    });
    const indTop = Object.entries(indCount).sort((a, b) => b[1] - a[1]);
    const indText = indTop.slice(0, 4).map(([k, v]) => `${k} ${v}`).join(" · ") +
      (indTop.length > 4 ? ` · 其他 ${indTop.length - 4} 个行业` : "");
    const scoreMode = res.select_mode === "score";
    const groups = new Set((res.conditions || []).map((c) => c.group || 0));
    const groupText = !scoreMode && groups.size > 1 ? ` · ${groups.size} 组（组内且 / 组间或）` : "";
    $("#scrMeta").textContent = `${res.as_of} · 股票池 ${res.universe_size} 只 · ` +
      `${scoreMode ? "候选" : "命中"} ${res.count} 只 · 显示 ${res.returned} 只 · ` +
      (scoreMode ? "综合分排序" : `逻辑${res.logic === "or" ? "或" : "且"}${groupText}`) +
      (res.confirm ? ` · 确认 ${res.confirm.min}/${res.confirm.days} 日` : "") +
      (res.industry_limit > 0 ? ` · 每行业≤${res.industry_limit} 只` : "") +
      (indText ? ` · ${indText}` : "");
    const extra = res.confirm ? 1 : 0;
    $("#scrHead").innerHTML = `<tr><th>代码</th><th>名称</th><th>行业</th>` +
      labels.map((l) => `<th style="text-align:right">${esc(l)}</th>`).join("") +
      `<th style="text-align:right">综合分</th><th style="text-align:right">最新价</th><th style="text-align:right">涨跌幅</th>` +
      (extra ? `<th style="text-align:right">确认</th>` : "") + `</tr>`;
    const body = $("#scrBody");
    if (!res.items.length) {
      body.innerHTML = `<tr><td colspan="${labels.length + 6 + extra}" style="text-align:center;color:var(--faint);padding:30px">没有股票满足全部条件</td></tr>`;
      return;
    }
    body.innerHTML = res.items.map((it) => `<tr style="cursor:default">
      <td><span class="cd">${esc(it.code)}</span></td>
      <td>${esc(it.name)}</td>
      <td>${esc(it.industry)}</td>
      ${it.values.map((v) => `<td style="text-align:right" class="num-mono">${v == null ? "—" : fmtNum(v, 3)}</td>`).join("")}
      <td style="text-align:right" class="num-mono">${fmtNum(it.score, 2)}</td>
      <td style="text-align:right" class="num-mono">${it.close == null ? "—" : it.close.toFixed(2)}</td>
      <td style="text-align:right" class="num-mono ${it.change_pct > 0 ? "up" : it.change_pct < 0 ? "down" : ""}">${it.change_pct == null ? "—" : it.change_pct.toFixed(2) + "%"}</td>
      ${extra ? `<td style="text-align:right" class="num-mono">${it.hit_days == null ? "—" : it.hit_days + "/" + res.confirm.days}</td>` : ""}
    </tr>`).join("");
  }

  function exportScreenerCsv() {
    const res = state.screenerResult;
    if (!res || !res.items.length) { toast("暂无可导出的结果", true); return; }
    const rows = [["code", "name", "industry", ...(res.labels || []), "score", "close", "change_pct"]];
    res.items.forEach((it) => rows.push([it.code, it.name, it.industry, ...it.values, it.score, it.close, it.change_pct]));
    downloadCsv(rows, `screen_${res.as_of || "latest"}.csv`);
    toast("选股结果已导出 CSV");
  }

  async function saveScreenerUniverse() {
    const res = state.screenerResult;
    if (!res || !res.items.length) { toast("暂无可保存的选股结果", true); return; }
    const name = window.prompt("自选池名称", `选股 ${res.as_of}`);
    if (!name) return;
    try {
      await api("/api/universes", { method: "POST", body: { name: name.trim(), codes: res.items.map((i) => i.code) } });
      toast("已保存为自选池");
      loadIndices();
    } catch (e) { toast(e.message, true); }
  }

  const SCHEME_KEY = "alphalab_screen_schemes";

  function loadScreenSchemes() {
    try {
      return JSON.parse(localStorage.getItem(SCHEME_KEY) || "[]");
    } catch (e) {
      return [];
    }
  }

  function saveScreenScheme() {
    const name = window.prompt("选股方案名称", `方案 ${new Date().toLocaleDateString("zh-CN")}`);
    if (!name) return;
    const schemes = loadScreenSchemes().filter((s) => s.name !== name.trim());
    schemes.unshift({
      name: name.trim(),
      conditions: state.screenerConds.map((c) => ({ ...c })),
      filters: {
        star: $("#scrStar").checked,
        bj: $("#scrBJ").checked,
        amount: Number($("#scrAmount").value) || 0,
        industries: Array.from($("#scrIndustries").selectedOptions).map((o) => o.value),
      },
      limit: Number($("#scrLimit").value) || 50,
      confirmDays: Number($("#scrConfirmDays").value) || 0,
      confirmMin: Number($("#scrConfirmMin").value) || 0,
      industryLimit: Number($("#scrIndustryLimit").value) || 0,
      mode: $("#scrMode").value || "conditions",
      logic: $("#scrLogic").value,
      sortBy: $("#scrSort").value || "score",
    });
    localStorage.setItem(SCHEME_KEY, JSON.stringify(schemes.slice(0, 30)));
    toast("选股方案已保存");
  }

  function openLoadScreenScheme() {
    const schemes = loadScreenSchemes();
    if (!schemes.length) { toast("暂无保存的选股方案", true); return; }
    openModal(`
      <div class="modal-head">
        <div><h3>载入选股方案</h3><div class="panel-sub" style="margin-top:3px">保存于当前浏览器</div></div>
        <button class="icon-btn" data-close>✕</button>
      </div>
      <div style="display:flex;flex-direction:column;gap:6px;max-height:360px;overflow-y:auto">
        ${schemes.map((s, i) => `<button class="btn ghost" data-scheme="${i}" style="justify-content:flex-start;text-align:left">
          ${esc(s.name)} <span class="panel-sub" style="margin-left:8px">${s.conditions.length} 个条件</span>
        </button>`).join("")}
      </div>`, true);
    $$("#modalBox [data-scheme]").forEach((btn) => btn.addEventListener("click", () => {
      const s = schemes[+btn.dataset.scheme];
      state.screenerConds = (s.conditions || []).map((c) => ({ ...c }));
      if (!state.screenerConds.length) state.screenerConds = [{ factorId: state.factors[0]?.id || "", mode: "rank_top", value: 20, weight: 1 }];
      const f = s.filters || {};
      $("#scrStar").checked = f.star !== false;
      $("#scrBJ").checked = f.bj !== false;
      $("#scrAmount").value = f.amount || 0;
      const wanted = new Set(f.industries || []);
      $$("#scrIndustries option").forEach((o) => { o.selected = wanted.has(o.value); });
      $("#scrLimit").value = s.limit || 50;
      $("#scrConfirmDays").value = s.confirmDays || 0;
      $("#scrConfirmMin").value = s.confirmMin || 0;
      $("#scrIndustryLimit").value = s.industryLimit || 0;
      $("#scrMode").value = s.mode || "conditions";
      $("#scrLogic").value = s.logic || "and";
      renderScreenerConds();
      if (s.sortBy) $("#scrSort").value = s.sortBy;
      closeModal();
      toast(`已载入方案「${s.name}」`);
    }));
  }

  function runScreenerBacktest() {
    const res = state.screenerResult;
    if (!res || !res.items.length) { toast("请先运行选股", true); return; }
    if (res.items.length < 2) { toast("选股结果不足 2 只，无法回测", true); return; }
    state.universe = new Set(res.items.map((i) => i.code));
    renderUniverse();
    switchPage("backtest");
    toast(`已将 ${res.items.length} 只股票载入回测股票池`);
  }

  /* ------------------------------ 回测记录 ------------------------------ */
  async function loadHistory() {
    const grid = $("#historyGrid");
    try {
      const data = await api("/api/backtests?limit=60");
      const rawItems = data.items || [];
      const existing = new Set(rawItems.map((it) => String(it.id)));
      state.compare = new Set([...state.compare].filter((id) => existing.has(id)));
      const q = state.historyQuery.trim().toLowerCase();
      const items = q ? rawItems.filter((it) => (it.name || "").toLowerCase().includes(q)) : rawItems;
      if (!items.length) {
        grid.innerHTML = `<div class="empty" style="grid-column:1/-1;padding:60px">
          <p>还没有回测记录</p><div class="sub">前往「回测引擎」运行第一次回测</div></div>`;
        return;
      }
      grid.innerHTML = items.map((it) => {
        const m = it.metrics || {};
        const selected = state.compare.has(String(it.id));
        return `<div class="bt-card ${selected ? "selected" : ""}" data-id="${it.id}">
          <div class="top">
            <div>
              <div class="name">${esc(it.name)}</div>
              <div class="time">#${it.id} · ${esc(it.created_at)}</div>
            </div>
            <div style="display:flex;flex-direction:column;align-items:flex-end;gap:6px">
              <span class="chip ${m.annual_return >= 0 ? "green" : "amber"}">${fmtPct(m.annual_return)}</span>
              <div class="form-row" style="gap:6px;align-items:center">
                <label class="cmp-check" title="加入对比">
                  <input type="checkbox" data-cmp="${it.id}" ${selected ? "checked" : ""}> 对比
                </label>
                <button class="btn sm" data-del="${it.id}" style="padding:2px 8px;font-size:11px">删除</button>
              </div>
            </div>
          </div>
          <div class="row">
            <div class="mini"><div class="k">累计收益</div><div class="v ${colorClass(m.total_return)}">${fmtPct(m.total_return)}</div></div>
            <div class="mini"><div class="k">夏普</div><div class="v">${fmtNum(m.sharpe)}</div></div>
            <div class="mini"><div class="k">最大回撤</div><div class="v">${fmtPct(m.max_drawdown)}</div></div>
          </div>
        </div>`;
      }).join("");
      $$("#historyGrid .bt-card").forEach((card) => {
        card.addEventListener("click", (ev) => {
          if (ev.target.closest(".cmp-check") || ev.target.closest("[data-del]")) return;
          openHistoryDetail(card.dataset.id);
        });
      });
      $$("#historyGrid [data-del]").forEach((btn) => {
        btn.addEventListener("click", async (ev) => {
          ev.stopPropagation();
          const id = btn.dataset.del;
          if (!window.confirm(`确定删除回测记录 #${id} 吗？此操作不可恢复。`)) return;
          try {
            await api(`/api/backtests/${id}`, { method: "DELETE" });
            state.compare.delete(id);
            toast(`已删除回测 #${id}`);
            loadHistory();
          } catch (e) {
            toast(e.message, true);
          }
        });
      });
      $$("#historyGrid .cmp-check input").forEach((cb) => {
        cb.addEventListener("change", () => {
          const id = cb.dataset.cmp;
          if (cb.checked) {
            if (state.compare.size >= 3) {
              cb.checked = false;
              toast("最多同时对比 3 个回测", true);
              return;
            }
            state.compare.add(id);
          } else {
            state.compare.delete(id);
          }
          cb.closest(".bt-card").classList.toggle("selected", cb.checked);
          updateCompareBar();
        });
      });
      updateCompareBar();
    } catch (e) {
      grid.innerHTML = `<div class="empty" style="grid-column:1/-1"><p>${esc(e.message)}</p></div>`;
    }
  }

  function updateCompareBar() {
    const bar = $("#historyCompareBar");
    const n = state.compare.size;
    bar.style.display = n ? "flex" : "none";
    $("#compareHint").textContent = `已选择 ${n} 个回测（最多 3 个）`;
    $("#compareBtn").disabled = n < 2;
    $("#compareBtn").style.opacity = n < 2 ? 0.5 : 1;
  }

  async function openCompare() {
    const ids = Array.from(state.compare);
    if (ids.length < 2) { toast("请至少选择两个回测", true); return; }
    openModal(`<div class="modal-head"><div><h3>回测对比</h3></div><button class="icon-btn" data-close>✕</button></div>
      <div class="empty" style="border:none;padding:50px"><span class="spinner" style="width:22px;height:22px;margin:0 auto 12px;display:block"></span><p>加载回测结果…</p></div>`, true);
    try {
      const details = await Promise.all(ids.map((id) => api(`/api/backtests/${id}`)));
      const colors = ["#3ddc97", "#6aa9ff", "#f5b56b"];
      const base = details[0].result;
      const labels = base.dates;
      const series = details.map((d, i) => {
        const navMap = {};
        d.result.dates.forEach((date, k) => { navMap[date] = d.result.nav[k]; });
        return {
          name: `#${d.id} ${d.name}`,
          color: colors[i % colors.length],
          data: labels.map((date) => navMap[date] != null ? (navMap[date] - 1) * 100 : null),
        };
      });
      openModal(`
        <div class="modal-head">
          <div><h3>回测对比</h3>
            <div class="panel-sub" style="margin-top:3px">${details.length} 条净值曲线 · 起点归一为 0%</div>
          </div>
          <button class="icon-btn" data-close>✕</button>
        </div>
        <div class="chart-wrap"><canvas id="compareChart"></canvas></div>
        <div class="table-wrap" style="margin-top:14px;max-height:220px">
          <table><thead><tr><th>回测</th><th>年化</th><th>夏普</th><th>最大回撤</th><th>换手</th></tr></thead>
          <tbody>${details.map((d) => `<tr style="cursor:default"><td>${esc(d.name)} <span class="cd">#${d.id}</span></td>
            <td class="num-mono ${colorClass(d.metrics.annual_return)}">${fmtPct(d.metrics.annual_return)}</td>
            <td class="num-mono">${fmtNum(d.metrics.sharpe)}</td>
            <td class="num-mono">${fmtPct(d.metrics.max_drawdown)}</td>
            <td class="num-mono">${fmtNum(d.metrics.annual_turnover, 1)}x</td></tr>`).join("")}</tbody></table>
        </div>`, true);
      Charts.line($("#compareChart"), {
        labels, series, height: 320,
        yFmt: (v) => v.toFixed(0) + "%", xFmt: shortDate,
      });
    } catch (e) {
      closeModal();
      toast(e.message, true);
    }
  }

  async function openStrategyCombine() {
    try {
      const data = await api("/api/backtests?limit=40");
      const items = data.items || [];
      if (items.length < 2) { toast("至少需要两条回测记录", true); return; }
      openModal(`
        <div class="modal-head">
          <div><h3>多策略组合</h3><div class="panel-sub" style="margin-top:3px">按权重合成已保存回测的收益曲线（日频再平衡）</div></div>
          <button class="icon-btn" data-close>✕</button>
        </div>
        <div style="display:flex;align-items:center;gap:10px;margin:4px 0 10px">
          <label style="font-size:12px;color:var(--muted)">权重方案</label>
          <select id="combineOptimizer" style="padding:5px 8px;font-size:12px">
            <option value="manual">手动权重</option>
            <option value="equal">等权</option>
            <option value="min_var">最小方差</option>
            <option value="max_sharpe">最大夏普</option>
            <option value="risk_parity">风险平价</option>
          </select>
          <span class="panel-sub" style="margin:0">自动方案将忽略手动权重</span>
        </div>
        <div style="display:flex;flex-direction:column;gap:5px;max-height:320px;overflow-y:auto">
          ${items.map((it, i) => `<label style="display:grid;grid-template-columns:auto 1fr 72px;gap:10px;align-items:center;border:1px solid var(--border-soft);background:#0a121c;border-radius:8px;padding:7px 10px">
            <input type="checkbox" data-id="${it.id}" ${i < 2 ? "checked" : ""} style="accent-color:var(--green)">
            <span style="font-size:12.5px">${esc(it.name)} <span class="cd">#${it.id}</span></span>
            <input type="number" class="combine-weight" data-wid="${it.id}" value="1" step="0.5" min="0" style="padding:3px 6px;font-size:11px;text-align:center">
          </label>`).join("")}
        </div>
        <div class="modal-foot">
          <button class="btn ghost" data-close>取消</button>
          <button class="btn primary" id="combineRun">计算组合</button>
        </div>
        <div id="combineResult"></div>`, true);

      $("#combineRun").addEventListener("click", async () => {
        const picked = $$("#modalBox input[type=checkbox]:checked");
        if (picked.length < 2) { toast("请至少选择两个回测", true); return; }
        if (picked.length > 6) { toast("最多组合 6 个策略", true); return; }
        const payload = {
          items: picked.map((cb) => {
            const wid = cb.dataset.id;
            const inp = $(`#modalBox .combine-weight[data-wid="${wid}"]`);
            return { id: Number(wid), weight: Number(inp && inp.value) || 1 };
          }),
          optimizer: $("#combineOptimizer").value,
        };
        const btn = $("#combineRun");
        btn.disabled = true;
        btn.innerHTML = `<span class="spinner"></span> 计算中`;
        try {
          const res = await api("/api/portfolio/combine", { method: "POST", body: payload });
          renderCombineResult(res);
          toast("策略组合计算完成");
        } catch (e) {
          toast(e.message, true);
        } finally {
          btn.disabled = false;
          btn.textContent = "计算组合";
        }
      });
    } catch (e) { toast(e.message, true); }
  }

  function renderCombineResult(res) {
    const m = res.metrics;
    const box = $("#combineResult");
    const OPT_LABELS = { manual: "手动权重", equal: "等权", min_var: "最小方差", max_sharpe: "最大夏普", risk_parity: "风险平价" };
    if (res.optimizer && res.optimizer !== "manual") {
      res.items.forEach((it) => {
        const inp = $(`#modalBox .combine-weight[data-wid="${it.id}"]`);
        if (inp) inp.value = (it.weight * 100).toFixed(1);
      });
    }
    box.innerHTML = `
      <div class="divider"></div>
      <div class="panel-sub" style="margin:14px 0 8px">权重方案：${OPT_LABELS[res.optimizer] || "手动权重"} · 组合权重：${res.items.map((i) => `${esc(i.name)} ${(i.weight * 100).toFixed(0)}%`).join(" · ")}</div>
      <div class="grid-3">
        ${[["年化收益", fmtPct(m.annual_return)], ["累计收益", fmtPct(m.total_return)], ["夏普", fmtNum(m.sharpe)], ["最大回撤", fmtPct(m.max_drawdown)]].map(([k, v]) => `<div class="perf"><div class="k">${k}</div><div class="v">${v}</div></div>`).join("")}
      </div>
      <div class="chart-wrap" style="margin-top:12px"><canvas id="combineChart"></canvas></div>
      <div class="panel-sub" style="margin-top:10px">策略相关性</div>
      <div class="table-wrap" style="max-height:170px">
        <table><thead><tr><th></th>${res.correlation.labels.map((l) => `<th>${esc(l)}</th>`).join("")}</tr></thead>
          <tbody>${res.correlation.matrix.map((row, i) => `<tr style="cursor:default"><td>${esc(res.correlation.labels[i])}</td>${row.map((v) => `<td class="num-mono">${fmtNum(v, 2)}</td>`).join("")}</tr>`).join("")}</tbody></table>
      </div>`;
    const nav = res.nav.map((v) => (v - 1) * 100);
    Charts.line($("#combineChart"), {
      labels: res.dates, series: [{ name: "组合净值", data: nav, color: "#3ddc97", fill: true }],
      height: 240, yFmt: (v) => v.toFixed(0) + "%", xFmt: shortDate,
    });
  }

  async function openHistoryDetail(id) {
    try {
      const detail = await api(`/api/backtests/${id}`);
      switchPage("backtest");
      detail.result.id = detail.id;
      renderBacktestResult(detail.result);
      window.scrollTo(0, 0);
      toast(`已载入历史回测 #${id}`);
    } catch (e) { toast(e.message, true); }
  }

  /* ------------------------------ Modal ------------------------------ */
  function openModal(html, wide = false) {
    const box = $("#modalBox");
    box.className = "modal" + (wide ? " wide" : "");
    box.innerHTML = html;
    $("#modalRoot").classList.add("on");
    $$("[data-close]", box).forEach((b) => b.addEventListener("click", closeModal));
  }
  function closeModal() {
    $("#modalRoot").classList.remove("on");
    $("#modalBox").innerHTML = "";
  }

  /* ------------------------------ 初始化 ------------------------------ */
  function bindEvents() {
    $$(".nav-item").forEach((b) => b.addEventListener("click", () => switchPage(b.dataset.page)));
    $$("[data-goto]").forEach((b) => b.addEventListener("click", () => switchPage(b.dataset.goto)));

    $("#syncBtn").addEventListener("click", startSync);
    $("#qualityBtn").addEventListener("click", openQuality);
    $("#stockSearch").addEventListener("input", (e) => {
      state.stockFilter = e.target.value;
      renderStocks();
    });
    $("#factorSearch").addEventListener("input", (e) => {
      state.factorQuery = e.target.value;
      renderFactorList();
    });
    $("#historySearch").addEventListener("input", (e) => {
      state.historyQuery = e.target.value;
      loadHistory();
    });
    $("#newFactorBtn").addEventListener("click", () => openFactorModal());
    $("#corrBtn").addEventListener("click", openCorrelationModal);
    $("#factoryBtn").addEventListener("click", openFactory);
    $("#previewBtn").addEventListener("click", previewFactor);
    $("#addFactorBtn").addEventListener("click", addFactorToCombo);
    $("#icWeightBtn").addEventListener("click", autoWeightByIC);
    $("#saveComboFactorBtn").addEventListener("click", saveComboAsFactor);
    $$(".tpl-btn").forEach((btn) => btn.addEventListener("click", () => applyTemplate(btn.dataset.tpl)));
    $("#signalBtn").addEventListener("click", openSignal);
    $("#saveConfigBtn").addEventListener("click", saveCurrentConfig);
    $("#loadConfigBtn").addEventListener("click", openLoadConfig);
    $("#runBacktestBtn").addEventListener("click", runBacktest);
    $("#btRebalanceMode").addEventListener("change", updateRebalanceUI);
    $("#scrRun").addEventListener("click", runScreener);
    $("#scrHistory").addEventListener("click", openScreenerHistory);
    $("#scrAdd").addEventListener("click", () => {
      if (state.screenerConds.length >= 6) { toast("最多支持 6 个条件", true); return; }
      state.screenerConds.push({ factorId: state.factors[0]?.id || "", mode: "rank_top", value: 20, weight: 1 });
      renderScreenerConds();
    });
    $("#scrExport").addEventListener("click", exportScreenerCsv);
    $("#scrSave").addEventListener("click", saveScreenerUniverse);
    $("#scrSaveScheme").addEventListener("click", saveScreenScheme);
    $("#scrLoadScheme").addEventListener("click", openLoadScreenScheme);
    $("#scrBacktest").addEventListener("click", runScreenerBacktest);
    $("#batchBtn").addEventListener("click", openBatch);
    $("#costBtn").addEventListener("click", openCostSensitivity);
    $("#weightOptBtn").addEventListener("click", openWeightOpt);
    $("#wfBtn").addEventListener("click", runWalkForward);
    $("#optimizeBtn").addEventListener("click", runParamScan);
    $("#refreshHistory").addEventListener("click", loadHistory);
    $("#strategyCombineBtn").addEventListener("click", openStrategyCombine);
    $("#compareBtn").addEventListener("click", openCompare);
    $("#clearCompare").addEventListener("click", () => {
      state.compare.clear();
      loadHistory();
    });
    $("#universeAll").addEventListener("click", (e) => {
      e.preventDefault();
      state.stocks.forEach((s) => state.universe.add(s.code));
      renderUniverse();
    });
    $("#universeNone").addEventListener("click", (e) => {
      e.preventDefault();
      state.universe.clear();
      renderUniverse();
    });
    $("#saveUniverseBtn").addEventListener("click", (e) => {
      e.preventDefault();
      saveUniverse();
    });    $("#loadUniverseBtn").addEventListener("click", (e) => {
      e.preventDefault();
      openLoadUniverse();
    });

    $("#modalRoot").addEventListener("click", (e) => {
      if (e.target === $("#modalRoot")) closeModal();
    });
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") closeModal();
    });

    setInterval(() => {
      const d = new Date();
      $("#clock").textContent = d.toLocaleTimeString("zh-CN", { hour12: false });
    }, 1000);
  }

  async function init() {
    bindEvents();
    try {
      const s = await loadStatus(true);
      await Promise.all([loadStocks(), loadFactors(), loadIndices()]);
      renderUniverse();
      initScreener();
      if (s.stock_count === 0) waitForData();
      const first = state.factors.find((f) => !f.is_custom) || state.factors[0];
      if (first) selectFactor(first.id);
      if (s.stock_count > 0) loadHistory();
    } catch (e) {
      toast(e.message, true);
    }
  }

  document.addEventListener("DOMContentLoaded", init);
})();

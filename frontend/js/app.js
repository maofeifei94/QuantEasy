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
    btResult: null,
    syncTimer: null,
    stockFilter: "",
    polling: false,
  };

  const PAGE_META = {
    overview: ["Workspace / Overview", "研究总览"],
    data: ["Workspace / Market Data", "行情数据"],
    factors: ["Research / Factor Lab", "因子实验室"],
    backtest: ["Research / Backtest", "回测引擎"],
    history: ["Archive / Backtests", "回测记录"],
  };

  const CAT_ORDER = ["动量/反转", "波动率", "量价", "流动性", "技术指标", "自定义"];

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
        <div class="chart-wrap"><canvas id="candleChart"></canvas></div>`, true);
      Charts.candle($("#candleChart"), {
        dates: data.dates, open: data.open, high: data.high,
        low: data.low, close: data.close, volume: data.volume, height: 380,
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
          await Promise.all([loadStocks(), loadFactors()]);
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
    renderFactorList();
    renderFactorSelect();
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
    const groups = factorGroups();
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
            ${f.is_custom ? `<span class="del" data-del="${esc(f.id)}" title="删除">✕</span>` : ""}
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
        <span class="chip green">IC 均值 ${fmtNum(ic.mean, 3)}</span>
      </div>
      <div class="metric-cards">
        <div class="mcard"><div class="k">IC 均值</div><div class="v ${colorClass(ic.mean)}">${fmtNum(ic.mean, 3)}</div><div class="s">Spearman 秩相关</div></div>
        <div class="mcard"><div class="k">ICIR</div><div class="v">${fmtNum(ic.ir, 2)}</div><div class="s">年化信息比</div></div>
        <div class="mcard"><div class="k">IC &gt; 0 占比</div><div class="v">${fmtPct(ic.positive_rate, 1, false)}</div><div class="s">方向稳定性</div></div>
        <div class="mcard"><div class="k">因子覆盖率</div><div class="v">${fmtPct(res.stats.coverage, 1, false)}</div><div class="s">有效值占比</div></div>
      </div>
      <div id="icBlock"></div>
      <div class="grid-2" style="margin-top:0">
        <div id="decayBlock"></div>
        <div id="quantileBlock"></div>
      </div>
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

    const decayBlock = document.createElement("div");
    decayBlock.innerHTML = `<div style="margin-top:18px"><div class="chart-title-row"><div><div class="panel-title" style="font-size:13px">IC 衰减</div><div class="panel-sub">不同预测窗口的平均 IC</div></div></div><div class="chart-wrap"><canvas></canvas></div></div>`;
    box.querySelector("#decayBlock").appendChild(decayBlock);
    Charts.bar(decayBlock.querySelector("canvas"), {
      labels: res.ic_decay.map((d) => d.horizon + "日"),
      values: res.ic_decay.map((d) => d.ic),
      height: 200, colorsByValue: true, yFmt: (v) => v.toFixed(2), includeZero: true,
    });

    const qBlock = document.createElement("div");
    qBlock.innerHTML = `<div style="margin-top:18px"><div class="chart-title-row"><div><div class="panel-title" style="font-size:13px">分位组合收益</div><div class="panel-sub">按因子值五等分，未来 ${res.horizon} 日平均收益</div></div></div><div class="chart-wrap"><canvas></canvas></div></div>`;
    box.querySelector("#quantileBlock").appendChild(qBlock);
    Charts.bar(qBlock.querySelector("canvas"), {
      labels: res.quantile_returns.map((d) => d.group),
      values: res.quantile_returns.map((d) => d.return * 100),
      height: 200, colorsByValue: true, yFmt: (v) => v.toFixed(1) + "%", includeZero: true,
    });

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

  function openFactorModal() {
    openModal(`
      <div class="modal-head">
        <div><h3>新建自定义因子</h3><div class="panel-sub" style="margin-top:3px">表达式将作用于 (交易日 × 股票) 面板数组</div></div>
        <button class="icon-btn" data-close>✕</button>
      </div>
      <div class="form-grid">
        <div class="field"><label>因子名称</label><input id="nfName" type="text" placeholder="如：量价背离_20"></div>
        <div class="field"><label>方向</label>
          <select id="nfDirection"><option value="1">值越大越好（正向）</option><option value="-1">值越小越好（反向）</option></select>
        </div>
      </div>
      <div class="field" style="margin-top:12px"><label>因子表达式</label>
        <textarea id="nfExpr" rows="4" placeholder="rank(ts_mean(close, 20) / ts_mean(close, 60) - 1)">rank(ts_mean(close, 20) / ts_mean(close, 60) - 1)</textarea>
      </div>
      <div class="field" style="margin-top:12px"><label>描述</label><input id="nfDesc" type="text" placeholder="说明因子逻辑（可选）"></div>
      <div style="margin-top:14px">
        <div class="panel-sub">可用字段：<code style="color:var(--green)">open high low close volume amount vwap returns</code></div>
        <div class="fn-chips" id="fnChips"></div>
      </div>
      <div class="modal-foot">
        <button class="btn ghost" data-close>取消</button>
        <button class="btn primary" id="nfSave">保存因子</button>
      </div>`);
    const fns = ["ts_mean(x, 20)", "ts_std(x, 20)", "ts_rank(x, 20)", "ts_delay(x, 20)",
      "ts_delta(x, 5)", "ts_corr(x, y, 20)", "rank(x)", "zscore(x)", "where(cond, a, b)", "clip(x, -3, 3)"];
    $("#fnChips").innerHTML = fns.map((f) => `<button type="button" data-fn="${esc(f)}">${esc(f)}</button>`).join("");
    $$("#fnChips button").forEach((b) => b.addEventListener("click", () => {
      const ta = $("#nfExpr");
      const pos = ta.selectionStart || ta.value.length;
      ta.value = ta.value.slice(0, pos) + b.dataset.fn + ta.value.slice(pos);
      ta.focus();
    }));
    $("#nfSave").addEventListener("click", async () => {
      const name = $("#nfName").value.trim();
      const expression = $("#nfExpr").value.trim();
      if (!name || !expression) { toast("名称与表达式不能为空", true); return; }
      try {
        const created = await api("/api/factors", {
          method: "POST",
          body: {
            name, expression,
            description: $("#nfDesc").value.trim(),
            direction: Number($("#nfDirection").value),
          },
        });
        closeModal();
        await loadFactors();
        selectFactor(created.id);
        toast("自定义因子已保存");
      } catch (e) { toast(e.message, true); }
    });
  }

  /* ------------------------------ 回测 ------------------------------ */
  function renderFactorSelect() {
    const sel = $("#btFactor");
    const prev = sel.value;
    sel.innerHTML = factorGroups().map(([cat, items]) =>
      `<optgroup label="${esc(cat)}">` + items.map((f) =>
        `<option value="${esc(f.id)}">${esc(f.label || f.name)}</option>`
      ).join("") + "</optgroup>"
    ).join("");
    if (prev && state.factors.some((f) => String(f.id) === String(prev))) sel.value = prev;
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

  async function runBacktest() {
    const btn = $("#runBacktestBtn");
    const codes = state.universe.size ? Array.from(state.universe) : null;
    const payload = {
      factor_id: $("#btFactor").value,
      codes,
      start: $("#btStart").value || null,
      end: $("#btEnd").value || null,
      rebalance_days: Number($("#btRebalance").value),
      top_n: Number($("#btTopN").value) || 0,
      top_pct: Number($("#btTopPct").value) || 0.2,
      commission_bps: Number($("#btCost").value) || 0,
      long_short: $("#btLongShort").checked,
    };
    btn.disabled = true;
    btn.innerHTML = `<span class="spinner"></span> JAX 编译中`;
    $("#btResults").innerHTML = `<div class="panel"><div class="empty" style="border:none;padding:80px 24px"><span class="spinner" style="width:26px;height:26px;margin:0 auto 16px;display:block"></span><p>正在执行向量化回测…</p><div class="sub">首次运行需要 JAX 编译，之后会复用缓存</div></div></div>`;
    try {
      const payload2 = { ...payload, name: payloadLabel() };
      const res = await api("/api/backtest", { method: "POST", body: payload2 });
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
    const opt = $("#btFactor").selectedOptions[0];
    const label = opt ? opt.textContent : "因子";
    const n = state.universe.size;
    return `${label} · ${n ? n + "只" : "全市场"}`;
  }

  function renderBacktestResult(res) {
    const m = res.metrics;
    const cfg = res.config;
    const navPct = res.nav.map((v) => (v - 1) * 100);
    const benchPct = res.benchmark.map((v) => (v - 1) * 100);
    const ddPct = res.drawdown.map((v) => v * 100);

    const perf = [
      ["年化收益", fmtPct(m.annual_return), colorClass(m.annual_return)],
      ["累计收益", fmtPct(m.total_return), colorClass(m.total_return)],
      ["夏普比率", fmtNum(m.sharpe), ""],
      ["最大回撤", fmtPct(m.max_drawdown), "neg"],
      ["年化波动", fmtPct(m.annual_volatility, 1, false), ""],
      ["胜率", fmtPct(m.win_rate, 1, false), ""],
      ["年化换手", fmtNum(m.annual_turnover, 1) + "x", ""],
      ["超额年化", fmtPct(m.excess_annual_return), colorClass(m.excess_annual_return)],
    ];

    $("#btResults").innerHTML = `
      <div class="panel">
        <div class="result-head">
          <div>
            <div class="result-title">${esc(cfg.name)} <small>#${res.id} · ${esc(res.factor.label)}</small></div>
            <div class="factor-meta" style="margin:8px 0 0">
              <span class="chip">${esc(res.universe.start)} → ${esc(res.universe.end)}</span>
              <span class="chip">${res.universe.stocks} 只股票</span>
              <span class="chip">${cfg.rebalance_days} 日调仓</span>
              <span class="chip">${cfg.top_n > 0 ? "持仓 " + cfg.top_n : "前 " + (cfg.top_pct * 100).toFixed(0) + "%"}</span>
              <span class="chip">成本 ${fmtNum(cfg.commission_bps, 0)} bps</span>
              ${cfg.long_short ? `<span class="chip amber">多空</span>` : ""}
              <span class="chip green">已存入 SQLite</span>
            </div>
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
            <div class="panel-sub">初始净值 = 0，单位：% · 基准为股票池等权组合</div>
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

      <div class="panel">
        <div class="chart-title-row"><div><div class="panel-title" style="font-size:13px">月度收益热力图</div><div class="panel-sub">红色为正收益，绿色为负收益</div></div></div>
        <div id="heatmap" class="heatmap"></div>
      </div>

      <div class="panel">
        <div class="chart-title-row"><div><div class="panel-title" style="font-size:13px">调仓持仓</div><div class="panel-sub">最近 ${res.holdings.length} 次调仓的持仓明细（权重）</div></div></div>
        <div class="holdings-grid" id="holdingsGrid"></div>
      </div>`;

    Charts.line($("#equityChart"), {
      labels: res.dates,
      series: [
        { name: "策略", data: navPct, color: "#3ddc97", fill: true },
        { name: "基准(等权)", data: benchPct, color: "#6aa9ff", dash: [5, 4], width: 1.3 },
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

  async function loadHistory() {
    const grid = $("#historyGrid");
    try {
      const data = await api("/api/backtests?limit=60");
      const items = data.items || [];
      if (!items.length) {
        grid.innerHTML = `<div class="empty" style="grid-column:1/-1;padding:60px">
          <p>还没有回测记录</p><div class="sub">前往「回测引擎」运行第一次回测</div></div>`;
        return;
      }
      grid.innerHTML = items.map((it) => {
        const m = it.metrics || {};
        return `<div class="bt-card" data-id="${it.id}">
          <div class="top">
            <div>
              <div class="name">${esc(it.name)}</div>
              <div class="time">#${it.id} · ${esc(it.created_at)}</div>
            </div>
            <span class="chip ${m.annual_return >= 0 ? "green" : "amber"}">${fmtPct(m.annual_return)}</span>
          </div>
          <div class="row">
            <div class="mini"><div class="k">累计收益</div><div class="v ${colorClass(m.total_return)}">${fmtPct(m.total_return)}</div></div>
            <div class="mini"><div class="k">夏普</div><div class="v">${fmtNum(m.sharpe)}</div></div>
            <div class="mini"><div class="k">最大回撤</div><div class="v">${fmtPct(m.max_drawdown)}</div></div>
          </div>
        </div>`;
      }).join("");
      $$("#historyGrid .bt-card").forEach((card) => {
        card.addEventListener("click", () => openHistoryDetail(card.dataset.id));
      });
    } catch (e) {
      grid.innerHTML = `<div class="empty" style="grid-column:1/-1"><p>${esc(e.message)}</p></div>`;
    }
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
    $("#stockSearch").addEventListener("input", (e) => {
      state.stockFilter = e.target.value;
      renderStocks();
    });
    $("#newFactorBtn").addEventListener("click", openFactorModal);
    $("#previewBtn").addEventListener("click", previewFactor);
    $("#runBacktestBtn").addEventListener("click", runBacktest);
    $("#refreshHistory").addEventListener("click", loadHistory);
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
      await Promise.all([loadStocks(), loadFactors()]);
      renderUniverse();
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

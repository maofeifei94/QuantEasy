/* 轻量 Canvas 图表库：折线 / 柱状 / K线，支持高 DPI 与悬浮提示 */
(function () {
  const Charts = {};
  const PALETTE = ["#3ddc97", "#6aa9ff", "#f5b56b", "#a78bfa", "#ff6b7a"];

  const fmtNum = (v) => {
    if (v === null || v === undefined || !isFinite(v)) return "—";
    const a = Math.abs(v);
    if (a >= 1e8) return (v / 1e8).toFixed(2) + "亿";
    if (a >= 1e4) return (v / 1e4).toFixed(1) + "万";
    if (a >= 100) return v.toFixed(1);
    if (a >= 1) return v.toFixed(2);
    return v.toFixed(3);
  };
  Charts.fmtNum = fmtNum;

  function setup(canvas, height) {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.parentElement.clientWidth || 600;
    canvas.style.width = "100%";
    canvas.style.height = height + "px";
    canvas.width = Math.floor(w * dpr);
    canvas.height = Math.floor(height * dpr);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w, h: height, dpr };
  }

  function ensureTip(canvas) {
    const wrap = canvas.parentElement;
    let tip = wrap.querySelector(".chart-tip");
    if (!tip) {
      tip = document.createElement("div");
      tip.className = "chart-tip";
      wrap.appendChild(tip);
    }
    return tip;
  }

  function niceTicks(min, max, count) {
    if (!isFinite(min) || !isFinite(max)) return { min: 0, max: 1, ticks: [0, 1] };
    if (min === max) { min -= 1; max += 1; }
    const span = max - min;
    const step0 = span / count;
    const mag = Math.pow(10, Math.floor(Math.log10(step0)));
    const norm = step0 / mag;
    let step;
    if (norm <= 1) step = 1; else if (norm <= 2) step = 2; else if (norm <= 5) step = 5; else step = 10;
    step *= mag;
    const lo = Math.floor(min / step) * step;
    const hi = Math.ceil(max / step) * step;
    const ticks = [];
    for (let v = lo; v <= hi + step * 0.5; v += step) ticks.push(v);
    return { min: lo, max: hi, ticks };
  }

  function drawGrid(ctx, w, h, pad, range, yFmt, opts) {
    const { min, max, ticks } = range;
    ctx.font = "10px 'SF Mono', ui-monospace, monospace";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (const t of ticks) {
      const y = pad.t + (h - pad.t - pad.b) * (1 - (t - min) / (max - min || 1));
      ctx.strokeStyle = opts.gridColor || "rgba(255,255,255,0.055)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(pad.l, y + 0.5);
      ctx.lineTo(w - pad.r, y + 0.5);
      ctx.stroke();
      ctx.fillStyle = opts.labelColor || "#5c6b80";
      ctx.fillText(yFmt(t), pad.l - 8, y);
    }
  }

  function drawXLabels(ctx, w, h, pad, labels, xFmt, opts) {
    ctx.fillStyle = opts.labelColor || "#5c6b80";
    ctx.font = "10px 'SF Mono', ui-monospace, monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    if (!labels.length) return;
    const n = Math.min(6, labels.length);
    for (let i = 0; i < n; i++) {
      const idx = Math.round((labels.length - 1) * (i / Math.max(n - 1, 1)));
      const x = pad.l + (w - pad.l - pad.r) * (labels.length === 1 ? 0.5 : idx / (labels.length - 1));
      ctx.fillText(xFmt ? xFmt(labels[idx]) : labels[idx], x, h - pad.b + 8);
    }
  }

  function drawLegend(canvas, series, colors) {
    let legend = canvas.parentElement.querySelector(".chart-legend");
    if (legend) legend.remove();
    if (series.length < 2) return;
    legend = document.createElement("div");
    legend.className = "legend chart-legend";
    legend.style.marginTop = "8px";
    legend.innerHTML = series.map((s, i) =>
      `<span class="it"><span class="sw" style="background:${s.color || colors[i % colors.length]}"></span>${s.name}</span>`
    ).join("");
    canvas.parentElement.appendChild(legend);
  }

  /* ------------------------- 折线图 ------------------------- */
  Charts.line = function (canvas, cfg) {
    const height = cfg.height || 260;
    const colors = cfg.colors || PALETTE;
    const yFmt = cfg.yFmt || ((v) => v.toFixed(2));
    const xFmt = cfg.xFmt || ((v) => String(v).slice(2, 10));
    const tip = ensureTip(canvas);
    let hover = -1;

    function render() {
      const { ctx, w, h } = setup(canvas, height);
      ctx.clearRect(0, 0, w, h);
      const pad = { l: 54, r: 14, t: 14, b: 26 };
      const labels = cfg.labels || [];
      let min = Infinity, max = -Infinity;
      cfg.series.forEach((s) => s.data.forEach((v) => {
        if (v !== null && v !== undefined && isFinite(v)) { min = Math.min(min, v); max = Math.max(max, v); }
      }));
      if (cfg.includeZero) { min = Math.min(min, 0); max = Math.max(max, 0); }
      if (!isFinite(min)) { min = 0; max = 1; }
      const padY = (max - min) * 0.08 || Math.abs(max) * 0.05 || 1;
      const range = niceTicks(cfg.minY !== undefined ? cfg.minY : min - padY,
        cfg.maxY !== undefined ? cfg.maxY : max + padY, 4);
      drawGrid(ctx, w, h, pad, range, yFmt, cfg);
      drawXLabels(ctx, w, h, pad, labels, xFmt, cfg);

      const xAt = (i) => pad.l + (w - pad.l - pad.r) * (labels.length <= 1 ? 0.5 : i / (labels.length - 1));
      const yAt = (v) => pad.t + (h - pad.t - pad.b) * (1 - (v - range.min) / (range.max - range.min || 1));

      cfg.series.forEach((s, si) => {
        const color = s.color || colors[si % colors.length];
        if (s.fill) {
          const grad = ctx.createLinearGradient(0, pad.t, 0, h - pad.b);
          grad.addColorStop(0, s.fillColor || "rgba(61,220,151,0.18)");
          grad.addColorStop(1, "rgba(61,220,151,0)");
          ctx.beginPath();
          let started = false;
          s.data.forEach((v, i) => {
            if (v === null || !isFinite(v)) return;
            if (!started) { ctx.moveTo(xAt(i), yAt(v)); started = true; }
            else ctx.lineTo(xAt(i), yAt(v));
          });
          if (started) {
            ctx.lineTo(xAt(s.data.length - 1), h - pad.b);
            ctx.lineTo(xAt(s.data.findIndex((v) => v !== null && isFinite(v))), h - pad.b);
            ctx.closePath();
            ctx.fillStyle = grad;
            ctx.fill();
          }
        }
        ctx.beginPath();
        ctx.strokeStyle = color;
        ctx.lineWidth = s.width || 1.8;
        if (s.dash) ctx.setLineDash(s.dash); else ctx.setLineDash([]);
        let started = false;
        s.data.forEach((v, i) => {
          if (v === null || v === undefined || !isFinite(v)) { started = false; return; }
          if (!started) { ctx.moveTo(xAt(i), yAt(v)); started = true; }
          else ctx.lineTo(xAt(i), yAt(v));
        });
        ctx.stroke();
        ctx.setLineDash([]);
      });

      if (hover >= 0 && hover < labels.length) {
        const x = xAt(hover);
        ctx.strokeStyle = "rgba(255,255,255,0.22)";
        ctx.setLineDash([3, 3]);
        ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, h - pad.b); ctx.stroke();
        ctx.setLineDash([]);
        cfg.series.forEach((s, si) => {
          const v = s.data[hover];
          if (v === null || v === undefined || !isFinite(v)) return;
          ctx.beginPath();
          ctx.arc(x, yAt(v), 2.6, 0, Math.PI * 2);
          ctx.fillStyle = s.color || colors[si % colors.length];
          ctx.fill();
        });
        tip.innerHTML =
          `<div class="tip-title">${xFmt(labels[hover])}</div>` +
          cfg.series.map((s, si) =>
            `<div class="tip-row"><span class="sw" style="background:${s.color || colors[si % colors.length]}"></span>` +
            `<span>${s.name} ${yFmt(s.data[hover])}</span></div>`
          ).join("");
        const rect = canvas.getBoundingClientRect();
        const px = xAt(hover);
        tip.style.left = Math.min(Math.max(px + 14, 8), rect.width - 130) + "px";
        tip.style.top = pad.t + 6 + "px";
        tip.style.opacity = 1;
      } else {
        tip.style.opacity = 0;
      }
    }

    const parent = canvas.parentElement;
    if (!parent.dataset.chartBound) {
      parent.dataset.chartBound = "1";
      parent.addEventListener("mousemove", (e) => {
        const c = parent.querySelector("canvas");
        if (!c || !c._draw) return;
        const rect = c.getBoundingClientRect();
        const padL = 54, padR = 14;
        const usable = rect.width - padL - padR;
        const x = e.clientX - rect.left;
        if (x < padL || x > rect.width - padR) { c._hover = -1; c._draw(); return; }
        const labels = c._labels || [];
        const idx = Math.round(((x - padL) / usable) * (labels.length - 1));
        c._hover = Math.max(0, Math.min(labels.length - 1, idx));
        c._draw();
      });
      parent.addEventListener("mouseleave", () => {
        const c = parent.querySelector("canvas");
        if (c && c._draw) { c._hover = -1; c._draw(); }
      });
      let raf = 0;
      const ro = new ResizeObserver(() => {
        cancelAnimationFrame(raf);
        raf = requestAnimationFrame(() => {
          const c = parent.querySelector("canvas");
          if (c && c._redraw) c._redraw();
        });
      });
      ro.observe(parent);
    }

    canvas._labels = cfg.labels || [];
    canvas._hover = -1;
    canvas._draw = () => { hover = canvas._hover; render(); };
    canvas._redraw = render;
    render();
    drawLegend(canvas, cfg.series, colors);
    return canvas;
  };

  /* ------------------------- 柱状图 ------------------------- */
  Charts.bar = function (canvas, cfg) {
    const height = cfg.height || 200;
    const colors = cfg.colors || PALETTE;
    const tip = ensureTip(canvas);
    let hover = -1;

    function render() {
      const { ctx, w, h } = setup(canvas, height);
      ctx.clearRect(0, 0, w, h);
      const pad = { l: 54, r: 14, t: 14, b: 28 };
      const labels = cfg.labels || [];
      const values = cfg.values || [];
      let min = Math.min(0, ...values.filter((v) => isFinite(v)));
      let max = Math.max(0, ...values.filter((v) => isFinite(v)));
      const range = niceTicks(min, max, 4);
      drawGrid(ctx, w, h, pad, range, cfg.yFmt || ((v) => v.toFixed(2)), cfg);
      const innerW = w - pad.l - pad.r;
      const step = innerW / Math.max(labels.length, 1);
      const barW = Math.min(step * 0.56, 46);
      const zeroY = pad.t + (h - pad.t - pad.b) * (1 - (0 - range.min) / (range.max - range.min || 1));

      labels.forEach((label, i) => {
        const v = values[i] || 0;
        const y = pad.t + (h - pad.t - pad.b) * (1 - (v - range.min) / (range.max - range.min || 1));
        const x = pad.l + step * (i + 0.5) - barW / 2;
        const color = cfg.colorsByValue
          ? (v >= 0 ? "#f87171" : "#3ddc97")
          : colors[i % colors.length];
        ctx.fillStyle = hover === i ? "#ffffff" : color;
        ctx.globalAlpha = hover === i ? 0.9 : 0.78;
        ctx.fillRect(x, Math.min(y, zeroY), barW, Math.max(1, Math.abs(zeroY - y)));
        ctx.globalAlpha = 1;
        ctx.fillStyle = "#5c6b80";
        ctx.font = "10px 'SF Mono', ui-monospace, monospace";
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        ctx.fillText(label, x + barW / 2, h - pad.b + 8);
      });

      if (hover >= 0) {
        tip.innerHTML = `<div class="tip-title">${labels[hover]}</div><div>${cfg.yFmt ? cfg.yFmt(values[hover]) : values[hover]}</div>`;
        const rect = canvas.getBoundingClientRect();
        tip.style.left = Math.min(pad.l + step * (hover + 0.5), rect.width - 120) + "px";
        tip.style.top = "10px";
        tip.style.opacity = 1;
      } else tip.style.opacity = 0;
    }

    const parent = canvas.parentElement;
    if (!parent.dataset.barBound) {
      parent.dataset.barBound = "1";
      parent.addEventListener("mousemove", (e) => {
        const c = parent.querySelector("canvas");
        if (!c || !c._bdraw) return;
        const rect = c.getBoundingClientRect();
        const innerW = rect.width - 68;
        const idx = Math.floor((e.clientX - rect.left - 54) / (innerW / Math.max(c._labels.length, 1)));
        c._bhover = Math.max(0, Math.min(c._labels.length - 1, idx));
        c._bdraw();
      });
      parent.addEventListener("mouseleave", () => {
        const c = parent.querySelector("canvas");
        if (c && c._bdraw) { c._bhover = -1; c._bdraw(); }
      });
      let raf = 0;
      const ro = new ResizeObserver(() => {
        cancelAnimationFrame(raf);
        raf = requestAnimationFrame(() => {
          const c = parent.querySelector("canvas");
          if (c && c._bredraw) c._bredraw();
        });
      });
      ro.observe(parent);
    }

    canvas._labels = cfg.labels || [];
    canvas._bhover = -1;
    canvas._bdraw = () => { hover = canvas._bhover; render(); };
    canvas._bredraw = render;
    render();
    return canvas;
  };

  /* ------------------------- K线图 ------------------------- */
  Charts.candle = function (canvas, cfg) {
    const height = cfg.height || 340;
    function render() {
      const { ctx, w, h } = setup(canvas, height);
      ctx.clearRect(0, 0, w, h);
      const pad = { l: 54, r: 14, t: 12, b: 24 };
      const volH = 52;
      const priceH = h - pad.t - pad.b - volH - 10;
      const { dates, open, high, low, close, volume } = cfg;
      const n = dates.length;
      if (!n) return;
      let min = Infinity, max = -Infinity, vmax = 0;
      for (let i = 0; i < n; i++) {
        min = Math.min(min, low[i]); max = Math.max(max, high[i]);
        vmax = Math.max(vmax, volume[i] || 0);
      }
      const padv = (max - min) * 0.06 || 1;
      const range = niceTicks(min - padv, max + padv, 4);
      drawGrid(ctx, w, h, pad, range, (v) => v.toFixed(1), cfg);
      const innerW = w - pad.l - pad.r;
      const xAt = (i) => pad.l + innerW * (i / Math.max(n - 1, 1));
      const yAt = (v) => pad.t + priceH * (1 - (v - range.min) / (range.max - range.min || 1));
      const step = Math.max(1, Math.floor(n / innerW * 2));
      const barW = Math.max(1.2, Math.min(6, innerW / n * 0.66));

      for (let i = 0; i < n; i += 1) {
        const up = close[i] >= open[i];
        const color = up ? "#ff6b7a" : "#3ddc97";
        const x = xAt(i);
        ctx.strokeStyle = color;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x, yAt(high[i]));
        ctx.lineTo(x, yAt(low[i]));
        ctx.stroke();
        const bodyTop = yAt(Math.max(open[i], close[i]));
        const bodyH = Math.max(1, Math.abs(yAt(open[i]) - yAt(close[i])));
        ctx.fillStyle = color;
        ctx.fillRect(x - barW / 2, bodyTop, barW, bodyH);
      }

      // 均线叠加
      (cfg.ma || []).forEach((line) => {
        ctx.beginPath();
        ctx.strokeStyle = line.color;
        ctx.lineWidth = 1.2;
        ctx.setLineDash([]);
        let started = false;
        line.data.forEach((v, i) => {
          if (v === null || !isFinite(v)) { started = false; return; }
          if (!started) { ctx.moveTo(xAt(i), yAt(v)); started = true; }
          else ctx.lineTo(xAt(i), yAt(v));
        });
        ctx.stroke();
      });

      // 成交量
      const volTop = pad.t + priceH + 12;
      const vBase = h - pad.b;
      for (let i = 0; i < n; i++) {
        const up = close[i] >= open[i];
        ctx.fillStyle = up ? "rgba(255,107,122,.35)" : "rgba(61,220,151,.35)";
        const bh = (volume[i] / (vmax || 1)) * volH;
        ctx.fillRect(xAt(i) - barW / 2, vBase - bh, barW, bh);
      }

      ctx.fillStyle = "#5c6b80";
      ctx.font = "10px 'SF Mono', ui-monospace, monospace";
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      const ticks = Math.min(5, n);
      for (let i = 0; i < ticks; i++) {
        const idx = Math.round((n - 1) * (i / Math.max(ticks - 1, 1)));
        ctx.fillText(dates[idx].slice(2), xAt(idx), h - pad.b + 7);
      }
    }
    render();
    let raf = 0;
    const ro = new ResizeObserver(() => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(render);
    });
    ro.observe(canvas.parentElement);
    return canvas;
  };

  window.Charts = Charts;
})();

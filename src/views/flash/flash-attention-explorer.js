import { setupDocOutline } from "../../lib/setupDocOutline.js";

export function initializeFlashAttentionExplorer(root) {
  const controller = new AbortController();
  const { signal } = controller;
  const byId = id => root.querySelector("#" + id);
  const document = {
    body: root,
    querySelector: selector => root.querySelector(selector),
    querySelectorAll: selector => root.querySelectorAll(selector),
    getElementById: byId,
    createElementNS: window.document.createElementNS.bind(window.document),
  };
  const intervals = new Set();
  const setInterval = (callback, delay, ...args) => {
    const timer = window.setInterval(callback, delay, ...args);
    intervals.add(timer);
    return timer;
  };
  const clearInterval = timer => {
    window.clearInterval(timer);
    intervals.delete(timer);
  };

  /* ================= 公共工具 ================= */
  const $ = s => document.querySelector(s);
  const SVGNS = "http://www.w3.org/2000/svg";
  const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
  function el(tag, attrs, parent) {
    const n = document.createElementNS(SVGNS, tag);
    for (const k in attrs) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  }
  function fmtBytes(b) {
    if (b >= 1073741824) return (b / 1073741824).toFixed(2) + " GiB";
    if (b >= 1048576) return (b / 1048576).toFixed(1) + " MiB";
    if (b >= 1024) return (b / 1024).toFixed(1) + " KB";
    return Math.round(b) + " B";
  }
  function fmtFlops(f) {
    if (f >= 1e12) return (f / 1e12).toFixed(1) + " TFLOP";
    if (f >= 1e9) return (f / 1e9).toFixed(1) + " GFLOP";
    return (f / 1e6).toFixed(1) + " MFLOP";
  }
  /* 确定性伪随机（演示用） */
  function hash01(a, b) {
    let x = Math.sin(a * 127.1 + b * 311.7) * 43758.5453;
    return x - Math.floor(x);
  }
  /* ================= HERO 循环动画 ================= */
  (function () {
    const gQ = $("#heroQ"),
      gKV = $("#heroKV"),
      gTile = $("#heroTile"),
      gO = $("#heroO");
    const qRects = [],
      kRects = [],
      vRects = [],
      oRects = [],
      oFills = [];
    for (let i = 0; i < 4; i++) {
      const y = 40 + i * 36;
      const r = el(
        "rect",
        {
          x: 18,
          y,
          width: 120,
          height: 28,
          fill: "oklch(45% .16 267)",
          stroke: "oklch(80% .08 260)",
        },
        gQ
      );
      el(
        "text",
        {
          x: 28,
          y: y + 18,
          "font-family": "IBM Plex Mono,monospace",
          "font-size": 11,
          fill: "oklch(92% .04 260)",
        },
        gQ
      ).textContent = "Q" + (i + 1) + " · 行 " + i * 4 + "–" + (i * 4 + 3);
      qRects.push(r);
      const or_ = el(
        "rect",
        {
          x: 404,
          y,
          width: 100,
          height: 28,
          fill: "none",
          stroke: "oklch(70% .09 260)",
        },
        gO
      );
      const f = el(
        "rect",
        { x: 404, y, width: 0, height: 28, fill: "oklch(58% .14 261)" },
        gO
      );
      oRects.push(or_);
      oFills.push(f);
      const kx = 200 + j2x(i);
      function j2x(j) {
        return j * 38;
      }
      const k = el(
        "rect",
        {
          x: 200 + i * 38,
          y: 40,
          width: 34,
          height: 28,
          fill: "oklch(50% .13 60)",
          stroke: "oklch(85% .06 55)",
        },
        gKV
      );
      el(
        "text",
        {
          x: 200 + i * 38 + 17,
          y: 58,
          "text-anchor": "middle",
          "font-family": "IBM Plex Mono,monospace",
          "font-size": 10,
          fill: "#fff",
        },
        gKV
      ).textContent = "K" + (i + 1);
      const v = el(
        "rect",
        {
          x: 200 + i * 38,
          y: 82,
          width: 34,
          height: 28,
          fill: "oklch(44% .09 195)",
          stroke: "oklch(85% .05 195)",
        },
        gKV
      );
      el(
        "text",
        {
          x: 200 + i * 38 + 17,
          y: 100,
          "text-anchor": "middle",
          "font-family": "IBM Plex Mono,monospace",
          "font-size": 10,
          fill: "#fff",
        },
        gKV
      ).textContent = "V" + (i + 1);
      kRects.push(k);
      vRects.push(v);
    }
    const cells = [];
    for (let r = 0; r < 4; r++)
      for (let c = 0; c < 4; c++) {
        cells.push(
          el(
            "rect",
            {
              x: 206 + c * 33,
              y: 150 + r * 16,
              width: 30,
              height: 13,
              fill: "oklch(55% .12 305)",
              opacity: 0,
            },
            gTile
          )
        );
      }
    let step = 0;
    function apply() {
      const i = Math.floor(step / 4),
        j = step % 4;
      qRects.forEach((r, x) => {
        r.setAttribute("stroke-width", x === i ? 2.5 : 1);
        r.setAttribute(
          "fill",
          x === i ? "oklch(55% .17 263)" : "oklch(45% .16 267)"
        );
      });
      kRects.forEach((r, x) => {
        r.setAttribute("stroke-width", x === j ? 2.5 : 1);
        r.style.opacity = x === j ? 1 : 0.55;
      });
      vRects.forEach((r, x) => {
        r.setAttribute("stroke-width", x === j ? 2.5 : 1);
        r.style.opacity = x === j ? 1 : 0.55;
      });
      for (let c = 0; c < 4; c++)
        cells[i * 4 + c].setAttribute(
          "opacity",
          c <= j
            ? 0.35 + 0.65 * hash01(i * 4 + c, j)
            : cells[i * 4 + c].getAttribute("data-done")
              ? cells[i * 4 + c].getAttribute("opacity")
              : 0
        );
      for (let c = 0; c <= j; c++) {
        cells[i * 4 + c].setAttribute("data-done", 1);
      }
      oFills.forEach((f, x) => {
        if (x < i) f.setAttribute("width", 100);
        else if (x === i) f.setAttribute("width", ((j + 1) / 4) * 100);
      });
      const S4 = "₁₂₃₄";
      $("#heroEq").textContent =
        "S" +
        S4[i] +
        S4[j] +
        " = Q" +
        S4[i] +
        "K" +
        S4[j] +
        "ᵀ → 并入 O" +
        S4[i];
      $("#heroCounter").textContent =
        "块对 " + (step + 1) + "/16 · HBM 上的 N×N 矩阵：0 B";
    }
    if (reduceMotion) {
      step = 15;
      apply();
      $("#heroCounter").textContent = "块对 16/16 · HBM 上的 N×N 矩阵：0 B";
    } else {
      apply();
      setInterval(() => {
        step = (step + 1) % 16;
        if (step === 0) {
          cells.forEach(c => {
            c.setAttribute("opacity", 0);
            c.removeAttribute("data-done");
          });
          oFills.forEach(f => f.setAttribute("width", 0));
        }
        apply();
      }, 850);
    }
  })();
  /* ================= TOOL 01 · 访存分析器 ================= */
  (function () {
    const nIn = $("#t1-n");
    let d = 64;
    function calc() {
      const N = Math.pow(2, +nIn.value);
      const qk = 4 * N * d,
        sRW = 4 * N * N,
        pRW = 4 * N * N,
        vo = 4 * N * d;
      const bytes = qk + sRW + pRW + vo;
      const flops = 4 * N * N * d;
      const ai = flops / bytes;
      const ridge = 312e12 / 1.6e12;
      const idle = Math.max(0, 1 - ai / ridge);
      $("#t1-n-out").textContent = N.toLocaleString();
      $("#t1-bytes").textContent = fmtBytes(bytes);
      $("#t1-detail").textContent =
        "S、P 两张 N×N 矩阵贡献 " +
        (((sRW + pRW) / bytes) * 100).toFixed(1) +
        "%";
      $("#t1-flops").textContent = fmtFlops(flops);
      $("#t1-ai").textContent = ai.toFixed(1);
      $("#t1-ridge").textContent = ridge.toFixed(0);
      $("#t1-verdict").textContent = ai < ridge ? "内存受限" : "算力受限";
      $("#t1-verdict-sub").textContent =
        ai < ridge
          ? "算力空转 ≈" + (idle * 100).toFixed(0) + "%"
          : "带宽已喂饱";
      const rows = [
        ["读 Q · K", qk, ""],
        ["S 写＋读 · QKᵀ", sRW, "c-s"],
        ["P 写＋读 · softmax", pRW, "c-p"],
        ["读 V · 写 O", vo, "c-kv"],
      ];
      $("#t1-bars").innerHTML = rows
        .map(
          ([lbl, v, cls]) =>
            '<div class="flowbar"><span class="lbl">' +
            lbl +
            '</span><div class="trk"><div class="fil ' +
            cls +
            '" style="width:' +
            Math.max(1.5, (v / bytes) * 100) +
            '%"></div></div><span class="val">' +
            fmtBytes(v) +
            "</span></div>"
        )
        .join("");
    }
    nIn.addEventListener("input", calc, signal);
    $("#t1-d").addEventListener(
      "click",
      e => {
        const b = e.target.closest("button");
        if (!b) return;
        d = +b.dataset.d;
        [...$("#t1-d").children].forEach(x =>
          x.classList.toggle("is-on", x === b)
        );
        calc();
      },
      signal
    );
    calc();
  })();
  /* ================= TOOL 02 · 内存层次 ================= */
  (function () {
    const svg = $("#hierSvg");
    const levels = [
      {
        name: "SRAM · 片上",
        cap: 192 * 1024,
        capLbl: "192 KB / SM",
        note: "×108 SM ≈ 20.25 MB · 聚合带宽 ~19 TB/s · 分块的舞台",
        color: "var(--blue)",
      },
      {
        name: "L2 缓存",
        cap: 40 * 1048576,
        capLbl: "40 MB",
        note: "全卡共享 · 带宽介于 SRAM 与 HBM 之间",
        color: "var(--teal)",
      },
      {
        name: "HBM · 显存",
        cap: 40 * 1073741824,
        capLbl: "40–80 GB",
        note: "1.5–2.0 TB/s · 慢一个数量级 · N×N 矩阵不该来这里",
        color: "var(--orange)",
      },
    ];
    const maxLog = Math.log10(40 * 1073741824),
      minLog = Math.log10(64 * 1024);
    const X = v => 150 + ((Math.log10(v) - minLog) / (maxLog - minLog)) * 440;
    levels.forEach((lv, i) => {
      const y = 24 + i * 44;
      el(
        "text",
        { x: 12, y: y + 15, class: "n-lbl", "font-size": 11 },
        svg
      ).textContent = lv.name;
      const r = el(
        "rect",
        {
          x: 150,
          y,
          height: 22,
          width: X(lv.cap) - 150,
          fill: lv.color,
          opacity: 0.85,
          cursor: "pointer",
          tabindex: 0,
          role: "button",
          "aria-label": lv.name,
        },
        svg
      );
      el(
        "text",
        { x: X(lv.cap) + 8, y: y + 15, class: "n-sub", "font-size": 10 },
        svg
      ).textContent = lv.capLbl;
      const show = () => {
        svg
          .querySelectorAll("rect")
          .forEach(x => x.setAttribute("opacity", 0.35));
        r.setAttribute("opacity", 0.95);
        $("#hier-hover").textContent = lv.name + " · " + lv.note;
      };
      r.addEventListener("click", show, signal);
      r.addEventListener(
        "keydown",
        e => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            show();
          }
        },
        signal
      );
    });
    const matBar = el(
      "rect",
      {
        x: 150,
        y: 170,
        height: 26,
        width: 0,
        fill: "var(--red)",
        opacity: 0.9,
      },
      svg
    );
    const matLbl = el(
      "text",
      { x: 12, y: 187, class: "n-lbl", "font-size": 11 },
      svg
    );
    matLbl.textContent = "N×N 矩阵";
    const matVal = el(
      "text",
      { x: 150, y: 214, class: "n-sub", "font-size": 10 },
      svg
    );
    const nIn = $("#t2-n");
    function calc() {
      const N = Math.pow(2, +nIn.value);
      const mat = 2 * N * N;
      $("#t2-n-out").textContent = N.toLocaleString();
      $("#t2-mat").textContent = fmtBytes(mat);
      $("#t2-sm").textContent =
        mat <= 192 * 1024
          ? "可以驻留"
          : "超出 " +
            (mat / (192 * 1024)).toFixed(mat / (192 * 1024) > 10 ? 0 : 1) +
            "×";
      $("#t2-all").textContent =
        mat <= 20.25 * 1048576
          ? "可以驻留"
          : "超出 " + (mat / (20.25 * 1048576)).toFixed(1) + "×";
      $("#t2-verdict").textContent =
        mat <= 192 * 1024 ? "尚可直接放" : "必须分块";
      matBar.setAttribute("width", Math.max(2, X(mat) - 150));
      matVal.setAttribute("x", Math.max(150, X(mat) + 8));
      matVal.textContent =
        fmtBytes(mat) +
        "（FP16，" +
        N.toLocaleString() +
        "×" +
        N.toLocaleString() +
        "）";
    }
    nIn.addEventListener("input", calc, signal);
    calc();
  })();
  /* ================= TOOL 03 · 分块回放 ================= */
  (function () {
    const svg = $("#tileSvg");
    const BR = 4,
      NB = 4,
      DB = 4,
      BYTES = 2; // 4 块 × 4 行 × d=4 · FP16
    const BLK = BR * DB * BYTES; // 每块 32 B
    const STD_TOTAL = 8 * 16 * 16 + 8 * 16 * DB; // 标准实现对照 2560 B
    const SPILL_STD = 8 * 16 * 16; // 其中 S/P 落盘 2048 B
    let causal = false,
      stepIdx = 0,
      timer = null;
    /* ---- 静态结构 ---- */
    const defs = el("defs", {}, svg);
    const pat = el(
      "pattern",
      {
        id: "hatch",
        width: 6,
        height: 6,
        patternUnits: "userSpaceOnUse",
        patternTransform: "rotate(45)",
      },
      defs
    );
    el("rect", { width: 6, height: 6, fill: "var(--paper-subtle)" }, pat);
    el("rect", { width: 2, height: 6, fill: "var(--line-strong)" }, pat);
    const qR = [],
      kR = [],
      vR = [],
      oR = [],
      oF = [],
      eQ = [],
      eK = [],
      eV = [],
      eO = [];
    const SUB = ["₁", "₂", "₃", "₄"];
    for (let i = 0; i < NB; i++) {
      const y = 66 + i * 66;
      el(
        "text",
        { x: 16, y: y - 8, class: "n-sub", "font-size": 9 },
        svg
      ).textContent = "行 " + i * BR + "–" + (i * BR + BR - 1);
      qR.push(
        el(
          "rect",
          { x: 16, y, width: 104, height: 40, class: "blk blk-q" },
          svg
        )
      );
      el(
        "text",
        { x: 26, y: 25, class: "n-lbl", "font-size": 13 },
        svg
      ).textContent = "Q" + SUB[i];
      oR.push(
        el(
          "rect",
          { x: 556, y, width: 104, height: 40, class: "blk blk-o" },
          svg
        )
      );
      oF.push(
        el(
          "rect",
          { x: 556, y, width: 0, height: 40, fill: "oklch(85% .08 260)" },
          svg
        )
      );
      el(
        "text",
        { x: 566, y: 25, class: "n-lbl", "font-size": 13 },
        svg
      ).textContent = "O" + SUB[i];
      const kx = 216 + i * 84;
      el(
        "text",
        {
          x: kx + 24,
          y: 22,
          class: "n-sub",
          "font-size": 9,
          "text-anchor": "middle",
        },
        svg
      ).textContent = "列 " + i * BR + "–" + (i * BR + BR - 1);
      kR.push(
        el(
          "rect",
          { x: kx, y: 30, width: 56, height: 32, class: "blk blk-k" },
          svg
        )
      );
      el(
        "text",
        {
          x: kx + 28,
          y: 50,
          class: "n-lbl",
          "font-size": 12,
          "text-anchor": "middle",
        },
        svg
      ).textContent = "K" + SUB[i];
      vR.push(
        el(
          "rect",
          { x: kx, y: 74, width: 56, height: 32, class: "blk blk-v" },
          svg
        )
      );
      el(
        "text",
        {
          x: kx + 28,
          y: 94,
          class: "n-lbl",
          "font-size": 12,
          "text-anchor": "middle",
        },
        svg
      ).textContent = "V" + SUB[i];
      eQ.push(
        el(
          "path",
          {
            d:
              "M120 " +
              (y + 20) +
              " C 160 " +
              (y + 20) +
              ", 170 " +
              (176 + i * 34) +
              ", 196 " +
              (176 + i * 34),
            class: "edge is-dim",
          },
          svg
        )
      );
      eK.push(
        el(
          "path",
          {
            d: "M" + (kx + 20) + " 62 L " + (kx + 20) + " 136",
            class: "edge is-dim",
          },
          svg
        )
      );
      eV.push(
        el(
          "path",
          {
            d: "M" + (kx + 40) + " 106 L " + (kx + 40) + " 136",
            class: "edge is-dim",
          },
          svg
        )
      );
      eO.push(
        el(
          "path",
          {
            d:
              "M524 " +
              (176 + i * 34) +
              " C 540 " +
              (176 + i * 34) +
              ", 544 " +
              (y + 20) +
              ", 556 " +
              (y + 20),
            class: "edge is-dim",
          },
          svg
        )
      );
    }
    el(
      "text",
      { x: 16, y: 36, class: "n-sub", "font-size": 9 },
      svg
    ).textContent = "HBM · 读";
    el(
      "text",
      { x: 556, y: 50, class: "n-sub", "font-size": 9 },
      svg
    ).textContent = "HBM · 写回";
    /* SRAM 容器 */
    el(
      "rect",
      {
        x: 196,
        y: 136,
        width: 328,
        height: 180,
        fill: "oklch(97% .01 305)",
        stroke: "var(--purple)",
        "stroke-dasharray": "6 4",
      },
      svg
    );
    el(
      "text",
      {
        x: 360,
        y: 156,
        class: "n-lbl",
        "font-size": 10,
        "text-anchor": "middle",
        fill: "var(--purple)",
      },
      svg
    ).textContent = "SRAM · 片上驻留（示意 192 KB）";
    /* S-tile 4×4 */
    const cells = [];
    for (let r = 0; r < BR; r++)
      for (let c = 0; c < BR; c++)
        cells.push(
          el(
            "rect",
            {
              x: 216 + c * 46,
              y: 172 + r * 30,
              width: 40,
              height: 24,
              fill: "var(--purple)",
              opacity: 0,
            },
            svg
          )
        );
    el(
      "text",
      { x: 216, y: 308, class: "n-sub", "font-size": 9 },
      svg
    ).textContent = "Sᵢⱼ 4×4 · 算完即弃";
    /* SRAM 内的更新公式 */
    const upd = [
      "m ← max(m, rowmax(Sᵢⱼ))",
      "ℓ ← ℓ·α + Σexp(Sᵢⱼ−m)",
      "O ← O·α + exp(Sᵢⱼ−m)·Vⱼ",
    ];
    upd.forEach(
      (t, k) =>
        (el(
          "text",
          {
            x: 408,
            y: 190 + k * 26,
            class: "n-sub",
            "font-size": 10,
            fill: "var(--ink-strong)",
          },
          svg
        ).textContent = t)
    );
    el(
      "text",
      {
        x: 408,
        y: 190 + 3 * 26 + 2,
        class: "n-sub",
        "font-size": 9,
        fill: "var(--purple)",
      },
      svg
    ).textContent = "α = e^(m_old − m_new)";
    /* 块对地图 */
    el(
      "text",
      { x: 16, y: 352, class: "n-sub", "font-size": 9 },
      svg
    ).textContent = "块对地图 (i, j)";
    const pmap = [];
    for (let r = 0; r < NB; r++)
      for (let c = 0; c < NB; c++)
        pmap.push(
          el(
            "rect",
            {
              x: 16 + c * 34,
              y: 360 + r * 24,
              width: 30,
              height: 20,
              fill: "var(--paper-subtle)",
              stroke: "var(--line-strong)",
            },
            svg
          )
        );
    const stepTxt = el(
      "text",
      { x: 170, y: 376, class: "n-lbl", "font-size": 12 },
      svg
    );
    const stepTxt2 = el(
      "text",
      { x: 170, y: 396, class: "n-sub", "font-size": 10 },
      svg
    );
    /* ---- 步骤序列 ---- */
    function seq() {
      const s = [];
      for (let i = 0; i < NB; i++)
        for (let j = 0; j < NB; j++) if (!causal || j <= i) s.push([i, j]);
      return s;
    }
    function fmt(b) {
      return b >= 1024
        ? (b / 1024).toFixed(2).replace(/\.?0+$/, "") + " KB"
        : b + " B";
    }
    function apply() {
      const S = seq(),
        total = S.length;
      $("#t3-prog").max = total;
      $("#t3-prog").value = stepIdx;
      $("#t3-count").textContent = stepIdx + " / " + total;
      $("#t3-mode").textContent = causal
        ? "CAUSAL · " + total + " 步（跳过 " + (16 - total) + " 个全掩码块）"
        : "FULL · " + total + " 步";
      [qR, kR, vR, oR].flat().forEach(r => r.classList.remove("is-hot"));
      [...eQ, ...eK, ...eV, ...eO].forEach(e => e.classList.add("is-dim"));
      pmap.forEach((p, idx) => {
        const r = Math.floor(idx / 4),
          c = idx % 4;
        p.setAttribute(
          "fill",
          causal && c > r ? "url(#hatch)" : "var(--paper-subtle)"
        );
      });
      cells.forEach(c => c.setAttribute("opacity", 0));
      oF.forEach(f => f.setAttribute("width", 0));
      oR.forEach(r => r.classList.remove("is-done"));
      let reads = 0,
        writes = 0,
        donePairs = 0;
      for (let s = 0; s < stepIdx; s++) {
        const [i, j] = S[s];
        reads += 2 * BLK;
        if (j === 0) reads += BLK;
        const idx = i * 4 + j;
        pmap[idx].setAttribute("fill", "oklch(88% .07 260)");
        const lastJ = causal ? i : NB - 1;
        if (j === lastJ) {
          writes += BLK;
          oR[i].classList.add("is-done");
          oF[i].setAttribute("width", 104);
        }
        donePairs++;
      }
      if (stepIdx > 0 && stepIdx <= total) {
        const [i, j] = S[stepIdx - 1];
        qR[i].classList.add("is-hot");
        kR[j].classList.add("is-hot");
        vR[j].classList.add("is-hot");
        eQ[i].classList.remove("is-dim");
        eK[j].classList.remove("is-dim");
        eV[j].classList.remove("is-dim");
        eO[i].classList.remove("is-dim");
        for (let r = 0; r < BR; r++)
          for (let c = 0; c < BR; c++)
            cells[r * BR + c].setAttribute(
              "opacity",
              (0.3 + 0.7 * hash01(i * 4 + j, r * 4 + c)).toFixed(2)
            );
        const lastJ = causal ? i : NB - 1;
        if (!oR[i].classList.contains("is-done"))
          oF[i].setAttribute("width", (104 * (j + 1)) / (lastJ + 1));
        stepTxt.textContent =
          "第 " +
          stepIdx +
          " 步 · S" +
          SUB[i] +
          SUB[j] +
          " = Q" +
          SUB[i] +
          "K" +
          SUB[j] +
          "ᵀ → online softmax → 并入 O" +
          SUB[i];
        stepTxt2.textContent =
          "HBM 读：K" +
          SUB[j] +
          "、V" +
          SUB[j] +
          " 各 32 B" +
          (j === 0 ? "，Q" + SUB[i] + " 32 B" : "") +
          " · S" +
          SUB[i] +
          SUB[j] +
          " 留在 SRAM，不落盘";
      } else {
        stepTxt.textContent = "准备就绪 —— 按播放或步进";
        stepTxt2.textContent = "Q 块在外层循环，K／V 块在内层循环";
      }
      $("#t3-read").textContent = fmt(reads);
      $("#t3-read-sub").textContent = "Q/K/V 块 · " + reads / BLK + " 块次";
      $("#t3-write").textContent = fmt(writes);
      $("#t3-spill").textContent = "0 B";
      const stdSpill = (SPILL_STD * donePairs) / Math.max(1, total);
      $("#t3-spill").nextElementSibling.textContent =
        "标准实现此刻 S/P 已落盘 ≈ " + fmt(stdSpill);
      $("#t3-std").textContent = fmt(STD_TOTAL);
    }
    function stop() {
      if (timer) {
        clearInterval(timer);
        timer = null;
      }
      $("#t3-play").textContent = "▶ 播放";
    }
    function next() {
      const total = seq().length;
      if (stepIdx >= total) {
        stop();
        return;
      }
      stepIdx++;
      apply();
      if (stepIdx >= total) stop();
    }
    $("#t3-play").addEventListener(
      "click",
      () => {
        if (timer) {
          stop();
          return;
        }
        if (stepIdx >= seq().length) {
          stepIdx = 0;
          apply();
        }
        $("#t3-play").textContent = "⏸ 暂停";
        next();
        timer = setInterval(next, reduceMotion ? 10 : 950);
      },
      signal
    );
    $("#t3-step").addEventListener(
      "click",
      () => {
        stop();
        if (stepIdx >= seq().length) stepIdx = 0;
        next();
      },
      signal
    );
    $("#t3-reset").addEventListener(
      "click",
      () => {
        stop();
        stepIdx = 0;
        apply();
      },
      signal
    );
    $("#t3-full").addEventListener(
      "click",
      () => {
        causal = false;
        stop();
        stepIdx = 0;
        $("#t3-full").classList.add("is-on");
        $("#t3-causal").classList.remove("is-on");
        apply();
      },
      signal
    );
    $("#t3-causal").addEventListener(
      "click",
      () => {
        causal = true;
        stop();
        stepIdx = 0;
        $("#t3-causal").classList.add("is-on");
        $("#t3-full").classList.remove("is-on");
        apply();
      },
      signal
    );
    apply();
  })();
  /* ================= TOOL 04 · ONLINE SOFTMAX ================= */
  (function () {
    const X = [2.1, -0.5, 1.3, 0.8, 3.2, -1.0, 0.4, 2.6];
    const V = [
      [0.5, -0.2],
      [1.1, 0.3],
      [-0.4, 0.9],
      [0.2, 0.6],
      [-0.8, 0.1],
      [0.7, -0.5],
      [0.3, 1.2],
      [-0.1, 0.4],
    ];
    const BLK1 = [0, 1, 2, 3],
      BLK2 = [4, 5, 6, 7];
    let step = 0;
    const f2 = x => (Math.round(x * 100) / 100).toFixed(2);
    const f4 = x => (Math.round(x * 1e4) / 1e4).toFixed(4);
    const tbody = document.querySelector("#t4-tab tbody");
    tbody.innerHTML = X.map(
      (x, k) =>
        '<tr data-k="' +
        k +
        '"><td class="l">k' +
        (k + 1) +
        (k < 4 ? " · 块1" : " · 块2") +
        "</td><td>" +
        f2(x) +
        '</td><td class="ev">—</td><td class="pv">—</td></tr>'
    ).join("");
    const rows = [...tbody.children];
    /* 精确对照 */
    const mx = Math.max(...X);
    const ex = X.map(x => Math.exp(x - mx));
    const lExact = ex.reduce((a, b) => a + b, 0);
    const pExact = ex.map(e => e / lExact);
    const oExact = [0, 1].map(c =>
      pExact.reduce((a, p, k) => a + p * V[k][c], 0)
    );
    $("#t4-exact").textContent = "[" + oExact.map(f4).join(", ") + "]";
    function apply() {
      let m = -Infinity,
        l = 0,
        O = [0, 0],
        msg = "";
      if (step >= 1) {
        const m1 = Math.max(...BLK1.map(k => X[k]));
        const e1 = BLK1.map(k => Math.exp(X[k] - m1));
        m = m1;
        l = e1.reduce((a, b) => a + b, 0);
        O = [0, 1].map(c => e1.reduce((a, e, t) => a + e * V[BLK1[t]][c], 0));
        msg =
          "块 1：m = " +
          f2(m) +
          "，ℓ = " +
          f4(l) +
          " —— 此刻还不知道 3.20 的存在";
      }
      if (step >= 2) {
        const mOld = m,
          m2 = Math.max(m, ...BLK2.map(k => X[k]));
        const alpha = Math.exp(mOld - m2);
        const e2 = BLK2.map(k => Math.exp(X[k] - m2));
        l = l * alpha + e2.reduce((a, b) => a + b, 0);
        O = [0, 1].map(
          c => O[c] * alpha + e2.reduce((a, e, t) => a + e * V[BLK2[t]][c], 0)
        );
        m = m2;
        msg =
          "块 2 出现 3.20：历史整体乘 α = e^(2.10−3.20) = " +
          f4(alpha) +
          " 再并入";
      }
      let probs = null;
      if (step >= 3) {
        O = O.map(o => o / l);
        probs = X.map(x => Math.exp(x - m) / l);
        msg = "收尾：O ← O/ℓ —— 与直接 softmax 完全一致";
      }
      rows.forEach((tr, k) => {
        const inB1 = BLK1.includes(k),
          seen = step >= 3 || (step >= 1 && inB1) || step >= 2;
        tr.className =
          step >= 2 && BLK2.includes(k)
            ? "in-block2"
            : step >= 1 && inB1
              ? "in-block"
              : "";
        tr.querySelector(".ev").textContent = seen
          ? f4(Math.exp(X[k] - m))
          : "—";
        tr.querySelector(".pv").textContent = probs
          ? f4(probs[k])
          : seen
            ? f4(Math.exp(X[k] - m)) + " ~"
            : "—";
      });
      $("#t4-m").textContent = step === 0 ? "−∞" : f2(m);
      $("#t4-m-sub").textContent =
        step < 2
          ? "块 1 的局部最大值"
          : m > 2.1
            ? "被块 2 的 3.20 刷新"
            : "保持不变";
      $("#t4-l").textContent = f4(l);
      $("#t4-o").textContent = "[" + O.map(f4).join(", ") + "]";
      $("#t4-o-sub").textContent =
        step < 3 ? "未归一 · 最后除以 ℓ" : "已归一 · 即最终输出行";
      ["#t4-vm", "#t4-vl", "#t4-vo"].forEach(s => $(s).classList.remove("hot"));
      if (step === 1 || step === 2) $("#t4-vm").classList.add("hot");
      if (step >= 2) $("#t4-vl").classList.add("hot");
      if (step === 3) $("#t4-vo").classList.add("hot");
      if (step >= 2)
        $("#t4-online").textContent = "[" + O.map(f4).join(", ") + "]";
      else $("#t4-online").textContent = "—";
      if (step >= 3) {
        const diff = Math.max(...O.map((o, c) => Math.abs(o - oExact[c])));
        $("#t4-diff").textContent =
          "与对照差值 = " + diff.toExponential(1) + "（浮点舍入量级）";
      } else $("#t4-diff").textContent = "差值 —";
      $("#t4-msg").textContent =
        step === 0 ? "初始：m = −∞，ℓ = 0，O = 0" : msg;
      $("#t4-prog").value = step;
      $("#t4-step-lbl").textContent =
        "第 " +
        step +
        " 步 · " +
        ["初始", "处理块 1", "处理块 2 · 触发 rescale", "归一收尾"][step];
      $("#t4-next").textContent = step >= 3 ? "已完成" : "下一步";
      $("#t4-next").disabled = step >= 3;
    }
    $("#t4-next").addEventListener(
      "click",
      () => {
        if (step < 3) {
          step++;
          apply();
        }
      },
      signal
    );
    $("#t4-reset").addEventListener(
      "click",
      () => {
        step = 0;
        apply();
      },
      signal
    );
    apply();
  })();
  /* ================= TOOL 05 · 存 or 重算 ================= */
  (function () {
    const nIn = $("#t5-n"),
      d = 64;
    function calc() {
      const N = Math.pow(2, +nIn.value);
      const store = 4 * N * N,
        recomp = 2 * N * d + 4 * N;
      $("#t5-n-out").textContent = N.toLocaleString();
      $("#t5-store").textContent = fmtBytes(store);
      $("#t5-recomp").textContent = fmtBytes(recomp);
      $("#t5-ratio").textContent = (store / recomp).toFixed(0) + "×";
      const maxL = Math.log10(store),
        minL = Math.log10(1024);
      const w = v =>
        Math.max(2, ((Math.log10(v) - minL) / (maxL - minL)) * 100);
      $("#t5-bars").innerHTML =
        '<div class="flowbar"><span class="lbl">存 S＋P · 4N²</span><div class="trk"><div class="fil" style="width:' +
        w(store) +
        '%;background:var(--orange)"></div></div><span class="val">' +
        fmtBytes(store) +
        "</span></div>" +
        '<div class="flowbar"><span class="lbl">只存 O＋L · O(Nd)</span><div class="trk"><div class="fil" style="width:' +
        w(recomp) +
        '%"></div></div><span class="val">' +
        fmtBytes(recomp) +
        "</span></div>" +
        '<div class="flowbar"><span class="lbl" style="color:var(--ink)">对数刻度 · d = 64 · FP16</span><div class="trk" style="background:transparent"></div><span class="val"></span></div>';
    }
    nIn.addEventListener("input", calc, signal);
    calc();
  })();
  /* ================= TOOL 06 · 基准面板 ================= */
  (function () {
    const TABS = [
      {
        src: "来源：FlashAttention 论文 Fig.2（arXiv:2205.14135）· GPT-2 medium · seq 1024 · d=64 · 16 头 · batch 64 · A100 · 前向+反向",
        groups: [
          {
            title: "HBM 读写（GB · 越短越好）",
            rows: [
              { lbl: "标准注意力", v: 40.3, txt: "40.3 GB", cls: "f-std" },
              { lbl: "FlashAttention", v: 4.4, txt: "4.4 GB · ↓9.2×", cls: "" },
            ],
            maxV: 44,
          },
          {
            title: "运行时间（ms · 越短越好）",
            rows: [
              { lbl: "标准注意力", v: 41.7, txt: "41.7 ms", cls: "f-std" },
              {
                lbl: "FlashAttention",
                v: 7.3,
                txt: "7.3 ms · 5.7× 更快",
                cls: "",
              },
            ],
            maxV: 44,
          },
          {
            title: "计算量（GFLOP · 重算的代价）",
            rows: [
              { lbl: "标准注意力", v: 66.6, txt: "66.6", cls: "f-std" },
              {
                lbl: "FlashAttention",
                v: 75.2,
                txt: "75.2 · +13%",
                cls: "f-o",
              },
            ],
            maxV: 80,
          },
        ],
      },
      {
        src: "来源：arXiv:2307.08691 §4 · arXiv:2407.08608 摘要 · 峰值为稠密理论值。2026 年，同一思想已适配第四代硬件：FlashAttention-4 在 Blackwell B200 上官方报告约 1605 TFLOP/s BF16（71% 利用率）。",
        groups: [
          {
            title: "注意力前向吞吐（TFLOP/s · 越长越好）",
            rows: [
              {
                lbl: "FA1 · A100",
                sub: "2022 · 312 峰值",
                v: 125,
                txt: "78–125 · 25–40%",
                cls: "f-t",
              },
              {
                lbl: "FA2 · A100",
                sub: "2023 · 312 峰值",
                v: 230,
                txt: "230 · 73%",
                cls: "",
              },
              {
                lbl: "FA2 · H100",
                sub: "沿用 Ampere 指令",
                v: 346,
                txt: "≈346 · 仅 35%",
                cls: "f-std",
              },
              {
                lbl: "FA3 · H100 · FP16",
                sub: "2024 · 989 峰值",
                v: 740,
                txt: "740 · 75%",
                cls: "f-p",
              },
              {
                lbl: "FA3 · H100 · FP8",
                sub: "1979 峰值",
                v: 1200,
                txt: "≈1200 · ~60%",
                cls: "f-o",
              },
            ],
            maxV: 2000,
          },
        ],
      },
      {
        src: "来源：arXiv:2205.14135 §4（BERT-large · MLPerf 1.1 提交记录；GPT-2 · OpenWebText）· arXiv:2307.08691 表 1（GPT 2.7B · seq 8K · 8×A100 80GB）",
        groups: [
          {
            title: "BERT-large 训练至目标准确率（分钟 · seq 512）",
            rows: [
              {
                lbl: "NVIDIA MLPerf 1.1",
                v: 20.0,
                txt: "20.0 min",
                cls: "f-std",
              },
              {
                lbl: "FlashAttention",
                v: 17.4,
                txt: "17.4 min · +15%",
                cls: "",
              },
            ],
            maxV: 22,
          },
          {
            title: "GPT-2 端到端训练吞吐（相对倍数 · seq 1K）",
            rows: [
              { lbl: "vs HuggingFace", v: 1.0, txt: "1.0×", cls: "f-std" },
              {
                lbl: "vs Megatron-LM",
                v: 1.76,
                txt: "1.7× 基线更快",
                cls: "f-t",
              },
              { lbl: "FlashAttention", v: 3.0, txt: "3.0×", cls: "" },
            ],
            maxV: 3.4,
          },
          {
            title: "GPT 2.7B · seq 8K 训练吞吐（TFLOP/s）",
            rows: [
              { lbl: "无 FlashAttention", v: 80, txt: "80", cls: "f-std" },
              { lbl: "FlashAttention-1", v: 175, txt: "175", cls: "f-t" },
              {
                lbl: "FlashAttention-2",
                v: 225,
                txt: "225 · 72% MFU · 2.8×",
                cls: "",
              },
            ],
            maxV: 250,
          },
        ],
      },
    ];
    const box = $("#bench-bars"),
      src = $("#bench-src");
    function render(t) {
      const tab = TABS[t];
      box.innerHTML = tab.groups
        .map(
          g =>
            '<div style="margin-bottom:18px"><div class="kicker" style="margin-bottom:10px">' +
            g.title +
            "</div>" +
            g.rows
              .map(
                r =>
                  '<div class="barrow"><span class="lbl">' +
                  r.lbl +
                  (r.sub ? "<small>" + r.sub + "</small>" : "") +
                  "</span>" +
                  '<div class="trk"><div class="fil ' +
                  (r.cls || "") +
                  '" style="width:' +
                  Math.max(1.2, (r.v / g.maxV) * 100) +
                  '%"></div></div>' +
                  '<span class="val"><b>' +
                  r.txt +
                  "</b></span></div>"
              )
              .join("") +
            "</div>"
        )
        .join("");
      src.textContent = tab.src;
    }
    document.querySelectorAll(".tabs button").forEach(b =>
      b.addEventListener(
        "click",
        () => {
          document
            .querySelectorAll(".tabs button")
            .forEach(x =>
              x.setAttribute("aria-selected", x === b ? "true" : "false")
            );
          render(+b.dataset.tab);
        },
        signal
      )
    );
    render(0);
  })();
  /* ================= TOOL 01 · 逐步对照实验 ================= */
  (function () {
    const SQ = Math.SQRT1_2; // 缩放因子 1/√d，d = 2
    const Q = [
      [1.0, 0.5],
      [0.2, 0.8],
      [0.9, -0.3],
      [0.4, 0.6],
    ];
    const K = [
      [0.8, 0.4],
      [0.5, -0.6],
      [1.0, 0.2],
      [0.3, 0.7],
    ];
    const V = [
      [1.0, 0.0],
      [0.5, 0.5],
      [-0.5, 1.0],
      [0.8, -0.2],
    ];
    const KT = K[0].map((_, c) => K.map(row => row[c])); // Kᵀ [2×4]
    const S = Q.map(q => K.map(k => (q[0] * k[0] + q[1] * k[1]) * SQ));
    const P = S.map(row => {
      const m = Math.max(...row);
      const e = row.map(x => Math.exp(x - m));
      const s = e.reduce((a, b) => a + b, 0);
      return e.map(x => x / s);
    });
    const O = P.map(p =>
      [0, 1].map(c => p.reduce((a, x, j) => a + x * V[j][c], 0))
    );
    const f2 = x => {
      const v = Math.round(x * 100) / 100,
        a = Math.abs(v).toFixed(2);
      return v < 0 ? "−" + a : a;
    };
    const f3 = x => {
      const v = Math.round(x * 1000) / 1000,
        a = Math.abs(v).toFixed(3);
      return v < 0 ? "−" + a : a;
    };
    const PAIRS = [
      [0, 0],
      [0, 1],
      [1, 0],
      [1, 1],
    ];
    const SUB = ["₁", "₂"];
    /* 字节账（FP16 = 2 B/元素） */
    const NV_RW = [
      [0, 0],
      [32, 32],
      [64, 64],
      [112, 64],
      [112, 80],
      [112, 80],
    ];
    const NV_SPILL = [0, 32, 64, 64, 64, 64];
    const FL_RW = [
      [0, 0],
      [24, 0],
      [40, 0],
      [64, 0],
      [80, 0],
      [80, 16],
    ];
    function matHTML(id, rows, cols, label) {
      let cells = "";
      for (let r = 0; r < rows; r++)
        for (let c = 0; c < cols; c++)
          cells +=
            '<div class="mcell emp" id="' +
            id +
            "-" +
            r +
            "-" +
            c +
            '">·</div>';
      return (
        '<div class="mwrap"><span>' +
        label +
        '</span><div class="mgrid" style="grid-template-columns:repeat(' +
        cols +
        ',auto)">' +
        cells +
        "</div></div>"
      );
    }
    function fillMat(id, M, cls) {
      M.forEach((row, r) =>
        row.forEach((v, c) => {
          const e = document.getElementById(id + "-" + r + "-" + c);
          e.textContent = f2(v);
          e.className = "mcell" + (cls ? " " + cls : "");
        })
      );
    }
    function dimMat(id, rows, cols) {
      for (let r = 0; r < rows; r++)
        for (let c = 0; c < cols; c++) {
          const e = document.getElementById(id + "-" + r + "-" + c);
          e.textContent = "·";
          e.className = "mcell emp";
        }
    }
    function setCls(id, rows, cols, cls) {
      for (let r = 0; r < rows; r++)
        for (let c = 0; c < cols; c++) {
          const e = document.getElementById(id + "-" + r + "-" + c);
          if (e.textContent !== "·") e.className = "mcell " + cls;
        }
    }
    function countersHTML(p) {
      return [
        ["read", "HBM 读"],
        ["write", "HBM 写"],
        ["spill", "S·P 落盘"],
        ["total", "合计"],
      ]
        .map(
          kv =>
            '<div class="counter"><span>' +
            kv[1] +
            '</span><b id="' +
            p +
            "-" +
            kv[0] +
            '">0 B</b></div>'
        )
        .join("");
    }
    /* flash 在线合并模拟（前 upto 个块对） */
    function flashSim(upto) {
      const m = [-1 / 0, -1 / 0, -1 / 0, -1 / 0],
        l = [0, 0, 0, 0],
        OA = [
          [0, 0],
          [0, 0],
          [0, 0],
          [0, 0],
        ];
      let alpha = [1, 1, 1, 1],
        tile = null;
      for (let s = 0; s < upto; s++) {
        const i = PAIRS[s][0],
          j = PAIRS[s][1],
          rows = [i * 2, i * 2 + 1],
          cols = [j * 2, j * 2 + 1];
        alpha = [1, 1, 1, 1];
        for (const r of rows) {
          const t = [S[r][cols[0]], S[r][cols[1]]];
          const mNew = Math.max(m[r], t[0], t[1]);
          alpha[r] = Math.exp(m[r] - mNew);
          l[r] =
            l[r] * alpha[r] + t.reduce((a, x) => a + Math.exp(x - mNew), 0);
          for (const c of [0, 1])
            OA[r][c] =
              OA[r][c] * alpha[r] +
              t.reduce(
                (a, x, tt) => a + Math.exp(x - mNew) * V[cols[tt]][c],
                0
              );
          m[r] = mNew;
        }
        tile = rows.map(r => cols.map(c => S[r][c]));
      }
      return { m, l, OA, alpha, tile };
    }
    /* ---- 构建两条车道 ---- */
    $("#lane-naive").innerHTML =
      '<div class="lane-head"><b>标准注意力 · NAIVE</b><span class="io-legend"><i class="in"></i>输入 · 读入<i class="out"></i>输出 · 写入</span><span class="chip purple" id="nv-chip">S/P 落盘 0 B</span></div>' +
      '<div class="replay-bar lane-bar">' +
      '<button type="button" class="btn" id="nv-reset">重置</button>' +
      '<button type="button" class="btn" id="nv-prev">上一步</button>' +
      '<button type="button" class="btn primary" id="nv-play">▶ 播放</button>' +
      '<button type="button" class="btn" id="nv-next">下一步</button>' +
      '<progress id="nv-prog" max="5" value="0"></progress>' +
      '<output id="nv-count">0 / 5</output>' +
      "</div>" +
      '<div class="lane-body">' +
      '<div class="hbm"><div class="hbm-title"><span>HBM · 显存驻留</span><b id="nv-hbm-b">48 B</b></div>' +
      '<div class="hbm-slots">' +
      ["Q", "K", "V", "S", "P", "O"]
        .map(x => '<span class="slot" id="nv-slot-' + x + '">' + x + "</span>")
        .join("") +
      "</div></div>" +
      '<div class="stage">' +
      matHTML("nvQ", 4, 2, "Q [4×2]") +
      '<span class="arw">·</span>' +
      matHTML("nvKt", 2, 4, "Kᵀ [2×4]") +
      '<span class="arw go">→</span>' +
      matHTML("nvS", 4, 4, "S [4×4]") +
      matHTML("nvP", 4, 4, "P [4×4]") +
      '<span class="arw">·</span>' +
      matHTML("nvV", 4, 2, "V [4×2]") +
      '<span class="arw go">→</span>' +
      matHTML("nvO", 4, 2, "O [4×2]") +
      "</div>" +
      '<div class="lane-msg" id="nv-msg"></div>' +
      '<div class="counters">' +
      countersHTML("nv") +
      "</div>" +
      "</div>";
    $("#lane-flash").innerHTML =
      '<div class="lane-head"><b>FLASHATTENTION · 块 = 2</b><span class="chip ok">S/P 落盘恒为 0</span></div>' +
      '<div class="replay-bar lane-bar">' +
      '<button type="button" class="btn" id="fl-reset">重置</button>' +
      '<button type="button" class="btn" id="fl-prev">上一步</button>' +
      '<button type="button" class="btn" id="fl-play">▶ 播放</button>' +
      '<button type="button" class="btn" id="fl-next">下一步</button>' +
      '<progress id="fl-prog" max="5" value="0"></progress>' +
      '<output id="fl-count">0 / 5</output>' +
      "</div>" +
      '<div class="lane-body">' +
      '<div class="hbm"><div class="hbm-title"><span>HBM · 显存驻留</span><b id="fl-hbm-b">48 B</b></div>' +
      '<div class="hbm-slots">' +
      '<span class="slot" id="fl-slot-Q">Q</span><span class="slot" id="fl-slot-K">K</span><span class="slot" id="fl-slot-V">V</span>' +
      '<span class="slot ban">S ✗</span><span class="slot ban">P ✗</span><span class="slot" id="fl-slot-O">O</span></div></div>' +
      '<div class="stage">' +
      matHTML("flQ0", 2, 2, "Q₁ · 行0–1") +
      matHTML("flQ1", 2, 2, "Q₂ · 行2–3") +
      matHTML("flK0", 2, 2, "K₁ · 列0–1") +
      matHTML("flK1", 2, 2, "K₂ · 列2–3") +
      matHTML("flV0", 2, 2, "V₁") +
      matHTML("flV2", 2, 2, "V₂") +
      '<div class="rowbrk"></div>' +
      '<div class="srambox"><span>SRAM · 算完即弃</span>' +
      matHTML("flT", 2, 2, "Sᵢⱼ 小块") +
      '<div class="mlrow"><div class="mlcard" id="fl-m">m = [—, —]</div><div class="mlcard" id="fl-l">ℓ = [—, —]</div><div class="mlcard" id="fl-a">α = [—, —]</div></div></div>' +
      matHTML("flO", 4, 2, "O · 未归一") +
      "</div>" +
      '<div class="lane-msg" id="fl-msg"></div>' +
      '<div class="counters">' +
      countersHTML("fl") +
      "</div>" +
      "</div>";
    const NV_MSG = [
      "准备：Q、K、V 驻留 HBM（共 48 B）。盯住 S、P 两个槽位——它们即将被反复读写。",
      "① 读 Q、K 上芯片（<u>+32 B</u>）→ 一次 GEMM 算出 <i>S = QKᵀ/√d</i>，16 个分数全部亮相 → 整张 S 写回 HBM（<u>+32 B</u>）",
      "② 读回 S（<u>+32 B</u>）→ 逐行 softmax 得 P（每行和 = 1.00）→ 整张 P 再写回 HBM（<u>+32 B</u>）",
      "③ 读回 P（<u>+32 B</u>）、读 V（<u>+16 B</u>）→ 第二次 GEMM：<i>O = P·V</i>",
      "④ 写 O 回 HBM（<u>+16 B</u>）——完成。S、P 那 <u>64 B</u> 的来回搬运与最终结果毫无关系",
      "✓ 完成：合计 <u>192 B</u>，其中 S/P 落盘 <u>64 B</u>。N=4096 时这两张矩阵的读写将达 128 MiB / 头 / 层（见 TOOL 02）",
    ];
    function flMsg(k, sim) {
      const rows =
        k >= 1 && k <= 4
          ? [PAIRS[k - 1][0] * 2, PAIRS[k - 1][0] * 2 + 1]
          : null;
      const ml = rws =>
        rws
          ? "m = [" +
            rws.map(r => f2(sim.m[r])).join(", ") +
            "] · ℓ = [" +
            rws.map(r => f3(sim.l[r])).join(", ") +
            "]"
          : "";
      switch (k) {
        case 0:
          return "准备：同一份 Q、K、V。S、P 槽位被禁用——它们永远不会出现在显存里。";
        case 1:
          return (
            "① 载入 Q₁、K₁、V₁（<s>+24 B</s>）→ SRAM 内算 S₁₁（2×2）→ " +
            ml(rows) +
            "，O 先在片上累加"
          );
        case 2:
          return (
            "② 载入 K₂、V₂（<s>+16 B</s>）→ S₁₂ 出现更大值：<i>m [0.71, 0.34] → [0.78, 0.44]</i>，历史 ×α = [" +
            sim.alpha.slice(0, 2).map(f2).join(", ") +
            "] 再并入"
          );
        case 3:
          return (
            "③ 换 Q 块：载入 Q₂、K₁、V₁（<s>+24 B</s>）→ SRAM 内 S₂₁ → " +
            ml(rows) +
            "（新的 Q 块，ℓ 从零起算）"
          );
        case 4:
          return (
            "④ 载入 K₂、V₂（<s>+16 B</s>）→ S₂₂ → 仅第 3 行刷新 m：<i>α = [" +
            sim.alpha.slice(2).map(f2).join(", ") +
            "]</i>——第 4 行无需 rescale"
          );
        case 5:
          return "⑤ 归一 <i>O ← O/ℓ</i>，写回 HBM（<s>+16 B</s>）。合计 <s>96 B</s>，S/P 落盘 0 B";
      }
    }
    function renderNaive(k) {
      fillMat("nvQ", Q);
      fillMat("nvKt", KT);
      fillMat("nvV", V);
      if (k >= 1) fillMat("nvS", S);
      else dimMat("nvS", 4, 4);
      if (k >= 2) fillMat("nvP", P);
      else dimMat("nvP", 4, 4);
      if (k >= 3) fillMat("nvO", O);
      else dimMat("nvO", 4, 2);
      if (k === 1) {
        setCls("nvQ", 4, 2, "hot-in");
        setCls("nvKt", 2, 4, "hot-in");
        setCls("nvS", 4, 4, "hot-out");
      }
      if (k === 2) {
        setCls("nvS", 4, 4, "hot-in");
        setCls("nvP", 4, 4, "hot-out");
      }
      if (k === 3) {
        setCls("nvP", 4, 4, "hot-in");
        setCls("nvV", 4, 2, "hot-in");
        setCls("nvO", 4, 2, "hot-out");
      }
      if (k === 4) {
        setCls("nvO", 4, 2, "hot-out");
      }
      if (k >= 5) {
        setCls("nvO", 4, 2, "done");
      }
      const on = (id, cls) => document.getElementById(id).classList.add(cls);
      ["Q", "K", "V", "S", "P", "O"].forEach(
        x => (document.getElementById("nv-slot-" + x).className = "slot")
      );
      if (k >= 1) {
        on("nv-slot-Q", "on");
        on("nv-slot-K", "on");
        on("nv-slot-V", "on");
        on("nv-slot-S", "on-s");
      }
      if (k >= 2) {
        on("nv-slot-P", "on-s");
      }
      if (k >= 4) {
        on("nv-slot-O", "on");
      }
      let hb = 48 + (k >= 1 ? 32 : 0) + (k >= 2 ? 32 : 0) + (k >= 4 ? 16 : 0);
      $("#nv-hbm-b").textContent = hb + " B";
      $("#nv-chip").textContent = "S/P 落盘 " + NV_SPILL[k] + " B";
      $("#nv-read").textContent = NV_RW[k][0] + " B";
      $("#nv-write").textContent = NV_RW[k][1] + " B";
      $("#nv-spill").textContent = NV_SPILL[k] + " B";
      $("#nv-spill").className = "orange";
      $("#nv-total").textContent = NV_RW[k][0] + NV_RW[k][1] + " B";
      $("#nv-msg").innerHTML = NV_MSG[k];
    }
    function renderFlash(k) {
      fillMat("flQ0", Q.slice(0, 2));
      fillMat("flQ1", Q.slice(2));
      fillMat("flK0", K.slice(0, 2));
      fillMat("flK1", K.slice(2));
      fillMat("flV0", V.slice(0, 2));
      fillMat("flV2", V.slice(2));
      const sim = flashSim(Math.min(k, 4));
      if (k >= 1) fillMat("flT", sim.tile, "sram");
      else dimMat("flT", 2, 2);
      if (k >= 1 && k <= 4) {
        const i = PAIRS[k - 1][0],
          j = PAIRS[k - 1][1];
        setCls("flQ" + i, 2, 2, "hot");
        setCls("flK" + j, 2, 2, "hot");
        setCls("flV" + (j === 0 ? 0 : 2), 2, 2, "hot");
        $("#fl-m").classList.add("hot");
        $("#fl-l").classList.add("hot");
        $("#fl-a").classList.add("hot");
      }
      /* m / ℓ / α 卡 */
      if (k === 0) {
        $("#fl-m").textContent = "m = [—, —]";
        $("#fl-l").textContent = "ℓ = [—, —]";
        $("#fl-a").textContent = "α = [—, —]";
      } else if (k <= 4) {
        const rws = [PAIRS[k - 1][0] * 2, PAIRS[k - 1][0] * 2 + 1];
        $("#fl-m").textContent =
          "m = [" + rws.map(r => f2(sim.m[r])).join(", ") + "]";
        $("#fl-l").textContent =
          "ℓ = [" + rws.map(r => f3(sim.l[r])).join(", ") + "]";
        $("#fl-a").textContent =
          "α = [" +
          rws
            .map(r => (sim.alpha[r] === 0 ? "0·新行" : f2(sim.alpha[r])))
            .join(", ") +
          "]";
      } else {
        $("#fl-m").textContent = "m 全行 = [" + sim.m.map(f2).join(", ") + "]";
        $("#fl-l").textContent = "ℓ 全行 = [" + sim.l.map(f3).join(", ") + "]";
        $("#fl-a").textContent = "α = [—, —]";
      }
      /* O 累加器 */
      if (k >= 1 && k < 5) {
        const iMax = PAIRS[k - 1][0];
        for (let r = 0; r < 4; r++)
          for (let c = 0; c < 2; c++) {
            const e = document.getElementById("flO-" + r + "-" + c);
            if (r <= iMax * 2 + 1) {
              e.textContent = f3(sim.OA[r][c]);
              e.className = "mcell " + (r >= iMax * 2 ? "hot" : "sram");
            } else {
              e.textContent = "·";
              e.className = "mcell emp";
            }
          }
        document.querySelector(
          "#lane-flash .mwrap:last-child>span"
        ).textContent = "O · 未归一";
      } else if (k === 5) {
        const OF = sim.OA.map((row, r) => row.map(v => v / sim.l[r]));
        fillMat("flO", OF, "done");
        document.querySelector(
          "#lane-flash .mwrap:last-child>span"
        ).textContent = "O · 已归一 ✓";
      } else dimMat("flO", 4, 2);
      ["Q", "K", "V", "O"].forEach(
        x => (document.getElementById("fl-slot-" + x).className = "slot")
      );
      if (k >= 1) {
        ["Q", "K", "V"].forEach(x =>
          document.getElementById("fl-slot-" + x).classList.add("on")
        );
      }
      if (k === 5) document.getElementById("fl-slot-O").classList.add("on");
      $("#fl-hbm-b").textContent = 48 + (k === 5 ? 16 : 0) + " B";
      $("#fl-read").textContent = FL_RW[k][0] + " B";
      $("#fl-write").textContent = FL_RW[k][1] + " B";
      $("#fl-spill").textContent = "0 B";
      $("#fl-spill").className = "blue";
      $("#fl-total").textContent = FL_RW[k][0] + FL_RW[k][1] + " B";
      $("#fl-msg").innerHTML = flMsg(k, sim);
    }
    /* 对照条 */
    const simF = flashSim(4);
    const OF = simF.OA.map((row, r) => row.map(v => v / simF.l[r]));
    const DIFF = Math.max(
      ...OF.map((row, r) =>
        Math.max(...row.map((v, c) => Math.abs(v - O[r][c])))
      )
    );
    $("#duel-cmp").innerHTML =
      '<span class="t">殊途同归：两侧 O 最大差值 <b>' +
      (DIFF === 0 ? "0（逐位一致）" : DIFF.toExponential(1)) +
      "</b></span>" +
      '<div class="cmp-bars">' +
      '<div class="cmp-bar"><span>标准 · 192 B</span><div class="trk"><div class="fil" id="cmp-nv" style="background:var(--orange);width:0%"></div></div><span id="cmp-nv-v">0%</span></div>' +
      '<div class="cmp-bar"><span>Flash · 96 B</span><div class="trk"><div class="fil" id="cmp-fl" style="background:var(--blue);width:0%"></div></div><span id="cmp-fl-v">0%</span></div>' +
      "</div>" +
      '<span class="t" style="color:var(--ink)">教学尺寸 N=4 只差 2×；流量差 ∝ N² —— N=4096 时 S/P 读写 128 MiB vs 0 B</span>';
    function renderCmp(nvK, flK) {
      const nvT = NV_RW[nvK][0] + NV_RW[nvK][1],
        flT = FL_RW[flK][0] + FL_RW[flK][1];
      $("#cmp-nv").style.width = (nvT / 192) * 100 + "%";
      $("#cmp-fl").style.width = (flT / 192) * 100 + "%";
      $("#cmp-nv-v").textContent = Math.round((nvT / 192) * 100) + "%";
      $("#cmp-fl-v").textContent = Math.round((flT / 192) * 100) + "%";
    }
    /* 两条车道：各自独立的步进状态 */
    const lanes = [
      { p: "nv", step: 0, timer: null, render: renderNaive },
      { p: "fl", step: 0, timer: null, render: renderFlash },
    ];
    function paint(L) {
      L.render(L.step);
      renderCmp(lanes[0].step, lanes[1].step);
      $("#" + L.p + "-prog").value = L.step;
      $("#" + L.p + "-count").textContent = L.step + " / 5";
      $("#" + L.p + "-next").textContent = L.step >= 5 ? "已完成" : "下一步";
    }
    function stopLane(L) {
      if (L.timer) {
        clearInterval(L.timer);
        L.timer = null;
      }
      $("#" + L.p + "-play").textContent = "▶ 播放";
    }
    lanes.forEach(L => {
      $("#" + L.p + "-reset").addEventListener(
        "click",
        () => {
          stopLane(L);
          L.step = 0;
          paint(L);
        },
        signal
      );
      $("#" + L.p + "-prev").addEventListener(
        "click",
        () => {
          stopLane(L);
          if (L.step > 0) {
            L.step--;
            paint(L);
          }
        },
        signal
      );
      $("#" + L.p + "-next").addEventListener(
        "click",
        () => {
          stopLane(L);
          if (L.step < 5) {
            L.step++;
            paint(L);
          }
        },
        signal
      );
      $("#" + L.p + "-play").addEventListener(
        "click",
        () => {
          if (L.timer) {
            stopLane(L);
            return;
          }
          if (L.step >= 5) {
            L.step = 0;
            paint(L);
          }
          $("#" + L.p + "-play").textContent = "⏸ 暂停";
          if (L.step < 5) {
            L.step++;
            paint(L);
          }
          L.timer = setInterval(
            () => {
              if (L.step >= 5) {
                stopLane(L);
                return;
              }
              L.step++;
              paint(L);
            },
            reduceMotion ? 10 : 1700
          );
        },
        signal
      );
      paint(L);
    });
  })();

  // 文档目录交互抽到共享模块：FlashAttention 页同步 aria-label 与 aria-current，无初始 hash 深链恢复。
  const cleanupOutline = setupDocOutline(root, { signal, ariaLabel: true, ariaCurrent: true });

  return () => {
    controller.abort();
    cleanupOutline();
    intervals.forEach(timer => window.clearInterval(timer));
    intervals.clear();
    root.classList.remove("outline-docked");
  };
}

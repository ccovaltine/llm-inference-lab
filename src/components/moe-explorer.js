import { setupDocOutline } from "../lib/setupDocOutline.js";

export function initializeMoEExplorer(root) {
  "use strict";
  const byId = id => root.querySelector("#" + id);
  const controller = new AbortController();
  const signal = controller.signal;
  // 数值与表格排版工具：原先在下方两个交互区里各有一份重复实现，提升到此处共用。
  // fmt 两份仅在 NaN 输入的渲染上有差异（有限值完全一致），此处保留前一份的写法。
  const fmt = x => {
    const v = Math.abs(x) < 0.0005 ? 0 : x;
    return (v >= 0 ? "" : "−") + Math.abs(v).toFixed(3);
  };
  function table(headers, rows, selected = []) {
    return `<div class="table-wrap"><table><thead><tr>${headers.map(h => `<th>${h}</th>`).join("")}</tr></thead><tbody>${rows.map((r, i) => `<tr>${r.map((c, j) => `<td class="${selected.includes(i) && j > 0 ? "selected" : ""}">${c}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
  }
  const cleanupSingleToken = (() => {
    "use strict";
    const $ = s => root.querySelector(s),
      $$ = s => [...root.querySelectorAll(s)];
    const tokens = [
      { name: "小猫", x: [0.4, -0.2, 0.8, 0.6] },
      { name: "坐", x: [-0.3, 0.7, 0.2, 0.5] },
      { name: "在", x: [0.6, 0.1, -0.4, 0.9] },
      { name: "垫子", x: [0.2, 0.9, 0.5, -0.1] },
    ];
    const routerW = [
        [0.8, -0.3, 0.5, 0.1],
        [0.2, 0.7, -0.6, 0.4],
        [-0.4, 0.2, 0.3, -0.5],
        [0.1, 0.5, -0.2, 0.6],
      ],
      routerB = [0.1, -0.05, 0, 0.15];
    const round = x => Math.round(x * 100) / 100;
    const experts = Array.from({ length: 4 }, (_, e) => ({
      W1: Array.from({ length: 4 }, (_, r) =>
        Array.from({ length: 3 }, (_, c) =>
          round(Math.sin((e + 1) * 1.7 + r * 0.9 + c * 1.3) * 0.55)
        )
      ),
      b1: Array.from({ length: 3 }, (_, c) =>
        round(Math.cos(e + c * 0.8) * 0.06)
      ),
      W2: Array.from({ length: 3 }, (_, r) =>
        Array.from({ length: 4 }, (_, c) =>
          round(Math.cos((e + 1) * 1.1 + r * 0.8 - c * 0.7) * 0.48)
        )
      ),
      b2: Array.from({ length: 4 }, (_, c) =>
        round(Math.sin(e * 0.7 + c) * 0.04)
      ),
    }));
    const titles = [
      "读取当前 token 的隐藏向量",
      "Router 计算每个专家的 logits",
      "Softmax 打分并选择 Top-k",
      "Dispatch：把 x 发给选中专家",
      "专家内部执行独立 FFN",
      "Combine：按路由权重合并",
      "残差相加，回到 Transformer 主干",
    ];
    const codes = [
      "01 / INPUT",
      "02 / ROUTER",
      "03 / TOP-K",
      "04 / DISPATCH",
      "05 / EXPERT FFN",
      "06 / COMBINE",
      "07 / RESIDUAL",
    ];
    let state = {
      token: 0,
      k: 2,
      temp: 1,
      step: 0,
      expert: 2,
      playing: false,
      timer: null,
    };
    const vec = v => `[${v.map(fmt).join(", ")}]`;
    function matVec(x, W, b = []) {
      return W[0].map((_, c) =>
        x.reduce((s, v, r) => s + v * W[r][c], b[c] || 0)
      );
    }
    function gelu(x) {
      return (
        0.5 *
        x *
        (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x)))
      );
    }
    function calc() {
      const x = tokens[state.token].x,
        z = matVec(x, routerW, routerB);
      const scaled = z.map(v => v / state.temp),
        m = Math.max(...scaled),
        exps = scaled.map(v => Math.exp(v - m)),
        sum = exps.reduce((a, b) => a + b, 0),
        soft = exps.map(v => v / sum);
      const rank = soft.map((p, e) => ({ e, p })).sort((a, b) => b.p - a.p),
        selected = rank.slice(0, state.k).map(o => o.e),
        weightSum = selected.reduce((s, e) => s + soft[e], 0),
        weights = soft.map((p, e) =>
          selected.includes(e) ? p / weightSum : 0
        );
      const outputs = experts.map(ex => {
        const pre = matVec(x, ex.W1, ex.b1),
          h = pre.map(gelu),
          y = matVec(h, ex.W2, ex.b2);
        return { pre, h, y };
      });
      const y = Array.from({ length: 4 }, (_, j) =>
        selected.reduce((s, e) => s + weights[e] * outputs[e].y[j], 0)
      );
      return {
        x,
        z,
        soft,
        rank,
        selected,
        weights,
        outputs,
        y,
        residual: x.map((v, i) => v + y[i]),
      };
    }
    function matrix(name, M) {
      return `<div class="matrix-caption">${name}</div>${table(
        M[0].map((_, i) => `列 ${i}`),
        M.map(row => row.map(fmt))
      )}`;
    }
    function facts(items) {
      return items
        .map(
          ([a, b]) =>
            `<div class="fact"><span>${a}</span><code>${b}</code></div>`
        )
        .join("");
    }
    function expertTabs(c) {
      return `<div class="expert-tabs" aria-label="检查专家">${experts.map((_, e) => `<button data-inspect-expert="${e}" class="${state.expert === e ? "active" : ""}">E${e}${c.selected.includes(e) ? " · 激活" : ""}</button>`).join("")}</div>`;
    }
    function inspector(c) {
      const step = state.step,
        x = c.x;
      let body = "";
      if (step === 0)
        body = `<span class="kicker">TOKEN REPRESENTATION</span><h2>先只看一个 token。</h2><span class="short">x = ${vec(x)}</span><div class="formula">token “${tokens[state.token].name}” 已经完成 Attention 与 Pre-norm；<br>本原型从 MoE FFN 的输入 x 开始。</div>${facts(
          [
            ["输入形状", "[1,4] · 一行，不是整段序列"],
            ["每个元素", "一个隐藏特征通道"],
            ["主干", "x 会保留到最后做残差相加"],
          ]
        )}<p>真实模型中批量 token 可以并行经过 Router。这里抽出一行，是为了能逐元素看清全过程。</p>`;
      if (step === 1)
        body = `<span class="kicker">ROUTER / LEARNED AFFINITY</span><h2>Router 给四个专家打分。</h2><div class="formula">z = xWᵣ + bᵣ<br>Wᵣ [4,4] · bᵣ [4] → z [1,4]</div>${matrix("Wᵣ · Router 权重", routerW)}${table(
          ["专家", "逐元素点积", "logit"],
          c.z.map((v, e) => [
            `E${e}`,
            `${x.map((xv, j) => `(${fmt(xv)}×${fmt(routerW[j][e])})`).join(" + ")} + ${fmt(routerB[e])}`,
            fmt(v),
          ])
        )}<p>logit 还不是概率，也没有“最大者一定最适合所有 token”的全局含义；它只描述当前 x 与各 Router 向量的匹配程度。</p>`;
      if (step === 2)
        body = `<span class="kicker">SOFTMAX + SPARSE SELECTION</span><h2>先得到概率，再只保留 ${state.k} 路。</h2><div class="formula">sₑ = exp(zₑ / T) / Σⱼexp(zⱼ / T)，T=${state.temp.toFixed(1)}<br>pₑ = sₑ / Σⱼ∈TopK sⱼ（e∈TopK），其余为 0</div>${table(
          ["专家", "logit", "Softmax", "Top-k", "最终权重"],
          c.soft.map((p, e) => [
            `E${e}`,
            fmt(c.z[e]),
            fmt(p),
            c.selected.includes(e) ? "✓" : "—",
            fmt(c.weights[e]),
          ]),
          c.selected
        )}${facts([
          [
            "激活顺序",
            c.rank
              .filter(o => c.selected.includes(o.e))
              .map(o => `E${o.e}`)
              .join(" > "),
          ],
          ["权重和", fmt(c.weights.reduce((a, b) => a + b, 0))],
          ["未选专家", "p = 0，不执行后续 FFN"],
        ])}<p>温度越低，分布越尖锐；但它不会保证负载均衡。若所有 token 都偏爱同一专家，就会出现拥塞——下一章会专门观察这件事。</p>`;
      if (step === 3)
        body = `<span class="kicker">DISPATCH / TOKEN ROUTING</span><h2>把同一份 x 复制到选中专家。</h2><div class="formula">${c.selected.map(e => `x → E${e}，p=${fmt(c.weights[e])}`).join("<br>")}<br>每一路输入仍为 [1,4]</div>${table(
          ["专家", "是否收到 x", "输入", "输出用途"],
          experts.map((_, e) => [
            `E${e}`,
            c.selected.includes(e) ? "是" : "否",
            c.selected.includes(e) ? vec(x) : "—",
            c.selected.includes(e) ? "执行 FFN" : "跳过",
          ]),
          c.selected
        )}<p>Dispatch 是数据搬运与分批，不是数学上的向量切分。在分布式设备上，这一步通常对应跨卡通信；专家随后可把收到的多行 token 组成批量矩阵。</p>`;
      if (step === 4) {
        const e = state.expert,
          ex = experts[e],
          out = c.outputs[e],
          active = c.selected.includes(e);
        body = `<span class="kicker">EXPERT INTERNAL / E${e}</span><h2>专家 E${e}：4 → 3 → 4。</h2>${expertTabs(c)}${active ? `<div class="notice">E${e} 已被当前 token 激活，权重 p=${fmt(c.weights[e])}。</div>` : `<div class="notice">E${e} 未被当前 token 激活。下列数值用于检查“如果运行会得到什么”，但不会进入本 token 的 Combine。</div>`}<div class="formula">h = GELU(xW1 + b1) · 形状 [1,3]<br>yₑ = hW2 + b2 · 形状 [1,4]</div>${matrix(`E${e} · W1 [4,3]`, ex.W1)}${table(["xW1+b1", "GELU 后 h"], [out.pre.map(fmt), out.h.map(fmt)])}${matrix(`E${e} · W2 [3,4]`, ex.W2)}${facts(
          [
            ["输入 x", vec(x)],
            ["中间 h", vec(out.h)],
            ["输出 y" + e, vec(out.y)],
            ["参数计数", "12+3+12+4 = 31"],
          ]
        )}`;
      }
      if (step === 5)
        body = `<span class="kicker">WEIGHTED COMBINATION</span><h2>不是拼接，而是逐元素加权求和。</h2><div class="formula">y = Σₑ∈TopK pₑyₑ<br>每个 yₑ 都是 [1,4]，所以 y 仍是 [1,4]</div>${table(
          ["专家", "权重 pₑ", "专家输出 yₑ", "pₑyₑ"],
          c.selected.map(e => [
            `E${e}`,
            fmt(c.weights[e]),
            vec(c.outputs[e].y),
            vec(c.outputs[e].y.map(v => v * c.weights[e])),
          ])
        )}<div class="formula">${Array.from({ length: 4 }, (_, j) => `y[${j}] = ${c.selected.map(e => `${fmt(c.weights[e])}×${fmt(c.outputs[e].y[j])}`).join(" + ")} = ${fmt(c.y[j])}`).join("<br>")}</div>${facts(
          [
            ["合并结果", vec(c.y)],
            ["权重和", "1.000"],
            ["维度", "没有变成 k×4，而是仍为 4"],
          ]
        )}`;
      if (step === 6)
        body = `<span class="kicker">RESIDUAL / BACK TO MAINSTREAM</span><h2>MoE 只是 Transformer 的一个分支。</h2><div class="formula">输出 = x + y<br>${vec(x)}<br>+ ${vec(c.y)}<br>= ${vec(c.residual)}</div>${facts(
          [
            ["MoE 输入", vec(x)],
            ["MoE 分支输出", vec(c.y)],
            ["残差后输出", vec(c.residual)],
            ["后续", "进入下一子层或下一层"],
          ]
        )}<p>整个过程中隐藏维始终是 4。改变的是“由哪些专家处理这一行”，不是张量主干的基本形状。</p>`;
      $("#inspector").innerHTML = body;
      $$("[data-inspect-expert]").forEach(b =>
        b.addEventListener(
          "click",
          () => {
            state.expert = +b.dataset.inspectExpert;
            render();
          },
          { signal }
        )
      );
    }
    function updateNetwork(c) {
      $$("#network .node").forEach(n => {
        const s = +n.dataset.node;
        n.classList.toggle(
          "current",
          s === state.step &&
            (n.dataset.expert === undefined ||
              +n.dataset.expert === state.expert)
        );
        n.classList.toggle(
          "done",
          s < state.step && !n.classList.contains("expert")
        );
        if (n.dataset.expert !== undefined) {
          const e = +n.dataset.expert;
          n.classList.toggle("active", c.selected.includes(e));
          $("#expertWeight" + e).textContent = c.selected.includes(e)
            ? `p = ${fmt(c.weights[e])}`
            : "未激活，本次跳过";
        }
      });
      $$("#network .expert-op").forEach(n => {
        const e = +n.dataset.expert,
          active = c.selected.includes(e),
          inspect = state.step === 4 && state.expert === e;
        n.classList.toggle("active", active);
        n.classList.toggle("current-op", inspect);
        const show = active || inspect;
        $("#expertPre" + e).textContent = show
          ? vec(c.outputs[e].pre)
          : "未参与本次计算";
        $("#expertH" + e).textContent = show
          ? `h = ${vec(c.outputs[e].h)}`
          : "h [1,3]";
        $("#expertY" + e).textContent = show
          ? vec(c.outputs[e].y)
          : "未参与加权合并";
      });
      $$("#network .expert-frame").forEach(n =>
        n.classList.toggle(
          "active-frame",
          c.selected.includes(+n.dataset.frameExpert)
        )
      );
      $$("#network .edge").forEach(p => {
        let active = false,
          current = false;
        if (p.dataset.routeExpert !== undefined) {
          const e = +p.dataset.routeExpert;
          active = c.selected.includes(e);
          current = active && (state.step === 3 || state.step === 4);
        } else if (p.dataset.outputExpert !== undefined) {
          const e = +p.dataset.outputExpert;
          active = c.selected.includes(e);
          current = active && state.step === 5;
        } else if (p.dataset.internalExpert !== undefined) {
          const e = +p.dataset.internalExpert;
          active = c.selected.includes(e);
          current = active && state.step === 4;
        } else current = +p.dataset.step === state.step - 1;
        p.classList.toggle("selected-route", active);
        p.classList.toggle("current-route", current);
      });
      $("#topkSvg").textContent =
        `选择 ${state.k} 个专家：${c.selected.map(e => "E" + e).join(" / ")}`;
      $("#networkCode").textContent = `TOP-${state.k} / 4 EXPERTS`;
    }
    function render() {
      const c = calc();
      $("#stepBadge").textContent = `STEP 0${state.step + 1} / 07`;
      $("#stepTitle").textContent = titles[state.step];
      $("#inspectCode").textContent = codes[state.step];
      $("#timeline").value = state.step;
      $("#timelineOut").value = `${state.step + 1} / 7`;
      $("#prev").disabled = state.step === 0;
      $("#next").disabled = state.step === 6;
      $("#next").textContent = state.step === 6 ? "已到残差输出" : "下一步";
      $("#play").textContent = state.playing ? "暂停播放" : "自动播放";
      $("#topK").value = state.k;
      $("#temperature").value = state.temp;
      $("#temperatureOut").value = state.temp.toFixed(1);
      $$("#tokenTabs button").forEach(b =>
        b.classList.toggle("active", +b.dataset.token === state.token)
      );
      const perExpert = 31;
      $("#totalParams").textContent = 4 * perExpert;
      $("#activeParams").textContent = state.k * perExpert;
      $("#activeFormula").textContent = `Top-${state.k} × 31（另有 Router 20）`;
      $("#skippedExperts").textContent = `${4 - state.k} / 4`;
      updateNetwork(c);
      inspector(c);
    }
    function stop() {
      state.playing = false;
      if (state.timer) {
        clearInterval(state.timer);
        state.timer = null;
      }
      $("#play").textContent = "自动播放";
    }
    function go(step) {
      state.step = Math.max(0, Math.min(6, step));
      render();
    }
    $$("#tokenTabs button").forEach(b =>
      b.addEventListener(
        "click",
        () => {
          state.token = +b.dataset.token;
          state.expert = calc().selected[0];
          render();
        },
        { signal }
      )
    );
    const resetSingleToken = () => {
      $("#reset").click();
    };
    $("#stepReset").addEventListener("click", resetSingleToken, { signal });
    $("#topK").addEventListener(
      "change",
      e => {
        state.k = +e.target.value;
        state.expert = calc().selected[0];
        render();
      },
      { signal }
    );
    $("#temperature").addEventListener(
      "input",
      e => {
        state.temp = +e.target.value;
        render();
      },
      { signal }
    );
    $("#prev").addEventListener(
      "click",
      () => {
        stop();
        go(state.step - 1);
      },
      { signal }
    );
    $("#next").addEventListener(
      "click",
      () => {
        stop();
        go(state.step + 1);
      },
      { signal }
    );
    $("#timeline").addEventListener(
      "input",
      e => {
        stop();
        go(+e.target.value);
      },
      { signal }
    );
    $("#play").addEventListener(
      "click",
      () => {
        if (state.playing) {
          stop();
          return;
        }
        if (state.step === 6) state.step = 0;
        state.playing = true;
        render();
        state.timer = setInterval(() => {
          if (state.step >= 6) {
            stop();
            render();
            return;
          }
          go(state.step + 1);
        }, 1800);
      },
      { signal }
    );
    $("#reset").addEventListener(
      "click",
      () => {
        stop();
        state = {
          token: 0,
          k: 2,
          temp: 1,
          step: 0,
          expert: 2,
          playing: false,
          timer: null,
        };
        render();
      },
      { signal }
    );
    $$("#network [data-node]").forEach(n => {
      const activate = () => {
        stop();
        if (n.dataset.expert !== undefined) state.expert = +n.dataset.expert;
        go(+n.dataset.node);
      };
      n.addEventListener("click", activate, { signal });
      n.addEventListener(
        "keydown",
        e => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            activate();
          }
        },
        { signal }
      );
    });
    render();
    requestAnimationFrame(() => {
      const sc = $(".network-scroll");
      if (sc.scrollWidth > sc.clientWidth)
        sc.scrollLeft = (sc.scrollWidth - sc.clientWidth) / 2;
    });
    return stop;
  })();

  (() => {
    "use strict";
    const moduleRoot = root.querySelector("#routeModule"),
      $ = s => moduleRoot.querySelector(s),
      $$ = s => [...moduleRoot.querySelectorAll(s)];
    const tokenNames = ["小猫", "坐", "在", "垫子", "上", "然后", "追", "蝴蝶"];
    let state = {
      E: 4,
      k: 1,
      factor: 1,
      dropless: false,
      scenario: "spread",
      seed: 0,
      token: 0,
      expert: 0,
      alpha: 0.01,
    };
    function calc() {
      const T = tokenNames.length,
        E = state.E,
        logits = [],
        probs = [];
      for (let t = 0; t < T; t++) {
        const row = [];
        for (let e = 0; e < E; e++) {
          let v =
            Math.sin((t + 1) * (e + 2) * 1.13 + state.seed * 0.77) * 0.65 +
            Math.cos(t * 0.7 - e * 0.9) * 0.25;
          if (state.scenario === "skew")
            v += e === 0 ? 2.25 : e === 1 ? 0.55 : 0;
          if (state.scenario === "spread")
            v += Math.sin(t * 1.9 + e * 2.4) * 1.25;
          if (state.scenario === "flat") v *= 0.12;
          row.push(v);
        }
        logits.push(row);
        const m = Math.max(...row),
          ex = row.map(v => Math.exp(v - m)),
          sum = ex.reduce((a, b) => a + b, 0);
        probs.push(ex.map(v => v / sum));
      }
      const capacity = Math.ceil((T * state.k * state.factor) / E),
        loads = Array(E).fill(0),
        assignments = [];
      for (let t = 0; t < T; t++) {
        const rank = probs[t]
            .map((p, e) => ({ e, p }))
            .sort((a, b) => b.p - a.p || a.e - b.e),
          accepted = [],
          dropped = [],
          original = new Set(rank.slice(0, state.k).map(o => o.e));
        for (const o of rank.slice(0, state.k)) {
          if (state.dropless || loads[o.e] < capacity) {
            accepted.push({ e: o.e, p: o.p, redirected: false });
            loads[o.e]++;
          } else dropped.push(o);
        }
        if (!state.dropless && dropped.length) {
          for (const o of rank.slice(state.k)) {
            if (accepted.length >= state.k) break;
            if (loads[o.e] < capacity) {
              accepted.push({ e: o.e, p: o.p, redirected: true });
              loads[o.e]++;
              dropped.shift();
            }
          }
        }
        const weightSum = accepted.reduce((s, o) => s + o.p, 0);
        accepted.forEach(o => (o.w = weightSum ? o.p / weightSum : 0));
        assignments.push({ rank, original, accepted, dropped });
      }
      const f = loads.map(v => v / (T * state.k)),
        P = Array.from(
          { length: E },
          (_, e) => probs.reduce((s, row) => s + row[e], 0) / T
        ),
        aux = E * f.reduce((s, v, e) => s + v * P[e], 0);
      const overflowed = assignments.reduce(
          (sum, a) =>
            sum +
            [...a.original].filter(e => !a.accepted.some(o => o.e === e))
              .length,
          0
        ),
        redirected = assignments.reduce(
          (sum, a) => sum + a.accepted.filter(o => o.redirected).length,
          0
        );
      return {
        logits,
        probs,
        capacity,
        loads,
        assignments,
        f,
        P,
        aux,
        overflowed,
        redirected,
        requested: T * state.k,
        accepted: loads.reduce((a, b) => a + b, 0),
      };
    }
    function heatmap(c) {
      const T = tokenNames.length,
        E = state.E;
      let gridStyle = `grid-template-columns:110px repeat(${E},minmax(62px,1fr))`;
      let content =
        "<span></span>" +
        Array.from(
          { length: E },
          (_, e) =>
            `<span class="heat-label expert">E${e}<br>${c.loads[e]} / ${state.dropless ? "∞" : c.capacity}</span>`
        ).join("");
      for (let t = 0; t < T; t++) {
        content += `<button class="heat-token ${state.token === t ? "active" : ""}" data-token="${t}">${t} · ${tokenNames[t]}</button>`;
        const a = c.assignments[t];
        for (let e = 0; e < E; e++) {
          const accepted = a.accepted.find(o => o.e === e),
            rank = a.rank.findIndex(o => o.e === e),
            isOriginal = a.original.has(e),
            dropped = isOriginal && !accepted,
            status = accepted
              ? accepted.redirected
                ? "redirected"
                : "accepted"
              : dropped
                ? "dropped"
                : "off",
            statusText = {
              accepted: "实际接受",
              redirected: "改派接受",
              dropped: "容量溢出，未执行",
              off: "未选择",
            }[status],
            mark = accepted
              ? accepted.redirected
                ? "↪"
                : `${rank + 1}`
              : dropped
                ? "×"
                : "";
          content += `<button class="heat-cell ${status}" style="--cell-bg:oklch(${98 - c.probs[t][e] * 22}% ${0.004 + c.probs[t][e] * 0.12} 267)" data-token="${t}" data-expert="${e}" data-status="${status}" title="${statusText}" aria-label="token ${tokenNames[t]} 到专家 ${e}，Softmax 概率 ${fmt(c.probs[t][e])}，${statusText}">${fmt(c.probs[t][e])}<span class="mark">${mark}</span></button>`;
        }
      }
      $("#route-heatmap").setAttribute("style", gridStyle);
      $("#route-heatmap").innerHTML = content;
      $$("#route-heatmap [data-token]").forEach(b =>
        b.addEventListener(
          "click",
          () => {
            state.token = +b.dataset.token;
            if (b.dataset.expert !== undefined)
              state.expert = +b.dataset.expert;
            render();
          },
          { signal }
        )
      );
      $("#route-heatCode").textContent = `8 × ${E}`;
    }
    function inspector(c) {
      const t = state.token,
        a = c.assignments[t],
        accepted = a.accepted;
      $("#route-inspectCode").textContent = `TOKEN ${t} / E${state.expert}`;
      $("#route-inspector").innerHTML =
        `<span class="kicker">SELECTED TOKEN</span><h2>${t} · “${tokenNames[t]}” 去了哪里？</h2><span class="short">${accepted.length ? accepted.map(o => `E${o.e} · w=${fmt(o.w)}`).join(" + ") : "MoE 分支为空"}</span><div class="formula">原始 Top-${state.k}：${a.rank
          .slice(0, state.k)
          .map(o => "E" + o.e)
          .join(
            " / "
          )}<br>实际接受：${accepted.length ? accepted.map(o => "E" + o.e + (o.redirected ? "（改派）" : "")).join(" / ") : "无"}<br>${accepted.length ? "接受分支权重重新归一化" : "y = 0；残差输出 = x + 0"}</div>${table(
          ["专家", "概率", "原始优先级", "实际状态"],
          c.probs[t].map((p, e) => {
            const ac = accepted.find(o => o.e === e),
              rank = a.rank.findIndex(o => o.e === e);
            return [
              `E${e}`,
              fmt(p),
              rank < state.k ? rank + 1 : "—",
              ac
                ? ac.redirected
                  ? "改派接受"
                  : "已接受"
                : a.original.has(e)
                  ? "容量溢出"
                  : "未选择",
            ];
          }),
          accepted.map(o => o.e)
        )}${accepted.length ? `<div class="notice">Combine 使用接受后的权重：${accepted.map(o => `p(E${o.e})=${fmt(o.p)}，w=${fmt(o.w)}`).join("；")}。</div>` : `<div class="notice">所有请求分支都未接受。本教学模型令 MoE 分支输出为零，因此不会阻断残差主干。</div>`}<div class="fact"><span>容量规则</span><code>${state.dropless ? "Dropless · 不限制" : `C = ceil(8×${state.k}×${state.factor}/${state.E}) = ${c.capacity}`}</code></div><div class="fact"><span>当前检查</span><code>E${state.expert} · 全批次 ${c.loads[state.expert]} 条接受分支</code></div><p>概率高只代表 Router 偏好；容量满员时，它仍可能无法执行。改派分支使用新专家的原始概率重新归一化。</p>`;
    }
    function flow(c) {
      const E = state.E,
        tokenY = t => 48 + t * 70,
        expertY = e => 48 + e * (540 / Math.max(1, E - 1));
      let paths = "",
        nodes = "";
      for (let t = 0; t < 8; t++) {
        const a = c.assignments[t];
        for (const o of a.rank.slice(0, state.k)) {
          const accepted = a.accepted.find(x => x.e === o.e),
            status = accepted ? "accepted" : "dropped",
            weight = accepted ? accepted.w : 0,
            focus = state.token === t,
            width = accepted
              ? focus
                ? 2 + weight * 5
                : 1 + weight * 2
              : focus
                ? 2
                : 1.2,
            d = accepted
              ? `M150 ${tokenY(t)} C310 ${tokenY(t)},430 ${expertY(o.e)},570 ${expertY(o.e)}`
              : `M150 ${tokenY(t)} C300 ${tokenY(t)},400 ${expertY(o.e)},545 ${expertY(o.e)}`;
          paths += `<path class="route ${status} ${focus ? "focus" : ""}" data-token="${t}" data-expert="${o.e}" data-status="${status}" style="stroke-width:${width}" d="${d}"/>`;
        }
        for (const o of a.accepted.filter(x => x.redirected)) {
          const focus = state.token === t,
            d = `M150 ${tokenY(t)} C310 ${tokenY(t)},430 ${expertY(o.e)},570 ${expertY(o.e)}`;
          paths += `<path class="route redirected ${focus ? "focus" : ""}" data-token="${t}" data-expert="${o.e}" data-status="redirected" style="stroke-width:${focus ? 2 + o.w * 5 : 1 + o.w * 2}" d="${d}"/>`;
        }
      }
      for (let t = 0; t < 8; t++)
        nodes += `<g><rect class="token-box ${state.token === t ? "active" : ""}" x="25" y="${tokenY(t) - 23}" width="125" height="46"/><text x="87" y="${tokenY(t) - 2}" text-anchor="middle">${t} · ${tokenNames[t]}</text><text class="sub" x="87" y="${tokenY(t) + 15}" text-anchor="middle">token ${t}</text></g>`;
      for (let e = 0; e < E; e++)
        nodes += `<g><rect class="expert-box" x="570" y="${expertY(e) - 26}" width="160" height="52"/><text x="650" y="${expertY(e) - 3}" text-anchor="middle">Expert ${e}</text><text class="sub" x="650" y="${expertY(e) + 16}" text-anchor="middle">load ${c.loads[e]} / ${state.dropless ? "∞" : c.capacity}</text></g>`;
      $("#route-flow").innerHTML = paths + nodes;
      $("#route-flowCode").textContent =
        `${c.accepted} EXECUTING · ${c.redirected} REDIRECTED · ${c.requested - c.accepted} UNASSIGNED`;
    }
    function loads(c) {
      $("#route-loadList").innerHTML = c.loads
        .map((v, e) => {
          const ratio = state.dropless
            ? Math.min(100, (v / (8 * state.k)) * 100)
            : Math.min(100, (v / c.capacity) * 100);
          return `<div class="load-row"><strong>Expert ${e}</strong><span class="load-track"><i class="${!state.dropless && v >= c.capacity && v > 0 ? "over" : ""}" style="width:${ratio}%"></i></span><code>${v} / ${state.dropless ? "∞" : c.capacity}</code></div>`;
        })
        .join("");
    }
    function balance(c) {
      $("#route-auxLoss").textContent = c.aux.toFixed(3);
      $("#route-totalLoss").textContent = (1.84 + state.alpha * c.aux).toFixed(
        3
      );
      $("#route-alphaOut").value = state.alpha.toFixed(2);
      $("#route-balanceBars").innerHTML = ["f", "P"]
        .map(
          kind =>
            `<div class="balance-row ${kind === "P" ? "importance" : ""}"><span>${kind}ₑ</span><div>${c[kind === "f" ? "f" : "P"].map((v, e) => `<div class="balance-row"><span>${kind}${e}</span><span class="balance-track"><i style="width:${Math.min(100, v * 100)}%"></i></span><code>${v.toFixed(2)}</code></div>`).join("")}</div><span></span></div>`
        )
        .join("");
    }
    function render() {
      const c = calc();
      $("#route-scenario").value = state.scenario;
      $("#route-expertCount").value = state.E;
      $("#route-topK").value = state.k;
      $("#route-capacity").value = state.factor;
      $("#route-capacity").disabled = state.dropless;
      $("#route-dropless").checked = state.dropless;
      $("#route-alpha").value = state.alpha;
      $("#route-capacityOut").textContent = state.dropless
        ? "∞ · dropless"
        : `${c.capacity} branches`;
      $("#route-acceptedOut").textContent = `${c.accepted} / ${c.requested}`;
      $("#route-overflowOut").textContent =
        `${c.overflowed}（改派 ${c.redirected}）`;
      $("#route-droppedOut").textContent = `${c.requested - c.accepted}`;
      $("#route-maxLoadOut").textContent =
        `${Math.max(...c.loads)} / ${state.dropless ? "∞" : c.capacity}`;
      heatmap(c);
      inspector(c);
      flow(c);
      loads(c);
      balance(c);
    }
    $("#route-scenario").addEventListener(
      "change",
      e => {
        state.scenario = e.target.value;
        render();
      },
      { signal }
    );
    $("#route-expertCount").addEventListener(
      "change",
      e => {
        state.E = +e.target.value;
        state.expert = Math.min(state.expert, state.E - 1);
        render();
      },
      { signal }
    );
    $("#route-topK").addEventListener(
      "change",
      e => {
        state.k = +e.target.value;
        render();
      },
      { signal }
    );
    $("#route-capacity").addEventListener(
      "change",
      e => {
        state.factor = +e.target.value;
        render();
      },
      { signal }
    );
    $("#route-dropless").addEventListener(
      "change",
      e => {
        state.dropless = e.target.checked;
        render();
      },
      { signal }
    );
    $("#route-alpha").addEventListener(
      "input",
      e => {
        state.alpha = +e.target.value;
        render();
      },
      { signal }
    );
    $("#route-newBatch").addEventListener(
      "click",
      () => {
        state.seed++;
        render();
      },
      { signal }
    );
    $("#route-reset").addEventListener(
      "click",
      () => {
        state = {
          E: 4,
          k: 1,
          factor: 1,
          dropless: false,
          scenario: "spread",
          seed: 0,
          token: 0,
          expert: 0,
          alpha: 0.01,
        };
        render();
      },
      { signal }
    );
    render();
  })();

  (() => {
    "use strict";
    const moduleRoot = root.querySelector("#costModule"),
      $ = s => moduleRoot.querySelector(s),
      $$ = s => [...moduleRoot.querySelectorAll(s)];
    let state = {
      E: 8,
      k: 2,
      d: 1024,
      ff: 4096,
      layers: 8,
      batch: 16,
      bytes: 2,
      focus: 0,
      order: "token",
    };
    const fmtInt = n => Math.round(n).toLocaleString("zh-CN");
    function big(n, unit = "") {
      const abs = Math.abs(n);
      if (abs >= 1e12) return (n / 1e12).toFixed(2) + " T" + unit;
      if (abs >= 1e9) return (n / 1e9).toFixed(2) + " B" + unit;
      if (abs >= 1e6) return (n / 1e6).toFixed(2) + " M" + unit;
      if (abs >= 1e3) return (n / 1e3).toFixed(2) + " K" + unit;
      return n.toFixed(0) + unit;
    }
    function bytes(n) {
      if (n >= 1024 ** 3) return (n / 1024 ** 3).toFixed(2) + " GiB";
      if (n >= 1024 ** 2) return (n / 1024 ** 2).toFixed(2) + " MiB";
      return (n / 1024).toFixed(2) + " KiB";
    }
    function flops(n) {
      if (n >= 1e12) return (n / 1e12).toFixed(2) + " TFLOPs";
      if (n >= 1e9) return (n / 1e9).toFixed(2) + " GFLOPs";
      return (n / 1e6).toFixed(2) + " MFLOPs";
    }
    function calc() {
      const expert = 3 * state.d * state.ff,
        router = state.d * state.E,
        dense = state.layers * expert,
        total = state.layers * (state.E * expert + router),
        active = state.k * expert + router,
        equalDense = state.k * expert,
        denseFlops = 2 * expert * state.layers * state.batch,
        moeFlops = 2 * active * state.layers * state.batch;
      return {
        expert,
        router,
        dense,
        total,
        active,
        equalDense,
        denseFlops,
        moeFlops,
        denseMemory: dense * state.bytes,
        totalMemory: total * state.bytes,
        activeMemory: active * state.bytes,
        ratio: (state.E * expert + router) / equalDense,
        computeRatio: active / expert,
      };
    }
    function kOptions() {
      const old = state.k;
      $("#cost-topK").innerHTML = Array.from(
        { length: Math.min(8, state.E) },
        (_, i) => `<option value="${i + 1}">${i + 1} 个活跃专家</option>`
      ).join("");
      state.k = Math.min(old, Math.min(8, state.E));
      $("#cost-topK").value = state.k;
    }
    function bar(label, dense, moe, formatter, active = false) {
      const max = Math.max(dense, moe),
        ratio = dense ? moe / dense : 0;
      return `<div class="ledger-row"><div class="ledger-head"><strong>${label}</strong><code>MoE / Dense = ${ratio.toFixed(2)}×</code></div><div class="bar-line"><span>Dense</span><span class="bar-track"><i style="width:${(dense / max) * 100}%"></i></span><code>${formatter(dense)}</code></div><div class="bar-line ${active ? "active" : "moe"}"><span>${active ? "MoE 活跃" : "MoE 总量"}</span><span class="bar-track"><i style="width:${(moe / max) * 100}%"></i></span><code>${formatter(moe)}</code></div></div>`;
    }
    function ledger(c) {
      $("#cost-ledger").innerHTML =
        bar("模型权重总量", c.dense, c.total, v => big(v)) +
        bar("权重占用估算", c.denseMemory, c.totalMemory, bytes) +
        bar(
          "单 token / 单层 FLOPs",
          2 * c.expert,
          2 * c.active,
          v => big(v),
          true
        ) +
        bar(
          `本批 ${state.batch} tokens 的 FLOPs`,
          c.denseFlops,
          c.moeFlops,
          v => big(v),
          true
        ) +
        `<p class="ledger-note">前两行比较“要保存什么”；后两行比较“实际算什么”。MoE 总量通常大于同尺寸 Dense，活跃计算则主要由 Top-k=${state.k} 决定。</p>`;
    }
    function inspector(c) {
      $("#cost-inspector").innerHTML =
        `<span class="kicker">CURRENT CALCULATION</span><h2>E${state.E} / Top-${state.k} / ${state.layers} 层</h2><span class="short">d_model ${state.d} / d_ff ${state.ff}</span><div class="formula">单专家参数 = 3·d·d_ff<br>= 3×${state.d}×${state.ff}<br>= ${fmtInt(c.expert)}<br><br>Router 参数 = d·E = ${fmtInt(c.router)}<br>MoE 活跃 / 层 = k·P_expert + P_router<br>= ${fmtInt(c.active)}</div><div class="fact"><span>三个矩阵</span><code>[d,d_ff] + [d,d_ff] + [d_ff,d]</code></div><div class="fact"><span>等计算 Dense</span><code>d_ff ≈ k·${state.ff} = ${state.k * state.ff}</code></div><div class="fact"><span>容量倍数</span><code>${fmtInt(state.E * c.expert + c.router)} / ${fmtInt(c.equalDense)} = ${c.ratio.toFixed(2)}×</code></div><div class="fact"><span>活跃计算比</span><code>MoE / 单专家 Dense = ${c.computeRatio.toFixed(2)}×</code></div><p>“等活跃计算容量倍数”使用 k 倍宽的 Dense FFN 作为对照，并保留 Router 参数，因此结果略高于直观的 E/k。</p><p>FLOPs 采用 2×活跃参数近似，忽略部分逐元素操作与实现差异；它适合建立数量级直觉，不适合代替性能分析。</p>`;
    }
    function experts(c) {
      $("#cost-expertShelf").innerHTML = Array.from(
        { length: state.E },
        (_, e) =>
          `<button class="expert ${e < state.k ? "active" : ""} ${state.focus === e ? "focus" : ""}" data-expert="${e}">E${e}<small>${e < state.k ? "ACTIVE" : "STORED"}</small></button>`
      ).join("");
      $$("[data-expert]").forEach(b =>
        b.addEventListener(
          "click",
          () => {
            state.focus = +b.dataset.expert;
            render();
          },
          { signal }
        )
      );
      $("#cost-shelfCode").textContent =
        `${state.E} EXPERTS / ${state.k} ACTIVE`;
      $("#cost-denseShape").textContent =
        `[${state.d} → ${state.ff} → ${state.d}]`;
      $("#cost-expertDetail").innerHTML =
        `<h3>Expert ${state.focus} ${state.focus < state.k ? "· 当前激活" : "· 仅保存"}</h3><div class="formula">W_gate [${state.d},${state.ff}]<br>W_up [${state.d},${state.ff}]<br>W_down [${state.ff},${state.d}]</div><div class="fact"><span>单专家参数</span><code>${big(c.expert)}</code></div><div class="fact"><span>当前 token</span><code>${state.focus < state.k ? "执行此专家 FFN" : "跳过此专家 FFN"}</code></div>`;
    }
    function assignments() {
      const list = [];
      for (let t = 0; t < 8; t++) {
        const used = new Set();
        for (let j = 0; j < state.k; j++) {
          let e = (t * 3 + j * 5 + 1) % state.E;
          while (used.has(e)) e = (e + 1) % state.E;
          used.add(e);
          list.push({ t, e, j });
        }
      }
      return list;
    }
    function queue() {
      const list = assignments(),
        counts = Array(state.E).fill(0);
      list.forEach(o => counts[o.e]++);
      const padded = counts.reduce((s, n) => s + Math.ceil(n / 4) * 4, 0);
      if (state.order === "expert") list.sort((a, b) => a.e - b.e || a.t - b.t);
      else list.sort((a, b) => a.t - b.t || a.j - b.j);
      let previous = null;
      $("#cost-queue").innerHTML = list
        .map(o => {
          const divider =
            state.order === "expert" && previous !== null && previous !== o.e
              ? '<span class="queue-divider" aria-hidden="true"></span>'
              : "";
          previous = o.e;
          return `${divider}<span class="queue-item">T${o.t} → E${o.e}<small>${state.order === "expert" ? `E${o.e} · ${counts[o.e]} rows` : `choice ${o.j + 1}`}</small></span>`;
        })
        .join("");
      $("#cost-queueBranches").textContent = list.length;
      $("#cost-queueSlots").textContent = padded;
      $("#cost-queuePadding").textContent = padded - list.length;
      $$("[data-order]").forEach(b =>
        b.classList.toggle("active", b.dataset.order === state.order)
      );
    }
    function render() {
      kOptions();
      const c = calc();
      $("#cost-experts").value = state.E;
      $("#cost-dModel").value = state.d;
      $("#cost-dFf").value = state.ff;
      $("#cost-layers").value = state.layers;
      $("#cost-batch").value = state.batch;
      $("#cost-bytes").value = state.bytes;
      $("#cost-totalWeight").textContent = bytes(c.totalMemory);
      $("#cost-totalWeightSub").textContent =
        `${big(c.total)} params · ${state.layers} layers`;
      $("#cost-activeWeight").textContent = bytes(c.activeMemory);
      $("#cost-capacityRatio").textContent = c.ratio.toFixed(2) + "×";
      $("#cost-batchFlops").textContent = flops(c.moeFlops);
      $("#cost-flopsSub").textContent = `Dense 对照 ${flops(c.denseFlops)}`;
      $("#cost-ledgerCode").textContent = `E${state.E} / K${state.k}`;
      ledger(c);
      inspector(c);
      experts(c);
      queue();
    }
    const presets = {
      toy: {
        E: 4,
        k: 2,
        d: 512,
        ff: 1024,
        layers: 8,
        batch: 16,
        bytes: 2,
        focus: 0,
        order: "token",
      },
      sparse: {
        E: 64,
        k: 2,
        d: 4096,
        ff: 4096,
        layers: 32,
        batch: 128,
        bytes: 2,
        focus: 0,
        order: "expert",
      },
      batch: {
        E: 8,
        k: 2,
        d: 2048,
        ff: 8192,
        layers: 16,
        batch: 8192,
        bytes: 1,
        focus: 0,
        order: "expert",
      },
    };
    $("#cost-experts").addEventListener(
      "change",
      e => {
        state.E = +e.target.value;
        state.focus = 0;
        render();
      },
      { signal }
    );
    $("#cost-topK").addEventListener(
      "change",
      e => {
        state.k = +e.target.value;
        render();
      },
      { signal }
    );
    $("#cost-dModel").addEventListener(
      "change",
      e => {
        state.d = +e.target.value;
        render();
      },
      { signal }
    );
    $("#cost-dFf").addEventListener(
      "change",
      e => {
        state.ff = +e.target.value;
        render();
      },
      { signal }
    );
    $("#cost-layers").addEventListener(
      "change",
      e => {
        state.layers = +e.target.value;
        render();
      },
      { signal }
    );
    $("#cost-batch").addEventListener(
      "change",
      e => {
        state.batch = +e.target.value;
        render();
      },
      { signal }
    );
    $("#cost-bytes").addEventListener(
      "change",
      e => {
        state.bytes = +e.target.value;
        render();
      },
      { signal }
    );
    $$("[data-order]").forEach(b =>
      b.addEventListener(
        "click",
        () => {
          state.order = b.dataset.order;
          queue();
        },
        { signal }
      )
    );
    $$("[data-preset]").forEach(b =>
      b.addEventListener(
        "click",
        () => {
          state = { ...presets[b.dataset.preset] };
          render();
        },
        { signal }
      )
    );
    $("#cost-reset").addEventListener(
      "click",
      () => {
        state = { ...presets.toy, E: 8, k: 2, d: 1024, ff: 4096 };
        render();
      },
      { signal }
    );
    render();
  })();

  // 文档目录交互抽到共享模块：MoE 页带初始 hash 深链恢复，无 aria-label / aria-current 同步。
  const cleanupOutline = setupDocOutline(root, { signal, hashRestore: true });
  return () => {
    controller.abort();
    cleanupSingleToken();
    cleanupOutline();
    root.classList.remove("outline-docked");
  };
}

# LLM 推理实验室

**把一个 token 的诞生过程，拆成可以点击、展开和逐项核对的真实计算。**

这是一个端到端交互式 Transformer 推理教学应用，以纯静态站点的形式发布。它不只展示算子名称或流程动画，而是围绕同一份输入，持续展示矩阵形状、模型参数、中间状态、注意力权重、Logits 与采样概率，并支持展开到单个输出元素的乘加过程。

- **GitHub Pages 静态演示**：<https://ccovaltine.github.io/llm-inference-lab/>
- **源码仓库**：<https://github.com/ccovaltine/llm-inference-lab>
- **技术栈**：React 19 · TypeScript · Vite（纯静态 SPA）

> 本项目使用确定性的微型教学模型，不调用云端大模型生成答案。页面中的矩阵、注意力权重和概率都来自浏览器内的真实前向计算。

## 页面与路由

六个页面均为真实路由，可直接深链、分享或刷新保持：

| 页面         | 路由         | 内容                                                         |
| ------------ | ------------ | ------------------------------------------------------------ |
| 实验台       | `/`          | 端到端推理：分词 → 两层 Transformer → Logits → 采样 → 自回归 |
| KV Cache     | `/kv`        | Prefill／Decode 回放与有无缓存双轨对照                       |
| 注意力网络   | `/attention` | MHA／GQA／MLA 可重构计算图                                   |
| FlashAttention | `/flash`   | 访存瓶颈、Tiling 分块与 Online Softmax 图解                  |
| MoE 网络     | `/moe`       | 路由、负载均衡与成本推演                                     |
| 概念速查     | `/concepts`  | Prefill／Decode／KV Cache 概念示意                           |

除实验台随首屏加载外，其余页面按需懒加载，空闲时自动预取。

## 在线体验与功能边界

本项目是纯静态教学演示站：没有任何服务端，不需要数据库、账户或密钥，全部计算（包括微型模型前向）都在浏览器本地完成，刷新与跨设备体验一致。子路由深链在 Pages 上由 `404.html` SPA 回退承载；历史遗留的 `/login` 链接会被重定向回实验台。

## 核心功能

### 1. 端到端推理实验台

- 输入上下文，依次观察分词、Token／Position Embedding、RMSNorm、因果自注意力、MLP、残差、LM Head、Softmax 与解码。
- **43 个真实计算节点**组成依赖网络，区分串行主干与并行计算，而不是把网络画成简单的步骤列表。
- 桌面三列分别呈现计算网络、输入到输出的数据流、算子内部与单元素数值展开。
- 点击任一 Head 分支，同步切换该 Head 的真实参数、中间结果与输出。
- 参数与超参数默认折叠在计算箭头中，需要时可展开检查。
- 点击输出元素，可查看实际乘法、加法、指数、归一化或查表过程，并定位对应权重行／列。
- 调整 Temperature 与 Top-k，观察候选 token、Logits 和概率分布的变化。
- 选择一个候选 token 回填上下文，继续下一轮自回归推理，直至 EOS 或长度上限。

### 2. KV Cache 生成回放

用一个共享注意力网络和两条执行路径解释缓存的作用：

- 从完整 **Prefill** 开始，逐轮推进 4 个 **Decode** 阶段，展示上下文、当前 token 与缓存状态如何变化。
- Prefill 完整展示所有提示 token 的 Q／K／V；Decode 展示当前一行 Q，以及本次注意力实际读取的全部历史与当前 K／V。
- 并列比较无缓存与有缓存路径：前者重算全部 K／V，后者只新算当前 K／V 并复用历史缓存；两轨的注意力数值保持一致。
- 点击 QKᵀ、Mask、Softmax 或 AV，可切换真实数值矩阵，并展开单个元素的点积、因果遮蔽、归一化或加权求和过程。
- 右侧合并式悬浮块包含①②③区域导航、上下文与步骤控制；折叠步骤后仍可跳转区域，推进 Decode 时保持当前阅读位置。
- 提供 KV Cache 显存滑块与计算公式，说明缓存随上下文和 Batch size 线性增长，以及为什么不缓存历史 Q 或完整注意力矩阵。

### 3. 可重构注意力网络

可以动态调整结构并观察计算图变化：

- **MHA**：每个 Query Head 使用独立的 K／V Head。
- **GQA**：多个 Query Head 按组共享 K／V，可调整 Query Head 数与 KV Group 数。
- **MLA**：展示共享低维 `cKV`、上游投影还原 K／V，以及共享解耦 RoPE Key 的教学计算路径。
- 保留 Q／K／V 分支、V 旁路、Concat、输出投影、MLP 与两处残差。
- 点击节点后，关联节点与从源到目标的完整路径按原语义颜色高亮。
- 支持维度标注、淡化其他 Head、手动缩放、自动适应与移动端布局。

### 4. FlashAttention 图解

按「为什么慢 → 怎么分块 → 实测差距」三段展开：

- **访存瓶颈**：同一张 N×N 注意力矩阵在 HBM 里走三个来回，与 SRAM 快一个数量级但只有约 20 MB 的内存层次对照。
- **Tiling 分块回放**：把 Q、K、V 切成块，让每个 S 小方块只活在 SRAM 里，逐步演示分块计算过程。
- **Online Softmax**：维护 m、ℓ、O 三个流动变量，新块到来时 rescale，不需要先算完整 softmax。
- **反向与实测**：反向传播重计算策略，以及三代内核的实测数据对照。

### 5. MoE 网络图解

从单 token 到整批 token，再到系统成本：

- **单 Token 计算**：MoE 在推理网络中的位置（注意力混合信息、MoE 做特征变换），Router 按 Top-k 分派专家并加权合并。
- **整批路由**：负载均衡损失如何鼓励「别只用一个专家」，容量与实际负载的区别，超载时丢弃的是专家分支。
- **成本推演**：专家分组执行顺序、参数很多不等于每个 token 都算很多、FLOPs 更低不保证 wall-clock 更快、总显存看全部专家。

### 6. 概念速查

- 用交互示意理解 Prefill、Decode、KV Cache 和自回归循环。

## 教学模型

核心实现位于 `src/lib/tinyTransformer.ts`。

| 配置            | 值                               |
| -------------- | -------------------------------- |
| 架构            | Decoder-only Transformer         |
| 层数            | 2                                |
| Attention Heads | 2                                |
| `d_model`       | 8                                |
| `d_ff`          | 16                               |
| 词表大小        | 29                               |
| 归一化          | Pre-norm RMSNorm                 |
| 注意力          | Causal Multi-Head Self-Attention |
| 权重            | 固定伪随机值，确定性可复现       |

**模型边界**：固定权重只用于解释计算过程，未经过真实语言任务训练，因此输出不具备实际语言能力。这里的重点不是生成质量，而是让每一个算子、形状和状态变化都可见、可查、可复算。MHA／GQA／MLA、KV Cache 回放、FlashAttention 与 MoE 页面各自使用独立的确定性教学数值，用于解释对应机制，不声称取自上述固定权重。

## 本地运行

### 环境要求

- Node.js 22（推荐）
- npm

### 安装与启动

```bash
git clone https://github.com/ccovaltine/llm-inference-lab.git
cd llm-inference-lab
npm ci
npm run dev
```

打开 <http://localhost:3000>。开发服务即纯 Vite，不需要任何环境变量或外部服务。

### 生产构建与预览

```bash
npm run check
npm test
npm run build
npm run preview
```

构建产物输出到 `dist/public`，可直接托管在任意静态文件服务上。

## GitHub Pages 静态演示

推送到 `main` 后，`.github/workflows/deploy-pages.yml` 会自动测试、构建并部署 Pages 静态演示：以 `/llm-inference-lab/` 为基路径构建，并生成 `404.html` 承载子路由深链回退。本地预览同一份构建：

```bash
npm run build:pages
npm run preview:pages
```

若 fork 后仓库名发生变化，需要同步修改 `package.json` 中 `build:pages` 的 `VITE_BASE_PATH`。

## 常用命令

| 命令                    | 用途                           |
| ----------------------- | ------------------------------ |
| `npm run dev`           | 启动 Vite 开发服务             |
| `npm run check`         | TypeScript 类型检查            |
| `npm test`              | 运行模型、算子与注意力网络测试 |
| `npm run build`         | 构建生产静态站点               |
| `npm run build:pages`   | 构建 GitHub Pages 静态演示     |
| `npm run preview:pages` | 按仓库子路径预览 Pages 构建    |
| `npm run lint`          | ESLint 检查                    |

## 项目结构

```text
.
├── public/                            # 静态资源
├── src/
│   ├── App.tsx                        # 路由表：/ 下 6 个子路由 + /login 重定向 + 404
│   ├── main.tsx                       # React 入口与基路径适配
│   ├── pages/
│   │   ├── Home.tsx                   # 布局壳：导航与全局实验状态
│   │   └── NotFound.tsx
│   ├── views/                         # 六个页面视图（按需懒加载，实验台除外）
│   │   ├── LabView.tsx                # 实验台首屏
│   │   ├── kv/                        # KV Cache 回放页
│   │   ├── attention/                 # 注意力网络页
│   │   ├── flash/                     # FlashAttention 图解页
│   │   ├── moe/                       # MoE 图解页
│   │   └── concepts/                  # 概念速查
│   ├── components/
│   │   ├── TransformerWorkbench.tsx   # 三列推理工作台
│   │   ├── InferenceVisual.tsx        # 矩阵、概率与结构可视化原件
│   │   └── ui/                        # button、card 基础件
│   └── lib/
│       ├── tinyTransformer.ts         # 可复现的微型 Transformer
│       ├── inferenceOperations.ts     # 43 个计算节点与逐元素定义
│       ├── attentionNetwork.ts        # 参数化注意力图布局与路径计算
│       ├── numeric.ts                 # softmax 等共享数值工具
│       └── setupDocOutline.js         # 长文页共享文档目录侧栏（flash/moe）
├── index.html
└── vite.config.ts
```

FlashAttention 与 MoE 是长文图解页：标记与交互逻辑以静态资产形式放在各自视图目录下（`*.html`／`*.js`），由 React 包装组件挂载并注入生命周期。

## 测试

```bash
npm run check
npm test
```

16 项测试覆盖微型模型前向（因果无未来泄漏、形状链、温度与 Top-k 重归一化）、43 节点算子链完整性，以及注意力网络图的连通性、mask 语义与布线几何。

## 实现说明

- 所有教学推理均在浏览器本地完成，不需要模型 API Key，也没有任何服务端。
- 路由使用 react-router 子路由；除实验台外按视图懒加载并空闲预取，首屏只包含实验台与其依赖。
- 普通构建保持 `/` 基路径；只有 Pages 构建使用仓库子路径。
- 本项目不依赖远程模型服务，固定权重使每次演示都可以复现和核对。

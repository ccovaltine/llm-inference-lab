# LLM 推理实验室

**把一个 token 的诞生过程，拆成可以点击、展开和逐项核对的真实计算。**

这是一个端到端交互式 Transformer 推理教学应用。它不只展示算子名称或流程动画，而是围绕同一份输入，持续展示矩阵形状、模型参数、中间状态、注意力权重、Logits 与采样概率，并支持展开到单个输出元素的乘加过程。

- **GitHub Pages 静态演示**：<https://ccovaltine.github.io/llm-inference-lab/>
- **源码仓库**：<https://github.com/ccovaltine/llm-inference-lab>
- **技术栈**：React 19 · TypeScript · Vite · Hono · tRPC · Drizzle ORM · MySQL

> 本项目使用确定性的微型教学模型，不调用云端大模型生成答案。页面中的矩阵、注意力权重和概率都来自浏览器内的真实前向计算。

## 在线体验与功能边界

| 运行方式     | 推理可视化 | KV Cache 回放 | MHA／GQA／MLA 网络 | Kimi 登录 | 服务端实验记录 |
| ------------ | ---------- | ------------- | ------------------ | --------- | -------------- |
| GitHub Pages | 支持       | 支持          | 支持               | 不提供    | 不提供         |
| 完整应用     | 支持       | 支持          | 支持               | 支持      | 支持           |

GitHub Pages 只托管静态文件，无法运行本项目的 Hono API、OAuth 回调或 MySQL 数据库。因此 Pages 版本会隐藏登录与保存入口；模型计算仍在本地浏览器中完整执行，不会用 `localStorage` 模拟服务端持久化。

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

在顶部「KV Cache」页面中，用一个共享注意力网络和两条执行路径解释缓存的作用：

- 从完整 **Prefill** 开始，逐轮推进 4 个 **Decode** 阶段，展示上下文、当前 token 与缓存状态如何变化。
- Prefill 完整展示所有提示 token 的 Q／K／V；Decode 展示当前一行 Q，以及本次注意力实际读取的全部历史与当前 K／V。
- 并列比较无缓存与有缓存路径：前者重算全部 K／V，后者只新算当前 K／V 并复用历史缓存；两轨的注意力数值保持一致。
- 点击 QKᵀ、Mask、Softmax 或 AV，可切换真实数值矩阵，并展开单个元素的点积、因果遮蔽、归一化或加权求和过程。
- 右侧合并式悬浮块包含①②③区域导航、上下文与步骤控制；折叠步骤后仍可跳转区域，推进 Decode 时保持当前阅读位置。
- 提供 KV Cache 显存滑块与计算公式，说明缓存随上下文和 Batch size 线性增长，以及为什么不缓存历史 Q 或完整注意力矩阵。

### 3. 可重构注意力网络

在顶部「注意力网络」页面中，可以动态调整结构并观察计算图变化：

- **MHA**：每个 Query Head 使用独立的 K／V Head。
- **GQA**：多个 Query Head 按组共享 K／V，可调整 Query Head 数与 KV Group 数。
- **MLA**：展示共享低维 `cKV`、上游投影还原 K／V，以及共享解耦 RoPE Key 的教学计算路径。
- 保留 Q／K／V 分支、V 旁路、Concat、输出投影、MLP 与两处残差。
- 点击节点后，关联节点与从源到目标的完整路径按原语义颜色高亮。
- 支持维度标注、淡化其他 Head、手动缩放、自动适应与移动端布局。

### 4. 概念速查与个人实验

- 用交互示意理解 Prefill、Decode、KV Cache 和自回归循环。
- 完整应用支持 Kimi 登录，并将输入、输出、Temperature 与 Top-k 保存到服务端。
- 实验记录通过鉴权 API 按用户隔离，可重新打开或删除。

## 教学模型

核心实现位于 `src/lib/tinyTransformer.ts`。

| 配置            | 值                               |
| --------------- | -------------------------------- |
| 架构            | Decoder-only Transformer         |
| 层数            | 2                                |
| Attention Heads | 2                                |
| `d_model`       | 8                                |
| `d_ff`          | 16                               |
| 词表大小        | 29                               |
| 归一化          | Pre-norm RMSNorm                 |
| 注意力          | Causal Multi-Head Self-Attention |
| 权重            | 固定伪随机值，确定性可复现       |

**模型边界**：固定权重只用于解释计算过程，未经过真实语言任务训练，因此输出不具备实际语言能力。这里的重点不是生成质量，而是让每一个算子、形状和状态变化都可见、可查、可复算。MHA／GQA／MLA 页面用于讲解结构差异，并不改变上述两层教学模型的真实前向权重。KV Cache 回放页使用另一组确定性教学 Q／K／V 数值，专门解释 Prefill、Decode 与缓存生命周期，不声称取自上述固定权重。

## 本地运行完整应用

### 环境要求

- Node.js 22（推荐）
- npm
- MySQL 8 或兼容数据库
- 如需登录：可用的 Kimi OAuth 应用配置

### 安装与配置

```bash
git clone https://github.com/ccovaltine/llm-inference-lab.git
cd llm-inference-lab
npm ci
cp .env.example .env
```

编辑 `.env`，至少配置数据库；需要登录时再补全 OAuth 参数：

```dotenv
APP_ID=your-app-id
APP_SECRET=your-long-random-secret
DATABASE_URL=mysql://user:password@localhost:3306/llm_inference_lab

VITE_KIMI_AUTH_URL=https://your-oauth-server.example.com
VITE_APP_ID=your-app-id
KIMI_AUTH_URL=https://your-oauth-server.example.com
KIMI_OPEN_URL=https://your-open-platform.example.com
OWNER_UNION_ID=
```

`.env` 已被 `.gitignore` 排除，不要提交真实凭据。`VITE_` 前缀变量会进入浏览器构建产物，其中不应放置服务端密钥。

### 初始化数据库并启动

```bash
npm run db:push
npm run dev
```

打开 <http://localhost:3000>。不配置 OAuth 时仍可使用全部本地推理可视化，只是无法登录和保存实验。

### 生产构建

```bash
npm run check
npm test
npm run build
npm start
```

生产服务由 `dist/boot.js` 启动，并托管 `dist/public` 中的前端构建产物。

## GitHub Pages 部署

Pages 构建会执行以下处理：

1. 使用 `/llm-inference-lab/` 作为 Vite `base`，避免项目页资源错误指向域名根目录。
2. 启用 `VITE_STATIC_DEMO`，关闭无后端环境下的账户请求与保存入口。
3. 输出到 `dist/public`，可生成 `404.html` 作为 SPA 回退。

首次使用时，在仓库 **Settings → Pages → Build and deployment** 中将 **Source** 设为 **GitHub Actions**。不要选择 `main / (root)`：那会直接发布源码目录的 `index.html`，浏览器无法把 `/src/main.tsx` 当作已编译应用运行。

在仓库新增 `.github/workflows/deploy-pages.yml`，即可在 `main` 更新后自动测试、构建和部署：

```yaml
name: Deploy GitHub Pages

on:
  push:
    branches: [main]
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: pages
  cancel-in-progress: false

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 22
      - run: npm install --no-audit --no-fund
      - run: npm run check && npm test
      - run: npm run build:pages
      - run: |
          touch dist/public/.nojekyll
          cp dist/public/index.html dist/public/404.html
      - uses: actions/configure-pages@v5
      - uses: actions/upload-pages-artifact@v3
        with:
          path: dist/public

  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - id: deployment
        uses: actions/deploy-pages@v4
```

本地验证同一份 Pages 构建：

```bash
npm run build:pages
npm run preview:pages -- --host 0.0.0.0
```

若 fork 后仓库名发生变化，需要同步修改 `package.json` 中 `build:pages` 的 `VITE_BASE_PATH`。

## 常用命令

| 命令                    | 用途                           |
| ----------------------- | ------------------------------ |
| `npm run dev`           | 启动 Vite 与 Hono 开发服务     |
| `npm run check`         | TypeScript 类型检查            |
| `npm test`              | 运行模型与算子测试             |
| `npm run build`         | 构建完整前端与生产服务         |
| `npm run build:pages`   | 构建 GitHub Pages 静态演示     |
| `npm run preview:pages` | 按仓库子路径预览 Pages 构建    |
| `npm start`             | 启动完整生产服务               |
| `npm run lint`          | ESLint 检查                    |
| `npm run db:push`       | 将 Drizzle Schema 推送到数据库 |
| `npm run db:generate`   | 生成数据库迁移                 |

## 项目结构

```text
.
├── api/                               # Hono、tRPC、鉴权与实验 API
├── contracts/                         # 前后端共享 API 类型
├── db/                                # Drizzle Schema 与数据库连接
├── public/                            # 静态资源
├── src/
│   ├── components/
│   │   ├── AttentionNetwork.tsx       # MHA／GQA／MLA 交互页面
│   │   ├── KVCacheReplay.tsx          # Prefill／Decode 与缓存双轨回放
│   │   ├── KVCacheReplay.css          # 回放页布局与合并式悬浮控制
│   │   ├── TransformerWorkbench.tsx   # 三列推理工作台
│   │   ├── InferenceVisual.tsx        # 矩阵、概率与结构可视化
│   │   └── Concepts.tsx               # Prefill／Decode／KV Cache
│   ├── lib/
│   │   ├── tinyTransformer.ts         # 可复现的微型 Transformer
│   │   ├── inferenceOperations.ts     # 43 个节点与逐元素计算定义
│   │   └── attentionNetwork.ts        # 参数化注意力图与路径计算
│   ├── pages/Home.tsx                 # 导航、实验台与个人记录
│   └── main.tsx                       # React 入口与子路径路由适配
├── index.html
└── vite.config.ts
```

## 测试与验证

```bash
npm run check
npm test
npm run build
npm run build:pages
```

当前测试覆盖微型模型前向计算、分词／解码约束，以及核心推理算子。Pages 构建还应检查：

- `dist/public/index.html` 中的 JS、CSS 与 favicon 均以 `/llm-inference-lab/` 开头。
- 页面直接打开时无 `/src/main.tsx` 请求，也无资源 404。
- 实验台、KV Cache 回放、注意力网络、概念速查和静态实验记录说明均可正常切换。
- KV Cache 页的合并式悬浮块在桌面与移动端均不越界，区域导航、步骤折叠和 Decode 滚动补偿可用。

## 实现说明

- 所有教学推理均在浏览器本地完成，不需要模型 API Key。
- 完整应用的服务端只负责账户与实验记录，不参与 Transformer 前向计算。
- 普通完整构建保持 `/` 基路径；只有 Pages 构建使用仓库子路径。
- 本项目不依赖远程模型服务，固定权重使每次演示都可以复现和核对。

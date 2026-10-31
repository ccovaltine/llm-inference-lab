# LLM 推理实验室

一个端到端交互式 Transformer 推理教学应用。用户可以输入上下文，逐阶段检查分词、Embedding、因果注意力、MLP、残差、LM Head、Top-k／温度采样和 token 回填，并保存实验记录。

## 教学模型

`src/lib/tinyTransformer.ts` 实现了一个确定性的 2 层 Decoder-only Transformer：2 个注意力头、d_model=8、d_ff=16、29 项词表。页面展示的矩阵、注意力权重和概率均由真实前向计算得到。权重为固定伪随机值，仅用于解释计算过程，不具备真实语言能力。

## 功能

- 43 个真实计算节点组成依赖网络，明确展示并行查表、双 Head 并行、各 Head 内 Q/K/V 独立投影、V 旁路，以及 X／R 主干跨过分支汇入残差的完整连线
- 左侧网络使用独立滚动窗口并自动定位当前节点；加宽的桌面三列分别展示计算网络、输入到输出的数据流、算子内部与单元素数值展开
- 大屏下右侧计算透视独立悬浮，超长公式与数值展开可在透视栏内部滚动，不随页面下划移出视口
- 点击任一 Head 分支可同步切换真实参数与输出；参数／超参数默认折叠在计算箭头中
- 点击任意输出元素可检查实际乘法、加法、指数、归一化或查表过程，并定位所用权重行／列
- 覆盖 Q/K/V、原始注意力分数、因果 mask、Softmax、Context、拼接投影、MLP 中间激活与 Logits
- 自回归 token 回填、停止条件、Prefill／Decode 与 KV Cache 说明
- 响应式桌面／手机界面
- Kimi 登录与个人实验记录持久化

## 运行

```bash
npm install
cp .env.example .env
```

在 `.env` 中填写自己的 MySQL 连接串、JWT 密钥与 Kimi OAuth 配置，然后初始化数据库并启动：

```bash
npm run db:push
npm run dev
```

开发服务位于 `http://localhost:3000`。核心 Transformer 推理演示可直接使用；实验保存与个人记录需要配置数据库和登录。自部署时需要自己的数据库和 OAuth 应用，不包含平台提供的云数据库或 Kimi 登录凭据。请勿提交 `.env`。

```bash
npm run check
npm run test
npm run build
npm start
```

## 结构

- `src/pages/Home.tsx`：实验台、导航与个人记录视图
- `src/components/TransformerWorkbench.tsx`：并行／串行依赖网络、三列数据流与计算透视、可折叠参数和单元素数值展开工作台
- `src/lib/inferenceOperations.ts`：43 个计算节点的输入、参数、输出和数值展开定义
- `src/components/InferenceVisual.tsx`：矩阵和概率可视化
- `src/components/Concepts.tsx`：Prefill／Decode 与 KV Cache 工具
- `api/experiment-router.ts`：受登录保护的实验 API
- `db/schema.ts`：用户和实验记录表

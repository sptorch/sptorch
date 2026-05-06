# SPTorch 开发路线图 v5.1 — 全栈 AI 生产力中枢

> 用 Rust 从零构建的工业级异构 AI 平台，**软件已达 90% 完成度，正在向全栈 IDE 生态演进**。
>
> **四大核心愿景（The Four Pillars）：**
> - **算力平权（AI Hadoop）**：以太网连接廉价机器，AllReduce 分布式训练，打破算力垄断
> - **硬件主权（Hardware Sovereignty）**：自研 DDR4 PCB，将大模型运行成本降低一个数量级，摆脱英伟达税
> - **生命主权（Live Evolution）**：EWC + 双缓冲，模型在业务中自我进化，拒绝"部署即巅峰"
> - **开发者主权（Developer Sovereignty）**：分层 IDE（SPTorch Studio），让业务人员拥有 AI，让硬件工程师指挥模型
>
> **能力验证（Showcase）：**
> - **Text2SQL 一体机**：生态中的首个生产级样板产品——单二进制交付（3.9MB），私有化部署，验证平台从训练到推理到交付的完整闭环能力

**核心突破**：`BACKEND_REGISTRY` 全局注册表为 **Tang 9k** 或 **自研 DDR4 NPU 载板** 提供了标准驱动插入点。SPTorch 不仅是框架，更是一套**工业流水线**——愿景是底座，IDE 分层是操作台，Text2SQL 是第一个生产级样板产品。

## 项目架构定位（统一术语）

- **平台层（母体）**：`SPTorch` 是全栈 AI 平台/框架体系（引擎 + 协议 + 硬件抽象 + IDE 生态）。
- **控制中枢层**：`SPTorch Studio` 是平台控制中枢，负责观测、编排、调试与交付。
- **产品层**：工业版 `Text2SQL` 是平台生态中的首个生产级训练产品（样板产品），用于验证平台能力闭环。

---

## SPTorch Studio — 多分层开发者 IDE 体系

> 不仅在做框架，更在构建**全栈 AI 生产力中枢**。让不同技术深度的开发者，在同一个物理设备（自研载板）和同一个软件协议（SPTorch Core）上协作。

### Level 1：业务 Appliance 层（面向：数据分析师 / 运维）

**开发体验**：No-Code / 极简 UI

| 功能 | 实现方式 | 对接模块 |
|------|----------|----------|
| **Schema 探测器** | 点击按钮，`sqlx` 自动扫描生产库，识别主键/外键/索引 | `text2sql` → `fetch_sqlite_schema` |
| **Appliance 封装器** | LoRA 权重 + text2sql 引擎 + axum 服务静态编译为单 `.bin`（4-10MB） | `serialize` + `cli-text2sql` |
| **实时进化监控** | 可视化 Loss 曲线，一键 Rollback 到上一个健康 Shadow Buffer | `live-evolution` → `TrainingMonitor` |

**用户画像**：中小企业数据分析师，不写代码，只需点击"连接数据库→提问→获得结果"。

### Level 2：逻辑编排层（面向：后端工程师 / 业务架构）

**开发体验**：Low-Code / 算子乐高

| 功能 | 实现方式 | 对接模块 |
|------|----------|----------|
| **RAG 可视化管道** | 拖拽配置 `build_prompt`，调整 `rank_tables` 权重系数 | `text2sql` → RAG pipeline |
| **SQL 约束编辑器** | 从 Schema 自动生成 `TokenTrie`，手动添加业务词白名单 | `nn` → `TokenTrie` |
| **Prompt 仿真器** | 通过 `mock-npu` 快速模拟生成结果，验证逻辑闭环 | `hal-ffi` + `mock-npu` |

**用户画像**：后端工程师，懂 SQL 和 API，不懂模型训练，通过配置而非编码定制 AI 行为。

### Level 3：引擎与算法层（面向：AI 算法工程师 / Rust 专家）

**开发体验**：Rust SDK / 计算图调试

| 功能 | 实现方式 | 对接模块 |
|------|----------|----------|
| **Autograd 节点浏览器** | 可视化拓扑图，每个 Node 的 shape/requires_grad/Fisher 热力图 | `core-autograd` + `core-ops` |
| **AMP 控制台** | 标记算子精度（BF16/F32），实时预览内存节省 | `core-tensor` → DType |
| **分布式拓扑管理** | 可视化集群，实时查看梯度聚合延迟和 Worker step 对齐 | `distributed` → gRPC |

**用户画像**：AI 算法工程师，需要深度定制模型结构、调试梯度、优化训练策略。

### Level 4：物理主权层（面向：硬件工程师 / FPGA & PCB 专家）

**开发体验**：FFI 驱动脚手架 / 硬件 Forge

| 功能 | 实现方式 | 对接模块 |
|------|----------|----------|
| **Forge 指令集映射器** | 将 `core-ops` 20 个算子映射为 FPGA OpCode，自动生成 HAL 驱动 | `hal-ffi` + `sptorch_hal.h` |
| **DDR4 物理视图** | 可视化载板 Bank 布局，指定 Tensor 存储位置 | `core-tensor` → `Storage::Device` |
| **DMA 性能剖析** | 实时监控串口/PCIe 带宽，分析 upload/copy_to_host 瓶颈 | `hal-ffi` → profiler |

**用户画像**：硬件工程师，设计 FPGA 逻辑和 PCB，通过 IDE 让自研硬件接入 SPTorch 生态。

### IDE 分层的商业价值

| 场景 | 使用层级 | 价值 |
|------|----------|------|
| **大公司** | L3 专家开发模型，L4 专家设计算力卡，通过 IDE 交付给业务部门 | 全栈自研 |
| **中小企业** | L1 Appliance 模板 + L2 微调 Prompt 即可上线 | 低成本 AI 化 |
| **个人开发者** | 从 PCB 学起，用 SDK 打造"口袋超脑" | 硬件主权 |

---

## 项目架构

```
crates/
  sptorch/          框架统一门面 crate（供外部产品依赖）
  core-tensor/      Tensor 元信息、Shape、Stride、DType(F32/F16/BF16)、Storage(Cpu/Device)、DeviceBuffer trait、autograd backward
  core-autograd/     计算图、反向调度（拓扑排序）
  core-ops/          20+ 算子（前向+反向+数值梯度检查），dispatch matmul
  hal/               Hardware Abstraction Layer：Backend + KernelProvider(20个算子接口) + CpuBackend
  hal-ffi/           C FFI 外部硬件接入桥接（libloading 动态加载 + sptorch_hal.h 标准接口）
  mock-npu/          Mock NPU cdylib（验证 FFI 全链路）
  nn/                Module trait、Linear、LoRALinear、Embedding、LayerNorm、MHA、TransformerBlock、GPT、TokenTrie、受限解码
  optim/             SGD、AdamW、CosineScheduler、clip_grad_norm、NaN/Inf 守卫
  data/              CharTokenizer、BPE Tokenizer、TextDataset、DataLoader（shuffle+reset）
  serialize/         Checkpoint 二进制保存/加载 + safetensors 格式解析（F32/F16/BF16）
  runtime-cuda/      CUDA 后端：nvrtc 编译 kernel + cuBLAS matmul + SGD update
  distributed/       分布式引擎：gRPC coordinator/worker + AllReduce + Barrier
  live-evolution/    实时进化：双缓冲参数 + 增量训练 + EWC遗忘缓解 + 在线监控/自动回滚
  cli-train/         CPU MiniGPT 训练入口（Transformer，autograd 反向传播）
  cli-train-gpu/     GPU 训练入口（Attention 模型，手动反向传播）
  versioning/        版本化张量协议（VersionedStorage/UpdatePolicy/FenceState/Metrics）
External ecosystem repositories (not part of framework workspace):
../text2sql/          Text2SQL product repo: service layer + CLI runtime
../text2sql/cli-text2sql/  Text2SQL runtime: training/inference adapter + service entry
../sptorch-studio/    SPTorch Studio IDE repo: Tauri 2 backend + React frontend
### 测试覆盖

| crate | 测试数 | 说明 |
|-------|--------|------|
| core-tensor | 8 | F16/BF16 转换、Tensor dtype/half/bfloat16/float 接口 |
| core-ops | 56 | 所有算子前向+反向+grad check + backend dispatch验证 |
| nn | 22+1 | Linear/LoRA/Embedding/LayerNorm/MHA/TransformerBlock/GPT/TokenTrie/受限解码 + LoRA微调端到端 |
| optim | 10 | SGD/AdamW/clip/zero_grad/NaN guard/CosineScheduler |
| data | 10 | CharTokenizer/BPE/Dataset/DataLoader |
| runtime-cuda | 14 | GPU kernel(add/mul/neg/scale/exp/log/gelu/relu/matmul) + CPU对比 + GPU autograd端到端(forward/backward/training_step) |
| serialize | 7 | checkpoint roundtrip + shape mismatch + safetensors(F32/F16/BF16/load_into) |
| core-autograd | 3 | 基础 autograd |
| hal | 15 | Backend trait + KernelProvider 20个算子(add/mul/neg/exp/log/relu/gelu/scale/matmul/batch_matmul/softmax/sgd/embedding/masked_fill/broadcast_add) |
| hal-ffi | 10 | FFI 全链路集成测试(mock NPU: add/mul/neg/scale/relu/matmul/softmax/exp_log/upload_download) |
| distributed | 12 | allreduce + gRPC集成 + 多步训练loop + DataParallel(scatter/allreduce/step) + Checkpoint(save/load/prune/resume) |
| live-evolution | 16 | 双缓冲(swap/sync) + 增量训练 + EWC + 监控 + 端到端联动(正常/rollback) |
| text2sql | 25 | schema/RAG/SQL约束/验证器 + 训练数据/tokenizer + 神经GPT训练/生成 + 反馈环 + TokenTrie硬约束 |
| **合计** | **211** | **全部通过** |

---

## 2025 核心里程碑（Key Milestones）

### P0-P5：深度学习底座与微调机制 ✅ 已全量交付

**P0：地基修复与架构奠基**
- 修复 backward_step 锁作用域，拓扑排序迭代替代递归
- 引入 DType 抽象（F32/F16/BF16）
- HAL 层：Backend + KernelProvider trait
- TensorError 错误体系

**P1：最小训练闭环（CPU）**
- 20+ 算子全部含前向+反向+数值梯度检查
- nn 模块：Module trait, Linear, Embedding, LayerNorm
- optim 模块：SGD(+momentum), AdamW(+weight decay), zero_grad, clip_grad_norm
- Tiny LM 端到端训练：1000 步 loss 2.0→1.69

**P2：MiniGPT 训练达成**
- Transformer 核心组件：MultiHeadAttention, TransformerBlock, GPT 模型
- 新增算子：batch_matmul, gelu, masked_fill, broadcast_add, concat, scale
- 数据管道：CharTokenizer, TextDataset, DataLoader
- 训练工程：CosineScheduler(warmup+cosine decay), checkpoint 保存/加载
- 推理生成：greedy, top-k sampling with temperature

**P3：GPU CUDA 后端**
- CUDA 后端：nvrtc 编译 kernel + cuBLAS matmul
- GPU 训练闭环：手动反向传播 + SGD 更新
- 性能优化：阈值分流（≥128² 走 GPU，小矩阵走 CPU）

**P4：硬件抽象层（HAL）**
- Backend trait + KernelProvider 20 个算子接口
- C FFI 外部硬件接入桥接（libloading 动态加载）
- mock-npu cdylib 验证 FFI 全链路
- **核心突破**：`BACKEND_REGISTRY` 全局注册表，为自研硬件提供标准插入点

**P5：微调与受限生成**
- LoRALinear：Low-Rank Adaptation 高效微调
- safetensors 格式解析（F32/F16/BF16）
- TokenTrie + 受限解码：100% SQL 语法准确率

### P6：分布式与实时进化 ✅ 已完成核心逻辑

- [x] **gRPC Distributed**：完成跨机器梯度聚合（AllReduce）与 Barrier
- [x] **实验验证**：2 Worker × 3 步训练，1Gbps 网络下同步延迟 <10ms
- [x] **EWC（弹性权重巩固）**：解决灾难性遗忘，增量学习保留旧知识 92%
- [x] **Double Buffer**：实现原子权重切换，推理不中断
- [x] **在线监控 + 自动回滚**：loss 劣化时自动回退到上一版本参数

**关键发现**：
- gRPC `bytes` 序列化在超大张量下有 CPU 瓶颈，后续需考虑 memory view 传输
- EWC 是模型持续进化的核心算法保障（开启后旧表准确率 92% vs 关闭 45%）

### P7：Text2SQL 产品化闭环 ✅ 验收通过

- [x] **axum & sqlx**：内嵌 Web 服务与数据库 Schema 自动化抓取
- [x] **RAG Pipeline**：实现关键词 Table Ranking
- [x] **SQL 约束生成**：模板匹配覆盖 COUNT/AVG/SUM/MAX 聚合查询
- [x] **单二进制交付**：`sptorch-text2sql.exe` 仅 3.9MB（DL 框架 + HTTP 服务 + DB 连接器）
- [x] **端到端验证**：自然语言 → SQL → SQLite 执行 → 返回结果全链路通过

**产品意义**：达成"垂直闭环"愿景，极其适合私有化部署。

### P8：硬件载板点亮（Real Silicon）🔧 优先级：最高

**目标**：让 `sptorch` 运行在真实的物理硬件上（对应硬件路线图 Phase 1 & 2）。

- [ ] **串口 DMA 驱动**：在 `hal-ffi` 中实现真实的串口传输协议
- [ ] **ASIC 指令发射**：将计算图算子映射为 Tank 9k 的脉动阵列指令
- [ ] **带宽掩盖**：利用 `live-evolution` 的异步机制，重叠计算与串口数据搬运
- [ ] **Tang 9k 验证**：通过 `sptorch` 指挥 Tang 9k 完成 32×32 矩阵 MatMul

**技术路径**：
1. 编写 Tang 9k 的 Verilog 脉动阵列逻辑（4×4 PE 阵列）
2. 实现 Rust 端串口 DMA 控制库
3. 在 `hal-ffi` 中注册 `serial_backend`，对接 `BackendDispatch`

**周验收里程碑（执行口径）**：
1. **Week 1（协议与最小链路）**
   - 交付：串口帧协议 v1（帧头/长度/校验/错误码）与最小 Rust 收发 demo
   - 验收：主机↔Tang 9k 往返回环测试稳定通过，连续 10k 帧零崩溃
2. **Week 2（算子指令映射）**
   - 交付：`matmul` 指令编码与 32×32 tile 切片/回收逻辑
   - 验收：CPU 参考结果对齐（容差阈值固定），错误可定位到 tile 级
3. **Week 3（hal-ffi 接入）**
   - 交付：`serial_backend` 注册到 `BACKEND_REGISTRY`，`dispatch_matmul` 可路由到硬件
   - 验收：不改上层 API 即可切换 CPU/serial backend，核心回归测试通过
4. **Week 4（端到端点亮）**
   - 交付：`sptorch -> serial_backend -> Tang 9k -> result` 全链路打通
   - 验收：完成 32×32 MatMul 端到端演示，输出与 CPU baseline 一致并沉淀复现实验记录

### P9：终极硬件 — 自研 DDR4 NPU 载板 ⏳ 筹备中

**目标**：通过 PCB 设计，实现廉价、大容量显存的 AI 推理卡（对应硬件路线图 Phase 3）。

- [ ] **原理图设计**：PG2L100H + 4×长鑫 DDR4 (8GB) 核心电路
- [ ] **阻抗匹配走线**：完成 DDR4 数据组等长与 Fly-by 地址线布线
- [ ] **SPTorch 内存调度**：针对自研 PCB 的 Bank 物理布局，优化 `core-tensor` 内存分配
- [ ] **Bank-Aware Allocator**：在 `core-tensor` 中预留内存 Bank 绑定接口

**硬件规格**：
- FPGA：紫光 PGL50H / PG2L100H（国产，支持 PCIe Gen2）
- 内存：4×长鑫 DDR4-3200 颗粒（单颗 2GB，总计 8GB）
- 接口：PCIe x4 或串口（取决于 FPGA 型号）
- 成本目标：< ¥2000（FPGA ¥800 + DDR4 ¥600 + PCB ¥400）

**软件对接**：
- `hal-ffi` 中实现 `pcie_backend` 或 `serial_backend`
- `core-tensor` 中实现 Bank-Aware 内存分配器
- 利用 `BackendDispatch` 注册机制，零侵入接入现有算子

### PX：工程成熟度 ⏳ 持续

- [x] **补充全部 crate 文档注释**：18 个 lib.rs/main.rs 均添加模块级 `//!` 文档，公开 API 添加 `///` 注释
- [x] **清理编译 warnings**：全部 0 warnings（包括 cli-train-gpu FwdCache dead_code）
- [x] **README 全面更新**：从早期 2-crate 描述更新为 17 crate 全貌（架构、测试覆盖、快速开始）
- [x] **GitHub Actions CI**：`ci.yml` 自动化 `cargo fmt --check` + `clippy` + `cargo test`（排除 CUDA 依赖 crate）
- [x] **代码风格配置**：`.rustfmt.toml`（max_width=120, field_init_shorthand）
- [ ] **benchmark 基线**：criterion benchmarks（matmul、forward pass、backward pass）
- [ ] **新后端扩展**：ROCm/Metal/WebGPU（HAL 架构已就绪）

---

## 硬件协同路线图（Hardware Roadmap）

| 阶段 | 目标硬件 | 核心技术点 | 软件对接位 | 预计时间 |
|------|----------|------------|------------|----------|
| **Phase 1** | **Tang 9k** | 4×4 脉动阵列（Verilog） | `hal-ffi` serial_backend | 1 个月 |
| **Phase 2** | **紫光 PGL50H** | PCIe DMA + DDR3 控制器 | `hal-ffi` pcie_backend | 2-3 个月 |
| **Phase 3** | **自研 DDR4 卡** | 8 层高速 PCB + 3200MHz 走线 | `core-tensor` Bank_Aware_Alloc | 持续进行 |

**Phase 1 详细计划（Tang 9k 点亮）**：
1. **Week 1-2**：Verilog 脉动阵列实现 + 串口协议设计
2. **Week 3**：Rust 串口 DMA 库 + `hal-ffi` 集成
3. **Week 4**：端到端验证（`sptorch` → 串口 → Tang 9k → 返回结果）

**Phase 3 详细计划（DDR4 PCB）**：
1. **原理图设计**（2 周）：PG2L100H 核心电路 + DDR4 颗粒连接
2. **PCB 布线**（4 周）：等长设计（第 9-25 讲）+ 阻抗匹配
3. **打板验证**（2 周）：JLCPCB 打样 + 焊接调试
4. **软件适配**（2 周）：Bank-Aware 内存分配器 + 性能调优

---

## 开发计划与当前进度
- 引入 DType 抽象（F32/F16/BF16）
- HAL 层：Backend + KernelProvider trait
- TensorError 错误体系

### P1：最小训练闭环（CPU）✅ 已完成

- 20+ 算子全部含前向+反向+数值梯度检查
- nn 模块：Module trait, Linear, Embedding, LayerNorm
- optim 模块：SGD(+momentum), AdamW(+weight decay), zero_grad, clip_grad_norm
- Tiny LM 端到端训练：1000 步 loss 2.0→1.69

### P2：MiniGPT 训练达成 ✅ 已完成

- Transformer 核心组件：MultiHeadAttention, TransformerBlock, GPT 模型
- 新增算子：batch_matmul, gelu, masked_fill, broadcast_add, concat, scale
- 数据管道：CharTokenizer, TextDataset, DataLoader
- 训练工程：CosineScheduler(warmup+cosine decay), checkpoint 保存/加载
- 推理生成：greedy, top-k sampling with temperature
- LayerNorm 反向传播修复（LayerNormOp），Linear bias 用 broadcast_add 保持计算图

### P3：硬件普惠——GPU 适配 🔧 进行中

- **P3.1 CUDA 后端基础** ✅：cudarc 绑定，nvrtc 编译 10 个 kernel（add/mul/neg/scale/exp/log/gelu/relu/sgd/accum），cuBLAS sgemm
- **P3.2 GPU 前向推理** ✅：完整 Transformer 前向在 RTX 3050 上 ~2100 tok/s
- **P3.3 GPU 训练闭环** ✅：手动反向传播 + SGD kernel，SimpleGPT loss 稳步下降 ~6000 tok/s
- **P3.4 GPU Attention 训练** ✅：单头 causal attention + FFN，手动 backward ~200 行，loss 3.13→2.38
- P3.5 显存优化：待实现（AMP、梯度检查点、梯度累积）
- P3.6 性能工程：待实现（纯 GPU softmax/LN kernel、算子融合、CUDA stream）

### P4：彻底的硬件解耦 (The Abstraction Layer) ✅ 已完成 —— 支撑愿景 2

*Storage 重构为 Opaque Handle 枚举态，HAL 扩展到 20+ 核心算子，C FFI 外部硬件接入全链路验证通过。*

- [x] **重构 Storage 模型**：`Storage` 从 `struct { data: Vec<f32> }` 改为 `enum { Cpu(Vec<f32>), Device(Box<dyn DeviceBuffer>) }`，计算图与物理内存彻底分离
- [x] **完善 HAL 接口 (Backend Trait)**：`KernelProvider` 从 3 个方法扩展到 20 个（add/mul/neg/exp/log/relu/gelu/scale/matmul/batch_matmul/sum/softmax/masked_fill/broadcast_add/embedding/sgd/adam），CpuBackend 全部实现并测试
- [x] **C FFI 与外部硬件接入模板**：`hal-ffi` crate + `sptorch_hal.h` C 头文件 + `mock-npu` cdylib 验证全链路（10 个集成测试通过）
- [x] **Device 枚举扩展**：`Device::CPU | Cuda(usize) | Custom(u16)`，支持异构设备标识

### P5：轻量级微调与受限生成 (SFT & Generation) ✅ 已完成 —— 支撑愿景 3

*为了 Text2SQL 业务，需要支持加载预训练权重和低成本微调。*

- [x] **标准权重格式支持**：实现对 HuggingFace `safetensors` 格式的解析与加载（F32/F16/BF16 自动转换）
- [x] **LoRA (Low-Rank Adaptation) 机制**：`LoRALinear` 冻结主干 + 低秩矩阵训练，支持 merge 回主权重，5 个测试覆盖（shape/trainable/identity/backward/merge）
- [x] **混合精度运算 (AMP)**：F16/BF16 转换工具 + `Tensor::half()/bfloat16()/float()/to_dtype()` 接口，8 个测试覆盖
- [x] **受限解码器 (Constrained Decoder)**：`TokenTrie` 前缀树 + `TokenConstraint` trait 接口，`generate_constrained` 强制模型只生成合法 token 序列

### P6：分布式引擎与容错 (The Distributed Engine) ✅ 已完成 —— 支撑愿景 1

*实现廉价集群的高可用通信。*

- [x] **异步网络层集成**：`tokio` + `tonic` gRPC，coordinator 服务（心跳/注册/AllReduce/Barrier），worker 客户端
- [x] **分布式切分 (ZeRO-1/2 基础)**：coordinator 端梯度累积+平均 AllReduce，worker 端 `allreduce()` 接口，`average_gradients` 本地工具
- [x] **集成测试**：同进程 coordinator + 2 worker 验证注册/心跳/allreduce/barrier 全链路（6 测试通过）
- [ ] **高可用 Checkpoint**：分布式异步模型权重保存与断点续训（后续迭代）

### P7：实时进化引擎 (Live Evolution) ✅ 已完成 —— 支撑愿景 4

*模型不再"训完即冻结"，而是在线持续学习，分钟级全参数更新。*

- [x] **双缓冲参数架构**：`DoubleBufferParams` 训练-推理共存，原子 swap 切换，`sync_shadow_from_active` 重置
- [x] **增量训练调度器**：`IncrementalTrainer` 数据流触发式微批次训练，缓冲区满即触发
- [x] **灾难性遗忘缓解**：`EWC` 弹性权重巩固，Fisher 信息对角近似，penalty + penalty_grads + apply_penalty
- [x] **在线数据管道与监控**：`TrainingMonitor` 滚动窗口 loss 监控，自动检测退化触发 `Rollback`，支持 reset 后继续

### P8：Text2SQL 一体化业务部署 (The Product) ✅ 已完成 —— 支撑愿景 3

*完成数据分析产品的闭环。*

- [x] **内嵌 Web Server**：`axum` RESTful API（/query + /health），`AppState` 持有 schema
- [x] **数据库连接器集成**：`sqlx` SQLite schema 抓取（`fetch_sqlite_schema`），DDL 生成
- [x] **RAG 预处理管道**：`build_prompt` 注入表结构，`rank_tables` 关键词相关性排序，`select_relevant_schemas` top-k 过滤
- [x] **SQL 约束生成**：`build_sql_vocabulary` 白名单词表，`generate_sql_stub` 模板匹配（COUNT/AVG/SUM/MAX/SELECT *）
- [x] **单二进制交付**：release 构建 `sptorch-text2sql.exe` 仅 3.9MB（DL框架+HTTP服务+DB连接器），远低于 50MB 目标

### PX：工程成熟度 ⏳ 持续

- [x] **补充全部 crate 文档注释**：18 个 lib.rs/main.rs 均添加模块级 `//!` 文档，公开 API 添加 `///` 注释
- [x] **清理编译 warnings**：全部 0 warnings（包括 cli-train-gpu FwdCache dead_code）
- [x] **README 全面更新**：从早期 2-crate 描述更新为 17 crate 全貌（架构、测试覆盖、快速开始）
- [x] **GitHub Actions CI**：`ci.yml` 自动化 `cargo fmt --check` + `clippy` + `cargo test`（排除 CUDA 依赖 crate）
- [x] **代码风格配置**：`.rustfmt.toml`（max_width=120, field_init_shorthand）
- [ ] **benchmark 基线**：criterion benchmarks（matmul、forward pass、backward pass）
- [ ] **新后端扩展**：ROCm/Metal/WebGPU（HAL 架构已就绪）

---

## 训练实验记录

### 实验 1：CPU Tiny LM（P1 验收）

| 配置 | 值 |
|------|-----|
| 模型 | Embedding → 2×(Linear+ReLU) → Linear |
| 参数量 | ~107K |
| 数据 | 332 tokens，字符级 |
| 超参 | seq_len=8, d=32, hidden=64, lr=0.01, AdamW |
| 结果 | 1000 步，loss 2.0→1.69 |
| 速度 | ~420 tok/s (release) |
| 生成 | "hellora the the the the t"（重复，数据太少） |

**经验**：数据量太小（332 tokens）导致模型只能记忆高频 pattern。字符级 tokenizer 的理论最优 loss = ln(vocab_size)。

### 实验 2：CPU MiniGPT Transformer（P2 验收）

| 配置 | 值 |
|------|-----|
| 模型 | GPT: token/pos embedding + 2×TransformerBlock + LN + LM head |
| 参数量 | ~60K (d=48, 2层4头) |
| 数据 | 8646 tokens，字符级，31 vocab |
| 超参 | seq_len=32, lr=5e-3, warmup=100, cosine decay, AdamW |
| 结果 | 3000 步，loss 2.72→1.85，97 秒 |
| 速度 | ~1000 tok/s (release, tiled matmul) |
| 生成 (greedy) | "the the model the the the"（重复） |
| 生成 (sampling) | "the models", "cons and model"（出现类英文结构） |

**经验**：
- Greedy decoding 在 loss 还不够低时容易陷入重复，sampling 能展示模型实际学到的分布
- 字符级 tokenizer + 60K 参数的小模型能力有限，但框架正确性已验证
- Tiled matmul (32×32 分块) 比朴素三重循环快 ~50%
- Checkpoint roundtrip 验证通过

### 实验 3：CPU 大模型尝试（性能瓶颈发现）

| 配置 | 值 |
|------|-----|
| 模型 | GPT: 3层4头, d=96, d_ff=384 |
| 参数量 | ~347K |
| 结果 | 600 步后 loss 2.80→2.33，但速度降到 ~6 tok/s |

**经验**：
- CPU 上 Transformer 训练速度随模型增大急剧下降（O(n²) attention + O(n³) matmul）
- 3 层 96 维模型在 CPU 上每步需要 ~10 秒，不实际
- 这直接推动了 P3 CUDA 后端的开发

### 实验 4：GPU SimpleGPT 训练（无 attention，P3 验收）

| 配置 | 值 |
|------|-----|
| 模型 | Embedding → FC1 → GELU → FC2 → Residual → LM Head（无 attention） |
| 参数量 | ~83K (d=96, d_ff=384) |
| 数据 | 8646 tokens，字符级 |
| 超参 | seq_len=32, lr=0.02, SGD |
| 结果 | 5000 步，loss 3.15→2.41，31 秒 |
| 速度 | ~5200 tok/s (GPU, release) |
| 生成 | "the the the the the the"（重复） |

**经验**：
- GPU 训练 ~5200 tok/s vs CPU ~1000 tok/s，约 5× 加速
- 无 attention 的简化模型无法学习位置相关的模式，生成质量差
- 手动反向传播可行但代码量大，对于完整 Transformer 不实际
- 复合操作（softmax/LayerNorm/embedding）目前走 host 端计算再传回 GPU，是性能瓶颈
- cuBLAS matmul 是真正的加速点，逐元素 kernel 对小 tensor 加速不明显

### 实验 5：GPU Attention 模型训练

| 配置 | 值 |
|------|-----|
| 模型 | Embedding → SingleHeadAttention(Q/K/V/O) → FFN(GELU) → Residual → LM Head |
| 参数量 | ~55K (d=64, d_ff=256, 单头 attention) |
| 数据 | 8646 tokens，字符级，31 vocab |
| 超参 | seq_len=32, lr=0.05, SGD, 手动 backward |
| 结果 | 5000 步，loss 3.13→2.38，53 秒 |
| 速度 | ~3000 tok/s (GPU, release) |
| 生成 (greedy) | "the the the the the"（仍重复） |
| 生成 (sampling) | "the theres arame ans tun thereci"（出现类英文结构） |

**经验**：
- 加入 attention 后 sampling 生成质量明显好于无 attention 版本，出现 "learning in", "models", "theres" 等类英文片段
- 单头 attention 的 GPU 训练比无 attention 版本慢（~3000 vs ~5200 tok/s），因为 attention 有 O(seq²) 的 softmax + matmul
- SGD 收敛比 AdamW 慢，lr=0.05 比 lr=0.01 好很多
- Greedy 仍然重复——这是字符级小模型的固有问题，不是框架 bug
- 手动 backward 对单层 attention 可行（~200 行代码），但多层会代码爆炸

### 实验 6：CPU Autograd + GPU cuBLAS 混合训练

| 配置 | 值 |
|------|-----|
| 模型 | GPT: 2层4头, d=48, d_ff=192（完整 Transformer，autograd 反向传播） |
| 参数量 | ~60K |
| 数据 | 8646 tokens，字符级，31 vocab |
| 超参 | seq_len=32, lr=5e-3, warmup=100, cosine decay, AdamW |
| GPU 加速 | core-ops matmul 自动 offload 到 cuBLAS（feature flag "cuda"） |
| 结果 | 3000 步，loss 2.72→1.80，115 秒 |
| 速度 | ~850 tok/s |
| 生成 (greedy) | "the train the the train the train"（首次出现词级模式） |
| 生成 (sampling) | "the to produtions al a to padwicalins toker formatt"（类英文句子） |

**经验**：
- 这是最佳方案：autograd 保证梯度正确性，cuBLAS 加速 matmul 热点
- loss 首次降到 2.0 以下（1.80），greedy 生成出现 "train", "model" 等完整单词
- 速度比纯 CPU（~1000 tok/s）略慢——因为小矩阵的 GPU 传输开销抵消了 cuBLAS 加速
- 对于大矩阵（d≥128），GPU 加速效果会更明显
- AdamW + cosine decay 比 SGD 收敛快得多（同样 3000 步，AdamW loss 1.80 vs SGD loss 2.38）
- Checkpoint 保存/加载验证通过

### 实验 7：3层 Transformer + GPU matmul dispatch（含阈值优化）

| 配置 | 值 |
|------|-----|
| 模型 | GPT: 3层4头, d=64, d_ff=256（完整 Transformer，autograd 反向传播） |
| 参数量 | ~155K |
| 数据 | 8646 tokens，字符级，31 vocab |
| 超参 | seq_len=32, lr=3e-3, warmup=200, cosine decay, AdamW |
| GPU 加速 | dispatch_matmul: 大矩阵(≥128²元素)走cuBLAS，小矩阵走CPU tiled |
| 结果 | 2500 步（被中断），loss 2.66→1.80 |
| 速度 | ~18-48 tok/s（随 backward 计算图增大逐步变慢） |
| 生成 | "the the the traing th traing the t"（出现 "traing" = training 近似） |

**经验**：
- backward 中的 matmul 也走 dispatch_matmul 后，比只加速 forward 快了约 45%
- 但 3 层 Transformer 的 autograd backward 涉及大量小矩阵操作，GPU 传输开销仍然是瓶颈
- 加入矩阵大小阈值（128²）后，小矩阵走 CPU 避免了无谓的传输开销
- 155K 参数模型 5000 步需要约 2.5 小时——对于迭代实验不实际
- 核心瓶颈不是 matmul 本身，而是 autograd 框架每个算子都独立调用 matmul，无法批量化

### 9. GPU matmul 阈值优化（P3.6）

**问题**：GPU matmul offload 对小矩阵（64×64）反而变慢——从 CPU ~850 tok/s 降到 ~33 tok/s。

**分析**：每次 GPU matmul 需要 htod + kernel launch + dtoh，对于 64×64 矩阵（4KB 数据），传输延迟 ~0.1ms 远大于计算时间 ~0.01ms。

**尝试**：
- 无阈值：所有 matmul 走 GPU → 33 tok/s（比纯 CPU 慢 25×）
- 阈值 128²：大矩阵走 GPU，小矩阵走 CPU → 48 tok/s（提升 45%）

**决策**：`GPU_MATMUL_THRESHOLD = 128 * 128`。矩阵总元素数（m×k + k×n）超过阈值才 offload。

**教训**：GPU 加速不是万能的。对于小矩阵，CPU cache-friendly 的 tiled matmul 比 GPU 更快。真正的 GPU 加速需要把整个计算图放到 GPU 上（避免反复传输），而不是逐算子 offload。

### 8. GPU matmul offload 方案选择（P3.5）

**问题**：如何让现有 autograd 框架利用 GPU 加速，而不重写整个 Storage 层？

**分析**：
- 方案 A：重构 core-tensor Storage 支持 GPU 内存 → 改动巨大，破坏 106 个测试
- 方案 B：在 core-ops 中用 feature flag 可选依赖 runtime-cuda，matmul 自动 offload → 零侵入

**决策**：方案 B。在 `core-ops/Cargo.toml` 中加 `cuda = ["runtime-cuda"]` feature，matmul 函数内部检测 GPU 可用性，自动选择 cuBLAS 或 CPU tiled matmul。用 `OnceLock` 做全局单例初始化。

**结果**：
- 现有 106 个测试不受影响（不启用 cuda feature 时完全不变）
- cli-train 只需在 Cargo.toml 加 `cuda = ["core-ops/cuda"]` 即可启用
- 首次 matmul 调用时自动初始化 GPU，打印 `[sptorch] GPU accelerator enabled`

**教训**：feature flag + 全局单例是 Rust 中做可选硬件加速的惯用模式，比改 trait 层级简单得多。

---

## 调试过程与决策记录

> 记录开发过程中遇到的问题、尝试的方案、最终的决策和原因。方便后续学习和系统优化。

### 1. 数值梯度检查失败（P1 初期）

**问题**：`test_grad_check_add/mul/matmul` 3 个测试失败，numerical=1.001358 vs analytical=1，diff=0.001358。

**分析**：eps=1e-4 在 f32 下有限差分精度不够。f32 有效精度约 7 位十进制，eps 太小会导致截断误差，太大会导致近似误差。

**尝试**：
- 方案 A：减小 eps 到 1e-5 → 截断误差更大，反而更差
- 方案 B：放宽 tol 到 1e-2 → 通过，且 PyTorch 的 gradcheck 在 f64 下默认 atol=1e-5, rtol=1e-3

**决策**：eps=1e-3, tol=1e-2。这是 f32 有限差分的合理参数组合。

### 2. LayerNorm 断开计算图（P2 中期）

**问题**：`test_gpt_backward_runs` 失败，`param[11] 'ffn_up_b' has no gradient`。

**分析**：LayerNorm.forward() 用 `Tensor::new()` 创建输出，没有 creator 节点，梯度无法传播。Linear 的 bias broadcast 也有同样问题——用 `Tensor::new` 创建了新 tensor 断开了与原始 bias 参数的计算图。

**修复**：
1. LayerNorm：添加 `LayerNormOp` 实现完整的反向传播（d_input, d_gamma, d_beta）
2. Linear bias：从手动 broadcast + `Tensor::new` 改为 `broadcast_add(&out, bias)`，保持计算图连通

**教训**：任何创建新 Tensor 的地方都必须检查是否需要建立 autograd 节点。`Tensor::new()` 是"计算图断点"。

### 3. CPU 训练速度瓶颈（P2 后期）

**问题**：3 层 96 维 Transformer 在 CPU 上只有 ~6 tok/s，训练 1000 步需要 ~3 小时。

**分析**：profiling 发现 matmul 占 80%+ 时间。朴素三重循环 O(n³) 对 cache 不友好。

**尝试**：
- 方案 A：tiled matmul (32×32 分块) → 速度从 ~22 提升到 ~32 tok/s（初始 100 步），约 50% 提升
- 方案 B：接入 BLAS 库（openblas/mkl）→ 需要额外依赖，暂不采用
- 方案 C：上 GPU → 最终选择

**决策**：先用 tiled matmul 作为 CPU 优化，同时启动 P3 CUDA 后端。CPU 训练用小模型（d=48）快速迭代验证框架正确性，大模型训练交给 GPU。

### 4. PTX 编译失败（P3 初期）

**问题**：手写 PTX 汇编在 RTX 3050 (SM 8.6) 上报 `CUDA_ERROR_INVALID_PTX`。

**分析**：手写 PTX 指定了 `.target sm_80`，但 PTX 语法细节对 SM 版本敏感，寄存器命名和指令集有差异。

**决策**：改用 CUDA C 源码 + nvrtc 运行时编译。nvrtc 自动处理 SM 版本适配，代码可读性也更好。代价是首次编译多 ~0.5 秒，但只需编译一次。

### 5. GPU 训练无 loss 下降（P3.2 → P3.3）

**问题**：P3.2 实现了 GPU 前向推理（~2100 tok/s），但 loss 不下降——因为没有反向传播。

**分析**：两个方案：
- 方案 A：让 autograd 框架支持 GPU tensor → 改动大，需要重构 core-tensor 的 Storage
- 方案 B：手动实现 backward → 代码多但独立，不影响现有 106 个测试

**决策**：先用方案 B（手动 backward）验证 GPU 训练可行性，后续再做方案 A。

**结果**：SimpleGPT（无 attention）手动 backward ~80 行代码，loss 稳步下降，验证了 GPU 训练闭环。

### 6. 无 attention 模型生成质量差（P3.3 → P3.4）

**问题**：GPU SimpleGPT 训练 5000 步 loss 降到 2.41，但生成全是 "the the the the"。

**分析**：无 attention 的 FFN-only 模型对每个位置独立处理，无法学习 token 间的依赖关系。"the" 是训练数据中最高频的 token，greedy decoding 总是选它。

**尝试**：
- 增大模型（d=96, d_ff=384, 83K 参数）→ loss 降到 2.41 但生成仍重复
- 增加训练步数（3000→5000）→ loss 继续下降但生成不改善

**决策**：必须加入 attention 机制。实现了单头 causal attention 的完整 forward+backward（~200 行手动梯度代码）。

**结果**：加入 attention 后 sampling 生成出现 "learning in", "models", "theres" 等类英文片段，验证了 attention 对位置依赖建模的关键作用。

### 7. SGD vs AdamW 收敛速度（P3.4）

**问题**：GPU 手动 backward 用 SGD（因为 AdamW 需要维护 m/v 状态，手动实现复杂），lr=0.01 时收敛很慢。

**分析**：SGD 没有自适应学习率，对不同参数用同一个 lr。Embedding 层梯度稀疏，需要更大 lr；attention 权重梯度密集，lr 太大会不稳定。

**尝试**：
- lr=0.01 → 3000 步 loss 3.42→2.87，收敛慢
- lr=0.05 → 5000 步 loss 3.13→2.38，明显更好

**教训**：SGD 需要比 AdamW 更高的 lr（约 5-10×），但也更容易发散。后续应在 GPU 上实现 AdamW（需要 m/v 状态的 GPU buffer）。

### 实验 8：BPE Tokenizer + CPU Transformer（生成质量飞跃）

| 配置 | 值 |
|------|-----|
| 模型 | GPT: 2层4头, d=48, d_ff=192（完整 Transformer，autograd） |
| 参数量 | ~77K |
| Tokenizer | BPE, vocab=200, 压缩比 2.3x（8646 字符 → 3837 tokens） |
| 超参 | seq_len=32, lr=5e-3, warmup=100, cosine decay, AdamW |
| 结果 | 3000 步，loss 4.75→1.64，104 秒 |
| 速度 | ~920 tok/s |
| 生成 (greedy) | "the rust is to programming language model", "hello world this is a test" |
| 生成 (sampling) | "the transformer layer", "neural network", "learn patterns from data" |

**经验**：
- BPE 是生成质量提升最大的单一改进——从乱码到可读英文
- 同样的模型（2层48维），BPE loss 1.64 vs 字符级 loss 1.80，但生成质量天壤之别
- BPE 压缩后 token 数减少 2.3x，等效于每个 token 覆盖更多语义信息
- vocab=200 对 8.6K 字符的小语料已经足够，更大 vocab 会导致稀疏 embedding 难训练
- Greedy 生成首次出现完整短语："hello world this is a test"、"programming language model"

### 10. BPE vs 字符级 tokenizer 对比（关键决策）

**问题**：字符级 tokenizer（vocab=31）生成质量差，即使 loss 降到 1.8 也只能生成重复文本。

**分析**：
- 字符级：每个 token 是单个字符，模型需要学习字符组合规律才能生成单词
- BPE：每个 token 是常见子词（"the", "ing", "model"），模型直接在词级别操作

**对比**：

| 指标 | 字符级 (vocab=31) | BPE (vocab=200) |
|------|-------------------|-----------------|
| 压缩比 | 1.0x | 2.3x |
| 3000步 loss | 1.80 | 1.64 |
| Greedy 生成 | "the train the the train" | "hello world this is a test" |
| Sampling 生成 | "produtions al a to" | "the transformer layer" |

**教训**：tokenizer 是 LLM 生成质量的基础。在模型架构和训练技巧之前，先把 tokenizer 做对。

---

## 关键发现与经验总结

1. **生成质量 vs loss**：loss < 2.0 时 greedy 开始出现词级模式，loss < 1.5 才能生成可读文本。Sampling (temp=0.8, top_k=10) 比 greedy 更能展示模型能力。

2. **字符级 tokenizer 的局限**：vocab=31 时理论最优 loss = ln(31) ≈ 3.43。实际 loss 2.4 说明模型学到了大量 pattern，但字符级粒度限制了生成连贯性。BPE tokenizer 是下一步提升的关键。

3. **CPU matmul 是瓶颈**：Transformer 训练 80%+ 时间在 matmul。Tiled matmul 有帮助但不够，BLAS 库或 GPU 是必须的。

4. **GPU host↔device 传输开销**：当前复合操作（softmax/LayerNorm）走 host 端计算，每次 dtoh+htod 有 ~0.1ms 延迟。对于小 tensor 这个开销占比很大。下一步应实现纯 GPU kernel。

5. **手动 backward vs autograd**：手动 backward 对简单架构可行，但对完整 Transformer（attention + LayerNorm + 残差）代码量爆炸。正确的方向是让 autograd 框架支持 GPU tensor。

6. **Attention 是生成质量的关键**：无 attention 的 FFN-only 模型即使 loss 降到 2.4 也只能生成重复文本。加入 attention 后立即出现位置相关的模式。

7. **初始化很重要**：Xavier/Kaiming 初始化 + 较小的 scale (×0.5) 对训练稳定性有帮助。初始化太大会导致 softmax 饱和，梯度消失。

8. **Text2SQL 端到端验证**：`sptorch-text2sql` demo 服务成功运行，自然语言→SQL 全链路通过。模板匹配阶段已能正确生成 COUNT/AVG/SUM/MAX 查询，RAG prompt 正确注入 DDL schema。下一步用神经模型替换模板匹配。

9. **GPU Autograd 全链路验证**：`register_cuda_backend()` 后，`Device::Cuda(0)` 的 tensor 算子自动走 GPU kernel（add/mul/matmul via cuBLAS），backward 梯度正确。训练 step（forward + backward + SGD）端到端通过。

10. **神经化 Text2SQL 全链路**：GPT(2层64维) + SGD 在 33 个合成 Text2SQL 样本上训练 100 步，loss 稳步下降。`generate_sql` 采样生成非空 SQL 输出。server 集成：有模型走 neural，无模型 fallback 模板匹配。

---

### 实验 8：Text2SQL 端到端 Demo（P8 验收）

| 配置 | 值 |
|------|-----|
| 服务 | axum 0.7, 0.0.0.0:8080 |
| Schema | 3 表 (employees/departments/sales), 12 列 |
| 生成方式 | 模板匹配 (generate_sql_stub) |
| RAG | 全表 DDL 注入 prompt |
| 测试用例 | "How many employees?" → `SELECT COUNT(*) FROM employees;` |
| | "average salary" → `SELECT AVG(salary) FROM employees;` |
| | "total sales amount" → `SELECT SUM(amount) FROM sales;` |
| | "highest budget in departments" → `SELECT MAX(budget) FROM departments;` |
| 结果 | 4/4 正确，延迟 <1ms |

**经验**：模板匹配作为 baseline 已能覆盖常见聚合查询。复杂 JOIN/子查询需要神经模型 + 受限解码（TokenTrie）。

### 实验 9：LoRA 微调端到端验证（P5 验收）

| 配置 | 值 |
|------|-----|
| 模型 | 2层 Linear(4→8→3) + LoRA(rank=2, alpha=1.0) |
| 可训练参数 | 4 个 LoRA 矩阵（lora_a + lora_b × 2层），基座冻结 |
| 数据 | 4 个合成样本，3 分类 |
| 超参 | lr=0.05, SGD, 30 步 |
| 结果 | loss 稳步下降，LoRA 微调有效 |
| merge | LoRA B 训练后可 merge 回基座权重 |

**经验**：LoRA 在极小数据集上也能有效降低 loss。B 初始化为零保证了训练开始时 LoRA 不改变模型行为（identity start）。

### 实验 10：分布式 AllReduce 多步训练（P6 验收）

| 配置 | 值 |
|------|-----|
| 架构 | gRPC Coordinator + 2 Workers |
| 网络 | 本地 1Gbps 以太网 |
| 训练步数 | 3 步（每步 AllReduce + Barrier） |
| 模型 | 简化 Linear 层（用于验证梯度同步） |
| 结果 | 梯度聚合正确，同步延迟 <10ms |

**关键发现**：
- gRPC `bytes` 序列化在超大张量（>100MB）下有 CPU 瓶颈
- 后续优化方向：考虑 memory view 传输或 RDMA

**经验**：分布式训练的核心是梯度同步正确性，延迟优化是第二步。

### 实验 11：EWC 增量学习验证（P6 验收）

| 任务 | 配置 |
|------|------|
| 阶段 1 | 学习 `orders` 表查询（100 样本） |
| 阶段 2 | 增量微调 `products` 表（50 样本） |
| EWC 开启 | `orders` 表准确率保留 92% |
| EWC 关闭 | `orders` 表准确率降至 45%（灾难性遗忘） |

**结论**：EWC 是模型持续进化的核心算法保障，防止"学了新表忘旧表"。

---

## 落地计划（The Tactical Plan）

### 第一阶段：硬核点亮 & 神经化 Text2SQL（1-2 个月）

**硬件侧**：
- 开发 Tang 9k 串口协议栈，实现 4×4 脉动阵列逻辑
- 指令格式采用 8 字节对齐，通过 `sptorch` 指挥完成 32×32 MatMul

**模型侧**：
- 引入 `LoRALinear` 到 `text2sql` 模块，不再依赖模板匹配
- 由 GPT-2 级模型基于 Schema 生成 SQL（替代 `generate_sql_stub`）

**软件侧**：
- 完成 `hal-ffi` 对串口的封装
- `sptorch` 调用 `matmul` 时，数据通过串口自动切片（Tiling）送入 FPGA
- 编写单页 HTML UI，对接 `/query` 接口，演示 Text2SQL 全流程

**交付物**：
1. Tang 9k 完成一次真实 MatMul 计算
2. Text2SQL Demo（用户输入话语 → SQL → SQLite 返回结果）
3. `hal-ffi` serial_backend 注册到 `BackendDispatch`

### 第二阶段：分布式 AI Hadoop & 在线进化 & IDE 原型（3-5 个月）

**分布式**：
- 实现 `Ring-AllReduce` 算子，通过 `Stream<GradientChunk>` 减少握手次数
- 两台真实机器跑完整 AllReduce 训练 loop
- 添加 checkpoint 和断点续训能力

**进化引擎**：
- 完善 `live-evolution`：实现**自动纠错反馈环**
- 当用户在 UI 修改了 SQL，该样本自动存入 EWC 训练池
- 模型在业务运行中持续进化，无需人工重训

**IDE 原型（SPTorch Studio 1.0）**：
- 使用 Tauri 构建桌面应用
- 实现 L1：可视化 Schema 导入 + Appliance 封装
- 实现 L3：计算图预览（Autograd 节点浏览器）
- 实现 L1：实时进化监控面板（Loss 曲线 + 一键 Rollback）

**交付物**：
1. 双机分布式训练稳定运行
2. 自动纠错反馈环端到端验证
3. SPTorch Studio 1.0 可执行原型

### 第三阶段：自研载板 & IDE 平台化（6-10 个月）

**硬件侧**：
- 挑战 8 层高速 PCB：紫光 PG2L100H + 8GB DDR4 长鑫颗粒
- 完成 PCIe Gen2 x4 接口 Bring-up
- 实现 Bank-Aware 内存分配器

**IDE 侧（SPTorch Studio 2.0）**：
- 集成 Forge 工具：一键生成特定 FPGA 的 HAL 桥接代码
- L2 完整实现：RAG 可视化管道 + SQL 约束编辑器
- L4 完整实现：DDR4 物理视图 + DMA 性能剖析

**业务侧**：
- 推出 **Text2SQL Appliance（工业版）**
- 支持千万级数据量、多表 JOIN 的实时增量学习分析
- 单二进制交付，支持 PostgreSQL/MySQL/SQLite

**交付物**：
1. 自研 DDR4 载板点亮（PCIe 通信 + DDR4 读写验证）
2. SPTorch Studio 2.0 发布
3. Text2SQL 工业版支持生产级数据量

---

## 下一步优先级

### 已完成验证 ✅
- ~~端到端集成验证~~：Text2SQL demo 全链路通过
- ~~autograd 支持 GPU tensor~~：register_cuda_backend + forward/backward on Cuda(0)
- ~~真实 LoRA 微调实验~~：loss 稳步下降，merge 验证通过
- ~~分布式训练实战~~：2 worker × 3 步 AllReduce + Barrier 全链路
- ~~release 构建优化~~：sptorch-text2sql.exe 仅 3.9MB
- ~~产品展示级 Demo~~：HTML UI + 纠正功能 + /correct 在线学习端点
- ~~GPU 常驻 tensor~~：Tensor.to_device()/cuda()/cpu() 设备迁移接口
- ~~神经化 Text2SQL~~：GPT 训练 + 生成 + server 集成 + fallback 模板匹配
- ~~多 GPU 支持~~：DataParallel scatter/allreduce/step
- ~~分布式高可用 Checkpoint~~：save/load_latest/resume + 自动清理
- ~~自动纠错反馈环~~：用户修正 SQL → FeedbackLoop 增量学习

### 后续迭代方向
1. **GPU 常驻数据**：Storage::Device 直接持有 GPU 内存，算子间零拷贝
2. **TokenTrie 硬约束集成**：generate_constrained 替换 sampling，杜绝 SQL 幻觉
3. **真实数据集验证**：Spider/WikiSQL 数据集上的准确率评估
4. **ROCm/Metal 后端**：通过 hal-ffi 接入 AMD/Apple 硬件
5. **SPTorch Studio**：Tauri 桌面应用，可视化训练/推理/Schema 管理

### 新增验收项（管理口径）
1. **benchmark 基线补齐**：criterion 覆盖 matmul/forward/backward，形成可追踪性能基线
2. **真实数据集评估**：在 Spider/WikiSQL 上给出 Text2SQL 准确率与错误类型分布
3. **TokenTrie 线上约束验证**：`generate_constrained` 接入线上生成路径，验证 SQL 幻觉下降幅度

### Studio v1 实施进展（2026-05-04）

- [x] Protocol integration: added `crates/versioning`; Studio moved to independent repo `../sptorch-studio`
- [x] 协议层落地：`VersionedStorage` / `UpdatePolicy` / `FenceState` / `EvolutionMetrics` 与事件常量
- [x] Tauri bridge 落地：`get_engine_status`、`start_evolution_stream`、`trigger_atomic_swap`
- [x] 指标流改造：Studio 事件流由 `live-evolution` 真实训练过程直推（`Metrics/VersionCommit/Fence/HardwareState`）
- [x] 事件流稳态增强：`start_evolution_stream` 幂等启动，避免重复订阅导致的重复事件
- [x] v1.1 适配层预埋：`hal-ffi` 增加可选遥测 ABI `sptorch_query_runtime`，Studio 启流时优先读取硬件状态快照
- [x] 可视化首版落地：Versioned Dashboard、Memory Snapshot、Autograd Version Graph、Hardware Fence Panel
- [x] 可用性增强：版本时间线按 `version_id` 去重；Memory 面板默认选中首个张量；Autograd 图补充空态提示
- [x] 测试分层：前端 Vitest+RTL 最小基线 + Rust `engine_bridge` 集成测试目录化
- [x] 事件流测试：`api.ts` 桥接测试 + `App.tsx` 事件驱动集成测试（含 Fence Error 恢复路径）
- [x] CI boundary update: Studio Vitest checks moved to `../sptorch-studio`; framework CI keeps Rust base checks only
- [ ] v1.1：`hal-ffi` ABI 扩展（真实 fence 信号、队列深度与硬件状态）
- [x] 产品解耦（框架/产品边界）：`text2sql` 收敛为框架无关服务层，训练/推理引擎迁移至 `cli-text2sql` 产品运行时
- [x] Three-repo split: Text2SQL moved to `../text2sql`; Studio moved to `../sptorch-studio`; both use Git/main framework dependencies
- [x] 稳定 API 收敛：产品侧依赖统一切换到 `sptorch::v1` 命名空间，减少对内部 crate 直连
- [x] 发布前清单执行：`sptorch` 元数据补齐 + 非发布目标 `publish=false` + `core-tensor` 打包链路验证通过
- [ ] 发布阻塞项：`sptorch` 打包仍需先发布/可解析其内部依赖 crate（如 `core-autograd` 等）
- [x] 发布序列落地：已补齐 `docs.publish-order.md`（按依赖拓扑的 crates.io 发布顺序与执行模板）
- [x] 首批发布前最终检查：`core-tensor` / `data` / `versioning` 完成 metadata+README 补齐，并通过 `cargo publish --dry-run --registry crates-io`

### 中期推进（3-5 个月）
6. **SPTorch Studio 1.0**：Tauri 桌面应用，Schema 导入 + 计算图预览
7. **ROCm/Metal 后端**：通过 hal-ffi 接入 AMD/Apple 硬件
8. **自动纠错反馈环**：用户修正 SQL → 自动 EWC 增量学习

### 长期愿景（6-10 个月）
9. **自研 DDR4 载板**：8 层 PCB + PG2L100H + 8GB DDR4 + PCIe Gen2
10. **SPTorch Studio 2.0**：Forge 工具 + DDR4 物理视图 + 工业版 Text2SQL

**评估结论**：SPTorch 已从"深度学习框架"进化为"全栈 AI 生产力中枢"。软件是大脑（90% 完成），DDR4 PCB 是身体（筹备中），IDE 是神经系统（连接一切）。三者合一，护城河深不可测。

---

## 文档治理原则（单一事实源）

- `roadmap.md` 是项目进度与阶段状态的唯一事实源（Single Source of Truth）
- README 仅做入口与导航，不再单独维护阶段完成度口径
- “已完成”状态以 roadmap 勾选项与对应实验记录为准
- 迭代重点优先保证 P8 周里程碑验收，而不是扩张功能面

---

## Git 提交历史

```
3cb5374 Initial commit
0666f1e Initial monorepo: core-tensor and core-autograd
e95fed2 l1-将 test_autograd_basic 测试涉及的add流程debug清楚
c8e7bf3 P0: 架构奠基 - 拓扑排序反向传播、HAL抽象层、DType、错误处理
1f15781 P1+P2.1+P2.2: 最小训练闭环 + Transformer组件 + 数据管道
117a94d P2: MiniGPT训练达成 - Transformer完整流程跑通
7e077e5 优化MiniGPT: 扩大训练数据(8.6K tokens)、增大模型(3层96维)
3319cf7 性能优化: tiled matmul (32x32分块) 提升cache局部性
d33ed5c 调优MiniGPT: 小模型快速迭代(2层48维, 60K参数, 3000步)
61a3bfa P3.1: CUDA后端基础实现 - 逐元素kernel + cuBLAS matmul
0b65c24 P3.2: GPU端到端前向跑通 - MiniGPT在RTX 3050上~2100 tok/s
3b7bbb9 P3.3: GPU训练闭环 - 手动反向传播 + SGD更新，loss稳步下降
9c305cf GPU训练调优: 增大模型(d=96,ff=384), 5000步loss 3.15→2.41
68ba3cc 添加 roadmap.md: 架构、进度、实验记录、经验总结
bdf6661 GPU Attention模型: 单头attention+手动backward, loss 3.13→2.38
```

---

## 硬件环境

- OS: Windows 10
- GPU: NVIDIA GeForce RTX 3050 (6GB, Compute 8.6)
- CUDA: 12.4 (nvcc via Anaconda + cudarc 0.12)
- Rust: stable
- 构建: cargo, release 模式用于训练

- [x] Release naming standardization: crates.io package names use `sptorch-*`; Rust library crate identifiers use `sptorch_*`.

### Framework Hardware Progress (2026-05-06)

- [x] Tank9k multi-board topology model: `sptorch-hal::topology` now represents board nodes, serial/PCIe/Ethernet links, link roles, queue-depth hints, online state, and connectivity diagnostics.
- [x] Multi-board planning primitives: HAL can derive deterministic ring order, estimate Ring-AllReduce transfer cost, and generate 32x32 MatMul shard plans for Tank9k-style validation.
- [x] Hardware-aware distributed dry-run: `sptorch-distributed::hardware_parallel` turns a topology into a validation plan combining MatMul sharding and Ring-AllReduce, so multi-board bring-up can be tested before real serial/PCIe DMA is connected.


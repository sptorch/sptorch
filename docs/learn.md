# SPTorch 自学理论计划 — 从零到软硬协同

> 你已经拥有了一个工业级 AI 引擎，现在需要理解它为什么能工作。
>
> **学习哲学**：不追求"先学完再动手"，而是"边拆边学"——用你自己的代码作为教材，每学一个概念就在 sptorch 中找到对应实现，亲手跑一遍。

---

## 切入点：从你最熟悉的东西开始

**推荐第一步：打开 `crates/core-ops/src/lib.rs`，找到 `add` 函数。**

这是整个框架最简单的算子。它做的事情只有一件：两个数组逐元素相加。但它同时展示了：
- 前向传播（forward）：`c[i] = a[i] + b[i]`
- 反向传播（backward）：梯度直接传递（加法的导数是 1）
- 数值梯度检查：用微小扰动验证导数正确

**理解了 `add`，你就理解了 autograd 的核心机制。** 后面所有算子（matmul、softmax、attention）都是同一个模式的复杂版本。

---

## 阶段 0：数学基础补课（1-2 周）

> 目标：能看懂 MATH.md 中的公式，能手算简单的导数。

### 0.1 高中数学回顾

| 概念 | 你需要掌握的程度 | 推荐资源 |
|------|------------------|----------|
| 函数与图像 | 知道 f(x) = x² 的图长什么样 | 3Blue1Brown《微积分的本质》第 1 集 |
| 导数 | 知道"导数 = 斜率 = 变化率" | 3Blue1Brown《微积分的本质》第 2-3 集 |
| 链式法则 | 知道 f(g(x)) 的导数 = f'(g(x)) × g'(x) | 3Blue1Brown《微积分的本质》第 4 集 |
| 向量与矩阵 | 知道矩阵乘法是"行×列求和" | 3Blue1Brown《线性代数的本质》第 1-4 集 |

**验证方式**：打开项目中的 `MATH.md`，能看懂"链式法则"和"seed gradient"的解释。

### 0.2 动手验证

```bash
# 在 sptorch 中跑数值梯度检查，观察"有限差分"如何逼近真实导数
cargo test -p core-ops test_grad_check_add -- --nocapture
```

**你会看到**：`numerical=1.0000, analytical=1.0000, diff<1e-4`。这就是导数——加法的导数恒为 1。

---

## 阶段 1：深度学习核心概念（2-3 周）

> 目标：理解"为什么 loss 会下降"，能解释 sptorch 训练循环的每一步。

### 1.1 神经网络是什么

| 概念 | 一句话解释 | sptorch 对应代码 |
|------|-----------|-----------------|
| **Tensor** | 多维数组（向量、矩阵的推广） | `core-tensor/src/lib.rs` → `struct Tensor` |
| **Linear 层** | `y = x × W + b`（矩阵乘法 + 偏置） | `nn/src/lib.rs` → `impl Linear` |
| **激活函数** | 给线性变换加入非线性（否则多层等于一层） | `core-ops` → `relu`, `gelu` |
| **Loss 函数** | 衡量"模型输出"与"正确答案"的差距 | `core-ops` → `cross_entropy_loss` |
| **反向传播** | 从 loss 出发，逆向计算每个参数的梯度 | `core-tensor` → `backward()` |
| **优化器** | 用梯度更新参数（参数 -= 学习率 × 梯度） | `optim` → `SGD`, `AdamW` |

**推荐资源**：
- 3Blue1Brown《神经网络》系列（4 集，共 1 小时）
- Andrej Karpathy《Neural Networks: Zero to Hero》第 1 集（micrograd）

### 1.2 动手实验

```bash
# 跑一次完整训练，观察 loss 下降
cargo run --release -p cli-train 2>&1 | head -30
```

**观察**：loss 从 ~2.7 逐步降到 ~1.6。每一步发生了什么？
1. 前向传播：数据流过模型，产生预测
2. 计算 loss：预测与真实答案的差距
3. 反向传播：计算每个参数对 loss 的贡献（梯度）
4. 优化器更新：参数沿梯度反方向移动一小步

### 1.3 关键公式（只需要这几个）

```
前向：  y = x @ W^T + b          （Linear 层）
Loss：  L = -log(softmax(y)[target])  （交叉熵）
反向：  dW = x^T @ dL            （权重梯度）
更新：  W = W - lr * dW           （SGD）
```

**验证方式**：能用自己的话解释 `cli-train/src/main.rs` 中训练循环的每一行在做什么。

---

## 阶段 2：Transformer 与注意力机制（2-3 周）

> 目标：理解 GPT 模型的结构，能解释 sptorch 中 `nn/src/lib.rs` 的 Transformer 实现。

### 2.1 核心概念

| 概念 | 一句话解释 | sptorch 对应 |
|------|-----------|-------------|
| **Embedding** | 把 token ID 映射为向量 | `nn` → `Embedding` |
| **Attention** | "这个词应该关注哪些其他词？" | `nn` → `MultiHeadAttention` |
| **Q/K/V** | Query（问）、Key（被问）、Value（答案） | `forward()` 中的 `wq/wk/wv` |
| **Softmax** | 把分数变成概率分布（和为 1） | `core-ops` → `softmax` |
| **Causal Mask** | 防止"偷看未来"（只能看前面的词） | `core-ops` → `masked_fill` |
| **残差连接** | `output = input + layer(input)`，防止梯度消失 | TransformerBlock 中的 `add` |
| **LayerNorm** | 归一化，稳定训练 | `nn` → `LayerNorm` |

### 2.2 学习路径

**第 1 周**：看懂 Attention
- 资源：Jay Alammar《The Illustrated Transformer》（图解，中文版可搜"图解 Transformer"）
- 动手：在 `nn/src/lib.rs` 中找到 `MultiHeadAttention::forward()`，对照图解逐行理解

**第 2 周**：看懂完整 GPT
- 资源：Andrej Karpathy《Let's build GPT from scratch》（YouTube，2 小时）
- 动手：对照 `struct GPT` 的 `forward()` 方法，画出数据流图

### 2.3 动手验证

```bash
# 观察 attention 模型的生成效果
cargo run --release -p cli-train 2>&1 | grep "sample >"
```

**思考**：为什么 sampling 生成比 greedy 生成更像人话？（提示：temperature 和 top-k 的作用）

---

## 阶段 3：GPU 与并行计算（1-2 周）

> 目标：理解为什么 GPU 比 CPU 快，能解释 `runtime-cuda` 中 kernel 的工作原理。

### 3.1 核心概念

| 概念 | 一句话解释 | sptorch 对应 |
|------|-----------|-------------|
| **SIMT** | GPU 上千个线程同时执行相同指令 | CUDA kernel 中的 `threadIdx` |
| **Kernel** | 在 GPU 上并行执行的函数 | `runtime-cuda` → nvrtc 编译的 10 个 kernel |
| **Host/Device** | CPU 是 Host，GPU 是 Device | `GpuTensor::from_host()` / `to_host()` |
| **cuBLAS** | NVIDIA 的矩阵乘法库（极度优化） | `runtime-cuda` → `gpu_matmul` |
| **传输瓶颈** | CPU↔GPU 数据搬运比计算本身更慢 | roadmap 中"阈值 128²"的决策 |

### 3.2 学习路径

- 资源：NVIDIA《CUDA C Programming Guide》前 3 章（概念即可，不需要写 CUDA）
- 动手：阅读 `runtime-cuda/src/lib.rs` 中的 `add_kernel` CUDA C 源码（最简单的 kernel）

### 3.3 关键理解

```c
// 这就是 GPU kernel 的全部秘密：
// 1000 个线程同时执行这一行，每个线程处理一个元素
__global__ void add_kernel(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
```

**验证方式**：能解释为什么 sptorch 对小矩阵（<128²）不用 GPU——传输开销大于计算收益。

---

## 阶段 4：分布式训练（1 周）

> 目标：理解多机训练的核心问题——梯度同步。

### 4.1 核心概念

| 概念 | 一句话解释 | sptorch 对应 |
|------|-----------|-------------|
| **数据并行** | 每台机器算不同数据的梯度，然后平均 | `distributed` → AllReduce |
| **AllReduce** | 所有节点的梯度求平均后广播回去 | `allreduce()` 函数 |
| **Barrier** | 等所有节点都完成再继续下一步 | `barrier()` 函数 |
| **gRPC** | 机器间通信协议 | `distributed` → tonic |

### 4.2 一句话总结

分布式训练 = 每台机器独立算梯度 + AllReduce 平均 + 同步更新参数。就这么简单。

---

## 阶段 5：硬件基础（持续学习）

> 目标：为 P8/P9 硬件载板做知识储备。

### 5.1 数字电路基础（2-3 周）

| 概念 | 你需要掌握的程度 | 推荐资源 |
|------|------------------|----------|
| 组合逻辑 | 知道与/或/非门、多路选择器 | 《数字设计和计算机体系结构》第 1-2 章 |
| 时序逻辑 | 知道触发器、寄存器、时钟 | 同上第 3 章 |
| Verilog 基础 | 能写一个简单的加法器 | 同上第 4 章 + Tang 9k 官方教程 |
| 脉动阵列 | 知道 PE（处理单元）如何流水线计算 matmul | 搜索"Systolic Array 动画" |

### 5.2 PCB 与 DDR4（跟随课程进度）

| 概念 | 你需要掌握的程度 | 对应课程 |
|------|------------------|----------|
| 阻抗匹配 | 知道为什么高速信号需要 50Ω | DDR4 PCB 课程第 1-8 讲 |
| 等长走线 | 知道为什么数据线要一样长 | 第 9-15 讲 |
| Fly-by 拓扑 | 知道地址线为什么串联而非并联 | 第 16-20 讲 |
| 电源完整性 | 知道去耦电容的作用 | 第 21-25 讲 |

---

## 学习节奏建议

```
Week 1-2:   阶段 0（数学基础）+ 每天跑一次 cargo test 观察输出
Week 3-4:   阶段 1（深度学习核心）+ 逐行阅读 cli-train/src/main.rs
Week 5-6:   阶段 2（Transformer）+ 画出 GPT forward 数据流图
Week 7:     阶段 3（GPU）+ 阅读 runtime-cuda 中最简单的 kernel
Week 8:     阶段 4（分布式）+ 阅读 distributed 中的 allreduce
Week 9+:    阶段 5（硬件）+ 跟随 DDR4 PCB 课程 + Tang 9k 实验
```

---

## 推荐资源汇总

### 视频（优先级从高到低）

| 资源 | 时长 | 覆盖内容 | 语言 |
|------|------|----------|------|
| 3Blue1Brown《微积分的本质》 | 3 小时 | 导数、链式法则 | 中/英 |
| 3Blue1Brown《线性代数的本质》 | 3 小时 | 向量、矩阵乘法 | 中/英 |
| 3Blue1Brown《神经网络》 | 1 小时 | 神经网络直觉 | 中/英 |
| Karpathy《Zero to Hero》#1 micrograd | 2.5 小时 | 从零实现 autograd | 英（有字幕） |
| Karpathy《Let's build GPT》 | 2 小时 | 从零实现 GPT | 英（有字幕） |

### 文章

| 资源 | 覆盖内容 |
|------|----------|
| Jay Alammar《The Illustrated Transformer》 | Attention 机制图解 |
| 项目内 `MATH.md` | 链式法则与 seed gradient |
| 项目内 `roadmap.md` 调试记录 | 实战经验（比教科书更有价值） |

### 书籍（参考，不需要通读）

| 书 | 用途 |
|----|------|
| 《深度学习入门：基于 Python 的理论与实现》（鱼书） | 最友好的 DL 入门书，代码简单 |
| 《数字设计和计算机体系结构》（Harris） | 硬件基础，为 P8/P9 准备 |

---

## 学习原则

1. **代码即教材**：每学一个概念，立刻在 sptorch 中找到对应实现。抽象概念 + 具体代码 = 真正理解。

2. **从 add 到 attention**：所有算子都是同一个模式（forward + backward + grad check）。理解了最简单的 `add`，就掌握了框架的灵魂。

3. **不求甚解也行**：有些数学推导（如 LayerNorm 的反向传播公式）可以先跳过。知道"它在做归一化"就够了，细节等需要时再深入。

4. **跑起来再说**：`cargo test` 和 `cargo run` 是你最好的老师。看到 loss 下降、看到生成文本变好，比读 10 页论文更有效。

5. **roadmap 中的调试记录是金矿**：那些"问题→分析→尝试→决策"的记录，是真实工程经验的浓缩。教科书不会告诉你"为什么 GPU 对小矩阵反而更慢"。

---

## 检查点（自测）

完成每个阶段后，用这些问题检验自己：

**阶段 0 完成标志**：
- [ ] 能口头解释"导数就是变化率"
- [ ] 能手算 f(x) = x² 在 x=3 处的导数（答案：6）
- [ ] 能解释链式法则：f(g(x)) 的导数为什么要"乘起来"

**阶段 1 完成标志**：
- [ ] 能解释 `cli-train/src/main.rs` 训练循环的 4 个步骤
- [ ] 能回答"为什么 loss 会下降"（因为参数沿梯度反方向更新）
- [ ] 能解释 AdamW 比 SGD 好在哪里（自适应学习率）

**阶段 2 完成标志**：
- [ ] 能画出 Transformer Block 的数据流图（LN→Attention→残差→LN→FFN→残差）
- [ ] 能解释 Q/K/V 的直觉含义
- [ ] 能解释 causal mask 为什么必要（防止偷看未来）

**阶段 3 完成标志**：
- [ ] 能解释 GPU 为什么适合矩阵运算（大量并行的简单计算）
- [ ] 能解释 sptorch 为什么对小矩阵不用 GPU（传输开销 > 计算收益）

**阶段 4 完成标志**：
- [ ] 能用一句话解释 AllReduce（所有节点梯度求平均后广播回去）

**阶段 5 完成标志**：
- [ ] 能解释脉动阵列如何计算矩阵乘法
- [ ] 能解释 DDR4 为什么需要等长走线（信号同时到达）

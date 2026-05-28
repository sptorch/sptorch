//! SPTorch 的神经网络模块层。
//!
//! 这一层把 `core-tensor` 和 `core-ops` 组合成更接近“模型搭建”的 API。
//! 它不追求复刻大型模型仓，而是优先提供训练闭环最常用、最稳定、最容易
//! 组合的小模型构件。
//!
//! 主要内容：
//! - [`Module`] 及常见实现：[`Linear`]、[`LoRALinear`]。
//! - 嵌入、归一化、注意力和 Transformer 块。
//! - [`GPT`] 级别的小型自回归语言模型骨架。
//! - 受约束解码相关的 [`TokenTrie`] 与 [`TokenConstraint`]。
//! - 贪心、采样和约束生成接口。

use rand::Rng;
use sptorch_core_ops::*;
use sptorch_core_tensor::Tensor;
use std::collections::VecDeque;

// ============ Module Trait ============

pub trait Module: Send + Sync {
    /// 执行前向计算。
    ///
    /// 这是模块层的最小行为契约：给定输入张量，返回输出张量。具体实现
    /// 可以在内部组合多个可微分算子，调用方只需要关心张量形状和训练语义。
    fn forward(&self, input: &Tensor) -> Tensor;

    /// 返回当前模块需要参与优化或保存的参数。
    ///
    /// 该列表用于优化器更新、checkpoint 保存和测试检查。默认只包含模型参数，
    /// 不包含临时缓存或推理中间态。
    fn parameters(&self) -> Vec<Tensor>;
}

/// 带稳定名称的参数条目。
///
/// 这是 `state_dict` 保存/加载的基础单元：名称负责稳定定位，Tensor 负责
/// 携带数值本体。只要模型结构没有语义变化，参数名就应尽量保持稳定。
#[derive(Debug, Clone)]
pub struct NamedParameter {
    /// 参数在模型中的稳定路径名。
    pub name: String,
    /// 参数张量本体。
    pub tensor: Tensor,
}

fn named_parameter(prefix: &str, suffix: &str, tensor: Tensor) -> NamedParameter {
    NamedParameter {
        name: format!("{prefix}.{suffix}"),
        tensor,
    }
}

// ============ Initialization ============
/// 使用 Xavier Uniform 初始化权重张量。
///
/// 适合线性层等近似对称的前向路径，能让初始激活和梯度在层间维持较稳定
/// 的尺度。返回值默认开启梯度追踪，可直接作为可训练参数。
pub fn xavier_uniform(rows: usize, cols: usize) -> Tensor {
    let mut rng = rand::thread_rng();
    let limit = (6.0 / (rows + cols) as f32).sqrt();
    let data: Vec<f32> = (0..rows * cols).map(|_| rng.gen_range(-limit..limit)).collect();
    Tensor::with_grad(data, vec![rows, cols], true)
}
/// 使用 Kaiming Normal 初始化权重张量。
///
/// 适合 ReLU / GELU 风格的非线性路径。当前实现使用 Box-Muller 采样，目标是
/// 提供稳定训练初始化，而不是复现某个外部框架的随机数序列。
pub fn kaiming_normal(rows: usize, cols: usize) -> Tensor {
    let mut rng = rand::thread_rng();
    let std = (2.0 / rows as f32).sqrt();
    let data: Vec<f32> = (0..rows * cols)
        .map(|_| {
            // Box-Muller transform for normal distribution
            let u1: f32 = rng.gen_range(1e-7..1.0);
            let u2: f32 = rng.gen_range(0.0..std::f32::consts::TAU);
            std * (-2.0 * u1.ln()).sqrt() * u2.cos()
        })
        .collect();
    Tensor::with_grad(data, vec![rows, cols], true)
}

// 创建全零可训练张量，通常用于初始化偏置。
fn zeros_grad(size: usize) -> Tensor {
    Tensor::with_grad(vec![0.0; size], vec![size], true)
}

// ============ Linear ============

pub struct Linear {
    /// 权重矩阵，形状为 `[out_features, in_features]`。
    pub weight: Tensor, // [out_features, in_features]
    /// 可选偏置，形状为 `[out_features]`。
    pub bias: Option<Tensor>, // [out_features]
}

impl Linear {
    /// 创建线性层并初始化权重和偏置。
    ///
    /// 权重采用 Xavier Uniform，偏置默认置零。前向语义为
    /// `input @ weight^T + bias`，输入最后一维必须等于 `in_features`。
    pub fn new(in_features: usize, out_features: usize, use_bias: bool) -> Self {
        let weight = xavier_uniform(out_features, in_features);
        let bias = if use_bias { Some(zeros_grad(out_features)) } else { None };
        Linear { weight, bias }
    }

    /// 返回线性层的命名参数。
    ///
    /// 名称使用 `prefix.weight` / `prefix.bias` 的形式，便于和更大模型的
    /// 层级路径拼接。
    pub fn named_parameters(&self, prefix: &str) -> Vec<NamedParameter> {
        let mut params = vec![named_parameter(prefix, "weight", self.weight.clone())];
        if let Some(ref bias) = self.bias {
            params.push(named_parameter(prefix, "bias", bias.clone()));
        }
        params
    }
}

impl Module for Linear {
    // 前向计算接口：接收输入张量并返回输出张量。
    fn forward(&self, input: &Tensor) -> Tensor {
        // input: [batch, in_features], weight: [out, in]
        // output = input @ weight^T + bias = [batch, out]
        let wt = transpose(&self.weight);
        let out = matmul(input, &wt);
        if let Some(ref bias) = self.bias {
            broadcast_add(&out, bias)
        } else {
            out
        }
    }

    // 返回当前模块需要优化或保存的参数张量列表。
    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![self.weight.clone()];
        if let Some(ref b) = self.bias {
            params.push(b.clone());
        }
        params
    }
}

// ============ Embedding ============

// ============ Dropout ============

pub struct Dropout {
    /// 丢弃概率。
    pub rate: f32,
    /// 是否处于训练模式。
    pub training: bool,
}

impl Dropout {
    /// 创建 Dropout 层并设置丢弃率。
    ///
    /// 默认处于训练模式。若用于评估、采样或权重导出，请显式调用
    /// [`Dropout::eval`]。
    pub fn new(rate: f32) -> Self {
        Dropout { rate, training: true }
    }
    /// 执行 Dropout 前向。
    ///
    /// 训练期会随机置零并使用 inverted dropout 缩放；评估期则直接透传输入。
    /// 这种语义适合大多数训练回路，能避免推理阶段再额外做缩放修正。
    pub fn forward(&self, input: &Tensor) -> Tensor {
        if !self.training || self.rate == 0.0 {
            return input.clone();
        }
        let mut rng = rand::thread_rng();
        let data = input.contiguous_data();
        let scale = 1.0 / (1.0 - self.rate);
        // Build mask tensor and use mul() to keep autograd graph connected
        let mask_data: Vec<f32> = data
            .iter()
            .map(|_| if rng.gen::<f32>() < self.rate { 0.0 } else { scale })
            .collect();
        let mask = Tensor::new(mask_data, input.shape());
        mul(input, &mask)
    }
    /// 切换到评估模式，禁用随机丢弃。
    pub fn eval(&mut self) {
        self.training = false;
    }
    /// 切换到训练模式，启用随机丢弃。
    pub fn train(&mut self) {
        self.training = true;
    }
}

// ============ LoRA Linear ============

/// LoRA 适配器线性层。
///
/// 它在冻结的基座线性层上叠加低秩增量，前向公式为：
/// `x @ W^T + x @ (B @ A)^T * (alpha / rank)`。
/// 其中 `W` 是冻结基座，`A` 和 `B` 是可训练参数。
pub struct LoRALinear {
    pub base: Linear,
    pub lora_a: Tensor, // [rank, in_features]
    pub lora_b: Tensor, // [out_features, rank]
    pub alpha: f32,
    pub rank: usize,
}

impl LoRALinear {
    /// 基于已有线性层创建 LoRA 适配器，并冻结基座参数。
    ///
    /// 这是轻量微调最常见的入口：保留原始模型的表达能力，再用低秩参数注入
    /// 任务特化增量。
    pub fn new(base: Linear, rank: usize, alpha: f32) -> Self {
        let in_features = base.weight.shape()[1];
        let out_features = base.weight.shape()[0];

        // Freeze base weight
        base.weight.0.write().unwrap().requires_grad = false;
        if let Some(ref b) = base.bias {
            b.0.write().unwrap().requires_grad = false;
        }

        // A: kaiming init, B: zeros (so LoRA starts as identity)
        let lora_a = kaiming_normal(rank, in_features);
        let lora_b = Tensor::with_grad(vec![0.0; out_features * rank], vec![out_features, rank], true);

        LoRALinear {
            base,
            lora_a,
            lora_b,
            alpha,
            rank,
        }
    }
    /// 按维度直接构造 LoRA 线性层。
    ///
    /// 适合在没有现成基座层的情况下快速搭建实验模型。
    pub fn from_dims(in_features: usize, out_features: usize, use_bias: bool, rank: usize, alpha: f32) -> Self {
        let base = Linear::new(in_features, out_features, use_bias);
        Self::new(base, rank, alpha)
    }
}

impl Module for LoRALinear {
    // 前向计算接口：接收输入张量并返回输出张量。
    fn forward(&self, input: &Tensor) -> Tensor {
        // Base: x @ W^T
        let base_out = self.base.forward(input);

        // LoRA: x @ A^T @ B^T * (alpha / rank)
        let at = transpose(&self.lora_a);
        let bt = transpose(&self.lora_b);
        let xa = matmul(input, &at); // [batch, rank]
        let xab = matmul(&xa, &bt); // [batch, out_features]
        let scaling = self.alpha / self.rank as f32;
        let lora_out = scale(&xab, scaling);

        add(&base_out, &lora_out)
    }

    // 返回当前模块需要优化或保存的参数张量列表。
    fn parameters(&self) -> Vec<Tensor> {
        // Only return trainable LoRA parameters
        vec![self.lora_a.clone(), self.lora_b.clone()]
    }
}

impl LoRALinear {
    /// 返回包含基座参数与 LoRA 参数的完整参数列表。
    ///
    /// 适合做完整保存、恢复或权重折叠前的审计。
    pub fn all_parameters(&self) -> Vec<Tensor> {
        let mut p = self.base.parameters();
        p.push(self.lora_a.clone());
        p.push(self.lora_b.clone());
        p
    }

    /// 返回包含基座参数与 LoRA 适配器参数的稳定命名参数。
    ///
    /// 命名路径采用 `prefix.base.*` 与 `prefix.lora_*`，便于 bundle 保存时区分
    /// 冻结基座和适配器参数。
    ///
    /// `parameters()` 只暴露可训练的 LoRA A/B；保存完整模型时还需要冻结的
    /// base 权重和 bias，否则加载后只能恢复适配器而无法还原完整前向。
    pub fn named_parameters(&self, prefix: &str) -> Vec<NamedParameter> {
        let mut params = self.base.named_parameters(&format!("{prefix}.base"));
        params.push(named_parameter(prefix, "lora_a", self.lora_a.clone()));
        params.push(named_parameter(prefix, "lora_b", self.lora_b.clone()));
        params
    }

    /// 将 LoRA 增量权重合并进基座权重。
    ///
    /// 这个操作会原地更新基座权重，适合训练完成后折叠适配器做部署或推理。
    pub fn merge(&self) {
        let a_data = self.lora_a.contiguous_data();
        let b_data = self.lora_b.contiguous_data();
        let out_features = self.base.weight.shape()[0];
        let in_features = self.base.weight.shape()[1];
        let scaling = self.alpha / self.rank as f32;

        // B @ A: [out, rank] @ [rank, in] = [out, in]
        let mut ba = vec![0.0f32; out_features * in_features];
        for i in 0..out_features {
            for j in 0..in_features {
                let mut sum = 0.0f32;
                for r in 0..self.rank {
                    sum += b_data[i * self.rank + r] * a_data[r * in_features + j];
                }
                ba[i * in_features + j] = sum * scaling;
            }
        }

        // W += B @ A * scaling
        let inner = self.base.weight.0.read().unwrap();
        let mut storage = inner.storage.write().unwrap();
        let w = storage.as_cpu_slice_mut();
        for i in 0..w.len() {
            w[i] += ba[i];
        }
    }
}

// ============ Embedding ============

pub struct Embedding {
    /// 嵌入表权重，形状为 `[num_embeddings, embedding_dim]`。
    pub weight: Tensor, // [num_embeddings, embedding_dim]
}

impl Embedding {
    /// 创建 Embedding 层。
    ///
    /// 权重以均匀随机值初始化，适合 token / position embedding 的训练起点。
    pub fn new(num_embeddings: usize, embedding_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let data: Vec<f32> = (0..num_embeddings * embedding_dim)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();
        Embedding {
            weight: Tensor::with_grad(data, vec![num_embeddings, embedding_dim], true),
        }
    }
    /// 按 token 索引查表并返回 embedding。
    ///
    /// 输入是 token ID 切片，输出是对应行向量按顺序堆叠的张量。
    pub fn forward_indices(&self, indices: &[usize]) -> Tensor {
        embedding_lookup(&self.weight, indices)
    }
    /// 返回 Embedding 可训练参数。
    pub fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone()]
    }

    /// 返回 Embedding 的命名参数。
    pub fn named_parameters(&self, prefix: &str) -> Vec<NamedParameter> {
        vec![named_parameter(prefix, "weight", self.weight.clone())]
    }
}

// ============ LayerNorm ============

pub struct LayerNorm {
    /// 归一化缩放参数，形状为 `[normalized_shape]`。
    pub gamma: Tensor, // [normalized_shape]
    /// 归一化平移参数，形状为 `[normalized_shape]`。
    pub beta: Tensor, // [normalized_shape]
    /// 数值稳定项。
    pub eps: f32,
    /// 被归一化的最后一维长度。
    pub normalized_shape: usize,
}

impl LayerNorm {
    /// 创建 LayerNorm 层。
    ///
    /// `gamma` 初始化为 1，`beta` 初始化为 0。这是训练中最常见的 LayerNorm 起点。
    pub fn new(normalized_shape: usize) -> Self {
        LayerNorm {
            gamma: Tensor::with_grad(vec![1.0; normalized_shape], vec![normalized_shape], true),
            beta: Tensor::with_grad(vec![0.0; normalized_shape], vec![normalized_shape], true),
            eps: 1e-5,
            normalized_shape,
        }
    }

    /// 返回 LayerNorm 的命名参数。
    pub fn named_parameters(&self, prefix: &str) -> Vec<NamedParameter> {
        vec![
            named_parameter(prefix, "gamma", self.gamma.clone()),
            named_parameter(prefix, "beta", self.beta.clone()),
        ]
    }
}

impl Module for LayerNorm {
    // 前向计算接口：接收输入张量并返回输出张量。
    fn forward(&self, input: &Tensor) -> Tensor {
        let shape = input.shape();
        let data = input.contiguous_data();
        let gamma = self.gamma.contiguous_data();
        let beta = self.beta.contiguous_data();
        let dim = self.normalized_shape;

        let leading: usize = data.len() / dim;
        let mut normalized = vec![0.0f32; data.len()];
        let mut out = vec![0.0f32; data.len()];
        for b in 0..leading {
            let off = b * dim;
            let mean: f32 = (0..dim).map(|i| data[off + i]).sum::<f32>() / dim as f32;
            let var: f32 = (0..dim).map(|i| (data[off + i] - mean).powi(2)).sum::<f32>() / dim as f32;
            let inv_std = 1.0 / (var + self.eps).sqrt();
            for i in 0..dim {
                normalized[off + i] = (data[off + i] - mean) * inv_std;
                out[off + i] = gamma[i] * normalized[off + i] + beta[i];
            }
        }

        let res = Tensor::new(out, shape.clone());

        if input.requires_grad() || self.gamma.requires_grad() || self.beta.requires_grad() {
            let mut inner = res.0.write().unwrap();
            inner.requires_grad = true;
            inner.creator = Some(std::sync::Arc::new(sptorch_core_tensor::Node {
                op: Box::new(LayerNormOp {
                    normalized,
                    gamma,
                    input_data: data,
                    eps: self.eps,
                    dim,
                    shape,
                }),
                inputs: vec![input.clone(), self.gamma.clone(), self.beta.clone()],
            }));
        }

        res
    }

    // 返回当前模块需要优化或保存的参数张量列表。
    fn parameters(&self) -> Vec<Tensor> {
        vec![self.gamma.clone(), self.beta.clone()]
    }
}

// ============ RMSNorm ============

/// RMSNorm 模块。
///
/// 与 LayerNorm 不同，RMSNorm 不减均值，只按均方根缩放最后一维。这是
/// Llama/Qwen 风格 decoder block 里更常见的归一化方式，计算路径更短，
/// 也更贴合后续 fused kernel 的实现边界。
pub struct RmsNorm {
    /// 缩放参数，形状为 `[normalized_shape]`。
    pub weight: Tensor,
    /// 数值稳定项。
    pub eps: f32,
    /// 被归一化的最后一维长度。
    pub normalized_shape: usize,
}

impl RmsNorm {
    /// 创建 RMSNorm，缩放参数初始化为 1。
    pub fn new(normalized_shape: usize) -> Self {
        RmsNorm {
            weight: Tensor::with_grad(vec![1.0; normalized_shape], vec![normalized_shape], true),
            eps: 1e-6,
            normalized_shape,
        }
    }

    /// 返回 RMSNorm 的命名参数。
    pub fn named_parameters(&self, prefix: &str) -> Vec<NamedParameter> {
        vec![named_parameter(prefix, "weight", self.weight.clone())]
    }
}

impl Module for RmsNorm {
    fn forward(&self, input: &Tensor) -> Tensor {
        rms_norm(input, &self.weight, self.eps)
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone()]
    }
}

#[derive(Debug)]
struct LayerNormOp {
    normalized: Vec<f32>, // (x - mean) / std for each element
    gamma: Vec<f32>,
    input_data: Vec<f32>,
    eps: f32,
    dim: usize,
    shape: Vec<usize>,
}

impl sptorch_core_tensor::Op for LayerNormOp {
    // 反向传播：根据上游梯度计算对输入/参数的梯度。
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        let g = grad_output.contiguous_data();
        let dim = self.dim;
        let leading = g.len() / dim;

        // d_gamma: sum over batch of grad * normalized
        let mut d_gamma = vec![0.0f32; dim];
        // d_beta: sum over batch of grad
        let mut d_beta = vec![0.0f32; dim];
        // d_input
        let mut d_input = vec![0.0f32; g.len()];

        for b in 0..leading {
            let off = b * dim;
            let mean: f32 = (0..dim).map(|i| self.input_data[off + i]).sum::<f32>() / dim as f32;
            let var: f32 = (0..dim).map(|i| (self.input_data[off + i] - mean).powi(2)).sum::<f32>() / dim as f32;
            let inv_std = 1.0 / (var + self.eps).sqrt();

            for i in 0..dim {
                d_gamma[i] += g[off + i] * self.normalized[off + i];
                d_beta[i] += g[off + i];
            }

            // d_input for this row
            // g_hat = g * gamma
            let mut g_hat = vec![0.0f32; dim];
            for i in 0..dim {
                g_hat[i] = g[off + i] * self.gamma[i];
            }
            let g_hat_mean: f32 = g_hat.iter().sum::<f32>() / dim as f32;
            let g_hat_norm_mean: f32 = g_hat
                .iter()
                .zip(self.normalized[off..off + dim].iter())
                .map(|(gh, n)| gh * n)
                .sum::<f32>()
                / dim as f32;

            for i in 0..dim {
                d_input[off + i] = inv_std * (g_hat[i] - g_hat_mean - self.normalized[off + i] * g_hat_norm_mean);
            }
        }

        vec![
            Some(Tensor::new(d_input, self.shape.clone())),
            Some(Tensor::new(d_gamma, vec![dim])),
            Some(Tensor::new(d_beta, vec![dim])),
        ]
    }
}

// ============ MultiHeadAttention ============

pub struct MultiHeadAttention {
    /// 注意力头数。
    pub n_head: usize,
    /// 模型维度。
    pub d_model: usize,
    /// 每个 head 的维度。
    pub head_dim: usize,
    /// query 投影。
    pub wq: Linear,
    /// key 投影。
    pub wk: Linear,
    /// value 投影。
    pub wv: Linear,
    /// 输出投影。
    pub wo: Linear,
}

impl MultiHeadAttention {
    /// 创建多头自注意力层，要求 `d_model % n_head == 0`。
    ///
    /// 这里默认不带 bias，便于把注意力路径和 FFN 路径的参数语义保持清楚。
    pub fn new(d_model: usize, n_head: usize) -> Self {
        assert_eq!(d_model % n_head, 0);
        let head_dim = d_model / n_head;
        MultiHeadAttention {
            n_head,
            d_model,
            head_dim,
            wq: Linear::new(d_model, d_model, false),
            wk: Linear::new(d_model, d_model, false),
            wv: Linear::new(d_model, d_model, false),
            wo: Linear::new(d_model, d_model, false),
        }
    }
    /// 执行带因果掩码的自注意力前向。
    ///
    /// 适合自回归语言模型，不适合双向编码器语义。
    fn forward_causal_impl(&self, input: &Tensor, rope_base: Option<f32>) -> Tensor {
        let shape = input.shape();
        let seq_len = shape[0];
        let h = self.n_head;
        let hd = self.head_dim;

        // Q, K, V projections: [seq_len, d_model]
        let q = self.wq.forward(input);
        let k = self.wk.forward(input);
        let v = self.wv.forward(input);

        // Reshape to [n_head, seq_len, head_dim]
        let mut q3 = reshape_to_heads(&q, seq_len, h, hd);
        let mut k3 = reshape_to_heads(&k, seq_len, h, hd);
        let v3 = reshape_to_heads(&v, seq_len, h, hd);

        if let Some(rope_base) = rope_base {
            q3 = rotary_position_embedding(&q3, rope_base);
            k3 = rotary_position_embedding(&k3, rope_base);
        }

        // Attention scores: [n_head, seq_len, seq_len]
        let kt = batch_transpose(&k3, h, seq_len, hd);
        let scores = batch_matmul(&q3, &kt);
        let scores = scale(&scores, 1.0 / (hd as f32).sqrt());

        // Causal mask
        let mut mask = vec![false; h * seq_len * seq_len];
        for hi in 0..h {
            for i in 0..seq_len {
                for j in 0..seq_len {
                    if j > i {
                        mask[hi * seq_len * seq_len + i * seq_len + j] = true;
                    }
                }
            }
        }
        let scores = masked_fill(&scores, &mask, f32::NEG_INFINITY);

        // Softmax per row: reshape to [n_head * seq_len, seq_len] for 2D softmax
        let scores_2d = reshape(&scores, vec![h * seq_len, seq_len]);
        let attn = softmax(&scores_2d);
        let attn_3d = reshape(&attn, vec![h, seq_len, seq_len]);

        // Weighted values: [n_head, seq_len, head_dim]
        let out = batch_matmul(&attn_3d, &v3);

        // Reshape back to [seq_len, d_model]
        let out_2d = reshape_from_heads(&out, seq_len, h, hd);

        // Output projection
        self.wo.forward(&out_2d)
    }

    /// 执行带因果掩码的自注意力前向。
    ///
    /// 适合自回归语言模型，不适合双向编码器语义。
    pub fn forward_causal(&self, input: &Tensor) -> Tensor {
        self.forward_causal_impl(input, None)
    }

    /// 执行带 RoPE 的因果自注意力前向。
    ///
    /// 这条路径在 Qwen/Llama 风格模型里更常见：位置信息先注入 Q/K，再做因果
    /// attention。
    pub fn forward_causal_rope(&self, input: &Tensor, rope_base: f32) -> Tensor {
        self.forward_causal_impl(input, Some(rope_base))
    }

    /// 计算当前输入对应的 Key/Value cache 张量。
    ///
    /// 返回的 K/V shape 均为 `[n_head, seq_len, head_dim]`。这不是增量 attention
    /// 的最终内核，只是把“本层应该缓存什么”从完整前向路径里抽出来，方便
    /// 推理运行态先建立 prefill/decode 生命周期。
    pub fn project_kv_cache(&self, input: &Tensor, rope_base: Option<f32>) -> (Tensor, Tensor) {
        let shape = input.shape();
        let seq_len = shape[0];
        let mut key = reshape_to_heads(&self.wk.forward(input), seq_len, self.n_head, self.head_dim);
        let value = reshape_to_heads(&self.wv.forward(input), seq_len, self.n_head, self.head_dim);

        if let Some(rope_base) = rope_base {
            key = rotary_position_embedding(&key, rope_base);
        }

        (key.detach(), value.detach())
    }

    /// 返回注意力层全部参数。
    pub fn parameters(&self) -> Vec<Tensor> {
        let mut p = Vec::new();
        p.extend(self.wq.parameters());
        p.extend(self.wk.parameters());
        p.extend(self.wv.parameters());
        p.extend(self.wo.parameters());
        p
    }

    /// 返回 MultiHeadAttention 的命名参数。
    pub fn named_parameters(&self, prefix: &str) -> Vec<NamedParameter> {
        let mut params = Vec::new();
        params.extend(self.wq.named_parameters(&format!("{prefix}.wq")));
        params.extend(self.wk.named_parameters(&format!("{prefix}.wk")));
        params.extend(self.wv.named_parameters(&format!("{prefix}.wv")));
        params.extend(self.wo.named_parameters(&format!("{prefix}.wo")));
        params
    }
}

/// [seq_len, d_model] -> [n_head, seq_len, head_dim]
fn reshape_to_heads(x: &Tensor, seq_len: usize, n_head: usize, head_dim: usize) -> Tensor {
    // x is [seq_len, d_model] where d_model = n_head * head_dim
    // We need [n_head, seq_len, head_dim]
    let data = x.contiguous_data();
    let d_model = n_head * head_dim;
    let mut out = vec![0.0f32; n_head * seq_len * head_dim];
    for s in 0..seq_len {
        for h in 0..n_head {
            for d in 0..head_dim {
                out[h * seq_len * head_dim + s * head_dim + d] = data[s * d_model + h * head_dim + d];
            }
        }
    }

    let res = Tensor::new(out, vec![n_head, seq_len, head_dim]);

    if x.requires_grad() {
        let mut inner = res.0.write().unwrap();
        inner.requires_grad = true;
        inner.creator = Some(std::sync::Arc::new(sptorch_core_tensor::Node {
            op: Box::new(ReshapeToHeadsOp {
                seq_len,
                n_head,
                head_dim,
            }),
            inputs: vec![x.clone()],
        }));
    }

    res
}

#[derive(Debug)]
struct ReshapeToHeadsOp {
    seq_len: usize,
    n_head: usize,
    head_dim: usize,
}

impl sptorch_core_tensor::Op for ReshapeToHeadsOp {
    // 反向传播：根据上游梯度计算对输入/参数的梯度。
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        // Reverse: [n_head, seq_len, head_dim] -> [seq_len, d_model]
        let g = grad_output.contiguous_data();
        let d_model = self.n_head * self.head_dim;
        let mut out = vec![0.0f32; self.seq_len * d_model];
        for s in 0..self.seq_len {
            for h in 0..self.n_head {
                for d in 0..self.head_dim {
                    out[s * d_model + h * self.head_dim + d] =
                        g[h * self.seq_len * self.head_dim + s * self.head_dim + d];
                }
            }
        }
        vec![Some(Tensor::new(out, vec![self.seq_len, d_model]))]
    }
}

/// [n_head, seq_len, head_dim] -> [seq_len, d_model]
fn reshape_from_heads(x: &Tensor, seq_len: usize, n_head: usize, head_dim: usize) -> Tensor {
    let data = x.contiguous_data();
    let d_model = n_head * head_dim;
    let mut out = vec![0.0f32; seq_len * d_model];
    for s in 0..seq_len {
        for h in 0..n_head {
            for d in 0..head_dim {
                out[s * d_model + h * head_dim + d] = data[h * seq_len * head_dim + s * head_dim + d];
            }
        }
    }

    let res = Tensor::new(out, vec![seq_len, d_model]);

    if x.requires_grad() {
        let mut inner = res.0.write().unwrap();
        inner.requires_grad = true;
        inner.creator = Some(std::sync::Arc::new(sptorch_core_tensor::Node {
            op: Box::new(ReshapeFromHeadsOp {
                seq_len,
                n_head,
                head_dim,
            }),
            inputs: vec![x.clone()],
        }));
    }

    res
}

#[derive(Debug)]
struct ReshapeFromHeadsOp {
    seq_len: usize,
    n_head: usize,
    head_dim: usize,
}

impl sptorch_core_tensor::Op for ReshapeFromHeadsOp {
    // 反向传播：根据上游梯度计算对输入/参数的梯度。
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        // Reverse: [seq_len, d_model] -> [n_head, seq_len, head_dim]
        let g = grad_output.contiguous_data();
        let d_model = self.n_head * self.head_dim;
        let mut out = vec![0.0f32; self.n_head * self.seq_len * self.head_dim];
        for s in 0..self.seq_len {
            for h in 0..self.n_head {
                for d in 0..self.head_dim {
                    out[h * self.seq_len * self.head_dim + s * self.head_dim + d] =
                        g[s * d_model + h * self.head_dim + d];
                }
            }
        }
        vec![Some(Tensor::new(out, vec![self.n_head, self.seq_len, self.head_dim]))]
    }
}

/// Transpose last two dims of [B, M, N] -> [B, N, M]
fn batch_transpose(x: &Tensor, batch: usize, rows: usize, cols: usize) -> Tensor {
    let data = x.contiguous_data();
    let mut out = vec![0.0f32; batch * rows * cols];
    for b in 0..batch {
        let off = b * rows * cols;
        for i in 0..rows {
            for j in 0..cols {
                out[b * cols * rows + j * rows + i] = data[off + i * cols + j];
            }
        }
    }

    let res = Tensor::new(out, vec![batch, cols, rows]);

    if x.requires_grad() {
        let mut inner = res.0.write().unwrap();
        inner.requires_grad = true;
        inner.creator = Some(std::sync::Arc::new(sptorch_core_tensor::Node {
            op: Box::new(BatchTransposeOp { batch, rows, cols }),
            inputs: vec![x.clone()],
        }));
    }

    res
}

#[derive(Debug)]
struct BatchTransposeOp {
    batch: usize,
    rows: usize,
    cols: usize,
}

impl sptorch_core_tensor::Op for BatchTransposeOp {
    // 反向传播：根据上游梯度计算对输入/参数的梯度。
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        // Transpose back: [B, cols, rows] -> [B, rows, cols]
        let g = grad_output.contiguous_data();
        let mut out = vec![0.0f32; self.batch * self.rows * self.cols];
        for b in 0..self.batch {
            for i in 0..self.cols {
                for j in 0..self.rows {
                    out[b * self.rows * self.cols + j * self.cols + i] =
                        g[b * self.cols * self.rows + i * self.rows + j];
                }
            }
        }
        vec![Some(Tensor::new(out, vec![self.batch, self.rows, self.cols]))]
    }
}

fn apply_rope_raw(
    data: &[f32],
    n_head: usize,
    seq_len: usize,
    head_dim: usize,
    rope_base: f32,
    inverse: bool,
) -> Vec<f32> {
    assert_eq!(head_dim % 2, 0, "rope requires an even head dimension");
    let half = head_dim / 2;
    let mut out = vec![0.0f32; data.len()];

    for h in 0..n_head {
        for pos in 0..seq_len {
            let off = h * seq_len * head_dim + pos * head_dim;
            for pair in 0..half {
                let theta = pos as f32 / rope_base.powf((2 * pair) as f32 / head_dim as f32);
                let (sin, cos) = theta.sin_cos();
                let sin = if inverse { -sin } else { sin };
                let even = data[off + 2 * pair];
                let odd = data[off + 2 * pair + 1];
                out[off + 2 * pair] = even * cos - odd * sin;
                out[off + 2 * pair + 1] = even * sin + odd * cos;
            }
        }
    }

    out
}

/// 对注意力头张量应用 Rotary Position Embedding。
///
/// 输入和输出 shape 都是 `[n_head, seq_len, head_dim]`。这个实现不绑定具体模型，
/// 只负责把位置信息以旋转方式注入到 query/key 的偶数维与奇数维配对中。
pub fn rotary_position_embedding(x: &Tensor, rope_base: f32) -> Tensor {
    let shape = x.shape();
    assert_eq!(shape.len(), 3, "rope expects [n_head, seq_len, head_dim]");
    let data = x.contiguous_data();
    let out = apply_rope_raw(&data, shape[0], shape[1], shape[2], rope_base, false);
    let res = Tensor::new(out, shape.clone());

    if x.requires_grad() {
        let mut inner = res.0.write().unwrap();
        inner.requires_grad = true;
        inner.creator = Some(std::sync::Arc::new(sptorch_core_tensor::Node {
            op: Box::new(RotaryEmbeddingOp {
                n_head: shape[0],
                seq_len: shape[1],
                head_dim: shape[2],
                rope_base,
            }),
            inputs: vec![x.clone()],
        }));
    }

    res
}

#[derive(Debug)]
struct RotaryEmbeddingOp {
    n_head: usize,
    seq_len: usize,
    head_dim: usize,
    rope_base: f32,
}

impl sptorch_core_tensor::Op for RotaryEmbeddingOp {
    // 反向传播时只需要把梯度沿相反角度旋回去即可，因为 RoPE 本身是正交旋转。
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        let g = grad_output.contiguous_data();
        let out = apply_rope_raw(&g, self.n_head, self.seq_len, self.head_dim, self.rope_base, true);
        vec![Some(Tensor::new(out, vec![self.n_head, self.seq_len, self.head_dim]))]
    }
}

// ============ TransformerBlock ============

pub struct TransformerBlock {
    /// 注意力前 LayerNorm。
    pub ln1: LayerNorm,
    /// 因果自注意力模块。
    pub attn: MultiHeadAttention,
    /// FFN 前 LayerNorm。
    pub ln2: LayerNorm,
    /// FFN 上投影。
    pub ffn_up: Linear,
    /// FFN 下投影。
    pub ffn_down: Linear,
    /// 注意力残差分支 dropout。
    pub attn_dropout: Dropout,
    /// FFN 残差分支 dropout。
    pub ffn_dropout: Dropout,
}

/// 以 RoPE 取代绝对位置 embedding 的 Transformer block。
///
/// 这个块更接近 Llama/Qwen 风格：位置语义进入 Q/K 旋转，不再依赖显式 position
/// embedding。它保留当前框架最稳定的 LayerNorm + causal attention + FFN 结构，
/// 只是把位置建模方式升级成更适合长上下文与 Text-to-SQL 任务的形式。
pub struct QwenLikeBlock {
    pub norm1: RmsNorm,
    pub attn: MultiHeadAttention,
    pub norm2: RmsNorm,
    pub ffn_up: Linear,
    pub ffn_gate: Linear,
    pub ffn_down: Linear,
    pub attn_dropout: Dropout,
    pub ffn_dropout: Dropout,
    pub rope_base: f32,
}

impl QwenLikeBlock {
    /// 创建一个 RoPE 风格的 Transformer block。
    pub fn new(d_model: usize, n_head: usize, d_ff: usize) -> Self {
        QwenLikeBlock {
            norm1: RmsNorm::new(d_model),
            attn: MultiHeadAttention::new(d_model, n_head),
            norm2: RmsNorm::new(d_model),
            ffn_up: Linear::new(d_model, d_ff, true),
            ffn_gate: Linear::new(d_model, d_ff, true),
            ffn_down: Linear::new(d_ff, d_model, true),
            attn_dropout: Dropout::new(0.1),
            ffn_dropout: Dropout::new(0.1),
            rope_base: 10000.0,
        }
    }

    /// 前向路径与标准 block 一致，只是注意力内部把 Q/K 改成 RoPE 旋转。
    pub fn forward_seq(&self, input: &Tensor) -> Tensor {
        let normed = self.norm1.forward(input);
        let attn_out = self.attn.forward_causal_rope(&normed, self.rope_base);
        let attn_out = self.attn_dropout.forward(&attn_out);
        let x = add(input, &attn_out);

        let normed2 = self.norm2.forward(&x);
        let ffn_out = swiglu(&self.ffn_up.forward(&normed2), &self.ffn_gate.forward(&normed2));
        let ffn_out = self.ffn_dropout.forward(&self.ffn_down.forward(&ffn_out));
        add(&x, &ffn_out)
    }

    /// 推理态的 block 前向，不经过 dropout。
    ///
    /// KV cache 的 materialize 过程需要一个稳定、可复现的残差流，因此这里
    /// 显式绕过 dropout，而不是依赖模型当前是否处于 eval 模式。
    pub fn forward_seq_inference(&self, input: &Tensor) -> Tensor {
        let normed = self.norm1.forward(input);
        let attn_out = self.attn.forward_causal_rope(&normed, self.rope_base);
        let x = add(input, &attn_out);

        let normed2 = self.norm2.forward(&x);
        let ffn_out = swiglu(&self.ffn_up.forward(&normed2), &self.ffn_gate.forward(&normed2));
        let ffn_out = self.ffn_down.forward(&ffn_out);
        add(&x, &ffn_out)
    }

    /// 切换训练/评估状态。
    pub fn set_training(&mut self, training: bool) {
        if training {
            self.attn_dropout.train();
            self.ffn_dropout.train();
        } else {
            self.attn_dropout.eval();
            self.ffn_dropout.eval();
        }
    }

    /// 返回块内全部参数。
    pub fn parameters(&self) -> Vec<Tensor> {
        let mut p = Vec::new();
        p.extend(self.norm1.parameters());
        p.extend(self.attn.parameters());
        p.extend(self.norm2.parameters());
        p.extend(self.ffn_up.parameters());
        p.extend(self.ffn_gate.parameters());
        p.extend(self.ffn_down.parameters());
        p
    }

    /// 返回块内命名参数。
    pub fn named_parameters(&self, prefix: &str) -> Vec<NamedParameter> {
        let mut params = Vec::new();
        params.extend(self.norm1.named_parameters(&format!("{prefix}.norm1")));
        params.extend(self.attn.named_parameters(&format!("{prefix}.attn")));
        params.extend(self.norm2.named_parameters(&format!("{prefix}.norm2")));
        params.extend(self.ffn_up.named_parameters(&format!("{prefix}.ffn_up")));
        params.extend(self.ffn_gate.named_parameters(&format!("{prefix}.ffn_gate")));
        params.extend(self.ffn_down.named_parameters(&format!("{prefix}.ffn_down")));
        params
    }
}

impl TransformerBlock {
    /// 创建 TransformerBlock。
    ///
    /// 结构为 `LayerNorm -> causal attention -> residual -> LayerNorm -> FFN -> residual`。
    /// 这是小型 GPT 模型里最常见的基础块。
    pub fn new(d_model: usize, n_head: usize, d_ff: usize) -> Self {
        TransformerBlock {
            ln1: LayerNorm::new(d_model),
            attn: MultiHeadAttention::new(d_model, n_head),
            ln2: LayerNorm::new(d_model),
            ffn_up: Linear::new(d_model, d_ff, true),
            ffn_down: Linear::new(d_ff, d_model, true),
            attn_dropout: Dropout::new(0.1),
            ffn_dropout: Dropout::new(0.1),
        }
    }
    /// 执行 TransformerBlock 前向，包含两条残差连接。
    ///
    /// 输入和输出 shape 都应为 `[seq_len, d_model]`。如果 dropout 处于训练模式，
    /// 该函数会引入随机性；评估和 checkpoint 验证时应切到 eval。
    pub fn forward_seq(&self, input: &Tensor) -> Tensor {
        let normed = self.ln1.forward(input);
        let attn_out = self.attn.forward_causal(&normed);
        let attn_out = self.attn_dropout.forward(&attn_out);
        let x = add(input, &attn_out);

        let normed2 = self.ln2.forward(&x);
        let ffn_out = gelu(&self.ffn_up.forward(&normed2));
        let ffn_out = self.ffn_dropout.forward(&self.ffn_down.forward(&ffn_out));
        add(&x, &ffn_out)
    }

    /// 推理态的 block 前向，不经过 dropout。
    ///
    /// 这条路径用于 KV cache materialize 和推理侧语义对齐：与 `forward_seq`
    /// 的主要区别是它不会引入训练时的随机失活。
    pub fn forward_seq_inference(&self, input: &Tensor) -> Tensor {
        let normed = self.ln1.forward(input);
        let attn_out = self.attn.forward_causal(&normed);
        let x = add(input, &attn_out);

        let normed2 = self.ln2.forward(&x);
        let ffn_out = gelu(&self.ffn_up.forward(&normed2));
        let ffn_out = self.ffn_down.forward(&ffn_out);
        add(&x, &ffn_out)
    }
    /// 设置块内 Dropout 的训练/评估状态。
    pub fn set_training(&mut self, training: bool) {
        if training {
            self.attn_dropout.train();
            self.ffn_dropout.train();
        } else {
            self.attn_dropout.eval();
            self.ffn_dropout.eval();
        }
    }
    /// 返回 TransformerBlock 全部参数。
    pub fn parameters(&self) -> Vec<Tensor> {
        let mut p = Vec::new();
        p.extend(self.ln1.parameters());
        p.extend(self.attn.parameters());
        p.extend(self.ln2.parameters());
        p.extend(self.ffn_up.parameters());
        p.extend(self.ffn_down.parameters());
        p
    }

    /// 返回 TransformerBlock 的命名参数。
    ///
    /// 参数名会继续沿用传入的 `prefix`，例如 `blocks.0.attn.wq.weight`。
    pub fn named_parameters(&self, prefix: &str) -> Vec<NamedParameter> {
        let mut params = Vec::new();
        params.extend(self.ln1.named_parameters(&format!("{prefix}.ln1")));
        params.extend(self.attn.named_parameters(&format!("{prefix}.attn")));
        params.extend(self.ln2.named_parameters(&format!("{prefix}.ln2")));
        params.extend(self.ffn_up.named_parameters(&format!("{prefix}.ffn_up")));
        params.extend(self.ffn_down.named_parameters(&format!("{prefix}.ffn_down")));
        params
    }
}

// ============ GPT Model ============

pub struct GPT {
    /// token embedding。
    pub token_emb: Embedding,
    /// position embedding。
    pub pos_emb: Embedding,
    /// Transformer block 堆叠。
    pub blocks: Vec<TransformerBlock>,
    /// 输出前最终 LayerNorm。
    pub ln_f: LayerNorm,
    /// 语言模型输出头。
    pub lm_head: Linear,
    /// 最大上下文长度。
    pub seq_len: usize,
}

/// 采用 RoPE + SwiGLU 的小型自回归模型骨架。
///
/// 这条路径更接近 Qwen/Llama 的训练语义：没有显式 position embedding，而是把
/// 位置编码直接注入注意力的 Q/K；FFN 侧使用门控激活而不是单一路径激活。
pub struct QwenLikeGPT {
    pub token_emb: Embedding,
    pub blocks: Vec<QwenLikeBlock>,
    pub norm_f: RmsNorm,
    pub lm_head: Linear,
    pub seq_len: usize,
}

/// 能为自回归推理运行态刷新 KV cache 的模型接口。
///
/// 这是在线推理 runtime 和具体模型之间的最小桥梁：runtime 不关心模型内部是
/// GPT 绝对位置编码、Qwen/Llama RoPE，还是未来的其他 decoder，只要求模型能
/// 声明 cache 规格，并把当前 token 上下文 materialize 到 cache 里。
pub trait PrefillCache {
    /// 返回模型需要的 KV cache 规格。
    fn kv_cache_spec(&self) -> KvCacheSpec;

    /// 将 `token_ids` 对应的上下文写入 `cache`。
    fn prefill_kv_cache(&self, token_ids: &[usize], cache: &mut KvCache);
}

impl GPT {
    /// 创建 GPT 模型。
    ///
    /// 这是面向训练语义的小型自回归语言模型骨架。它强调参数名稳定、
    /// forward/backward 清晰和 checkpoint 可回放，而不是追求大模型完整特性。
    pub fn new(vocab_size: usize, d_model: usize, n_head: usize, n_layer: usize, d_ff: usize, seq_len: usize) -> Self {
        let blocks = (0..n_layer)
            .map(|_| TransformerBlock::new(d_model, n_head, d_ff))
            .collect();
        GPT {
            token_emb: Embedding::new(vocab_size, d_model),
            pos_emb: Embedding::new(seq_len, d_model),
            blocks,
            ln_f: LayerNorm::new(d_model),
            lm_head: Linear::new(d_model, vocab_size, false),
            seq_len,
        }
    }
    /// 将 token 序列前向为 logits。
    ///
    /// 输入是 token ID 序列，长度不能超过 `seq_len`；输出 shape 为
    /// `[input_len, vocab_size]`，可直接用于 next-token cross entropy。
    pub fn forward_ids(&self, token_ids: &[usize]) -> Tensor {
        let slen = token_ids.len();
        assert!(slen <= self.seq_len);

        let positions: Vec<usize> = (0..slen).collect();
        let tok = self.token_emb.forward_indices(token_ids);
        let pos = self.pos_emb.forward_indices(&positions);
        let mut x = add(&tok, &pos);

        for block in &self.blocks {
            x = block.forward_seq(&x);
        }

        let x = self.ln_f.forward(&x);
        self.lm_head.forward(&x)
    }
    /// 返回 GPT 全部可训练参数。
    pub fn parameters(&self) -> Vec<Tensor> {
        let mut p = Vec::new();
        p.extend(self.token_emb.parameters());
        p.extend(self.pos_emb.parameters());
        for block in &self.blocks {
            p.extend(block.parameters());
        }
        p.extend(self.ln_f.parameters());
        p.extend(self.lm_head.parameters());
        p
    }

    /// 返回 GPT 的稳定命名参数。
    ///
    /// 名字采用层级路径格式，例如 `blocks.0.attn.wq.weight`。这使得保存、
    /// 恢复和局部调试都可以按名字定位，而不是依赖参数顺序。
    pub fn named_parameters(&self) -> Vec<NamedParameter> {
        let mut params = Vec::new();
        params.extend(self.token_emb.named_parameters("token_emb"));
        params.extend(self.pos_emb.named_parameters("pos_emb"));
        for (idx, block) in self.blocks.iter().enumerate() {
            params.extend(block.named_parameters(&format!("blocks.{idx}")));
        }
        params.extend(self.ln_f.named_parameters("ln_f"));
        params.extend(self.lm_head.named_parameters("lm_head"));
        params
    }

    /// 统一切换 GPT 内部块的训练/评估状态。
    ///
    /// 当前真正受影响的是各层 dropout。把开关放到模型级别，可以避免调用方
    /// 手动遍历 block，减少训练/评估模式混淆。
    pub fn set_training(&mut self, training: bool) {
        for block in &mut self.blocks {
            block.set_training(training);
        }
    }

    /// 返回该模型推理时需要的 KV cache 规格。
    ///
    /// GPT 的每个 Transformer block 都有一组自注意力 K/V；这里从第一层读取
    /// head 结构，避免调用方重复传入容易写错的模型元数据。空 block 模型仍
    /// 可以作为极简测试模型运行，此时 cache 层数为 0。
    pub fn kv_cache_spec(&self) -> KvCacheSpec {
        <Self as PrefillCache>::kv_cache_spec(self)
    }

    /// 将当前上下文写入给定 KV cache。
    pub fn prefill_kv_cache(&self, token_ids: &[usize], cache: &mut KvCache) {
        <Self as PrefillCache>::prefill_kv_cache(self, token_ids, cache)
    }
}

impl QwenLikeGPT {
    /// 创建 RoPE 风格的 GPT 变体。
    pub fn new(vocab_size: usize, d_model: usize, n_head: usize, n_layer: usize, d_ff: usize, seq_len: usize) -> Self {
        let blocks = (0..n_layer)
            .map(|_| QwenLikeBlock::new(d_model, n_head, d_ff))
            .collect();
        QwenLikeGPT {
            token_emb: Embedding::new(vocab_size, d_model),
            blocks,
            norm_f: RmsNorm::new(d_model),
            lm_head: Linear::new(d_model, vocab_size, false),
            seq_len,
        }
    }

    /// 将 token 序列前向为 logits。
    pub fn forward_ids(&self, token_ids: &[usize]) -> Tensor {
        let slen = token_ids.len();
        assert!(slen <= self.seq_len);

        let tok = self.token_emb.forward_indices(token_ids);
        let mut x = tok;
        for block in &self.blocks {
            x = block.forward_seq(&x);
        }

        let x = self.norm_f.forward(&x);
        self.lm_head.forward(&x)
    }

    /// 返回可训练参数。
    pub fn parameters(&self) -> Vec<Tensor> {
        let mut p = Vec::new();
        p.extend(self.token_emb.parameters());
        for block in &self.blocks {
            p.extend(block.parameters());
        }
        p.extend(self.norm_f.parameters());
        p.extend(self.lm_head.parameters());
        p
    }

    /// 返回命名参数。
    pub fn named_parameters(&self) -> Vec<NamedParameter> {
        let mut params = Vec::new();
        params.extend(self.token_emb.named_parameters("token_emb"));
        for (idx, block) in self.blocks.iter().enumerate() {
            params.extend(block.named_parameters(&format!("blocks.{idx}")));
        }
        params.extend(self.norm_f.named_parameters("norm_f"));
        params.extend(self.lm_head.named_parameters("lm_head"));
        params
    }

    /// 切换训练/评估状态。
    pub fn set_training(&mut self, training: bool) {
        for block in &mut self.blocks {
            block.set_training(training);
        }
    }

    /// 返回该 Qwen/Llama 风格模型推理时需要的 KV cache 规格。
    pub fn kv_cache_spec(&self) -> KvCacheSpec {
        <Self as PrefillCache>::kv_cache_spec(self)
    }

    /// 将当前上下文写入给定 KV cache。
    pub fn prefill_kv_cache(&self, token_ids: &[usize], cache: &mut KvCache) {
        <Self as PrefillCache>::prefill_kv_cache(self, token_ids, cache)
    }
}

impl PrefillCache for GPT {
    fn kv_cache_spec(&self) -> KvCacheSpec {
        if let Some(block) = self.blocks.first() {
            KvCacheSpec::new(self.blocks.len(), self.seq_len, block.attn.n_head, block.attn.head_dim)
        } else {
            KvCacheSpec::token_only(self.seq_len)
        }
    }

    fn prefill_kv_cache(&self, token_ids: &[usize], cache: &mut KvCache) {
        sptorch_core_tensor::no_grad(|| {
            assert_eq!(
                cache.spec(),
                self.kv_cache_spec(),
                "kv cache spec does not match GPT model"
            );
            cache.reset();
            if token_ids.is_empty() || self.blocks.is_empty() {
                return;
            }

            let slen = token_ids.len();
            assert!(slen <= self.seq_len);
            let positions: Vec<usize> = (0..slen).collect();
            let tok = self.token_emb.forward_indices(token_ids);
            let pos = self.pos_emb.forward_indices(&positions);
            let mut x = add(&tok, &pos);

            for (idx, block) in self.blocks.iter().enumerate() {
                let normed = block.ln1.forward(&x);
                let (key, value) = block.attn.project_kv_cache(&normed, None);
                cache.prefill_layer(idx, key, value);
                x = block.forward_seq_inference(&x);
            }
        });
    }
}

impl PrefillCache for QwenLikeGPT {
    fn kv_cache_spec(&self) -> KvCacheSpec {
        if let Some(block) = self.blocks.first() {
            KvCacheSpec::new(self.blocks.len(), self.seq_len, block.attn.n_head, block.attn.head_dim)
        } else {
            KvCacheSpec::token_only(self.seq_len)
        }
    }

    fn prefill_kv_cache(&self, token_ids: &[usize], cache: &mut KvCache) {
        sptorch_core_tensor::no_grad(|| {
            assert_eq!(
                cache.spec(),
                self.kv_cache_spec(),
                "kv cache spec does not match QwenLikeGPT model"
            );
            cache.reset();
            if token_ids.is_empty() || self.blocks.is_empty() {
                return;
            }

            let slen = token_ids.len();
            assert!(slen <= self.seq_len);
            let mut x = self.token_emb.forward_indices(token_ids);
            for (idx, block) in self.blocks.iter().enumerate() {
                let normed = block.norm1.forward(&x);
                let (key, value) = block.attn.project_kv_cache(&normed, Some(block.rope_base));
                cache.prefill_layer(idx, key, value);
                x = block.forward_seq_inference(&x);
            }
        });
    }
}

// ============ Text Generation ============

/// 自回归生成的通用解码配置。
///
/// 这个结构体只描述“如何从最后一行 logits 选择下一个 token”，不绑定具体模型。
/// 它覆盖 Text-to-SQL 在线推理最先需要稳定下来的几个控制旋钮：确定性生成、
/// top-k/top-p 候选截断、temperature 和重复惩罚。后续真实 KV cache、批量调度
/// 或硬件后端可以继续沿用这组语义，而不必改调用方配置。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GenerationConfig {
    /// 最多生成多少个新 token，不包含 prompt 本身。
    pub max_new_tokens: usize,
    /// 模型输出头对应的词表大小。生成函数会用它截取最后一个位置的 logits。
    pub vocab_size: usize,
    /// logits 缩放温度。较低温度更确定，较高温度更发散；会被下限保护到 `1e-8`。
    pub temperature: f32,
    /// 只保留 logits 最高的前 k 个候选；为 0 时表示不启用 top-k 截断。
    pub top_k: usize,
    /// nucleus sampling 阈值；小于 1 时按概率从高到低保留累计概率达到该值的最小集合。
    pub top_p: f32,
    /// 重复惩罚系数。`1.0` 表示关闭；大于 1 时会压低历史 token 的再次出现概率。
    pub repetition_penalty: f32,
    /// 可选的结束 token。生成器采到该 token 后会把它写入输出并停止继续解码。
    pub eos_token_id: Option<usize>,
}

impl GenerationConfig {
    /// 创建一组偏确定性的默认配置。
    ///
    /// 默认不启用 top-k/top-p/repetition penalty，只设置生成长度和词表大小。
    /// 调用方可以按业务需要再覆盖字段。
    pub fn new(max_new_tokens: usize, vocab_size: usize) -> Self {
        Self {
            max_new_tokens,
            vocab_size,
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            eos_token_id: None,
        }
    }

    /// 创建完全确定的贪心配置。
    pub fn greedy(max_new_tokens: usize, vocab_size: usize) -> Self {
        Self {
            top_k: 1,
            ..Self::new(max_new_tokens, vocab_size)
        }
    }
}

/// 解码阶段的轻量状态对象。
///
/// 这里故意只保存 token 级状态，而不假装已经有真实的 K/V 张量缓存。它的价值是
/// 先把 prefill、decode、上下文裁剪和 prompt/generated 边界固定下来；等 HAL/CUDA
/// 路径具备真实 KV cache 后，可以把每层 Key/Value buffer 挂在这个状态旁边。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InferenceState {
    tokens: Vec<usize>,
    prompt_len: usize,
    max_context: usize,
}

/// KV cache 的结构规格。
///
/// 真实在线推理里，K/V cache 的内存布局必须和模型结构绑定：层数决定需要
/// 多少组缓存，`n_head * head_dim` 决定每个 token 在每层需要保存多少元素，
/// `max_context` 则决定滑动窗口最多保留多少历史。把这些信息显式建模出来，
/// 可以先在 CPU 语义层验证 cache 生命周期，后续再替换成 CUDA/HAL 设备缓冲。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvCacheSpec {
    /// Transformer block 数量；每层各有一份 K/V。
    pub n_layer: usize,
    /// cache 滑动窗口长度，超过后丢弃最旧 token。
    pub max_context: usize,
    /// 注意力头数。
    pub n_head: usize,
    /// 每个注意力头的维度。
    pub head_dim: usize,
}

impl KvCacheSpec {
    /// 创建一份真实模型 cache 规格。
    pub fn new(n_layer: usize, max_context: usize, n_head: usize, head_dim: usize) -> Self {
        assert!(max_context > 0, "max_context must be positive");
        assert!(n_head > 0, "n_head must be positive");
        assert!(head_dim > 0, "head_dim must be positive");
        Self {
            n_layer,
            max_context,
            n_head,
            head_dim,
        }
    }

    /// 创建只跟踪 token 状态、不持有任何层缓存的规格。
    ///
    /// 这个入口用于兼容旧的 `InferenceSession::new(request, max_context)`，
    /// 也适合还没有模型结构信息的调度测试。它不是高性能推理形态，只是一个
    /// 明确的“无层缓存”语义，而不是再把 KV cache 混同成 token 状态。
    pub fn token_only(max_context: usize) -> Self {
        Self::new(0, max_context, 1, 1)
    }
}

/// 单层 Key/Value 缓存。
///
/// `key` 和 `value` 的 shape 统一为 `[n_head, cached_len, head_dim]`。当前实现
/// 用普通 [`Tensor`] 保存语义数据，重点验证 append、reset 和窗口裁剪是否正确；
/// 后续 CUDA/HAL 路径可以在保持这个外层契约的前提下把 Tensor 换成设备缓冲。
#[derive(Debug, Clone)]
pub struct KvCacheLayer {
    key: Option<Tensor>,
    value: Option<Tensor>,
    cached_len: usize,
}

impl KvCacheLayer {
    fn empty() -> Self {
        Self {
            key: None,
            value: None,
            cached_len: 0,
        }
    }

    /// 当前层已经缓存的 token 数。
    pub fn cached_len(&self) -> usize {
        self.cached_len
    }

    /// 判断当前层是否还没有任何 K/V。
    pub fn is_empty(&self) -> bool {
        self.cached_len == 0
    }

    /// 返回当前层缓存的 Key 张量。
    pub fn key(&self) -> Option<&Tensor> {
        self.key.as_ref()
    }

    /// 返回当前层缓存的 Value 张量。
    pub fn value(&self) -> Option<&Tensor> {
        self.value.as_ref()
    }
}

impl PartialEq for KvCacheLayer {
    fn eq(&self, other: &Self) -> bool {
        self.cached_len == other.cached_len
            && tensor_option_eq(&self.key, &other.key)
            && tensor_option_eq(&self.value, &other.value)
    }
}

fn tensor_option_eq(a: &Option<Tensor>, b: &Option<Tensor>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => a.shape() == b.shape() && a.contiguous_data() == b.contiguous_data(),
        _ => false,
    }
}

/// 自回归推理的 per-layer K/V cache。
///
/// 这个结构只处理 cache 的生命周期和形状不变量：prefill 时写入一段历史，
/// decode 时追加一个或多个新 token，超过窗口后裁掉最旧 token。它暂时不参与
/// `forward_ids` 的数学加速，但已经把上层服务和 batch runtime 需要依赖的
/// cache 语义固定下来。
#[derive(Debug, Clone, PartialEq)]
pub struct KvCache {
    spec: KvCacheSpec,
    layers: Vec<KvCacheLayer>,
}

impl KvCache {
    /// 创建空 KV cache。
    pub fn new(spec: KvCacheSpec) -> Self {
        Self {
            spec,
            layers: (0..spec.n_layer).map(|_| KvCacheLayer::empty()).collect(),
        }
    }

    /// 返回 cache 结构规格。
    pub fn spec(&self) -> KvCacheSpec {
        self.spec
    }

    /// 返回层数。
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// 返回最大上下文窗口。
    pub fn max_context(&self) -> usize {
        self.spec.max_context
    }

    /// 返回注意力头数。
    pub fn n_head(&self) -> usize {
        self.spec.n_head
    }

    /// 返回单头维度。
    pub fn head_dim(&self) -> usize {
        self.spec.head_dim
    }

    /// 返回所有层。
    pub fn layers(&self) -> &[KvCacheLayer] {
        &self.layers
    }

    /// 返回指定层。
    pub fn layer(&self, layer_idx: usize) -> Option<&KvCacheLayer> {
        self.layers.get(layer_idx)
    }

    /// 当前 cache 是否完全为空。
    pub fn is_empty(&self) -> bool {
        self.layers.iter().all(KvCacheLayer::is_empty)
    }

    /// 所有层中最长的缓存长度。
    ///
    /// 正常推理里每层长度应一致；这里取最大值可以让诊断代码在半更新状态下
    /// 仍然看到“最坏占用”。测试会覆盖常规同步更新路径。
    pub fn cached_len(&self) -> usize {
        self.layers.iter().map(KvCacheLayer::cached_len).max().unwrap_or(0)
    }

    /// 清空全部层缓存。
    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            *layer = KvCacheLayer::empty();
        }
    }

    /// 清空指定层缓存。
    pub fn reset_layer(&mut self, layer_idx: usize) {
        let layer = self
            .layers
            .get_mut(layer_idx)
            .unwrap_or_else(|| panic!("kv cache layer {layer_idx} out of range"));
        *layer = KvCacheLayer::empty();
    }

    /// 用一段完整 K/V 覆盖指定层，通常对应 prefill。
    pub fn prefill_layer(&mut self, layer_idx: usize, key: Tensor, value: Tensor) {
        self.reset_layer(layer_idx);
        self.append_layer(layer_idx, key, value);
    }

    /// 向指定层追加新的 K/V。
    ///
    /// `key` 和 `value` 必须都是 `[n_head, seq_len, head_dim]`。`seq_len` 可以是
    /// prompt prefill 的多 token，也可以是 decode 阶段的单 token。追加后如果
    /// 超过 `max_context`，会保留最新窗口，丢弃最旧 token。
    pub fn append_layer(&mut self, layer_idx: usize, key: Tensor, value: Tensor) {
        self.validate_kv_shape(&key, "key");
        self.validate_kv_shape(&value, "value");
        assert_eq!(key.shape()[1], value.shape()[1], "key/value seq_len mismatch");

        let layer = self
            .layers
            .get_mut(layer_idx)
            .unwrap_or_else(|| panic!("kv cache layer {layer_idx} out of range"));

        let next_key = append_kv_tensor(
            layer.key.as_ref(),
            &key,
            self.spec.max_context,
            self.spec.n_head,
            self.spec.head_dim,
        );
        let next_value = append_kv_tensor(
            layer.value.as_ref(),
            &value,
            self.spec.max_context,
            self.spec.n_head,
            self.spec.head_dim,
        );
        let cached_len = next_key.shape()[1];
        layer.key = Some(next_key);
        layer.value = Some(next_value);
        layer.cached_len = cached_len;
    }

    fn validate_kv_shape(&self, tensor: &Tensor, name: &str) {
        let shape = tensor.shape();
        assert_eq!(
            shape.len(),
            3,
            "{name} cache tensor must have shape [n_head, seq_len, head_dim]"
        );
        assert_eq!(shape[0], self.spec.n_head, "{name} n_head mismatch");
        assert_eq!(shape[2], self.spec.head_dim, "{name} head_dim mismatch");
    }
}

fn append_kv_tensor(
    existing: Option<&Tensor>,
    incoming: &Tensor,
    max_context: usize,
    n_head: usize,
    head_dim: usize,
) -> Tensor {
    let incoming_shape = incoming.shape();
    let incoming_len = incoming_shape[1];
    let incoming_data = incoming.contiguous_data();

    let (old_len, old_data) = if let Some(existing) = existing {
        let shape = existing.shape();
        assert_eq!(shape, vec![n_head, shape[1], head_dim], "existing cache shape mismatch");
        (shape[1], existing.contiguous_data())
    } else {
        (0, Vec::new())
    };

    let total_len = old_len + incoming_len;
    let keep_len = total_len.min(max_context);
    let skip_len = total_len - keep_len;
    let mut out = Vec::with_capacity(n_head * keep_len * head_dim);

    for h in 0..n_head {
        for logical_pos in skip_len..total_len {
            if logical_pos < old_len {
                let start = h * old_len * head_dim + logical_pos * head_dim;
                out.extend_from_slice(&old_data[start..start + head_dim]);
            } else {
                let incoming_pos = logical_pos - old_len;
                let start = h * incoming_len * head_dim + incoming_pos * head_dim;
                out.extend_from_slice(&incoming_data[start..start + head_dim]);
            }
        }
    }

    Tensor::new(out, vec![n_head, keep_len, head_dim])
}

impl InferenceState {
    /// 用 prompt 初始化一次推理状态。
    ///
    /// `max_context` 必须大于 0；当累计 token 超过该长度时，[`InferenceState::context`]
    /// 会返回最后一个窗口，模拟自回归模型的固定上下文限制。
    pub fn new(prompt: &[usize], max_context: usize) -> Self {
        assert!(max_context > 0, "max_context must be positive");
        Self {
            tokens: prompt.to_vec(),
            prompt_len: prompt.len(),
            max_context,
        }
    }

    /// 重置为新的 prompt，复用同一个状态对象。
    pub fn reset(&mut self, prompt: &[usize]) {
        self.tokens.clear();
        self.tokens.extend_from_slice(prompt);
        self.prompt_len = prompt.len();
    }

    /// 追加一个新生成 token。
    pub fn push_token(&mut self, token: usize) {
        self.tokens.push(token);
    }

    /// 返回当前完整 token 序列，包含 prompt 和已生成部分。
    pub fn tokens(&self) -> &[usize] {
        &self.tokens
    }

    /// 返回 prompt 之后新增的 token。
    pub fn generated_tokens(&self) -> &[usize] {
        &self.tokens[self.prompt_len..]
    }

    /// 返回模型本次 forward 应该看到的上下文窗口。
    pub fn context(&self) -> &[usize] {
        if self.tokens.len() > self.max_context {
            &self.tokens[self.tokens.len() - self.max_context..]
        } else {
            &self.tokens
        }
    }

    /// prompt 的 token 数量。
    pub fn prompt_len(&self) -> usize {
        self.prompt_len
    }

    /// 已生成的新 token 数量。
    pub fn generated_len(&self) -> usize {
        self.tokens.len().saturating_sub(self.prompt_len)
    }

    /// 消费状态并返回完整 token 序列。
    pub fn into_tokens(self) -> Vec<usize> {
        self.tokens
    }
}

/// 采样候选项。
///
/// `logit` 是经过 temperature、重复惩罚和候选过滤后的值；`probability` 是在最终
/// 候选集合内重新归一化后的概率。把它暴露出来可以让测试、日志和未来 Studio
/// 面板看到解码器真实做了哪些裁剪，而不是只看到最终 token。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TokenCandidate {
    pub token_id: usize,
    pub logit: f32,
    pub probability: f32,
}

/// 单步解码停止原因。
///
/// `finished = true` 只告诉调用方“不要再继续”，但线上服务还需要知道为什么停止：
/// EOS 是正常结束，候选为空更像约束或数值问题，已经结束则说明调用方重复 decode。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeStopReason {
    /// 本步正常生成了 token，仍可继续解码。
    NotStopped,
    /// 状态进入解码前已经以 EOS 结束。
    AlreadyEnded,
    /// 当前约束或 logits 过滤后没有任何合法候选。
    NoCandidates,
    /// 本步生成了 EOS token。
    EosToken,
    /// 达到配置允许的最大新 token 数量。
    MaxNewTokens,
}

/// 单步解码的返回值。
///
/// 在线推理通常需要逐 token 流式返回，而不是一次性等待整段序列生成完成。
/// 该结构体记录本步是否真的生成了 token、最终候选分布以及生成后是否应该停止，
/// 方便上层把 token 推送给客户端，同时把候选概率写入调试日志或 Studio 面板。
#[derive(Debug, Clone, PartialEq)]
pub struct DecodeStep {
    pub token_id: Option<usize>,
    pub candidates: Vec<TokenCandidate>,
    pub finished: bool,
    pub stop_reason: DecodeStopReason,
}

trait AutoregressiveForward {
    fn max_context_len(&self) -> usize;
    fn forward_token_ids(&self, token_ids: &[usize]) -> Tensor;
}

impl AutoregressiveForward for GPT {
    fn max_context_len(&self) -> usize {
        self.seq_len
    }

    fn forward_token_ids(&self, token_ids: &[usize]) -> Tensor {
        self.forward_ids(token_ids)
    }
}

impl AutoregressiveForward for QwenLikeGPT {
    fn max_context_len(&self) -> usize {
        self.seq_len
    }

    fn forward_token_ids(&self, token_ids: &[usize]) -> Tensor {
        self.forward_ids(token_ids)
    }
}

fn penalize_repeated_logit(logit: f32, token_id: usize, history: &[usize], repetition_penalty: f32) -> f32 {
    let penalty = repetition_penalty.max(1.0);
    if penalty == 1.0 || !history.contains(&token_id) {
        logit
    } else if logit.is_sign_negative() {
        logit * penalty
    } else {
        logit / penalty
    }
}

fn normalize_candidates(mut candidates: Vec<(usize, f32)>) -> Vec<TokenCandidate> {
    if candidates.is_empty() {
        return Vec::new();
    }

    let max_val = candidates
        .iter()
        .map(|(_, logit)| *logit)
        .fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = candidates.iter().map(|(_, logit)| (logit - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    candidates
        .drain(..)
        .zip(exps)
        .map(|((token_id, logit), exp)| TokenCandidate {
            token_id,
            logit,
            probability: exp / sum,
        })
        .collect()
}

/// 根据解码配置把原始 logits 变成最终采样候选。
///
/// 该函数不依赖模型，适合单独测试 top-k/top-p/repetition penalty 的边界语义。
/// 过滤顺序为：重复惩罚与 temperature 缩放、按 logit 排序、top-k、softmax、
/// top-p、再次归一化。top-p 始终至少保留一个候选，避免高置信场景下空集合。
pub fn sampling_candidates(logits: &[f32], config: &GenerationConfig, history: &[usize]) -> Vec<TokenCandidate> {
    assert!(
        config.vocab_size <= logits.len(),
        "vocab_size must not exceed logits length"
    );

    let temperature = config.temperature.max(1e-8);
    let mut indexed: Vec<(usize, f32)> = logits
        .iter()
        .take(config.vocab_size)
        .enumerate()
        .map(|(token_id, &logit)| {
            (
                token_id,
                penalize_repeated_logit(logit, token_id, history, config.repetition_penalty) / temperature,
            )
        })
        .filter(|(_, logit)| logit.is_finite())
        .collect();

    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    if config.top_k > 0 && indexed.len() > config.top_k {
        indexed.truncate(config.top_k);
    }

    let candidates = normalize_candidates(indexed);
    let top_p = config.top_p.clamp(0.0, 1.0);
    if top_p >= 1.0 || candidates.len() <= 1 {
        return candidates;
    }

    let mut cumulative = 0.0;
    let mut kept = Vec::new();
    for candidate in candidates {
        cumulative += candidate.probability;
        kept.push((candidate.token_id, candidate.logit));
        if cumulative >= top_p {
            break;
        }
    }
    normalize_candidates(kept)
}

fn sample_from_candidates(candidates: &[TokenCandidate], rng: &mut impl Rng) -> Option<usize> {
    let mut threshold: f32 = rng.gen();
    for candidate in candidates {
        threshold -= candidate.probability;
        if threshold <= 0.0 {
            return Some(candidate.token_id);
        }
    }
    candidates.last().map(|candidate| candidate.token_id)
}

fn decode_next_token_inner<M: AutoregressiveForward>(
    model: &M,
    state: &mut InferenceState,
    config: &GenerationConfig,
    constraint: Option<&dyn TokenConstraint>,
    rng: &mut impl Rng,
) -> DecodeStep {
    assert!(config.vocab_size > 0, "vocab_size must be positive");

    if config.eos_token_id.is_some() && state.tokens().last().copied() == config.eos_token_id {
        return DecodeStep {
            token_id: None,
            candidates: Vec::new(),
            finished: true,
            stop_reason: DecodeStopReason::AlreadyEnded,
        };
    }

    let logits = model.forward_token_ids(state.context());
    let logits_data = logits.contiguous_data();
    let last = &logits_data[logits_data.len() - config.vocab_size..];
    let mut step_logits = last.to_vec();

    if let Some(constraint) = constraint {
        if let Some(allowed) = constraint.allowed_next(state.generated_tokens()) {
            for (token_id, logit) in step_logits.iter_mut().enumerate() {
                if !allowed.contains(&token_id) {
                    *logit = f32::NEG_INFINITY;
                }
            }
        }
    }

    let candidates = sampling_candidates(&step_logits, config, state.tokens());
    let Some(next) = sample_from_candidates(&candidates, rng) else {
        return DecodeStep {
            token_id: None,
            candidates,
            finished: true,
            stop_reason: DecodeStopReason::NoCandidates,
        };
    };

    state.push_token(next);
    let stop_reason = if Some(next) == config.eos_token_id {
        DecodeStopReason::EosToken
    } else {
        DecodeStopReason::NotStopped
    };
    DecodeStep {
        token_id: Some(next),
        candidates,
        finished: stop_reason != DecodeStopReason::NotStopped,
        stop_reason,
    }
}

fn generate_with_config_inner<M: AutoregressiveForward>(
    model: &M,
    prompt: &[usize],
    config: GenerationConfig,
    constraint: Option<&dyn TokenConstraint>,
) -> Vec<usize> {
    assert!(config.vocab_size > 0, "vocab_size must be positive");

    let mut rng = rand::thread_rng();
    let mut state = InferenceState::new(prompt, model.max_context_len());

    if config.eos_token_id.is_some() && prompt.last().copied() == config.eos_token_id {
        return state.into_tokens();
    }

    for _ in 0..config.max_new_tokens {
        let step = decode_next_token_inner(model, &mut state, &config, constraint, &mut rng);
        if step.token_id.is_none() {
            break;
        }
        if step.finished {
            break;
        }
    }

    state.into_tokens()
}

/// 贪心解码生成 token 序列。
///
/// 每一步选择最后一个位置上 logits 最大的 token。该方法确定性强、便于测试，
/// 但缺乏采样多样性。
pub fn generate_greedy(model: &GPT, prompt: &[usize], max_new_tokens: usize, vocab_size: usize) -> Vec<usize> {
    generate_with_config(model, prompt, GenerationConfig::greedy(max_new_tokens, vocab_size))
}

/// 按 temperature 与 top-k 采样生成序列。
///
/// `temperature` 越低越接近贪心，越高越随机；`top_k` 限制候选 token 数量。
/// 调用方需要确保 `vocab_size` 与模型输出头一致。
pub fn generate_with_sampling(
    model: &GPT,
    prompt: &[usize],
    max_new_tokens: usize,
    vocab_size: usize,
    temperature: f32,
    top_k: usize,
) -> Vec<usize> {
    generate_with_config(
        model,
        prompt,
        GenerationConfig {
            temperature,
            top_k,
            ..GenerationConfig::new(max_new_tokens, vocab_size)
        },
    )
}

/// 使用完整解码配置生成 GPT 序列。
pub fn generate_with_config(model: &GPT, prompt: &[usize], config: GenerationConfig) -> Vec<usize> {
    generate_with_config_inner(model, prompt, config, None)
}

/// 使用完整解码配置生成 Qwen/Llama 风格模型序列。
///
/// 这条入口让 SFT 训练出来的小型 Qwen-like 模型也能复用同一套在线解码语义。
pub fn generate_qwen_like_with_config(model: &QwenLikeGPT, prompt: &[usize], config: GenerationConfig) -> Vec<usize> {
    generate_with_config_inner(model, prompt, config, None)
}

/// 对 GPT 状态执行一步解码。
///
/// 调用方负责持有并复用 [`InferenceState`]；函数会在成功采样时把 token 追加到状态中。
/// 这就是后续流式推理、请求队列和真实 KV cache 共享的最小同步原语。
pub fn decode_next_token(model: &GPT, state: &mut InferenceState, config: &GenerationConfig) -> DecodeStep {
    let mut rng = rand::thread_rng();
    decode_next_token_inner(model, state, config, None, &mut rng)
}

/// 对 Qwen/Llama 风格模型状态执行一步解码。
pub fn decode_qwen_like_next_token(
    model: &QwenLikeGPT,
    state: &mut InferenceState,
    config: &GenerationConfig,
) -> DecodeStep {
    let mut rng = rand::thread_rng();
    decode_next_token_inner(model, state, config, None, &mut rng)
}

/// 对 GPT 状态执行一步带约束解码。
///
/// 约束读取的是 prompt 之后已经生成的 token，因此可以自然表达 SQL 前缀树、
/// schema 候选或语法状态机。
pub fn decode_constrained_next_token(
    model: &GPT,
    state: &mut InferenceState,
    config: &GenerationConfig,
    constraint: &dyn TokenConstraint,
) -> DecodeStep {
    let mut rng = rand::thread_rng();
    decode_next_token_inner(model, state, config, Some(constraint), &mut rng)
}

// ============ Inference Scheduling ============

/// 单个在线推理请求的框架级描述。
///
/// 这里不包含 HTTP header、用户身份或产品业务字段，只保留模型调度真正需要的
/// 信息：请求 ID、prompt、解码配置和到达顺序。这样产品仓可以把任意服务协议
/// 映射到这个结构，而框架只负责做稳定、可测试的准入和批处理计划。
#[derive(Debug, Clone, PartialEq)]
pub struct InferenceRequest {
    pub request_id: u64,
    pub prompt: Vec<usize>,
    pub config: GenerationConfig,
    pub arrival_order: u64,
}

impl InferenceRequest {
    /// 创建推理请求。
    ///
    /// `arrival_order` 由调用方按入队顺序递增传入，调度器用它保持 FIFO 公平性。
    pub fn new(request_id: u64, prompt: Vec<usize>, config: GenerationConfig, arrival_order: u64) -> Self {
        Self {
            request_id,
            prompt,
            config,
            arrival_order,
        }
    }

    /// 请求在调度阶段声明的 token 预算。
    ///
    /// 预算包含 prompt token 和最多可能生成的新 token。它不是精确显存估算，
    /// 但足以作为第一层过载保护，避免单个请求把一个微批次撑爆。
    pub fn token_budget(&self) -> usize {
        self.prompt.len() + self.config.max_new_tokens
    }
}

/// 推理调度器配置。
///
/// `max_batch_size` 限制同一轮最多取多少个请求，`max_batch_tokens` 限制这些请求
/// 的总 token 预算，`max_queue_len` 则是入队前的背压阈值。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InferenceSchedulerConfig {
    pub max_batch_size: usize,
    pub max_batch_tokens: usize,
    pub max_queue_len: usize,
}

impl InferenceSchedulerConfig {
    /// 创建调度器配置。
    pub fn new(max_batch_size: usize, max_batch_tokens: usize, max_queue_len: usize) -> Self {
        assert!(max_batch_size > 0, "max_batch_size must be positive");
        assert!(max_batch_tokens > 0, "max_batch_tokens must be positive");
        Self {
            max_batch_size,
            max_batch_tokens,
            max_queue_len,
        }
    }
}

/// 推理调度拒绝原因。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferenceAdmissionError {
    /// 队列已经达到上限，调用方应触发过载保护或稍后重试。
    QueueFull,
    /// 单个请求预算已经超过批次 token 上限，无法安全接入当前调度器。
    RequestTooLarge,
}

/// 调度器产出的一个微批计划。
///
/// 当前批次仍然按请求独立执行；它的意义是把“哪些请求可以在同一轮被调度”
/// 这件事稳定下来。后续接真实 batched prefill/decode 时，可沿用这个计划结构。
#[derive(Debug, Clone, PartialEq)]
pub struct InferenceBatch {
    pub requests: Vec<InferenceRequest>,
    pub total_token_budget: usize,
}

impl InferenceBatch {
    /// 批次中的请求数量。
    pub fn len(&self) -> usize {
        self.requests.len()
    }

    /// 批次是否为空。
    pub fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }

    /// 批次请求 ID，便于日志和测试断言。
    pub fn request_ids(&self) -> Vec<u64> {
        self.requests.iter().map(|request| request.request_id).collect()
    }
}

/// 面向在线推理的 FIFO 微批调度器。
///
/// 这个结构故意不执行模型，也不创建线程；它只负责准入、排队和微批计划。
/// 这样我们先把调度不变量练扎实，再让产品服务层决定用 Tokio、线程池还是硬件队列。
#[derive(Debug, Clone)]
pub struct InferenceScheduler {
    config: InferenceSchedulerConfig,
    pending: VecDeque<InferenceRequest>,
}

impl InferenceScheduler {
    /// 创建空调度器。
    pub fn new(config: InferenceSchedulerConfig) -> Self {
        Self {
            config,
            pending: VecDeque::new(),
        }
    }

    /// 尝试接纳一个请求。
    pub fn enqueue(&mut self, request: InferenceRequest) -> Result<(), InferenceAdmissionError> {
        if request.token_budget() > self.config.max_batch_tokens {
            return Err(InferenceAdmissionError::RequestTooLarge);
        }
        if self.pending.len() >= self.config.max_queue_len {
            return Err(InferenceAdmissionError::QueueFull);
        }
        self.pending.push_back(request);
        Ok(())
    }

    /// 规划下一批请求。
    ///
    /// 调度器保持 FIFO：如果队首请求无法放入当前批次，说明后面的请求即使更小
    /// 也不应越过它，否则高 token 请求会长期饥饿。
    pub fn plan_next_batch(&mut self) -> Option<InferenceBatch> {
        let mut requests = Vec::new();
        let mut total_token_budget = 0usize;

        while let Some(next) = self.pending.front() {
            if requests.len() >= self.config.max_batch_size {
                break;
            }

            let next_budget = next.token_budget();
            if !requests.is_empty() && total_token_budget + next_budget > self.config.max_batch_tokens {
                break;
            }

            let next = self.pending.pop_front().expect("front exists");
            total_token_budget += next_budget;
            requests.push(next);
        }

        if requests.is_empty() {
            None
        } else {
            Some(InferenceBatch {
                requests,
                total_token_budget,
            })
        }
    }

    /// 当前等待中的请求数量。
    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    /// 队列是否为空。
    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    /// 返回调度器配置。
    pub fn config(&self) -> InferenceSchedulerConfig {
        self.config
    }
}

/// 单个已激活推理请求的运行态。
///
/// 调度器只负责把请求编成批；真正开始生成时，需要把 prompt 转成
/// [`InferenceState`]，并记录已生成步数、是否结束和最终停止原因。
#[derive(Debug, Clone, PartialEq)]
pub struct InferenceSession {
    pub request: InferenceRequest,
    pub state: InferenceState,
    pub kv_cache: KvCache,
    pub generated_steps: usize,
    pub finished: bool,
    pub stop_reason: DecodeStopReason,
}

impl InferenceSession {
    /// 从请求创建运行态。
    pub fn new(request: InferenceRequest, max_context: usize) -> Self {
        Self::with_kv_cache_spec(request, KvCacheSpec::token_only(max_context))
    }

    /// 从请求和模型 KV cache 规格创建运行态。
    ///
    /// `InferenceState` 负责 token 序列，`KvCache` 负责每层 K/V 的窗口和生命周期。
    /// 把两者并列放在 session 中，是为了避免后续真实增量推理接入时继续把
    /// “已经生成了哪些 token”和“硬件侧缓存了哪些 K/V”混在一个对象里。
    pub fn with_kv_cache_spec(request: InferenceRequest, cache_spec: KvCacheSpec) -> Self {
        let prompt_ended =
            request.config.eos_token_id.is_some() && request.prompt.last().copied() == request.config.eos_token_id;
        Self {
            state: InferenceState::new(&request.prompt, cache_spec.max_context),
            kv_cache: KvCache::new(cache_spec),
            request,
            generated_steps: 0,
            finished: prompt_ended,
            stop_reason: if prompt_ended {
                DecodeStopReason::AlreadyEnded
            } else {
                DecodeStopReason::NotStopped
            },
        }
    }

    /// 重新装载一个 prompt，并清空历史生成与 K/V cache。
    ///
    /// 这个方法对应一轮新的 prefill 生命周期：token 状态回到 prompt 起点，
    /// cache 也被同步刷新为干净状态，避免上一轮请求污染新会话。
    pub fn prefill<M: PrefillCache>(&mut self, model: &M) {
        self.state.reset(&self.request.prompt);
        self.generated_steps = 0;
        self.finished = self.request.config.eos_token_id.is_some()
            && self.request.prompt.last().copied() == self.request.config.eos_token_id;
        self.stop_reason = if self.finished {
            DecodeStopReason::AlreadyEnded
        } else {
            DecodeStopReason::NotStopped
        };
        model.prefill_kv_cache(self.state.tokens(), &mut self.kv_cache);
    }

    /// 重置 session 为新的请求。
    ///
    /// 与 `prefill` 不同，这里会替换请求本身，适合调度器复用 session 对象。
    pub fn reset<M: PrefillCache>(&mut self, request: InferenceRequest, model: &M) {
        self.request = request;
        self.state = InferenceState::new(&self.request.prompt, self.kv_cache.max_context());
        self.generated_steps = 0;
        self.finished = self.request.config.eos_token_id.is_some()
            && self.request.prompt.last().copied() == self.request.config.eos_token_id;
        self.stop_reason = if self.finished {
            DecodeStopReason::AlreadyEnded
        } else {
            DecodeStopReason::NotStopped
        };
        self.kv_cache.reset();
        model.prefill_kv_cache(self.state.tokens(), &mut self.kv_cache);
    }

    /// 请求 ID。
    pub fn request_id(&self) -> u64 {
        self.request.request_id
    }

    /// 完整 token 序列，包含 prompt 和已生成内容。
    pub fn tokens(&self) -> &[usize] {
        self.state.tokens()
    }

    /// 已生成的新 token。
    pub fn generated_tokens(&self) -> &[usize] {
        self.state.generated_tokens()
    }

    fn mark_finished(&mut self, reason: DecodeStopReason) {
        self.finished = true;
        self.stop_reason = reason;
    }
}

/// 一个已激活微批的运行态。
#[derive(Debug, Clone, PartialEq)]
pub struct InferenceBatchState {
    pub sessions: Vec<InferenceSession>,
}

impl InferenceBatchState {
    /// 把调度计划转成运行态。
    pub fn from_batch(batch: InferenceBatch, max_context: usize) -> Self {
        Self::from_batch_with_kv_cache_spec(batch, KvCacheSpec::token_only(max_context))
    }

    /// 用模型 cache 规格把调度计划转成运行态。
    pub fn from_batch_with_kv_cache_spec(batch: InferenceBatch, cache_spec: KvCacheSpec) -> Self {
        Self {
            sessions: batch
                .requests
                .into_iter()
                .map(|request| InferenceSession::with_kv_cache_spec(request, cache_spec))
                .collect(),
        }
    }

    /// 用模型结构创建运行态并立即执行 prompt prefill。
    pub fn from_batch_with_prefill<M: PrefillCache>(batch: InferenceBatch, model: &M) -> Self {
        let cache_spec = model.kv_cache_spec();
        let mut state = Self::from_batch_with_kv_cache_spec(batch, cache_spec);
        for session in &mut state.sessions {
            session.prefill(model);
        }
        state
    }

    /// 是否所有请求都已经结束。
    pub fn is_finished(&self) -> bool {
        self.sessions.iter().all(|session| session.finished)
    }

    /// 尚未结束的请求数量。
    pub fn active_len(&self) -> usize {
        self.sessions.iter().filter(|session| !session.finished).count()
    }

    /// 批次中的请求 ID。
    pub fn request_ids(&self) -> Vec<u64> {
        self.sessions.iter().map(|session| session.request_id()).collect()
    }
}

/// 批次单轮推进结果。
#[derive(Debug, Clone, PartialEq)]
pub struct InferenceBatchStep {
    pub request_id: u64,
    pub step: DecodeStep,
    pub generated_steps: usize,
}

/// 将一个调度批次激活为 GPT 运行态。
pub fn activate_inference_batch(model: &GPT, batch: InferenceBatch) -> InferenceBatchState {
    InferenceBatchState::from_batch_with_prefill(batch, model)
}

/// 将一个调度批次激活为 Qwen/Llama 风格模型运行态。
pub fn activate_qwen_like_inference_batch(model: &QwenLikeGPT, batch: InferenceBatch) -> InferenceBatchState {
    InferenceBatchState::from_batch_with_prefill(batch, model)
}

fn decode_session_with<M: AutoregressiveForward + PrefillCache>(
    model: &M,
    session: &mut InferenceSession,
) -> DecodeStep {
    if session.finished {
        return DecodeStep {
            token_id: None,
            candidates: Vec::new(),
            finished: true,
            stop_reason: session.stop_reason,
        };
    }

    if session.generated_steps >= session.request.config.max_new_tokens {
        session.mark_finished(DecodeStopReason::MaxNewTokens);
        return DecodeStep {
            token_id: None,
            candidates: Vec::new(),
            finished: true,
            stop_reason: DecodeStopReason::MaxNewTokens,
        };
    }

    let mut rng = rand::thread_rng();
    let step = decode_next_token_inner(model, &mut session.state, &session.request.config, None, &mut rng);
    if step.token_id.is_some() {
        session.generated_steps += 1;
        model.prefill_kv_cache(session.state.context(), &mut session.kv_cache);
    }
    if step.finished {
        session.mark_finished(step.stop_reason);
    } else if session.generated_steps >= session.request.config.max_new_tokens {
        session.mark_finished(DecodeStopReason::MaxNewTokens);
    }
    step
}

fn decode_batch_round_with<M: AutoregressiveForward + PrefillCache>(
    model: &M,
    batch: &mut InferenceBatchState,
) -> Vec<InferenceBatchStep> {
    let mut steps = Vec::new();
    for session in batch.sessions.iter_mut().filter(|session| !session.finished) {
        let request_id = session.request_id();
        let step = decode_session_with(model, session);
        steps.push(InferenceBatchStep {
            request_id,
            step,
            generated_steps: session.generated_steps,
        });
    }
    steps
}

/// 对 GPT 批次执行一轮逐请求解码。
///
/// 当前实现是语义层 round-robin：每个未结束请求各走一步。它先保证流式调度
/// 不变量正确，后续可以把内部替换为真正 batched decode。
pub fn decode_inference_batch_round(model: &GPT, batch: &mut InferenceBatchState) -> Vec<InferenceBatchStep> {
    decode_batch_round_with(model, batch)
}

/// 对 Qwen/Llama 风格模型批次执行一轮逐请求解码。
pub fn decode_qwen_like_inference_batch_round(
    model: &QwenLikeGPT,
    batch: &mut InferenceBatchState,
) -> Vec<InferenceBatchStep> {
    decode_batch_round_with(model, batch)
}

// ============ Constrained Decoding ============

/// 用于约束 token 生成的前缀 Trie。
///
/// 每个节点保存“当前前缀下允许的下一 token”。它适合表达有限集合、语法片段
/// 或 schema 引导的候选路径。
#[derive(Debug, Clone)]
pub struct TokenTrie {
    children: std::collections::HashMap<usize, TokenTrie>,
    is_terminal: bool,
}

impl Default for TokenTrie {
    // 默认构造一个空 Trie。
    fn default() -> Self {
        Self::new()
    }
}

impl TokenTrie {
    /// 创建空 TokenTrie。
    pub fn new() -> Self {
        TokenTrie {
            children: std::collections::HashMap::new(),
            is_terminal: false,
        }
    }
    /// 向 Trie 插入一条合法 token 路径。
    ///
    /// 插入后，生成器可以沿着该路径逐步约束下一 token。
    pub fn insert(&mut self, tokens: &[usize]) {
        let mut node = self;
        for &t in tokens {
            node = node.children.entry(t).or_default();
        }
        node.is_terminal = true;
    }
    /// 查询给定前缀下允许的下一 token 集合。
    ///
    /// 返回 `None` 表示没有继续约束，通常代表已到达终止节点或前缀不存在。
    pub fn allowed_tokens(&self, prefix: &[usize]) -> Option<Vec<usize>> {
        let mut node = self;
        for &t in prefix {
            match node.children.get(&t) {
                Some(child) => node = child,
                None => return None,
            }
        }
        if node.children.is_empty() {
            None // 到达终止节点或死路时不再施加约束
        } else {
            Some(node.children.keys().copied().collect())
        }
    }
}

/// 每一步解码可插拔约束的抽象接口。
/// 可用于接入 SQL 语法、正则或 schema 约束。
pub trait TokenConstraint: Send + Sync {
    /// 根据当前已生成序列，返回下一步允许的 token ID 集合。
    ///
    /// 返回 `None` 表示放开约束，允许全部 token。实现者需要保证返回的 token
    /// ID 与模型 vocabulary 对齐。
    fn allowed_next(&self, generated: &[usize]) -> Option<Vec<usize>>;
}

impl TokenConstraint for TokenTrie {
    // 按当前已生成序列返回下一步允许的 token 集合。
    fn allowed_next(&self, generated: &[usize]) -> Option<Vec<usize>> {
        self.allowed_tokens(generated)
    }
}
/// 在约束条件下执行采样生成。
///
/// 该函数先应用外部约束，再执行 temperature 与 top-k 采样。若约束导致当前步
/// 没有任何合法 token，生成会提前结束。
pub fn generate_constrained(
    model: &GPT,
    prompt: &[usize],
    max_new_tokens: usize,
    vocab_size: usize,
    temperature: f32,
    top_k: usize,
    constraint: &dyn TokenConstraint,
) -> Vec<usize> {
    generate_with_config_inner(
        model,
        prompt,
        GenerationConfig {
            temperature,
            top_k,
            ..GenerationConfig::new(max_new_tokens, vocab_size)
        },
        Some(constraint),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // 验证线性层前向输出形状正确。
    #[test]
    fn test_linear_forward_shape() {
        let linear = Linear::new(4, 3, true);
        let input = Tensor::new(vec![1.0; 8], vec![2, 4]);
        let out = linear.forward(&input);
        assert_eq!(out.shape(), vec![2, 3]);
    }

    // 验证关闭 bias 时参数数量符合预期。
    #[test]
    fn test_linear_no_bias() {
        let linear = Linear::new(3, 2, false);
        assert!(linear.bias.is_none());
        assert_eq!(linear.parameters().len(), 1);
    }

    // 验证启用 bias 时参数数量符合预期。
    #[test]
    fn test_linear_with_bias() {
        let linear = Linear::new(3, 2, true);
        assert!(linear.bias.is_some());
        assert_eq!(linear.parameters().len(), 2);
    }

    // 验证 embedding 查表输出形状正确。
    #[test]
    fn test_embedding_forward() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[0, 3, 7]);
        assert_eq!(out.shape(), vec![3, 4]);
    }

    // 验证一维 LayerNorm 输出均值接近 0。
    #[test]
    fn test_layer_norm_forward_1d() {
        let ln = LayerNorm::new(3);
        let input = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let out = ln.forward(&input);
        let d = out.data();
        // After layer norm with gamma=1, beta=0: mean should be ~0
        let mean: f32 = d.iter().sum::<f32>() / 3.0;
        assert!(mean.abs() < 1e-5);
    }

    // 验证二维 LayerNorm 每行归一化有效。
    #[test]
    fn test_layer_norm_forward_2d() {
        let ln = LayerNorm::new(4);
        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 4]);
        let out = ln.forward(&input);
        assert_eq!(out.shape(), vec![2, 4]);
        let d = out.data();
        // Each row should have mean ~0
        let mean0: f32 = d[0..4].iter().sum::<f32>() / 4.0;
        let mean1: f32 = d[4..8].iter().sum::<f32>() / 4.0;
        assert!(mean0.abs() < 1e-5);
        assert!(mean1.abs() < 1e-5);
    }

    // LayerNorm 的参数图不应因为输入本身不追踪梯度而断掉。
    #[test]
    fn test_layer_norm_tracks_beta_even_if_gamma_frozen() {
        let ln = LayerNorm::new(4);
        ln.gamma.set_requires_grad(false);
        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]);

        let out = ln.forward(&input);
        assert!(out.requires_grad());

        let loss = sum(&out);
        loss.backward();

        assert!(ln.gamma.grad().is_none());
        assert!(ln.beta.grad().is_some());
        assert_eq!(ln.beta.grad().unwrap().len(), 4);
    }

    // 验证 Xavier Uniform 采样值位于理论区间。
    #[test]
    fn test_xavier_uniform_range() {
        let t = xavier_uniform(100, 100);
        let d = t.data();
        let limit = (6.0 / 200.0f32).sqrt();
        for v in &d {
            assert!(*v >= -limit && *v <= limit);
        }
    }

    // 验证 Kaiming Normal 样本均值接近 0。
    #[test]
    fn test_kaiming_normal_stats() {
        let t = kaiming_normal(1000, 100);
        let d = t.data();
        let mean: f32 = d.iter().sum::<f32>() / d.len() as f32;
        // Mean should be close to 0
        assert!(mean.abs() < 0.1);
    }

    // --- MultiHeadAttention tests ---

    #[test]
    fn test_mha_forward_shape() {
        let mha = MultiHeadAttention::new(8, 2); // d_model=8, n_head=2
        let input = Tensor::new(vec![0.1; 4 * 8], vec![4, 8]); // seq_len=4
        let out = mha.forward_causal(&input);
        assert_eq!(out.shape(), vec![4, 8]);
    }

    #[test]
    fn test_rope_keeps_position_zero_and_rotates_later_positions() {
        let input = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, //
                1.0, 0.0, 0.0, 1.0,
            ],
            vec![1, 2, 4],
        );
        let out = rotary_position_embedding(&input, 10000.0);
        let data = out.data();
        assert_eq!(&data[..4], &[1.0, 2.0, 3.0, 4.0]);
        assert!(
            (data[4] - 1.0).abs() > 1e-3 || data[5].abs() > 1e-3,
            "position 1 should be rotated"
        );
    }

    // 验证多头注意力参数数量与结构一致。
    #[test]
    fn test_mha_parameters_count() {
        let mha = MultiHeadAttention::new(8, 2);
        // 4 weight matrices (no bias): Wq, Wk, Wv, Wo
        assert_eq!(mha.parameters().len(), 4);
    }

    // --- TransformerBlock tests ---

    #[test]
    fn test_transformer_block_forward_shape() {
        let block = TransformerBlock::new(8, 2, 32); // d_model=8, n_head=2, d_ff=32
        let input = Tensor::new(vec![0.1; 4 * 8], vec![4, 8]);
        let out = block.forward_seq(&input);
        assert_eq!(out.shape(), vec![4, 8]);
    }

    #[test]
    fn test_qwen_like_gpt_forward_and_backward() {
        let mut model = QwenLikeGPT::new(12, 8, 2, 1, 16, 6);
        model.set_training(false);
        let logits = model.forward_ids(&[0, 1, 2, 3]);
        assert_eq!(logits.shape(), vec![4, 12]);
        let loss = cross_entropy_loss(&logits, &[1, 2, 3, 4]);
        loss.backward();
        assert!(model.parameters().iter().any(|param| param.grad().is_some()));
    }

    // --- GPT tests ---

    #[test]
    fn test_gpt_forward_shape() {
        let gpt = GPT::new(16, 8, 2, 2, 32, 8);
        // vocab=16, d_model=8, n_head=2, n_layer=2, d_ff=32, seq_len=8
        let logits = gpt.forward_ids(&[0, 1, 2, 3]);
        assert_eq!(logits.shape(), vec![4, 16]); // [seq_len, vocab_size]
    }

    // 验证 GPT 参数总数与模块拆分一致。
    #[test]
    fn test_gpt_parameters() {
        let gpt = GPT::new(16, 8, 2, 1, 32, 8);
        let params = gpt.parameters();
        // token_emb(1) + pos_emb(1) + 1 block(ln1:2 + attn:4 + ln2:2 + ffn_up:2 + ffn_down:2 = 12) + ln_f(2) + lm_head(1) = 17
        assert_eq!(params.len(), 17);
    }

    // 验证 GPT 反向传播可执行且参数梯度存在。
    #[test]
    fn test_gpt_backward_runs() {
        let gpt = GPT::new(8, 4, 2, 1, 16, 4);
        let logits = gpt.forward_ids(&[0, 1, 2]);
        let loss = cross_entropy_loss(&logits, &[1, 2, 3]);
        loss.backward();
        // Check that gradients exist on parameters
        let param_names = [
            "token_emb",
            "pos_emb",
            "ln1_gamma",
            "ln1_beta",
            "wq",
            "wk",
            "wv",
            "wo",
            "ln2_gamma",
            "ln2_beta",
            "ffn_up_w",
            "ffn_up_b",
            "ffn_down_w",
            "ffn_down_b",
            "ln_f_gamma",
            "ln_f_beta",
            "lm_head",
        ];
        for (i, p) in gpt.parameters().iter().enumerate() {
            let name = if i < param_names.len() {
                param_names[i]
            } else {
                "unknown"
            };
            assert!(p.grad().is_some(), "param[{}] '{}' has no gradient", i, name);
        }
    }

    // --- LoRA tests ---

    #[test]
    fn test_lora_forward_shape() {
        let lora = LoRALinear::from_dims(4, 3, true, 2, 1.0);
        let input = Tensor::new(vec![1.0; 8], vec![2, 4]);
        let out = lora.forward(&input);
        assert_eq!(out.shape(), vec![2, 3]);
    }

    // 验证 LoRA 仅适配器参数参与训练。
    #[test]
    fn test_lora_only_adapters_trainable() {
        let lora = LoRALinear::from_dims(4, 3, true, 2, 1.0);
        // parameters() should only return lora_a and lora_b
        assert_eq!(lora.parameters().len(), 2);
        // all_parameters() should include base weight + bias + lora_a + lora_b
        assert_eq!(lora.all_parameters().len(), 4);
        // Base weight should be frozen
        assert!(!lora.base.weight.requires_grad());
    }

    // 验证 LoRA 初始输出与基座线性层一致。
    #[test]
    fn test_lora_starts_as_base() {
        // With B initialized to zeros, LoRA output should equal base output
        let base = Linear::new(4, 3, false);
        let base_weight_data = base.weight.data();
        let lora = LoRALinear::new(base, 2, 1.0);

        let input = Tensor::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], vec![2, 4]);
        let lora_out = lora.forward(&input);

        // Manually compute base output: input @ W^T
        let wt_data = {
            let w = base_weight_data;
            // W is [3, 4], W^T is [4, 3]
            let mut wt = vec![0.0f32; 4 * 3];
            for i in 0..3 {
                for j in 0..4 {
                    wt[j * 3 + i] = w[i * 4 + j];
                }
            }
            wt
        };
        let mut expected = vec![0.0f32; 2 * 3];
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    expected[i * 3 + j] += input.data()[i * 4 + k] * wt_data[k * 3 + j];
                }
            }
        }

        let out_data = lora_out.data();
        for i in 0..6 {
            assert!(
                (out_data[i] - expected[i]).abs() < 1e-5,
                "mismatch at {}: got {} expected {}",
                i,
                out_data[i],
                expected[i]
            );
        }
    }

    // 验证 LoRA 反向传播后适配器梯度可用。
    #[test]
    fn test_lora_backward_runs() {
        let lora = LoRALinear::from_dims(4, 3, false, 2, 1.0);
        let input = Tensor::with_grad(vec![0.1; 8], vec![2, 4], true);
        let out = lora.forward(&input);
        let loss = sum(&out);
        loss.backward();

        // LoRA adapters should have gradients
        assert!(lora.lora_a.grad().is_some(), "lora_a has no gradient");
        assert!(lora.lora_b.grad().is_some(), "lora_b has no gradient");
    }

    // 验证 LoRA merge 会按公式更新基座权重。
    #[test]
    fn test_lora_merge() {
        let lora = LoRALinear::from_dims(4, 3, false, 2, 2.0);

        // Manually set lora_a and lora_b to known values
        {
            let inner = lora.lora_a.0.read().unwrap();
            let mut s = inner.storage.write().unwrap();
            let slice = s.as_cpu_slice_mut();
            // rank=2, in=4 -> [2, 4]
            for (i, v) in [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0].iter().enumerate() {
                slice[i] = *v;
            }
        }
        {
            let inner = lora.lora_b.0.read().unwrap();
            let mut s = inner.storage.write().unwrap();
            let slice = s.as_cpu_slice_mut();
            // out=3, rank=2 -> [3, 2]
            for (i, v) in [1.0, 0.0, 0.0, 1.0, 0.0, 0.0].iter().enumerate() {
                slice[i] = *v;
            }
        }

        let w_before = lora.base.weight.data();
        lora.merge();
        let w_after = lora.base.weight.data();

        // B @ A = [[1,0],[0,1],[0,0]] @ [[1,0,0,0],[0,1,0,0]]
        //       = [[1,0,0,0],[0,1,0,0],[0,0,0,0]]
        // scaling = alpha/rank = 2/2 = 1.0
        // W[0][0] should increase by 1.0, W[1][1] by 1.0
        assert!((w_after[0] - w_before[0] - 1.0).abs() < 1e-6);
        assert!((w_after[5] - w_before[5] - 1.0).abs() < 1e-6);
        // W[2][2] should be unchanged
        assert!((w_after[10] - w_before[10]).abs() < 1e-6);
    }

    // --- TokenTrie tests ---

    #[test]
    fn test_trie_basic() {
        let mut trie = TokenTrie::new();
        // "SELECT" = [0, 1, 2]
        // "SET"    = [0, 1, 3]
        trie.insert(&[0, 1, 2]);
        trie.insert(&[0, 1, 3]);

        // At root, only token 0 is allowed
        assert_eq!(trie.allowed_tokens(&[]), Some(vec![0]));

        // After [0], only token 1
        assert_eq!(trie.allowed_tokens(&[0]), Some(vec![1]));

        // After [0, 1], tokens 2 and 3
        let mut allowed = trie.allowed_tokens(&[0, 1]).unwrap();
        allowed.sort();
        assert_eq!(allowed, vec![2, 3]);

        // 前缀 [0, 1, 2] 到达终止节点，没有子分支
        assert_eq!(trie.allowed_tokens(&[0, 1, 2]), None);

        // Invalid prefix
        assert_eq!(trie.allowed_tokens(&[5]), None);
    }

    // 验证 TokenConstraint trait 在 Trie 上的行为。
    #[test]
    fn test_trie_constraint_trait() {
        let mut trie = TokenTrie::new();
        trie.insert(&[10, 20]);
        trie.insert(&[10, 30]);

        let constraint: &dyn TokenConstraint = &trie;
        assert_eq!(constraint.allowed_next(&[]), Some(vec![10]));

        let mut allowed = constraint.allowed_next(&[10]).unwrap();
        allowed.sort();
        assert_eq!(allowed, vec![20, 30]);
    }

    // 验证约束生成严格遵守 Trie 可选 token。
    #[test]
    fn test_constrained_generation_respects_trie() {
        // Build a tiny model and a trie that only allows token 0 -> 1 -> 2
        let gpt = GPT::new(4, 4, 2, 1, 8, 4);
        let mut trie = TokenTrie::new();
        trie.insert(&[0, 1, 2]);

        let result = generate_constrained(&gpt, &[0], 3, 4, 0.01, 4, &trie);
        // First generated token must be 0 (trie root allows only 0)
        assert_eq!(result[1], 0, "first generated token should be 0 per trie constraint");
        // Second must be 1
        assert_eq!(result[2], 1, "second generated token should be 1 per trie constraint");
        // Third must be 2
        assert_eq!(result[3], 2, "third generated token should be 2 per trie constraint");
    }

    #[test]
    fn test_inference_state_tracks_prompt_generated_and_context_window() {
        let mut state = InferenceState::new(&[10, 11, 12], 4);
        state.push_token(13);
        state.push_token(14);

        assert_eq!(state.prompt_len(), 3);
        assert_eq!(state.generated_len(), 2);
        assert_eq!(state.tokens(), &[10, 11, 12, 13, 14]);
        assert_eq!(state.generated_tokens(), &[13, 14]);
        assert_eq!(state.context(), &[11, 12, 13, 14]);

        state.reset(&[7, 8]);
        assert_eq!(state.tokens(), &[7, 8]);
        assert_eq!(state.generated_tokens(), &[]);
        assert_eq!(state.context(), &[7, 8]);
    }

    #[test]
    fn test_kv_cache_appends_and_clips_sliding_window() {
        let mut cache = KvCache::new(KvCacheSpec::new(2, 3, 2, 2));
        assert_eq!(cache.layer_count(), 2);
        assert!(cache.is_empty());

        let key_prefill = Tensor::new(
            vec![
                1.0, 1.1, 2.0, 2.1, //
                3.0, 3.1, 4.0, 4.1,
            ],
            vec![2, 2, 2],
        );
        let value_prefill = Tensor::new(
            vec![
                10.0, 10.1, 20.0, 20.1, //
                30.0, 30.1, 40.0, 40.1,
            ],
            vec![2, 2, 2],
        );
        cache.prefill_layer(0, key_prefill, value_prefill);
        assert_eq!(cache.layer(0).unwrap().cached_len(), 2);

        let key_decode = Tensor::new(
            vec![
                5.0, 5.1, 6.0, 6.1, //
                7.0, 7.1, 8.0, 8.1,
            ],
            vec![2, 2, 2],
        );
        let value_decode = Tensor::new(
            vec![
                50.0, 50.1, 60.0, 60.1, //
                70.0, 70.1, 80.0, 80.1,
            ],
            vec![2, 2, 2],
        );
        cache.append_layer(0, key_decode, value_decode);

        let layer = cache.layer(0).unwrap();
        assert_eq!(layer.cached_len(), 3);
        assert_eq!(layer.key().unwrap().shape(), vec![2, 3, 2]);
        assert_eq!(
            layer.key().unwrap().contiguous_data(),
            vec![
                2.0, 2.1, 5.0, 5.1, 6.0, 6.1, //
                4.0, 4.1, 7.0, 7.1, 8.0, 8.1,
            ]
        );
        assert_eq!(
            layer.value().unwrap().contiguous_data(),
            vec![
                20.0, 20.1, 50.0, 50.1, 60.0, 60.1, //
                40.0, 40.1, 70.0, 70.1, 80.0, 80.1,
            ]
        );

        cache.reset_layer(0);
        assert!(cache.layer(0).unwrap().is_empty());
    }

    #[test]
    fn test_inference_batch_sessions_keep_model_kv_cache_spec() {
        let model = QwenLikeGPT::new(8, 6, 3, 2, 12, 5);
        let config = GenerationConfig::greedy(2, 8);
        let batch = InferenceBatch {
            requests: vec![InferenceRequest::new(7, vec![1, 2], config, 0)],
            total_token_budget: 4,
        };

        let state = activate_qwen_like_inference_batch(&model, batch);
        let session = &state.sessions[0];

        assert_eq!(session.kv_cache.spec(), KvCacheSpec::new(2, 5, 3, 2));
        assert_eq!(session.kv_cache.layer_count(), 2);
        assert_eq!(session.state.context(), &[1, 2]);
        assert_eq!(session.kv_cache.cached_len(), 2);
        assert_eq!(session.kv_cache.layer(0).unwrap().key().unwrap().shape(), vec![3, 2, 2]);
        assert_eq!(
            session.kv_cache.layer(1).unwrap().value().unwrap().shape(),
            vec![3, 2, 2]
        );
    }

    #[test]
    fn test_sampling_candidates_apply_top_p_after_probability_sort() {
        let config = GenerationConfig {
            top_p: 0.75,
            ..GenerationConfig::new(1, 4)
        };
        let candidates = sampling_candidates(&[3.0, 2.0, 1.0, 0.0], &config, &[]);
        let ids: Vec<usize> = candidates.iter().map(|candidate| candidate.token_id).collect();
        let prob_sum: f32 = candidates.iter().map(|candidate| candidate.probability).sum();

        assert_eq!(ids, vec![0, 1]);
        assert!((prob_sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_sampling_candidates_repetition_penalty_can_suppress_seen_token() {
        let config = GenerationConfig {
            repetition_penalty: 3.0,
            top_k: 1,
            ..GenerationConfig::new(1, 3)
        };
        let candidates = sampling_candidates(&[3.0, 2.5, 1.0], &config, &[0]);

        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].token_id, 1);
    }

    #[test]
    fn test_qwen_like_generation_uses_shared_config() {
        let model = QwenLikeGPT::new(6, 4, 2, 1, 8, 5);
        let output = generate_qwen_like_with_config(&model, &[0, 1], GenerationConfig::greedy(2, 6));

        assert_eq!(output.len(), 4);
    }

    #[test]
    fn test_generation_stops_immediately_when_prompt_already_has_eos() {
        let model = GPT::new(6, 4, 2, 1, 8, 5);
        let output = generate_with_config(
            &model,
            &[1, 2],
            GenerationConfig {
                eos_token_id: Some(2),
                ..GenerationConfig::greedy(3, 6)
            },
        );

        assert_eq!(output, vec![1, 2]);
    }

    #[test]
    fn test_generation_stops_after_sampling_eos_token() {
        let model = QwenLikeGPT::new(6, 1, 1, 0, 1, 5);
        {
            let token_weight = model.token_emb.weight.0.write().unwrap();
            let mut storage = token_weight.storage.write().unwrap();
            let slice = storage.as_cpu_slice_mut();
            slice.fill(0.0);
            slice[1] = 1.0;
        }
        {
            let lm_head_weight = model.lm_head.weight.0.write().unwrap();
            let mut storage = lm_head_weight.storage.write().unwrap();
            let slice = storage.as_cpu_slice_mut();
            slice.fill(0.0);
            slice[3] = 5.0;
        }

        let config = GenerationConfig {
            eos_token_id: Some(3),
            ..GenerationConfig::greedy(4, 6)
        };
        let output = generate_qwen_like_with_config(&model, &[1], config);

        assert_eq!(output, vec![1, 3]);
    }

    #[test]
    fn test_decode_next_token_appends_to_state_and_returns_candidates() {
        let mut model = GPT::new(6, 4, 2, 1, 8, 5);
        model.set_training(false);
        let mut state = InferenceState::new(&[1], model.seq_len);
        let config = GenerationConfig::greedy(4, 6);

        let step = decode_next_token(&model, &mut state, &config);

        assert!(step.token_id.is_some());
        assert_eq!(step.stop_reason, DecodeStopReason::NotStopped);
        assert!(!step.candidates.is_empty());
        assert_eq!(state.tokens().len(), 2);
        assert_eq!(state.generated_tokens(), &[step.token_id.unwrap()]);
    }

    #[test]
    fn test_decode_constrained_next_token_respects_trie_prefix() {
        let model = GPT::new(4, 4, 2, 1, 8, 4);
        let mut state = InferenceState::new(&[1], model.seq_len);
        let config = GenerationConfig::greedy(3, 4);
        let mut trie = TokenTrie::new();
        trie.insert(&[2, 3]);

        let step = decode_constrained_next_token(&model, &mut state, &config, &trie);

        assert_eq!(step.token_id, Some(2));
        assert_eq!(step.stop_reason, DecodeStopReason::NotStopped);
        assert_eq!(state.generated_tokens(), &[2]);
    }

    #[test]
    fn test_decode_next_token_stops_when_state_already_ended() {
        let model = GPT::new(6, 4, 2, 1, 8, 5);
        let mut state = InferenceState::new(&[1, 2], model.seq_len);
        let config = GenerationConfig {
            eos_token_id: Some(2),
            ..GenerationConfig::greedy(4, 6)
        };

        let step = decode_next_token(&model, &mut state, &config);

        assert!(step.finished);
        assert_eq!(step.token_id, None);
        assert_eq!(step.stop_reason, DecodeStopReason::AlreadyEnded);
        assert!(step.candidates.is_empty());
        assert_eq!(state.tokens(), &[1, 2]);
    }

    #[test]
    fn test_decode_constrained_next_token_reports_no_candidates() {
        let model = GPT::new(4, 4, 2, 1, 8, 4);
        let mut state = InferenceState::new(&[1], model.seq_len);
        let config = GenerationConfig::greedy(3, 4);
        let mut trie = TokenTrie::new();
        trie.insert(&[99]);

        let step = decode_constrained_next_token(&model, &mut state, &config, &trie);

        assert!(step.finished);
        assert_eq!(step.token_id, None);
        assert_eq!(step.stop_reason, DecodeStopReason::NoCandidates);
        assert_eq!(state.tokens(), &[1]);
    }

    #[test]
    fn test_inference_scheduler_batches_fifo_with_token_budget() {
        let mut scheduler = InferenceScheduler::new(InferenceSchedulerConfig::new(3, 10, 8));
        let config = GenerationConfig::greedy(2, 16);

        scheduler
            .enqueue(InferenceRequest::new(1, vec![1, 2], config, 0))
            .unwrap();
        scheduler
            .enqueue(InferenceRequest::new(2, vec![3, 4, 5], config, 1))
            .unwrap();
        scheduler
            .enqueue(InferenceRequest::new(3, vec![6, 7, 8], config, 2))
            .unwrap();

        let first = scheduler.plan_next_batch().unwrap();
        assert_eq!(first.request_ids(), vec![1, 2]);
        assert_eq!(first.total_token_budget, 9);
        assert_eq!(scheduler.pending_len(), 1);

        let second = scheduler.plan_next_batch().unwrap();
        assert_eq!(second.request_ids(), vec![3]);
        assert_eq!(second.total_token_budget, 5);
        assert!(scheduler.is_empty());
    }

    #[test]
    fn test_inference_scheduler_rejects_queue_full() {
        let mut scheduler = InferenceScheduler::new(InferenceSchedulerConfig::new(2, 10, 1));
        let config = GenerationConfig::greedy(1, 8);

        scheduler.enqueue(InferenceRequest::new(1, vec![1], config, 0)).unwrap();
        let err = scheduler
            .enqueue(InferenceRequest::new(2, vec![2], config, 1))
            .unwrap_err();

        assert_eq!(err, InferenceAdmissionError::QueueFull);
        assert_eq!(scheduler.pending_len(), 1);
    }

    #[test]
    fn test_inference_scheduler_rejects_request_too_large() {
        let mut scheduler = InferenceScheduler::new(InferenceSchedulerConfig::new(2, 4, 4));
        let config = GenerationConfig::greedy(3, 8);
        let err = scheduler
            .enqueue(InferenceRequest::new(1, vec![1, 2], config, 0))
            .unwrap_err();

        assert_eq!(err, InferenceAdmissionError::RequestTooLarge);
        assert!(scheduler.is_empty());
    }

    #[test]
    fn test_inference_scheduler_respects_batch_size_limit() {
        let mut scheduler = InferenceScheduler::new(InferenceSchedulerConfig::new(2, 100, 8));
        let config = GenerationConfig::greedy(1, 8);
        for idx in 0..3 {
            scheduler
                .enqueue(InferenceRequest::new(idx + 1, vec![idx as usize], config, idx))
                .unwrap();
        }

        let batch = scheduler.plan_next_batch().unwrap();
        assert_eq!(batch.request_ids(), vec![1, 2]);
        assert_eq!(scheduler.pending_len(), 1);
    }

    #[test]
    fn test_inference_batch_state_activates_sessions() {
        let model = GPT::new(8, 4, 2, 1, 8, 6);
        let config = GenerationConfig::greedy(2, 8);
        let batch = InferenceBatch {
            requests: vec![
                InferenceRequest::new(10, vec![1, 2], config, 0),
                InferenceRequest::new(11, vec![3], config, 1),
            ],
            total_token_budget: 0,
        };

        let state = activate_inference_batch(&model, batch);

        assert_eq!(state.request_ids(), vec![10, 11]);
        assert_eq!(state.active_len(), 2);
        assert_eq!(state.sessions[0].tokens(), &[1, 2]);
        assert_eq!(state.sessions[0].kv_cache.cached_len(), 2);
        assert_eq!(state.sessions[1].kv_cache.cached_len(), 1);
    }

    #[test]
    fn test_decode_inference_batch_round_advances_each_active_session() {
        let mut model = GPT::new(8, 4, 2, 1, 8, 6);
        model.set_training(false);
        let config = GenerationConfig::greedy(2, 8);
        let batch = InferenceBatch {
            requests: vec![
                InferenceRequest::new(1, vec![1], config, 0),
                InferenceRequest::new(2, vec![2], config, 1),
            ],
            total_token_budget: 0,
        };
        let mut state = activate_inference_batch(&model, batch);

        let steps = decode_inference_batch_round(&model, &mut state);

        assert_eq!(steps.len(), 2);
        assert_eq!(steps.iter().map(|step| step.request_id).collect::<Vec<_>>(), vec![1, 2]);
        assert_eq!(state.sessions[0].generated_steps, 1);
        assert_eq!(state.sessions[1].generated_steps, 1);
        assert_eq!(state.sessions[0].generated_tokens().len(), 1);
        assert_eq!(state.sessions[1].generated_tokens().len(), 1);
        assert_eq!(
            state.sessions[0].kv_cache.cached_len(),
            state.sessions[0].state.context().len()
        );
        assert_eq!(
            state.sessions[1].kv_cache.cached_len(),
            state.sessions[1].state.context().len()
        );
    }

    #[test]
    fn test_decode_inference_batch_round_marks_max_new_tokens() {
        let mut model = GPT::new(8, 4, 2, 1, 8, 6);
        model.set_training(false);
        let config = GenerationConfig::greedy(1, 8);
        let batch = InferenceBatch {
            requests: vec![InferenceRequest::new(1, vec![1], config, 0)],
            total_token_budget: 0,
        };
        let mut state = activate_inference_batch(&model, batch);

        let first = decode_inference_batch_round(&model, &mut state);
        assert_eq!(first.len(), 1);
        assert_eq!(state.sessions[0].generated_steps, 1);
        assert!(state.is_finished());
        assert_eq!(state.sessions[0].stop_reason, DecodeStopReason::MaxNewTokens);

        let second = decode_inference_batch_round(&model, &mut state);
        assert!(second.is_empty());
    }

    #[test]
    fn test_inference_batch_state_skips_prompt_that_already_ended() {
        let model = GPT::new(8, 4, 2, 1, 8, 6);
        let config = GenerationConfig {
            eos_token_id: Some(2),
            ..GenerationConfig::greedy(3, 8)
        };
        let batch = InferenceBatch {
            requests: vec![InferenceRequest::new(1, vec![1, 2], config, 0)],
            total_token_budget: 0,
        };
        let mut state = activate_inference_batch(&model, batch);

        assert!(state.is_finished());
        assert_eq!(state.sessions[0].stop_reason, DecodeStopReason::AlreadyEnded);
        assert!(decode_inference_batch_round(&model, &mut state).is_empty());
    }

    // --- Dropout tests ---

    #[test]
    fn test_dropout_eval_passthrough() {
        let mut drop = Dropout::new(0.5);
        drop.eval();
        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.forward(&input);
        assert_eq!(out.data(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    // 验证 dropout 率为 0 时输出不变。
    #[test]
    fn test_dropout_zero_rate() {
        let drop = Dropout::new(0.0);
        let input = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let out = drop.forward(&input);
        assert_eq!(out.data(), vec![1.0, 2.0, 3.0]);
    }
    // 验证训练模式下高 dropout 率会产生大量 0。
    #[test]
    fn test_dropout_train_has_zeros() {
        let drop = Dropout::new(0.9); // 90% dropout，理论上大多数位置会被置零
        let input = Tensor::new(vec![1.0; 100], vec![100]);
        let out = drop.forward(&input);
        let zeros = out.data().iter().filter(|&&x| x == 0.0).count();
        assert!(
            zeros > 50,
            "with 90% dropout, most values should be zero, got {} zeros",
            zeros
        );
    }

    // 验证 inverted dropout 近似保持期望均值。
    #[test]
    fn test_dropout_scale_preserves_mean() {
        // inverted dropout 下，输出期望值应接近输入期望值
        let drop = Dropout::new(0.5);
        let input = Tensor::new(vec![2.0; 10000], vec![10000]);
        let out = drop.forward(&input);
        let mean: f32 = out.data().iter().sum::<f32>() / 10000.0;
        // Mean should be close to 2.0 (within statistical noise)
        assert!((mean - 2.0).abs() < 0.2, "mean should be ~2.0, got {}", mean);
    }
}

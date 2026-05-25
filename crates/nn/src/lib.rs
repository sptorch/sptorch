//! Neural network modules for sptorch.
//!
//! - `Module` trait, `Linear`, `LoRALinear` (low-rank adaptation)
//! - `Embedding`, `LayerNorm`, `MultiHeadAttention`, `TransformerBlock`, `GPT`
//! - `TokenTrie` + `TokenConstraint` for constrained decoding
//! - `generate_greedy`, `generate_with_sampling`, `generate_constrained`

use rand::Rng;
use sptorch_core_ops::*;
use sptorch_core_tensor::Tensor;

// ============ Module Trait ============

pub trait Module: Send + Sync {
    // 前向计算接口：接收输入张量并返回输出张量。
    fn forward(&self, input: &Tensor) -> Tensor;
    // 返回当前模块需要优化或保存的参数张量列表。
    fn parameters(&self) -> Vec<Tensor>;
}

// ============ Initialization ============
/// 使用 Xavier Uniform 初始化权重张量。
pub fn xavier_uniform(rows: usize, cols: usize) -> Tensor {
    let mut rng = rand::thread_rng();
    let limit = (6.0 / (rows + cols) as f32).sqrt();
    let data: Vec<f32> = (0..rows * cols).map(|_| rng.gen_range(-limit..limit)).collect();
    Tensor::with_grad(data, vec![rows, cols], true)
}
/// 使用 Kaiming Normal 初始化权重张量。
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
    pub weight: Tensor,       // [out_features, in_features]
    pub bias: Option<Tensor>, // [out_features]
}

impl Linear {
    /// 创建线性层并初始化权重/偏置。
    pub fn new(in_features: usize, out_features: usize, use_bias: bool) -> Self {
        let weight = xavier_uniform(out_features, in_features);
        let bias = if use_bias { Some(zeros_grad(out_features)) } else { None };
        Linear { weight, bias }
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
    pub rate: f32,
    pub training: bool,
}

impl Dropout {
    /// 创建 Dropout 层并设置丢弃率。
    pub fn new(rate: f32) -> Self {
        Dropout { rate, training: true }
    }
    /// 执行 Dropout 前向：训练期随机置零并缩放，评估期直通。
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
    /// 切换到评估模式（禁用随机丢弃）。
    pub fn eval(&mut self) {
        self.training = false;
    }
    /// 切换到训练模式（启用随机丢弃）。
    pub fn train(&mut self) {
        self.training = true;
    }
}

// ============ LoRA Linear ============

/// LoRA adapter wrapping a frozen Linear layer.
/// Forward: output = x @ W^T + x @ (B @ A)^T * (alpha / rank)
/// Only A and B are trainable; W is frozen.
pub struct LoRALinear {
    pub base: Linear,
    pub lora_a: Tensor, // [rank, in_features]
    pub lora_b: Tensor, // [out_features, rank]
    pub alpha: f32,
    pub rank: usize,
}

impl LoRALinear {
    /// 基于已有线性层创建 LoRA 适配器并冻结基座参数。
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
    pub fn all_parameters(&self) -> Vec<Tensor> {
        let mut p = self.base.parameters();
        p.push(self.lora_a.clone());
        p.push(self.lora_b.clone());
        p
    }
    /// 将 LoRA 增量权重合并进基座权重。
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
    pub weight: Tensor, // [num_embeddings, embedding_dim]
}

impl Embedding {
    /// 创建 Embedding 层。
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
    pub fn forward_indices(&self, indices: &[usize]) -> Tensor {
        embedding_lookup(&self.weight, indices)
    }
    /// 返回 Embedding 可训练参数。
    pub fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone()]
    }
}

// ============ LayerNorm ============

pub struct LayerNorm {
    pub gamma: Tensor, // [normalized_shape]
    pub beta: Tensor,  // [normalized_shape]
    pub eps: f32,
    pub normalized_shape: usize,
}

impl LayerNorm {
    /// 创建 LayerNorm 层。
    pub fn new(normalized_shape: usize) -> Self {
        LayerNorm {
            gamma: Tensor::with_grad(vec![1.0; normalized_shape], vec![normalized_shape], true),
            beta: Tensor::with_grad(vec![0.0; normalized_shape], vec![normalized_shape], true),
            eps: 1e-5,
            normalized_shape,
        }
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
    pub n_head: usize,
    pub d_model: usize,
    pub head_dim: usize,
    pub wq: Linear,
    pub wk: Linear,
    pub wv: Linear,
    pub wo: Linear,
}

impl MultiHeadAttention {
    /// 创建多头自注意力层，要求 `d_model % n_head == 0`。
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
    pub fn forward_causal(&self, input: &Tensor) -> Tensor {
        let shape = input.shape();
        let seq_len = shape[0];
        let h = self.n_head;
        let hd = self.head_dim;

        // Q, K, V projections: [seq_len, d_model]
        let q = self.wq.forward(input);
        let k = self.wk.forward(input);
        let v = self.wv.forward(input);

        // Reshape to [n_head, seq_len, head_dim]
        let q3 = reshape_to_heads(&q, seq_len, h, hd);
        let k3 = reshape_to_heads(&k, seq_len, h, hd);
        let v3 = reshape_to_heads(&v, seq_len, h, hd);

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
    /// 返回注意力层全部参数。
    pub fn parameters(&self) -> Vec<Tensor> {
        let mut p = Vec::new();
        p.extend(self.wq.parameters());
        p.extend(self.wk.parameters());
        p.extend(self.wv.parameters());
        p.extend(self.wo.parameters());
        p
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

// ============ TransformerBlock ============

pub struct TransformerBlock {
    pub ln1: LayerNorm,
    pub attn: MultiHeadAttention,
    pub ln2: LayerNorm,
    pub ffn_up: Linear,
    pub ffn_down: Linear,
    pub attn_dropout: Dropout,
    pub ffn_dropout: Dropout,
}

impl TransformerBlock {
    /// 创建 TransformerBlock（注意力 + FFN + LayerNorm）。
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
    /// 执行 TransformerBlock 前向（含残差连接）。
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
}

// ============ GPT Model ============

pub struct GPT {
    pub token_emb: Embedding,
    pub pos_emb: Embedding,
    pub blocks: Vec<TransformerBlock>,
    pub ln_f: LayerNorm,
    pub lm_head: Linear,
    pub seq_len: usize,
}

impl GPT {
    /// 创建 GPT 模型。
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

    /// 统一切换 GPT 内部块的训练/评估状态。
    ///
    /// 当前真正持有随机行为的是每个 `TransformerBlock` 里的 dropout；把这个
    /// 开关放到模型层，调用者就不用手动逐层遍历，训练和推理模式也更不容易
    /// 在测试里被误配。
    pub fn set_training(&mut self, training: bool) {
        for block in &mut self.blocks {
            block.set_training(training);
        }
    }
}

// ============ Text Generation ============
/// 贪心解码生成 token 序列。
pub fn generate_greedy(model: &GPT, prompt: &[usize], max_new_tokens: usize, vocab_size: usize) -> Vec<usize> {
    let mut ids = prompt.to_vec();
    for _ in 0..max_new_tokens {
        let ctx = if ids.len() > model.seq_len {
            &ids[ids.len() - model.seq_len..]
        } else {
            &ids
        };
        let logits = model.forward_ids(ctx);
        let logits_data = logits.contiguous_data();
        let last = &logits_data[logits_data.len() - vocab_size..];
        let next = last
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        ids.push(next);
    }
    ids
}
/// 按 temperature 与 top-k 采样生成序列。
pub fn generate_with_sampling(
    model: &GPT,
    prompt: &[usize],
    max_new_tokens: usize,
    vocab_size: usize,
    temperature: f32,
    top_k: usize,
) -> Vec<usize> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut ids = prompt.to_vec();

    for _ in 0..max_new_tokens {
        let ctx = if ids.len() > model.seq_len {
            &ids[ids.len() - model.seq_len..]
        } else {
            &ids
        };
        let logits = model.forward_ids(ctx);
        let logits_data = logits.contiguous_data();
        let last = &logits_data[logits_data.len() - vocab_size..];

        // Apply temperature
        let scaled: Vec<f32> = last.iter().map(|x| x / temperature.max(1e-8)).collect();

        // Top-k filtering
        let mut indexed: Vec<(usize, f32)> = scaled.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let k = top_k.min(vocab_size);
        let top = &indexed[..k];

        // Softmax over top-k
        let max_val = top[0].1;
        let exps: Vec<f32> = top.iter().map(|(_, v)| (v - max_val).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let probs: Vec<f32> = exps.iter().map(|e| e / sum).collect();

        // Sample
        let r: f32 = rng.gen();
        let mut cumsum = 0.0;
        let mut next = top[0].0;
        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if r < cumsum {
                next = top[i].0;
                break;
            }
        }
        ids.push(next);
    }
    ids
}

// ============ Constrained Decoding ============

/// Prefix trie for constraining token generation.
/// Each node maps token_id -> child node. A node with `is_terminal = true`
/// marks the end of a valid sequence.
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
    pub fn insert(&mut self, tokens: &[usize]) {
        let mut node = self;
        for &t in tokens {
            node = node.children.entry(t).or_default();
        }
        node.is_terminal = true;
    }
    /// 查询给定前缀下允许的下一 token 集合。
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
    /// 返回 `None` 表示放开约束，允许全部 token。
    fn allowed_next(&self, generated: &[usize]) -> Option<Vec<usize>>;
}

impl TokenConstraint for TokenTrie {
    // 按当前已生成序列返回下一步允许的 token 集合。
    fn allowed_next(&self, generated: &[usize]) -> Option<Vec<usize>> {
        self.allowed_tokens(generated)
    }
}
/// 在约束条件下执行采样生成。
pub fn generate_constrained(
    model: &GPT,
    prompt: &[usize],
    max_new_tokens: usize,
    vocab_size: usize,
    temperature: f32,
    top_k: usize,
    constraint: &dyn TokenConstraint,
) -> Vec<usize> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut ids = prompt.to_vec();
    let prompt_len = prompt.len();

    for _ in 0..max_new_tokens {
        let ctx = if ids.len() > model.seq_len {
            &ids[ids.len() - model.seq_len..]
        } else {
            &ids
        };
        let logits = model.forward_ids(ctx);
        let logits_data = logits.contiguous_data();
        let last = &logits_data[logits_data.len() - vocab_size..];

        // Apply constraint mask
        let generated_so_far = &ids[prompt_len..];
        let allowed = constraint.allowed_next(generated_so_far);

        let mut masked: Vec<f32> = last.to_vec();
        if let Some(ref allowed_set) = allowed {
            for (i, m) in masked.iter_mut().enumerate() {
                if !allowed_set.contains(&i) {
                    *m = f32::NEG_INFINITY;
                }
            }
        }

        // Apply temperature
        let scaled: Vec<f32> = masked.iter().map(|x| x / temperature.max(1e-8)).collect();

        // Top-k filtering
        let mut indexed: Vec<(usize, f32)> = scaled
            .iter()
            .enumerate()
            .filter(|(_, &v)| v.is_finite())
            .map(|(i, &v)| (i, v))
            .collect();
        if indexed.is_empty() {
            break; // no valid tokens
        }
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let k = top_k.min(indexed.len());
        let top = &indexed[..k];

        // Softmax over top-k
        let max_val = top[0].1;
        let exps: Vec<f32> = top.iter().map(|(_, v)| (v - max_val).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let probs: Vec<f32> = exps.iter().map(|e| e / sum).collect();

        // Sample
        let r: f32 = rng.gen();
        let mut cumsum = 0.0;
        let mut next = top[0].0;
        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if r < cumsum {
                next = top[i].0;
                break;
            }
        }
        ids.push(next);
    }
    ids
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

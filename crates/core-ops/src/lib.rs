//! SPTorch 的可微分核心算子库。
//!
//! 本 crate 连接 `core-tensor` 的计算图协议和实际数学算子：每个可导算子都
//! 同时实现前向计算、反向传播公式，并在可能时通过后端注册表分发到设备
//! kernel。这里的目标是把数学语义写清楚，让 CPU fallback、CUDA 后端和未来
//! Tank9k kernel 都能对齐同一套行为。
//!
//! 当前覆盖：add、mul、neg、sub、sum、mean、matmul、exp、log、reshape、
//! transpose、softmax、log_softmax、cross_entropy_loss、embedding_lookup、relu、
//! gelu、scale、masked_fill、batch_matmul、broadcast_add、concat。

use sptorch_core_tensor::{get_backend, Device, Node, Op, Tensor};
use std::sync::Arc;

// ============ Backend-aware dispatch helper ============

// 二元算子统一走这个入口：如果张量设备注册了后端，就把连续 F32 切片交给
// 后端 kernel；否则执行 CPU 闭包。shape 检查由具体算子负责，这里只处理设备派发。
fn dispatch_binary(
    a_data: &[f32],
    b_data: &[f32],
    device: &Device,
    cpu_fn: impl Fn(&[f32], &[f32]) -> Vec<f32>,
    backend_fn: impl Fn(&dyn sptorch_core_tensor::BackendDispatch, &[f32], &[f32], &mut [f32]),
) -> Vec<f32> {
    if let Some(backend) = get_backend(device) {
        let mut out = vec![0.0f32; a_data.len()];
        backend_fn(&*backend, a_data, b_data, &mut out);
        out
    } else {
        cpu_fn(a_data, b_data)
    }
}

// 一元算子分发：优先使用设备后端，缺失时回退到 CPU 实现。
fn dispatch_unary(
    a_data: &[f32],
    device: &Device,
    cpu_fn: impl Fn(&[f32]) -> Vec<f32>,
    backend_fn: impl Fn(&dyn sptorch_core_tensor::BackendDispatch, &[f32], &mut [f32]),
) -> Vec<f32> {
    if let Some(backend) = get_backend(device) {
        let mut out = vec![0.0f32; a_data.len()];
        backend_fn(&*backend, a_data, &mut out);
        out
    } else {
        cpu_fn(a_data)
    }
}

// ============ GPU Accelerator (optional, via "cuda" feature) ============

#[cfg(feature = "cuda")]
mod gpu_accel {
    use sptorch_runtime_cuda::{CudaBackend, GpuTensor};
    use std::sync::OnceLock;

    static GPU: OnceLock<Option<CudaBackend>> = OnceLock::new();
    /// 获取全局 CUDA 后端实例；不可用时返回 `None`。
    pub fn get_gpu() -> Option<&'static CudaBackend> {
        GPU.get_or_init(|| match CudaBackend::new(0) {
            Ok(b) => {
                if b.load_kernels().is_ok() {
                    eprintln!("[sptorch] GPU accelerator enabled (cuBLAS matmul)");
                    Some(b)
                } else {
                    None
                }
            }
            Err(_) => None,
        })
        .as_ref()
    }
    /// 尝试在 GPU 上执行矩阵乘法并返回主机结果。
    pub fn gpu_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Option<Vec<f32>> {
        let backend = get_gpu()?;
        let ga = GpuTensor::from_host(backend, a, vec![m, k]).ok()?;
        let gb = GpuTensor::from_host(backend, b, vec![k, n]).ok()?;
        let gc = backend.gpu_matmul(&ga, &gb).ok()?;
        gc.to_host(backend).ok()
    }
}

// ============ Tiled Matmul (cache-friendly CPU fallback) ============

// CPU fallback 使用 matrixmultiply::sgemm。这里的 unsafe 仅用于把 Rust 切片
// 指针交给成熟 BLAS 风格内核；前置 assert 保证输入长度满足 row-major 矩阵读取边界。
fn tiled_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    assert!(a.len() >= m * k, "matmul: a.len()={} but need m*k={}", a.len(), m * k);
    assert!(b.len() >= k * n, "matmul: b.len()={} but need k*n={}", b.len(), k * n);
    let mut c = vec![0.0f32; m * n];
    unsafe {
        matrixmultiply::sgemm(
            m,
            k,
            n,
            1.0,
            a.as_ptr(),
            k as isize,
            1,
            b.as_ptr(),
            n as isize,
            1,
            0.0,
            c.as_mut_ptr(),
            n as isize,
            1,
        );
    }
    c
}

const _GPU_MATMUL_THRESHOLD: usize = 128 * 128;

// 矩阵乘法分发：CUDA 失败不会中断训练，而是回退到 CPU 参考实现。
fn dispatch_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    #[cfg(feature = "cuda")]
    {
        if m * k + k * n >= GPU_MATMUL_THRESHOLD {
            if let Some(result) = gpu_accel::gpu_matmul(a, b, m, k, n) {
                return result;
            }
        }
    }
    tiled_matmul(a, b, m, k, n)
}

// ============ Add ============

#[derive(Debug)]
pub struct AddOp;

impl Op for AddOp {
    // 根据上游梯度计算当前节点对输入的梯度。
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        vec![Some(grad_output.clone()), Some(grad_output.clone())]
    }
}
/// 逐元素加法：`a + b`，输入形状需要一致。
pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
    let a_data = a.data();
    let b_data = b.data();
    let shape = a.shape();
    let device = a.device();

    let res_data = dispatch_binary(
        &a_data,
        &b_data,
        &device,
        |a, b| a.iter().zip(b.iter()).map(|(x, y)| x + y).collect(),
        |backend, a, b, out| backend.add_f32(a, b, out),
    );
    let res = Tensor::new(res_data, shape);

    if a.requires_grad() || b.requires_grad() {
        let mut inner = res.0.write().unwrap();
        inner.requires_grad = true;
        inner.creator = Some(Arc::new(Node {
            op: Box::new(AddOp),
            inputs: vec![a.clone(), b.clone()],
        }));
    }

    res
}

// ============ Mul ============

#[derive(Debug)]
pub struct MulOp {
    saved_a: Tensor,
    saved_b: Tensor,
}

impl Op for MulOp {
    // 根据上游梯度计算当前节点对输入的梯度。
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        let g = grad_output.data();
        let a = self.saved_a.data();
        let b = self.saved_b.data();

        let da: Vec<f32> = g.iter().zip(b.iter()).map(|(g, b)| g * b).collect();
        let db: Vec<f32> = g.iter().zip(a.iter()).map(|(g, a)| g * a).collect();

        vec![
            Some(Tensor::new(da, grad_output.shape())),
            Some(Tensor::new(db, grad_output.shape())),
        ]
    }
}
/// 逐元素乘法：`a * b`，输入形状需要一致。
pub fn mul(a: &Tensor, b: &Tensor) -> Tensor {
    let a_data = a.data();
    let b_data = b.data();
    let shape = a.shape();
    let device = a.device();

    let res_data = dispatch_binary(
        &a_data,
        &b_data,
        &device,
        |a, b| a.iter().zip(b.iter()).map(|(x, y)| x * y).collect(),
        |backend, a, b, out| backend.mul_f32(a, b, out),
    );
    let res = Tensor::new(res_data, shape);

    if a.requires_grad() || b.requires_grad() {
        let mut inner = res.0.write().unwrap();
        inner.requires_grad = true;
        inner.creator = Some(Arc::new(Node {
            op: Box::new(MulOp {
                saved_a: a.clone(),
                saved_b: b.clone(),
            }),
            inputs: vec![a.clone(), b.clone()],
        }));
    }

    res
}

// ============ Neg ============

#[derive(Debug)]
pub struct NegOp;

impl Op for NegOp {
    // 根据上游梯度计算当前节点对输入的梯度。
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        let grad_data: Vec<f32> = grad_output.data().iter().map(|x| -x).collect();
        vec![Some(Tensor::new(grad_data, grad_output.shape()))]
    }
}
/// 逐元素取负：`-a`。
pub fn neg(a: &Tensor) -> Tensor {
    let a_data = a.data();
    let shape = a.shape();
    let device = a.device();

    let res_data = dispatch_unary(
        &a_data,
        &device,
        |a| a.iter().map(|x| -x).collect(),
        |backend, a, out| backend.neg_f32(a, out),
    );
    let res = Tensor::new(res_data, shape);

    if a.requires_grad() {
        let mut inner = res.0.write().unwrap();
        inner.requires_grad = true;
        inner.creator = Some(Arc::new(Node {
            op: Box::new(NegOp),
            inputs: vec![a.clone()],
        }));
    }

    res
}

// ============ Sub ============
/// 逐元素减法：`a - b`，通过 `add(a, -b)` 复用实现。
pub fn sub(a: &Tensor, b: &Tensor) -> Tensor {
    add(a, &neg(b))
}

// ============ Sum ============

#[derive(Debug)]
pub struct SumOp {
    input_shape: Vec<usize>,
}

impl Op for SumOp {
    // 根据上游梯度计算当前节点对输入的梯度。
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        let g_val = grad_output.data()[0];
        let size: usize = self.input_shape.iter().product();
        vec![Some(Tensor::new(vec![g_val; size], self.input_shape.clone()))]
    }
}
/// 对张量全部元素求和，返回标量张量 `[1]`。
pub fn sum(a: &Tensor) -> Tensor {
    let a_data = a.data();
    let total: f32 = a_data.iter().sum();
    let res = Tensor::new(vec![total], vec![1]);

    if a.requires_grad() {
        let mut inner = res.0.write().unwrap();
        inner.requires_grad = true;
        inner.creator = Some(Arc::new(Node {
            op: Box::new(SumOp { input_shape: a.shape() }),
            inputs: vec![a.clone()],
        }));
    }

    res
}

// ============ Mean ============

#[derive(Debug)]
pub struct MeanOp {
    input_shape: Vec<usize>,
    numel: f32,
}

impl Op for MeanOp {
    // 根据上游梯度计算当前节点对输入的梯度。
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        let g_val = grad_output.data()[0];
        let size: usize = self.input_shape.iter().product();
        vec![Some(Tensor::new(
            vec![g_val / self.numel; size],
            self.input_shape.clone(),
        ))]
    }
}
/// 对张量全部元素求均值，返回标量张量 `[1]`。
pub fn mean(a: &Tensor) -> Tensor {
    let a_data = a.data();
    let n = a_data.len() as f32;
    let total: f32 = a_data.iter().sum();
    let res = Tensor::new(vec![total / n], vec![1]);

    if a.requires_grad() {
        let mut inner = res.0.write().unwrap();
        inner.requires_grad = true;
        inner.creator = Some(Arc::new(Node {
            op: Box::new(MeanOp {
                input_shape: a.shape(),
                numel: n,
            }),
            inputs: vec![a.clone()],
        }));
    }

    res
}

// ============ Matmul (2D) ============

#[derive(Debug)]
pub struct MatmulOp {
    saved_a: Tensor,
    saved_b: Tensor,
}

impl Op for MatmulOp {
    // 根据上游梯度计算当前节点对输入的梯度。
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        let a_shape = self.saved_a.shape();
        let b_shape = self.saved_b.shape();
        let (m, k) = (a_shape[0], a_shape[1]);
        let n = b_shape[1];

        let g = grad_output.data();
        let a = self.saved_a.data();
        let b = self.saved_b.data();

        // dA = dY @ B^T  [m,n] @ [n,k] = [m,k]
        // Transpose B first, then use tiled_matmul
        let mut bt = vec![0.0f32; n * k];
        for i in 0..k {
            for j in 0..n {
                bt[j * k + i] = b[i * n + j];
            }
        }
        let da = dispatch_matmul(&g, &bt, m, n, k);

        // dB = A^T @ dY  [k,m] @ [m,n] = [k,n]
        let mut at = vec![0.0f32; k * m];
        for i in 0..m {
            for j in 0..k {
                at[j * m + i] = a[i * k + j];
            }
        }
        let db = dispatch_matmul(&at, &g, k, m, n);

        vec![Some(Tensor::new(da, vec![m, k])), Some(Tensor::new(db, vec![k, n]))]
    }
}
/// 二维矩阵乘法：`[m,k] x [k,n] -> [m,n]`。
pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    let a_shape = a.shape();
    let b_shape = b.shape();
    let (m, k) = (a_shape[0], a_shape[1]);
    let n = b_shape[1];

    let a_data = a.data();
    let b_data = b.data();
    let device = a.device();

    let res_data = if let Some(backend) = get_backend(&device) {
        let mut out = vec![0.0f32; m * n];
        backend.matmul_f32(&a_data, &b_data, &mut out, m, k, n);
        out
    } else {
        dispatch_matmul(&a_data, &b_data, m, k, n)
    };

    let res = Tensor::new(res_data, vec![m, n]);

    if a.requires_grad() || b.requires_grad() {
        let mut inner = res.0.write().unwrap();
        inner.requires_grad = true;
        inner.creator = Some(Arc::new(Node {
            op: Box::new(MatmulOp {
                saved_a: a.clone(),
                saved_b: b.clone(),
            }),
            inputs: vec![a.clone(), b.clone()],
        }));
    }

    res
}

// ============ Exp ============

#[derive(Debug)]
pub struct ExpOp {
    output: Tensor,
}

impl Op for ExpOp {
    // 根据上游梯度计算当前节点对输入的梯度。
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        // d/da exp(a) = exp(a) * grad
        let g = grad_output.data();
        let out = self.output.data();
        let da: Vec<f32> = g.iter().zip(out.iter()).map(|(g, o)| g * o).collect();
        vec![Some(Tensor::new(da, grad_output.shape()))]
    }
}
/// 逐元素指数函数。
pub fn exp(a: &Tensor) -> Tensor {
    let a_data = a.data();
    let shape = a.shape();
    let device = a.device();
    let res_data = dispatch_unary(
        &a_data,
        &device,
        |a| a.iter().map(|x| x.exp()).collect(),
        |backend, a, out| backend.exp_f32(a, out),
    );
    let res = Tensor::new(res_data, shape);

    if a.requires_grad() {
        let mut inner = res.0.write().unwrap();
        inner.requires_grad = true;
        inner.creator = Some(Arc::new(Node {
            op: Box::new(ExpOp { output: res.clone() }),
            inputs: vec![a.clone()],
        }));
    }

    res
}

// ============ Log ============

#[derive(Debug)]
pub struct LogOp {
    saved_input: Tensor,
}

impl Op for LogOp {
    // 根据上游梯度计算当前节点对输入的梯度。
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        // d/da ln(a) = grad / a
        let g = grad_output.data();
        let a = self.saved_input.data();
        let da: Vec<f32> = g.iter().zip(a.iter()).map(|(g, a)| g / a).collect();
        vec![Some(Tensor::new(da, grad_output.shape()))]
    }
}
/// 逐元素自然对数；输入应大于 0。
pub fn log(a: &Tensor) -> Tensor {
    let a_data = a.data();
    let shape = a.shape();
    let device = a.device();
    let res_data = dispatch_unary(
        &a_data,
        &device,
        |a| a.iter().map(|x| x.ln()).collect(),
        |backend, a, out| backend.log_f32(a, out),
    );
    let res = Tensor::new(res_data, shape);

    if a.requires_grad() {
        let mut inner = res.0.write().unwrap();
        inner.requires_grad = true;
        inner.creator = Some(Arc::new(Node {
            op: Box::new(LogOp { saved_input: a.clone() }),
            inputs: vec![a.clone()],
        }));
    }

    res
}

// ============ Reshape (零拷贝视图) ============

#[derive(Debug)]
pub struct ReshapeOp {
    input_shape: Vec<usize>,
}

impl Op for ReshapeOp {
    // 根据上游梯度计算当前节点对输入的梯度。
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        // Reshape grad back to input shape
        let g = grad_output.contiguous_data();
        vec![Some(Tensor::new(g, self.input_shape.clone()))]
    }
}
/// 仅改变张量形状视图，元素总数必须保持一致。
pub fn reshape(a: &Tensor, new_shape: Vec<usize>) -> Tensor {
    let old_shape = a.shape();
    let old_numel: usize = old_shape.iter().product();
    let new_numel: usize = new_shape.iter().product();
    assert_eq!(
        old_numel, new_numel,
        "reshape: numel mismatch {} vs {}",
        old_numel, new_numel
    );

    let data = a.contiguous_data();
    let res = Tensor::new(data, new_shape.clone());

    if a.requires_grad() {
        let mut inner = res.0.write().unwrap();
        inner.requires_grad = true;
        inner.creator = Some(Arc::new(Node {
            op: Box::new(ReshapeOp { input_shape: old_shape }),
            inputs: vec![a.clone()],
        }));
    }

    res
}

// ============ Transpose (2D stride 交换) ============

#[derive(Debug)]
pub struct TransposeOp;

impl Op for TransposeOp {
    // 根据上游梯度计算当前节点对输入的梯度。
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        // Transpose the grad back
        let g = grad_output.contiguous_data();
        let shape = grad_output.shape();
        let (rows, cols) = (shape[0], shape[1]);
        let mut out = vec![0.0f32; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                out[j * rows + i] = g[i * cols + j];
            }
        }
        vec![Some(Tensor::new(out, vec![cols, rows]))]
    }
}
/// 转置二维矩阵最后两个维度。
pub fn transpose(a: &Tensor) -> Tensor {
    let shape = a.shape();
    assert_eq!(shape.len(), 2, "transpose: only 2D tensors supported");
    let (rows, cols) = (shape[0], shape[1]);
    let data = a.contiguous_data();

    let mut out = vec![0.0f32; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            out[j * rows + i] = data[i * cols + j];
        }
    }

    let res = Tensor::new(out, vec![cols, rows]);

    if a.requires_grad() {
        let mut inner = res.0.write().unwrap();
        inner.requires_grad = true;
        inner.creator = Some(Arc::new(Node {
            op: Box::new(TransposeOp),
            inputs: vec![a.clone()],
        }));
    }

    res
}

// ============ Softmax ============

#[derive(Debug)]
pub struct SoftmaxOp {
    output: Tensor,
}

impl Op for SoftmaxOp {
    // 根据上游梯度计算当前节点对输入的梯度。
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        // d softmax / d input_i = s_i * (delta_ij - s_j)
        // For each row: grad_input = s * (grad_output - sum(grad_output * s))
        let s = self.output.contiguous_data();
        let g = grad_output.contiguous_data();
        let shape = grad_output.shape();

        if shape.len() == 1 {
            let dot: f32 = s.iter().zip(g.iter()).map(|(si, gi)| si * gi).sum();
            let da: Vec<f32> = s.iter().zip(g.iter()).map(|(si, gi)| si * (gi - dot)).collect();
            vec![Some(Tensor::new(da, shape))]
        } else {
            // 2D: softmax along last axis
            let (rows, cols) = (shape[0], shape[1]);
            let mut da = vec![0.0f32; rows * cols];
            for r in 0..rows {
                let off = r * cols;
                let dot: f32 = (0..cols).map(|c| s[off + c] * g[off + c]).sum();
                for c in 0..cols {
                    da[off + c] = s[off + c] * (g[off + c] - dot);
                }
            }
            vec![Some(Tensor::new(da, shape))]
        }
    }
}
/// 按最后一维执行 softmax，输出同形状概率分布。
pub fn softmax(a: &Tensor) -> Tensor {
    let data = a.contiguous_data();
    let shape = a.shape();

    let out = if shape.len() == 1 {
        let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = data.iter().map(|x| (x - max_val).exp()).collect();
        let sum: f32 = exps.iter().sum();
        if sum == 0.0 {
            vec![1.0 / data.len() as f32; data.len()]
        } else {
            exps.iter().map(|e| e / sum).collect()
        }
    } else {
        // 2D: softmax along last axis
        let (rows, cols) = (shape[0], shape[1]);
        let mut out = vec![0.0f32; rows * cols];
        for r in 0..rows {
            let off = r * cols;
            let max_val = (0..cols).map(|c| data[off + c]).fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = (0..cols).map(|c| (data[off + c] - max_val).exp()).sum();
            if sum == 0.0 {
                let uniform = 1.0 / cols as f32;
                for c in 0..cols {
                    out[off + c] = uniform;
                }
            } else {
                for c in 0..cols {
                    out[off + c] = (data[off + c] - max_val).exp() / sum;
                }
            }
        }
        out
    };

    let res = Tensor::new(out, shape.clone());

    if a.requires_grad() {
        let mut inner = res.0.write().unwrap();
        inner.requires_grad = true;
        inner.creator = Some(Arc::new(Node {
            op: Box::new(SoftmaxOp { output: res.clone() }),
            inputs: vec![a.clone()],
        }));
    }

    res
}

// ============ LogSoftmax ============

#[derive(Debug)]
pub struct LogSoftmaxOp {
    softmax_output: Tensor,
}

impl Op for LogSoftmaxOp {
    // 根据上游梯度计算当前节点对输入的梯度。
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        // d log_softmax / d input = grad - softmax * sum(grad)
        let s = self.softmax_output.contiguous_data();
        let g = grad_output.contiguous_data();
        let shape = grad_output.shape();

        if shape.len() == 1 {
            let sum_g: f32 = g.iter().sum();
            let da: Vec<f32> = g.iter().zip(s.iter()).map(|(gi, si)| gi - si * sum_g).collect();
            vec![Some(Tensor::new(da, shape))]
        } else {
            let (rows, cols) = (shape[0], shape[1]);
            let mut da = vec![0.0f32; rows * cols];
            for r in 0..rows {
                let off = r * cols;
                let sum_g: f32 = (0..cols).map(|c| g[off + c]).sum();
                for c in 0..cols {
                    da[off + c] = g[off + c] - s[off + c] * sum_g;
                }
            }
            vec![Some(Tensor::new(da, shape))]
        }
    }
}
/// 按最后一维执行 log-softmax，数值上更稳定。
pub fn log_softmax(a: &Tensor) -> Tensor {
    let data = a.contiguous_data();
    let shape = a.shape();

    let (log_sm, sm) = if shape.len() == 1 {
        let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = data.iter().map(|x| (x - max_val).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let log_sum = sum.ln();
        let log_sm: Vec<f32> = data.iter().map(|x| x - max_val - log_sum).collect();
        let sm: Vec<f32> = exps.iter().map(|e| e / sum).collect();
        (log_sm, sm)
    } else {
        let (rows, cols) = (shape[0], shape[1]);
        let mut log_sm = vec![0.0f32; rows * cols];
        let mut sm = vec![0.0f32; rows * cols];
        for r in 0..rows {
            let off = r * cols;
            let max_val = (0..cols).map(|c| data[off + c]).fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = (0..cols).map(|c| (data[off + c] - max_val).exp()).sum();
            let log_sum = sum.ln();
            for c in 0..cols {
                log_sm[off + c] = data[off + c] - max_val - log_sum;
                sm[off + c] = (data[off + c] - max_val).exp() / sum;
            }
        }
        (log_sm, sm)
    };

    let sm_tensor = Tensor::new(sm, shape.clone());
    let res = Tensor::new(log_sm, shape.clone());

    if a.requires_grad() {
        let mut inner = res.0.write().unwrap();
        inner.requires_grad = true;
        inner.creator = Some(Arc::new(Node {
            op: Box::new(LogSoftmaxOp {
                softmax_output: sm_tensor,
            }),
            inputs: vec![a.clone()],
        }));
    }

    res
}

// ============ CrossEntropyLoss (融合 log_softmax + nll_loss) ============

#[derive(Debug)]
pub struct CrossEntropyLossOp {
    softmax_output: Tensor,
    targets: Vec<usize>,
    batch_size: usize,
}

impl Op for CrossEntropyLossOp {
    // 根据上游梯度计算当前节点对输入的梯度。
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        // d CE / d logits = softmax - one_hot(target)
        // Scaled by grad_output (scalar) and 1/batch_size
        let g_val = grad_output.contiguous_data()[0];
        let s = self.softmax_output.contiguous_data();
        let shape = self.softmax_output.shape();
        let n = self.batch_size;

        let mut da = s.clone();
        if shape.len() == 1 {
            // Single sample
            da[self.targets[0]] -= 1.0;
            for v in da.iter_mut() {
                *v *= g_val;
            }
        } else {
            let cols = shape[1];
            for (r, &t) in self.targets.iter().enumerate() {
                da[r * cols + t] -= 1.0;
            }
            let scale = g_val / n as f32;
            for v in da.iter_mut() {
                *v *= scale;
            }
        }

        vec![Some(Tensor::new(da, shape))]
    }
}
/// 计算交叉熵损失（内部基于 log-softmax 与目标索引）。
pub fn cross_entropy_loss(logits: &Tensor, targets: &[usize]) -> Tensor {
    let data = logits.contiguous_data();
    let shape = logits.shape();

    if shape.len() == 1 {
        let num_classes = shape[0];
        assert!(
            targets[0] < num_classes,
            "cross_entropy: target {} out of range {}",
            targets[0],
            num_classes
        );
        let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = data.iter().map(|x| (x - max_val).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let log_sum = sum.ln();
        let log_prob = data[targets[0]] - max_val - log_sum;
        let loss = -log_prob;

        let sm: Vec<f32> = exps.iter().map(|e| e / sum).collect();
        let sm_tensor = Tensor::new(sm, vec![num_classes]);
        let res = Tensor::new(vec![loss], vec![1]);

        if logits.requires_grad() {
            let mut inner = res.0.write().unwrap();
            inner.requires_grad = true;
            inner.creator = Some(Arc::new(Node {
                op: Box::new(CrossEntropyLossOp {
                    softmax_output: sm_tensor,
                    targets: targets.to_vec(),
                    batch_size: 1,
                }),
                inputs: vec![logits.clone()],
            }));
        }

        res
    } else {
        // Batched: [batch, num_classes]
        let (batch, num_classes) = (shape[0], shape[1]);
        assert_eq!(targets.len(), batch);
        for &t in targets {
            assert!(
                t < num_classes,
                "cross_entropy: target {} out of range {}",
                t,
                num_classes
            );
        }

        let mut total_loss = 0.0f32;
        let mut sm = vec![0.0f32; batch * num_classes];

        for r in 0..batch {
            let off = r * num_classes;
            let max_val = (0..num_classes)
                .map(|c| data[off + c])
                .fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = (0..num_classes).map(|c| (data[off + c] - max_val).exp()).sum();
            let log_sum = sum.ln();
            total_loss += -(data[off + targets[r]] - max_val - log_sum);
            for c in 0..num_classes {
                sm[off + c] = (data[off + c] - max_val).exp() / sum;
            }
        }

        let loss = total_loss / batch as f32;
        let sm_tensor = Tensor::new(sm, vec![batch, num_classes]);
        let res = Tensor::new(vec![loss], vec![1]);

        if logits.requires_grad() {
            let mut inner = res.0.write().unwrap();
            inner.requires_grad = true;
            inner.creator = Some(Arc::new(Node {
                op: Box::new(CrossEntropyLossOp {
                    softmax_output: sm_tensor,
                    targets: targets.to_vec(),
                    batch_size: batch,
                }),
                inputs: vec![logits.clone()],
            }));
        }

        res
    }
}

// ============ Embedding Lookup ============

#[derive(Debug)]
pub struct EmbeddingLookupOp {
    indices: Vec<usize>,
    num_embeddings: usize,
    embedding_dim: usize,
}

impl Op for EmbeddingLookupOp {
    // 根据上游梯度计算当前节点对输入的梯度。
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        // Scatter grad back to embedding table
        let g = grad_output.contiguous_data();
        let mut dw = vec![0.0f32; self.num_embeddings * self.embedding_dim];
        for (i, &idx) in self.indices.iter().enumerate() {
            for d in 0..self.embedding_dim {
                dw[idx * self.embedding_dim + d] += g[i * self.embedding_dim + d];
            }
        }
        vec![Some(Tensor::new(dw, vec![self.num_embeddings, self.embedding_dim]))]
    }
}
/// 按索引从 embedding 权重矩阵中抽取行向量。
pub fn embedding_lookup(weight: &Tensor, indices: &[usize]) -> Tensor {
    let w_shape = weight.shape();
    assert_eq!(w_shape.len(), 2, "embedding weight must be 2D");
    let (num_embeddings, embedding_dim) = (w_shape[0], w_shape[1]);
    let w_data = weight.contiguous_data();

    let seq_len = indices.len();
    let mut out = vec![0.0f32; seq_len * embedding_dim];
    for (i, &idx) in indices.iter().enumerate() {
        assert!(
            idx < num_embeddings,
            "embedding index {} out of range {}",
            idx,
            num_embeddings
        );
        out[i * embedding_dim..(i + 1) * embedding_dim]
            .copy_from_slice(&w_data[idx * embedding_dim..(idx + 1) * embedding_dim]);
    }

    let res = Tensor::new(out, vec![seq_len, embedding_dim]);

    if weight.requires_grad() {
        let mut inner = res.0.write().unwrap();
        inner.requires_grad = true;
        inner.creator = Some(Arc::new(Node {
            op: Box::new(EmbeddingLookupOp {
                indices: indices.to_vec(),
                num_embeddings,
                embedding_dim,
            }),
            inputs: vec![weight.clone()],
        }));
    }

    res
}

// ============ ReLU ============

#[derive(Debug)]
pub struct ReluOp {
    mask: Vec<bool>,
}

impl Op for ReluOp {
    // 根据上游梯度计算当前节点对输入的梯度。
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        let g = grad_output.contiguous_data();
        let da: Vec<f32> = g
            .iter()
            .zip(self.mask.iter())
            .map(|(gi, &m)| if m { *gi } else { 0.0 })
            .collect();
        vec![Some(Tensor::new(da, grad_output.shape()))]
    }
}
/// ReLU 激活：`max(x, 0)`。
pub fn relu(a: &Tensor) -> Tensor {
    let data = a.contiguous_data();
    let shape = a.shape();
    let device = a.device();
    let mask: Vec<bool> = data.iter().map(|x| *x > 0.0).collect();
    let out = dispatch_unary(
        &data,
        &device,
        |a| a.iter().map(|x| x.max(0.0)).collect(),
        |backend, a, o| backend.relu_f32(a, o),
    );
    let res = Tensor::new(out, shape);

    if a.requires_grad() {
        let mut inner = res.0.write().unwrap();
        inner.requires_grad = true;
        inner.creator = Some(Arc::new(Node {
            op: Box::new(ReluOp { mask }),
            inputs: vec![a.clone()],
        }));
    }

    res
}

// ============ GELU ============

#[derive(Debug)]
pub struct GeluOp {
    saved_input: Tensor,
}

impl Op for GeluOp {
    // 根据上游梯度计算当前节点对输入的梯度。
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        // GELU'(x) = 0.5*(1+tanh(k)) + 0.5*x*(1-tanh(k)^2)*k'
        // where k = sqrt(2/pi)*(x + 0.044715*x^3), k' = sqrt(2/pi)*(1 + 0.134145*x^2)
        let g = grad_output.contiguous_data();
        let x = self.saved_input.contiguous_data();
        let sqrt_2_pi: f32 = (2.0 / std::f32::consts::PI).sqrt();
        let da: Vec<f32> = g
            .iter()
            .zip(x.iter())
            .map(|(gi, &xi)| {
                let k = sqrt_2_pi * (xi + 0.044715 * xi.powi(3));
                let tanh_k = k.tanh();
                let dk = sqrt_2_pi * (1.0 + 0.134145 * xi * xi);
                let gelu_grad = 0.5 * (1.0 + tanh_k) + 0.5 * xi * (1.0 - tanh_k * tanh_k) * dk;
                gi * gelu_grad
            })
            .collect();
        vec![Some(Tensor::new(da, grad_output.shape()))]
    }
}
/// GELU 激活（近似公式）。
pub fn gelu(a: &Tensor) -> Tensor {
    let data = a.contiguous_data();
    let shape = a.shape();
    let device = a.device();
    let out = dispatch_unary(
        &data,
        &device,
        |a| {
            let sqrt_2_pi: f32 = (2.0 / std::f32::consts::PI).sqrt();
            a.iter()
                .map(|&x| 0.5 * x * (1.0 + (sqrt_2_pi * (x + 0.044715 * x.powi(3))).tanh()))
                .collect()
        },
        |backend, a, o| backend.gelu_f32(a, o),
    );
    let res = Tensor::new(out, shape);

    if a.requires_grad() {
        let mut inner = res.0.write().unwrap();
        inner.requires_grad = true;
        inner.creator = Some(Arc::new(Node {
            op: Box::new(GeluOp { saved_input: a.clone() }),
            inputs: vec![a.clone()],
        }));
    }

    res
}

// ============ Scale (标量乘) ============

#[derive(Debug)]
pub struct ScaleOp {
    scalar: f32,
}

impl Op for ScaleOp {
    // 根据上游梯度计算当前节点对输入的梯度。
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        let g = grad_output.contiguous_data();
        let da: Vec<f32> = g.iter().map(|gi| gi * self.scalar).collect();
        vec![Some(Tensor::new(da, grad_output.shape()))]
    }
}
/// 标量缩放：`a * scalar`。
pub fn scale(a: &Tensor, scalar: f32) -> Tensor {
    let data = a.contiguous_data();
    let shape = a.shape();
    let device = a.device();
    let out = if let Some(backend) = get_backend(&device) {
        let mut o = vec![0.0f32; data.len()];
        backend.scale_f32(&data, scalar, &mut o);
        o
    } else {
        data.iter().map(|x| x * scalar).collect()
    };
    let res = Tensor::new(out, shape);

    if a.requires_grad() {
        let mut inner = res.0.write().unwrap();
        inner.requires_grad = true;
        inner.creator = Some(Arc::new(Node {
            op: Box::new(ScaleOp { scalar }),
            inputs: vec![a.clone()],
        }));
    }

    res
}

// ============ Masked Fill ============

#[derive(Debug)]
pub struct MaskedFillOp {
    mask: Vec<bool>,
}

impl Op for MaskedFillOp {
    // 根据上游梯度计算当前节点对输入的梯度。
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        let g = grad_output.contiguous_data();
        let da: Vec<f32> = g
            .iter()
            .zip(self.mask.iter())
            .map(|(gi, &m)| if m { 0.0 } else { *gi })
            .collect();
        vec![Some(Tensor::new(da, grad_output.shape()))]
    }
}
/// 按布尔 mask 将指定位置填充为常数。
pub fn masked_fill(a: &Tensor, mask: &[bool], fill_value: f32) -> Tensor {
    let data = a.contiguous_data();
    let shape = a.shape();
    assert_eq!(data.len(), mask.len());
    let out: Vec<f32> = data
        .iter()
        .zip(mask.iter())
        .map(|(&x, &m)| if m { fill_value } else { x })
        .collect();
    let res = Tensor::new(out, shape);

    if a.requires_grad() {
        let mut inner = res.0.write().unwrap();
        inner.requires_grad = true;
        inner.creator = Some(Arc::new(Node {
            op: Box::new(MaskedFillOp { mask: mask.to_vec() }),
            inputs: vec![a.clone()],
        }));
    }

    res
}

// ============ Batch Matmul [B, M, K] @ [B, K, N] -> [B, M, N] ============

#[derive(Debug)]
pub struct BatchMatmulOp {
    saved_a: Tensor,
    saved_b: Tensor,
}

impl Op for BatchMatmulOp {
    // 根据上游梯度计算当前节点对输入的梯度。
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        let a_shape = self.saved_a.shape();
        let b_shape = self.saved_b.shape();
        let (batch, m, k) = (a_shape[0], a_shape[1], a_shape[2]);
        let n = b_shape[2];

        let g = grad_output.contiguous_data();
        let a = self.saved_a.contiguous_data();
        let b = self.saved_b.contiguous_data();

        let mut da = vec![0.0f32; batch * m * k];
        let mut db = vec![0.0f32; batch * k * n];

        for bi in 0..batch {
            let g_off = bi * m * n;
            let a_off = bi * m * k;
            let b_off = bi * k * n;

            // dA[bi] = dY[bi] @ B[bi]^T
            for i in 0..m {
                for j in 0..k {
                    let mut s = 0.0f32;
                    for p in 0..n {
                        s += g[g_off + i * n + p] * b[b_off + j * n + p];
                    }
                    da[a_off + i * k + j] = s;
                }
            }

            // dB[bi] = A[bi]^T @ dY[bi]
            for i in 0..k {
                for j in 0..n {
                    let mut s = 0.0f32;
                    for p in 0..m {
                        s += a[a_off + p * k + i] * g[g_off + p * n + j];
                    }
                    db[b_off + i * n + j] = s;
                }
            }
        }

        vec![
            Some(Tensor::new(da, vec![batch, m, k])),
            Some(Tensor::new(db, vec![batch, k, n])),
        ]
    }
}
/// 批量矩阵乘法：`[b,m,k] x [b,k,n] -> [b,m,n]`。
pub fn batch_matmul(a: &Tensor, b: &Tensor) -> Tensor {
    let a_shape = a.shape();
    let b_shape = b.shape();
    assert_eq!(a_shape.len(), 3);
    assert_eq!(b_shape.len(), 3);
    let (batch, m, k) = (a_shape[0], a_shape[1], a_shape[2]);
    assert_eq!(b_shape[0], batch);
    assert_eq!(b_shape[1], k);
    let n = b_shape[2];

    let a_data = a.contiguous_data();
    let b_data = b.contiguous_data();

    let mut out = vec![0.0f32; batch * m * n];
    for bi in 0..batch {
        let a_slice = &a_data[bi * m * k..(bi + 1) * m * k];
        let b_slice = &b_data[bi * k * n..(bi + 1) * k * n];
        let c_slice = dispatch_matmul(a_slice, b_slice, m, k, n);
        out[bi * m * n..(bi + 1) * m * n].copy_from_slice(&c_slice);
    }

    let res = Tensor::new(out, vec![batch, m, n]);

    if a.requires_grad() || b.requires_grad() {
        let mut inner = res.0.write().unwrap();
        inner.requires_grad = true;
        inner.creator = Some(Arc::new(Node {
            op: Box::new(BatchMatmulOp {
                saved_a: a.clone(),
                saved_b: b.clone(),
            }),
            inputs: vec![a.clone(), b.clone()],
        }));
    }

    res
}

// ============ Broadcast Add [B, M, N] + [1, 1, N] or [M, N] + [N] ============

#[derive(Debug)]
pub struct BroadcastAddOp {
    b_shape: Vec<usize>,
}

impl Op for BroadcastAddOp {
    // 根据上游梯度计算当前节点对输入的梯度。
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        let g = grad_output.contiguous_data();
        let g_shape = grad_output.shape();

        // dA = grad (same shape)
        let da = Tensor::new(g.clone(), g_shape.clone());

        // dB = sum over broadcast dims
        let b_numel: usize = self.b_shape.iter().product();
        let mut db_data = vec![0.0f32; b_numel];
        for i in 0..g.len() {
            db_data[i % b_numel] += g[i];
        }

        vec![Some(da), Some(Tensor::new(db_data, self.b_shape.clone()))]
    }
}
/// 将一维偏置向量广播加到二维张量行上。
pub fn broadcast_add(a: &Tensor, b: &Tensor) -> Tensor {
    let a_data = a.contiguous_data();
    let b_data = b.contiguous_data();
    let a_shape = a.shape();
    let b_shape = b.shape();
    let b_numel = b_data.len();

    assert_eq!(
        a_data.len() % b_numel,
        0,
        "broadcast_add: a.numel()={} not divisible by b.numel()={}",
        a_data.len(),
        b_numel
    );

    let out: Vec<f32> = a_data
        .iter()
        .enumerate()
        .map(|(i, &av)| av + b_data[i % b_numel])
        .collect();
    let res = Tensor::new(out, a_shape.clone());

    if a.requires_grad() || b.requires_grad() {
        let mut inner = res.0.write().unwrap();
        inner.requires_grad = true;
        inner.creator = Some(Arc::new(Node {
            op: Box::new(BroadcastAddOp { b_shape }),
            inputs: vec![a.clone(), b.clone()],
        }));
    }

    res
}

// ============ Concat along axis 0 ============

#[derive(Debug)]
pub struct ConcatOp {
    split_sizes: Vec<usize>, // numel of each input
    shapes: Vec<Vec<usize>>,
}

impl Op for ConcatOp {
    // 根据上游梯度计算当前节点对输入的梯度。
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        let g = grad_output.contiguous_data();
        let mut offset = 0;
        let mut grads = Vec::new();
        for (size, shape) in self.split_sizes.iter().zip(self.shapes.iter()) {
            grads.push(Some(Tensor::new(g[offset..offset + size].to_vec(), shape.clone())));
            offset += size;
        }
        grads
    }
}
/// 在第 0 维拼接多个形状兼容的张量。
pub fn concat(tensors: &[&Tensor]) -> Tensor {
    assert!(!tensors.is_empty());
    let first_shape = tensors[0].shape();
    let ndim = first_shape.len();

    let mut total_dim0 = 0;
    let mut all_data = Vec::new();
    let mut split_sizes = Vec::new();
    let mut shapes = Vec::new();

    for t in tensors {
        let s = t.shape();
        assert_eq!(s.len(), ndim);
        for d in 1..ndim {
            assert_eq!(s[d], first_shape[d]);
        }
        total_dim0 += s[0];
        let data = t.contiguous_data();
        split_sizes.push(data.len());
        shapes.push(s);
        all_data.extend(data);
    }

    let mut out_shape = first_shape;
    out_shape[0] = total_dim0;
    let res = Tensor::new(all_data, out_shape);

    let any_grad = tensors.iter().any(|t| t.requires_grad());
    if any_grad {
        let mut inner = res.0.write().unwrap();
        inner.requires_grad = true;
        inner.creator = Some(Arc::new(Node {
            op: Box::new(ConcatOp { split_sizes, shapes }),
            inputs: tensors.iter().map(|t| (*t).clone()).collect(),
        }));
    }

    res
}

// ============ 数值梯度检查 ============

pub fn numerical_grad_check<F>(f: F, inputs: &[&Tensor], eps: f32, tol: f32) -> bool
where
    F: Fn(&[Tensor]) -> Tensor,
{
    for (idx, input) in inputs.iter().enumerate() {
        let data = input.data();
        let shape = input.shape();

        for i in 0..data.len() {
            let mut data_plus = data.clone();
            data_plus[i] += eps;
            let mut data_minus = data.clone();
            data_minus[i] -= eps;

            let inputs_plus: Vec<Tensor> = inputs
                .iter()
                .enumerate()
                .map(|(j, t)| {
                    if j == idx {
                        Tensor::new(data_plus.clone(), shape.clone())
                    } else {
                        Tensor::new(t.data(), t.shape())
                    }
                })
                .collect();

            let inputs_minus: Vec<Tensor> = inputs
                .iter()
                .enumerate()
                .map(|(j, t)| {
                    if j == idx {
                        Tensor::new(data_minus.clone(), shape.clone())
                    } else {
                        Tensor::new(t.data(), t.shape())
                    }
                })
                .collect();

            let out_plus = f(&inputs_plus);
            let out_minus = f(&inputs_minus);

            let numerical = (out_plus.data()[0] - out_minus.data()[0]) / (2.0 * eps);

            let analytical_inputs: Vec<Tensor> = inputs
                .iter()
                .map(|t| Tensor::with_grad(t.data(), t.shape(), true))
                .collect();

            let out = f(&analytical_inputs);
            out.backward();

            let analytical = analytical_inputs[idx].grad().unwrap()[i];

            let diff = (numerical - analytical).abs();
            let scale = numerical.abs().max(analytical.abs()).max(1e-8);
            if diff / scale > tol {
                eprintln!(
                    "grad check failed: input[{}][{}] numerical={} analytical={} diff={}",
                    idx, i, numerical, analytical, diff
                );
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    // 前向测试锁定基础数学语义，防止后端优化改变可观察结果。
    #[test]
    fn test_add_forward() {
        let a = Tensor::new(vec![1.0, 2.0], vec![2]);
        let b = Tensor::new(vec![3.0, 4.0], vec![2]);
        assert_eq!(add(&a, &b).data(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_mul_forward() {
        let a = Tensor::new(vec![2.0, 3.0], vec![2]);
        let b = Tensor::new(vec![4.0, 5.0], vec![2]);
        assert_eq!(mul(&a, &b).data(), vec![8.0, 15.0]);
    }

    #[test]
    fn test_neg_forward() {
        let a = Tensor::new(vec![2.0, -3.0], vec![2]);
        assert_eq!(neg(&a).data(), vec![-2.0, 3.0]);
    }

    #[test]
    fn test_sub_forward() {
        let a = Tensor::new(vec![5.0, 3.0], vec![2]);
        let b = Tensor::new(vec![2.0, 1.0], vec![2]);
        assert_eq!(sub(&a, &b).data(), vec![3.0, 2.0]);
    }

    #[test]
    fn test_sum_forward() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(sum(&a).data(), vec![6.0]);
    }

    #[test]
    fn test_mean_forward() {
        let a = Tensor::new(vec![2.0, 4.0, 6.0], vec![3]);
        assert_eq!(mean(&a).data(), vec![4.0]);
    }

    #[test]
    fn test_matmul_forward() {
        // [2,3] @ [3,2] = [2,2]
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = Tensor::new(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]);
        assert_eq!(matmul(&a, &b).data(), vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_exp_forward() {
        let a = Tensor::new(vec![0.0, 1.0], vec![2]);
        let r = exp(&a).data();
        assert!((r[0] - 1.0).abs() < 1e-6);
        assert!((r[1] - std::f32::consts::E).abs() < 1e-5);
    }

    #[test]
    fn test_log_forward() {
        let a = Tensor::new(vec![1.0, std::f32::consts::E], vec![2]);
        let r = log(&a).data();
        assert!((r[0] - 0.0).abs() < 1e-6);
        assert!((r[1] - 1.0).abs() < 1e-5);
    }

    // 反向测试覆盖手写梯度公式，是 autograd 正确性的第一层保护。
    #[test]
    fn test_add_backward() {
        let x = Tensor::with_grad(vec![2.0], vec![1], true);
        let y = Tensor::with_grad(vec![3.0], vec![1], true);
        let z = add(&x, &y);
        z.backward();
        assert_eq!(x.grad().unwrap(), vec![1.0]);
        assert_eq!(y.grad().unwrap(), vec![1.0]);
    }

    #[test]
    fn test_mul_backward() {
        let x = Tensor::with_grad(vec![3.0], vec![1], true);
        let y = Tensor::with_grad(vec![4.0], vec![1], true);
        let z = mul(&x, &y);
        z.backward();
        assert_eq!(x.grad().unwrap(), vec![4.0]); // dz/dx = y
        assert_eq!(y.grad().unwrap(), vec![3.0]); // dz/dy = x
    }

    #[test]
    fn test_neg_backward() {
        let a = Tensor::with_grad(vec![2.0], vec![1], true);
        let b = neg(&a);
        b.backward();
        assert_eq!(a.grad().unwrap(), vec![-1.0]);
    }

    #[test]
    fn test_sub_backward() {
        let x = Tensor::with_grad(vec![5.0], vec![1], true);
        let y = Tensor::with_grad(vec![3.0], vec![1], true);
        let z = sub(&x, &y);
        z.backward();
        assert_eq!(x.grad().unwrap(), vec![1.0]);
        assert_eq!(y.grad().unwrap(), vec![-1.0]);
    }

    #[test]
    fn test_sum_backward() {
        let a = Tensor::with_grad(vec![1.0, 2.0, 3.0], vec![3], true);
        let s = sum(&a);
        s.backward();
        assert_eq!(a.grad().unwrap(), vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_mean_backward() {
        let a = Tensor::with_grad(vec![2.0, 4.0, 6.0], vec![3], true);
        let m = mean(&a);
        m.backward();
        let g = a.grad().unwrap();
        let expected = 1.0 / 3.0;
        for v in &g {
            assert!((v - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_matmul_backward() {
        // A=[1,2; 3,4] B=[5,6; 7,8]
        let a = Tensor::with_grad(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], true);
        let b = Tensor::with_grad(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2], true);
        let c = matmul(&a, &b);
        // c = [19, 22; 43, 50]
        // loss = sum(c) 需要用 sum 包一层才能 backward（因为 backward 只对标量）
        let loss = sum(&c);
        loss.backward();

        // dL/dA = ones @ B^T = [[1,1],[1,1]] @ [[5,7],[6,8]] = [[11,15],[11,15]]
        let da = a.grad().unwrap();
        assert_eq!(da, vec![11.0, 15.0, 11.0, 15.0]);

        // dL/dB = A^T @ ones = [[1,3],[2,4]] @ [[1,1],[1,1]] = [[4,4],[6,6]]
        let db = b.grad().unwrap();
        assert_eq!(db, vec![4.0, 4.0, 6.0, 6.0]);
    }

    // 数值梯度检查用有限差分对照解析梯度，专门捕捉公式写错但样例碰巧通过的问题。
    #[test]
    fn test_grad_check_add() {
        let a = Tensor::new(vec![2.0, 3.0], vec![2]);
        let b = Tensor::new(vec![4.0, 5.0], vec![2]);
        assert!(numerical_grad_check(
            |inputs| sum(&add(&inputs[0], &inputs[1])),
            &[&a, &b],
            1e-3,
            1e-2
        ));
    }

    #[test]
    fn test_grad_check_mul() {
        let a = Tensor::new(vec![2.0, 3.0], vec![2]);
        let b = Tensor::new(vec![4.0, 5.0], vec![2]);
        assert!(numerical_grad_check(
            |inputs| sum(&mul(&inputs[0], &inputs[1])),
            &[&a, &b],
            1e-3,
            1e-2
        ));
    }

    #[test]
    fn test_grad_check_matmul() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        assert!(numerical_grad_check(
            |inputs| sum(&matmul(&inputs[0], &inputs[1])),
            &[&a, &b],
            1e-3,
            1e-2
        ));
    }

    #[test]
    fn test_grad_check_exp() {
        let a = Tensor::new(vec![0.5, 1.0], vec![2]);
        assert!(numerical_grad_check(|inputs| sum(&exp(&inputs[0])), &[&a], 1e-3, 1e-2));
    }

    #[test]
    fn test_grad_check_log() {
        let a = Tensor::new(vec![1.0, 2.0], vec![2]);
        assert!(numerical_grad_check(|inputs| sum(&log(&inputs[0])), &[&a], 1e-3, 1e-2));
    }

    // 组合图测试验证梯度能跨多个算子连续传播。
    #[test]
    fn test_composite_mul_add() {
        // z = x*y + x => dz/dx = y+1, dz/dy = x
        let x = Tensor::with_grad(vec![3.0], vec![1], true);
        let y = Tensor::with_grad(vec![4.0], vec![1], true);
        let z = add(&mul(&x, &y), &x);
        z.backward();
        assert_eq!(x.grad().unwrap(), vec![5.0]); // y + 1 = 5
        assert_eq!(y.grad().unwrap(), vec![3.0]); // x = 3
    }

    #[test]
    fn test_composite_exp_log() {
        // log(exp(x)) = x => grad = 1
        let x = Tensor::with_grad(vec![2.0], vec![1], true);
        let z = log(&exp(&x));
        z.backward();
        let g = x.grad().unwrap()[0];
        assert!((g - 1.0).abs() < 1e-5);
    }

    // reshape 测试关注 shape 变化不应改变元素顺序和梯度归属。
    #[test]
    fn test_reshape_forward() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = reshape(&a, vec![3, 2]);
        assert_eq!(b.shape(), vec![3, 2]);
        assert_eq!(b.data(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_reshape_backward() {
        let a = Tensor::with_grad(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], true);
        let b = reshape(&a, vec![3, 2]);
        let loss = sum(&b);
        loss.backward();
        assert_eq!(a.grad().unwrap(), vec![1.0; 6]);
        assert_eq!(a.shape(), vec![2, 3]); // grad shape matches input
    }

    // transpose 测试固定二维转置的前向布局与反向还原语义。
    #[test]
    fn test_transpose_forward() {
        // [[1,2,3],[4,5,6]] -> [[1,4],[2,5],[3,6]]
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = transpose(&a);
        assert_eq!(b.shape(), vec![3, 2]);
        assert_eq!(b.data(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_transpose_backward() {
        let a = Tensor::with_grad(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], true);
        let b = transpose(&a);
        let loss = sum(&b);
        loss.backward();
        assert_eq!(a.grad().unwrap(), vec![1.0; 4]);
    }

    // softmax 测试同时约束概率归一化和雅可比反向实现。
    #[test]
    fn test_softmax_forward_1d() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let s = softmax(&a);
        let d = s.data();
        let total: f32 = d.iter().sum();
        assert!((total - 1.0).abs() < 1e-6);
        assert!(d[2] > d[1] && d[1] > d[0]);
    }

    #[test]
    fn test_softmax_forward_2d() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0], vec![2, 3]);
        let s = softmax(&a);
        let d = s.data();
        let row0_sum: f32 = d[0..3].iter().sum();
        let row1_sum: f32 = d[3..6].iter().sum();
        assert!((row0_sum - 1.0).abs() < 1e-6);
        assert!((row1_sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_backward() {
        let a = Tensor::with_grad(vec![1.0, 2.0, 3.0], vec![3], true);
        let s = softmax(&a);
        let loss = sum(&s);
        loss.backward();
        // sum(softmax) = 1 always, so grad should be ~0
        let g = a.grad().unwrap();
        for v in &g {
            assert!(v.abs() < 1e-6);
        }
    }

    // log_softmax 测试确保对数域输出和 softmax 语义一致。
    #[test]
    fn test_log_softmax_forward() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let ls = log_softmax(&a);
        let s = softmax(&a);
        let ls_data = ls.data();
        let s_data = s.data();
        for (l, s) in ls_data.iter().zip(s_data.iter()) {
            assert!((l - s.ln()).abs() < 1e-5);
        }
    }

    #[test]
    fn test_grad_check_softmax() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        assert!(numerical_grad_check(
            |inputs| sum(&softmax(&inputs[0])),
            &[&a],
            1e-3,
            1e-2
        ));
    }

    #[test]
    fn test_grad_check_log_softmax() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        assert!(numerical_grad_check(
            |inputs| sum(&log_softmax(&inputs[0])),
            &[&a],
            1e-3,
            1e-2
        ));
    }

    // cross entropy 测试锁定 logits 到 NLL loss 的组合语义。
    #[test]
    fn test_cross_entropy_forward_1d() {
        // Uniform logits => loss = ln(num_classes)
        let a = Tensor::new(vec![0.0, 0.0, 0.0], vec![3]);
        let loss = cross_entropy_loss(&a, &[0]);
        let l = loss.data()[0];
        assert!((l - (3.0f32).ln()).abs() < 1e-5);
    }

    #[test]
    fn test_cross_entropy_forward_2d() {
        let a = Tensor::new(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0], vec![2, 3]);
        let loss = cross_entropy_loss(&a, &[0, 1]);
        let l = loss.data()[0];
        assert!((l - (3.0f32).ln()).abs() < 1e-5);
    }

    #[test]
    fn test_cross_entropy_backward() {
        let a = Tensor::with_grad(vec![1.0, 2.0, 3.0], vec![3], true);
        let loss = cross_entropy_loss(&a, &[2]);
        loss.backward();
        let g = a.grad().unwrap();
        // grad = softmax - one_hot => sum should be ~0
        let sum_g: f32 = g.iter().sum();
        assert!(sum_g.abs() < 1e-6);
    }

    #[test]
    fn test_grad_check_cross_entropy() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        assert!(numerical_grad_check(
            |inputs| cross_entropy_loss(&inputs[0], &[0, 1]),
            &[&a],
            1e-3,
            1e-2
        ));
    }

    // embedding 测试重点覆盖重复索引时的梯度累加。
    #[test]
    fn test_embedding_forward() {
        // 4 embeddings of dim 3
        let w = Tensor::new(
            vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
            vec![4, 3],
        );
        let out = embedding_lookup(&w, &[0, 2, 3]);
        assert_eq!(out.shape(), vec![3, 3]);
        assert_eq!(out.data(), vec![0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2,]);
    }

    #[test]
    fn test_embedding_backward() {
        let w = Tensor::with_grad(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2], true);
        let out = embedding_lookup(&w, &[0, 2, 0]); // index 0 appears twice
        let loss = sum(&out);
        loss.backward();
        let g = w.grad().unwrap();
        // index 0: grad accumulated twice => [2.0, 2.0]
        // index 1: not used => [0.0, 0.0]
        // index 2: once => [1.0, 1.0]
        assert_eq!(g, vec![2.0, 2.0, 0.0, 0.0, 1.0, 1.0]);
    }

    // ReLU 测试固定 0 点及负区间的梯度截断语义。
    #[test]
    fn test_relu_forward() {
        let a = Tensor::new(vec![-1.0, 0.0, 1.0, 2.0], vec![4]);
        assert_eq!(relu(&a).data(), vec![0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_relu_backward() {
        let a = Tensor::with_grad(vec![-1.0, 0.0, 1.0, 2.0], vec![4], true);
        let b = relu(&a);
        let loss = sum(&b);
        loss.backward();
        assert_eq!(a.grad().unwrap(), vec![0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn test_grad_check_relu() {
        // Avoid 0 (non-differentiable point)
        let a = Tensor::new(vec![-1.0, 0.5, 1.0, 2.0], vec![4]);
        assert!(numerical_grad_check(|inputs| sum(&relu(&inputs[0])), &[&a], 1e-3, 1e-2));
    }

    // GELU 测试锁定 tanh 近似公式的前向与反向精度。
    #[test]
    fn test_gelu_forward() {
        let a = Tensor::new(vec![-1.0, 0.0, 1.0, 2.0], vec![4]);
        let d = gelu(&a).data();
        assert!((d[1] - 0.0).abs() < 1e-6); // gelu(0) = 0
        assert!(d[0] < 0.0); // gelu(-1) ~ -0.159
        assert!(d[2] > 0.8); // gelu(1) ~ 0.841
    }

    #[test]
    fn test_grad_check_gelu() {
        let a = Tensor::new(vec![-1.0, 0.5, 1.0, 2.0], vec![4]);
        assert!(numerical_grad_check(|inputs| sum(&gelu(&inputs[0])), &[&a], 1e-3, 1e-2));
    }

    // scale 测试服务于梯度累积和混合精度缩放路径。
    #[test]
    fn test_scale_forward() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(scale(&a, 2.0).data(), vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_scale_backward() {
        let a = Tensor::with_grad(vec![1.0, 2.0], vec![2], true);
        let b = scale(&a, 3.0);
        let loss = sum(&b);
        loss.backward();
        assert_eq!(a.grad().unwrap(), vec![3.0, 3.0]);
    }

    // masked_fill 测试约束 attention mask 常用的 true/false 写入语义。
    #[test]
    fn test_masked_fill_forward() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let mask = vec![false, true, false, true];
        let out = masked_fill(&a, &mask, f32::NEG_INFINITY);
        let d = out.data();
        assert_eq!(d[0], 1.0);
        assert_eq!(d[1], f32::NEG_INFINITY);
        assert_eq!(d[2], 3.0);
        assert_eq!(d[3], f32::NEG_INFINITY);
    }

    #[test]
    fn test_masked_fill_backward() {
        let a = Tensor::with_grad(vec![1.0, 2.0, 3.0, 4.0], vec![4], true);
        let mask = vec![false, true, false, true];
        let out = masked_fill(&a, &mask, 0.0);
        let loss = sum(&out);
        loss.backward();
        assert_eq!(a.grad().unwrap(), vec![1.0, 0.0, 1.0, 0.0]);
    }

    // batch matmul 测试覆盖 batch stride 切片和对应梯度。
    #[test]
    fn test_batch_matmul_forward() {
        // batch=1, [1,2,2] @ [1,2,2]
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![1, 2, 2]);
        let c = batch_matmul(&a, &b);
        assert_eq!(c.shape(), vec![1, 2, 2]);
        assert_eq!(c.data(), vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_batch_matmul_multi_batch() {
        // batch=2, each [2,2] @ [2,2]
        let a = Tensor::new(
            vec![
                1.0, 0.0, 0.0, 1.0, // identity
                2.0, 0.0, 0.0, 2.0, // 2*identity
            ],
            vec![2, 2, 2],
        );
        let b = Tensor::new(vec![3.0, 4.0, 5.0, 6.0, 3.0, 4.0, 5.0, 6.0], vec![2, 2, 2]);
        let c = batch_matmul(&a, &b);
        assert_eq!(c.shape(), vec![2, 2, 2]);
        let d = c.data();
        // batch 0: I @ B = B
        assert_eq!(&d[0..4], &[3.0, 4.0, 5.0, 6.0]);
        // batch 1: 2I @ B = 2B
        assert_eq!(&d[4..8], &[6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_grad_check_batch_matmul() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![1, 2, 2]);
        assert!(numerical_grad_check(
            |inputs| sum(&batch_matmul(&inputs[0], &inputs[1])),
            &[&a, &b],
            1e-3,
            1e-2
        ));
    }

    // broadcast_add 测试固定一维 bias 在 batch 维上的复用和梯度归约。
    #[test]
    fn test_broadcast_add_2d() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = Tensor::new(vec![10.0, 20.0, 30.0], vec![3]);
        let c = broadcast_add(&a, &b);
        assert_eq!(c.data(), vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
    }

    #[test]
    fn test_broadcast_add_backward() {
        let a = Tensor::with_grad(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], true);
        let b = Tensor::with_grad(vec![10.0, 20.0, 30.0], vec![3], true);
        let c = broadcast_add(&a, &b);
        let loss = sum(&c);
        loss.backward();
        assert_eq!(a.grad().unwrap(), vec![1.0; 6]);
        // b grad: summed over batch dim => [2.0, 2.0, 2.0]
        assert_eq!(b.grad().unwrap(), vec![2.0, 2.0, 2.0]);
    }

    // concat 测试确保拼接后的梯度能按原 shape 切回每个输入。
    #[test]
    fn test_concat_forward() {
        let a = Tensor::new(vec![1.0, 2.0], vec![1, 2]);
        let b = Tensor::new(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);
        let c = concat(&[&a, &b]);
        assert_eq!(c.shape(), vec![3, 2]);
        assert_eq!(c.data(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_concat_backward() {
        let a = Tensor::with_grad(vec![1.0, 2.0], vec![1, 2], true);
        let b = Tensor::with_grad(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2], true);
        let c = concat(&[&a, &b]);
        let loss = sum(&c);
        loss.backward();
        assert_eq!(a.grad().unwrap(), vec![1.0, 1.0]);
        assert_eq!(b.grad().unwrap(), vec![1.0, 1.0, 1.0, 1.0]);
    }

    // 后端分发测试证明自定义设备注册后会覆盖 CPU fallback。
    #[test]
    fn test_backend_dispatch_custom() {
        use sptorch_core_tensor::{register_backend, BackendDispatch};
        use std::sync::Arc;

        struct DoubleBackend;
        impl BackendDispatch for DoubleBackend {
            // 后端逐元素加法内核接口。
            fn add_f32(&self, a: &[f32], b: &[f32], out: &mut [f32]) {
                for i in 0..out.len() {
                    out[i] = (a[i] + b[i]) * 2.0;
                }
            }
            // 后端逐元素乘法内核接口。
            fn mul_f32(&self, a: &[f32], b: &[f32], out: &mut [f32]) {
                for i in 0..out.len() {
                    out[i] = a[i] * b[i] * 2.0;
                }
            }
            // 后端逐元素取负内核接口。
            fn neg_f32(&self, a: &[f32], out: &mut [f32]) {
                for i in 0..out.len() {
                    out[i] = -a[i] * 2.0;
                }
            }
            // 后端逐元素指数内核接口。
            fn exp_f32(&self, a: &[f32], out: &mut [f32]) {
                for i in 0..out.len() {
                    out[i] = a[i].exp();
                }
            }
            // 后端逐元素对数内核接口。
            fn log_f32(&self, a: &[f32], out: &mut [f32]) {
                for i in 0..out.len() {
                    out[i] = a[i].ln();
                }
            }
            // 后端 ReLU 激活内核接口。
            fn relu_f32(&self, a: &[f32], out: &mut [f32]) {
                for i in 0..out.len() {
                    out[i] = if a[i] > 0.0 { a[i] } else { 0.0 };
                }
            }
            // 后端 GELU 激活内核接口。
            fn gelu_f32(&self, a: &[f32], out: &mut [f32]) {
                for i in 0..out.len() {
                    out[i] = a[i];
                }
            }
            // 后端标量缩放内核接口。
            fn scale_f32(&self, a: &[f32], s: f32, out: &mut [f32]) {
                for i in 0..out.len() {
                    out[i] = a[i] * s;
                }
            }
            // 后端矩阵乘法内核接口。
            fn matmul_f32(&self, a: &[f32], b: &[f32], out: &mut [f32], m: usize, k: usize, n: usize) {
                for i in 0..m {
                    for j in 0..n {
                        let mut s = 0.0f32;
                        for p in 0..k {
                            s += a[i * k + p] * b[p * n + j];
                        }
                        out[i * n + j] = s;
                    }
                }
            }
        }

        let dev = Device::Custom(99);
        register_backend(dev, Arc::new(DoubleBackend));

        let a = Tensor::new(vec![1.0, 2.0], vec![2]);
        let b = Tensor::new(vec![3.0, 4.0], vec![2]);
        a.0.write().unwrap().device = dev;
        b.0.write().unwrap().device = dev;

        let c = add(&a, &b);
        assert_eq!(c.data(), vec![8.0, 12.0]);

        let d = mul(&a, &b);
        assert_eq!(d.data(), vec![6.0, 16.0]);

        let e = neg(&a);
        assert_eq!(e.data(), vec![-2.0, -4.0]);
    }
}

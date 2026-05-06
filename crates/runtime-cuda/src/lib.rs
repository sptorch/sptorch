use cudarc::cublas::{sys::cublasOperation_t, CudaBlas};
use cudarc::driver::*;
use std::sync::Arc;

#[derive(Debug)]
pub enum CudaError {
    Driver(DriverError),
    Cublas(String),
    Compile(String),
}

impl From<DriverError> for CudaError {
    // 中文注释：关键逻辑说明。
    fn from(e: DriverError) -> Self {
        CudaError::Driver(e)
    }
}

pub struct CudaBackend {
    pub dev: Arc<CudaDevice>,
    pub blas: Arc<CudaBlas>,
}

impl CudaBackend {
    /// `new`：中文注释，说明函数用途、输入约束与输出语义。
    pub fn new(ordinal: usize) -> Result<Self, CudaError> {
        let dev = CudaDevice::new(ordinal)?;
        let blas = Arc::new(CudaBlas::new(dev.clone()).map_err(|e| CudaError::Cublas(e.to_string()))?);
        Ok(CudaBackend { dev, blas })
    }
}

// ============ GPU Buffer wrapper ============

pub struct GpuTensor {
    pub data: CudaSlice<f32>,
    pub shape: Vec<usize>,
    pub numel: usize,
}

impl GpuTensor {
    /// `from_host`：中文注释，说明函数用途、输入约束与输出语义。
    pub fn from_host(backend: &CudaBackend, data: &[f32], shape: Vec<usize>) -> Result<Self, CudaError> {
        let numel = data.len();
        let gpu_data = backend.dev.htod_sync_copy(data)?;
        Ok(GpuTensor {
            data: gpu_data,
            shape,
            numel,
        })
    }
    /// `to_host`：中文注释，说明函数用途、输入约束与输出语义。
    pub fn to_host(&self, backend: &CudaBackend) -> Result<Vec<f32>, CudaError> {
        let host = backend.dev.dtoh_sync_copy(&self.data)?;
        Ok(host)
    }
    /// `zeros`：中文注释，说明函数用途、输入约束与输出语义。
    pub fn zeros(backend: &CudaBackend, shape: Vec<usize>) -> Result<Self, CudaError> {
        let numel: usize = shape.iter().product();
        let gpu_data = backend.dev.alloc_zeros::<f32>(numel)?;
        Ok(GpuTensor {
            data: gpu_data,
            shape,
            numel,
        })
    }
}

// ============ CUDA C Kernels ============

const KERNEL_SRC: &str = r#"
extern "C" __global__ void add_kernel(const float* a, const float* b, float* c, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

extern "C" __global__ void mul_kernel(const float* a, const float* b, float* c, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] * b[idx];
}

extern "C" __global__ void neg_kernel(const float* a, float* b, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) b[idx] = -a[idx];
}

extern "C" __global__ void scale_kernel(const float* a, float* b, float scalar, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) b[idx] = a[idx] * scalar;
}

extern "C" __global__ void exp_kernel(const float* a, float* b, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) b[idx] = expf(a[idx]);
}

extern "C" __global__ void log_kernel(const float* a, float* b, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) b[idx] = logf(a[idx]);
}

extern "C" __global__ void gelu_kernel(const float* a, float* b, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = a[idx];
        float k = 0.7978845608f * (x + 0.044715f * x * x * x);
        b[idx] = 0.5f * x * (1.0f + tanhf(k));
    }
}

extern "C" __global__ void relu_kernel(const float* a, float* b, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) b[idx] = a[idx] > 0.0f ? a[idx] : 0.0f;
}

// SGD update: param[i] -= lr * grad[i]
extern "C" __global__ void sgd_kernel(float* param, const float* grad, float lr, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) param[idx] -= lr * grad[idx];
}

// Accumulate: a[i] += b[i]
extern "C" __global__ void accum_kernel(float* a, const float* b, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) a[idx] += b[idx];
}
"#;

const KERNEL_NAMES: &[&str] = &[
    "add_kernel",
    "mul_kernel",
    "neg_kernel",
    "scale_kernel",
    "exp_kernel",
    "log_kernel",
    "gelu_kernel",
    "relu_kernel",
    "sgd_kernel",
    "accum_kernel",
];

// ============ Kernel Operations ============

fn launch_cfg(n: usize) -> LaunchConfig {
    let block = 256;
    let grid = (n + block - 1) / block;
    LaunchConfig {
        grid_dim: (grid as u32, 1, 1),
        block_dim: (block as u32, 1, 1),
        shared_mem_bytes: 0,
    }
}

impl CudaBackend {
    /// `load_kernels`：中文注释，说明函数用途、输入约束与输出语义。
    pub fn load_kernels(&self) -> Result<(), CudaError> {
        if self.dev.has_func("elem", "add_kernel") {
            return Ok(());
        }
        let ptx = cudarc::nvrtc::compile_ptx(KERNEL_SRC).map_err(|e| CudaError::Compile(e.to_string()))?;
        self.dev.load_ptx(ptx, "elem", KERNEL_NAMES)?;
        Ok(())
    }
    /// `gpu_add`：中文注释，说明函数用途、输入约束与输出语义。
    pub fn gpu_add(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor, CudaError> {
        assert_eq!(a.numel, b.numel);
        let mut out = GpuTensor::zeros(self, a.shape.clone())?;
        let f = self.dev.get_func("elem", "add_kernel").unwrap();
        let cfg = launch_cfg(a.numel);
        unsafe { f.launch(cfg, (&a.data, &b.data, &mut out.data, a.numel as u32)) }?;
        Ok(out)
    }
    /// `gpu_mul`：中文注释，说明函数用途、输入约束与输出语义。
    pub fn gpu_mul(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor, CudaError> {
        assert_eq!(a.numel, b.numel);
        let mut out = GpuTensor::zeros(self, a.shape.clone())?;
        let f = self.dev.get_func("elem", "mul_kernel").unwrap();
        let cfg = launch_cfg(a.numel);
        unsafe { f.launch(cfg, (&a.data, &b.data, &mut out.data, a.numel as u32)) }?;
        Ok(out)
    }
    /// `gpu_neg`：中文注释，说明函数用途、输入约束与输出语义。
    pub fn gpu_neg(&self, a: &GpuTensor) -> Result<GpuTensor, CudaError> {
        let mut out = GpuTensor::zeros(self, a.shape.clone())?;
        let f = self.dev.get_func("elem", "neg_kernel").unwrap();
        let cfg = launch_cfg(a.numel);
        unsafe { f.launch(cfg, (&a.data, &mut out.data, a.numel as u32)) }?;
        Ok(out)
    }
    /// `gpu_scale`：中文注释，说明函数用途、输入约束与输出语义。
    pub fn gpu_scale(&self, a: &GpuTensor, scalar: f32) -> Result<GpuTensor, CudaError> {
        let mut out = GpuTensor::zeros(self, a.shape.clone())?;
        let f = self.dev.get_func("elem", "scale_kernel").unwrap();
        let cfg = launch_cfg(a.numel);
        unsafe { f.launch(cfg, (&a.data, &mut out.data, scalar, a.numel as u32)) }?;
        Ok(out)
    }
    /// `gpu_exp`：中文注释，说明函数用途、输入约束与输出语义。
    pub fn gpu_exp(&self, a: &GpuTensor) -> Result<GpuTensor, CudaError> {
        let mut out = GpuTensor::zeros(self, a.shape.clone())?;
        let f = self.dev.get_func("elem", "exp_kernel").unwrap();
        let cfg = launch_cfg(a.numel);
        unsafe { f.launch(cfg, (&a.data, &mut out.data, a.numel as u32)) }?;
        Ok(out)
    }
    /// `gpu_log`：中文注释，说明函数用途、输入约束与输出语义。
    pub fn gpu_log(&self, a: &GpuTensor) -> Result<GpuTensor, CudaError> {
        let mut out = GpuTensor::zeros(self, a.shape.clone())?;
        let f = self.dev.get_func("elem", "log_kernel").unwrap();
        let cfg = launch_cfg(a.numel);
        unsafe { f.launch(cfg, (&a.data, &mut out.data, a.numel as u32)) }?;
        Ok(out)
    }
    /// `gpu_gelu`：中文注释，说明函数用途、输入约束与输出语义。
    pub fn gpu_gelu(&self, a: &GpuTensor) -> Result<GpuTensor, CudaError> {
        let mut out = GpuTensor::zeros(self, a.shape.clone())?;
        let f = self.dev.get_func("elem", "gelu_kernel").unwrap();
        let cfg = launch_cfg(a.numel);
        unsafe { f.launch(cfg, (&a.data, &mut out.data, a.numel as u32)) }?;
        Ok(out)
    }
    /// `gpu_relu`：中文注释，说明函数用途、输入约束与输出语义。
    pub fn gpu_relu(&self, a: &GpuTensor) -> Result<GpuTensor, CudaError> {
        let mut out = GpuTensor::zeros(self, a.shape.clone())?;
        let f = self.dev.get_func("elem", "relu_kernel").unwrap();
        let cfg = launch_cfg(a.numel);
        unsafe { f.launch(cfg, (&a.data, &mut out.data, a.numel as u32)) }?;
        Ok(out)
    }
    /// `gpu_matmul`：中文注释，说明函数用途、输入约束与输出语义。
    pub fn gpu_matmul(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor, CudaError> {
        let (m, k) = (a.shape[0], a.shape[1]);
        let n = b.shape[1];
        assert_eq!(a.shape[1], b.shape[0]);

        let mut out = GpuTensor::zeros(self, vec![m, n])?;

        // cuBLAS is column-major. For row-major A@B:
        // treat as col-major B^T @ A^T = (A@B)^T, read back as row-major = A@B
        unsafe {
            cudarc::cublas::result::sgemm(
                *self.blas.handle(),
                cublasOperation_t::CUBLAS_OP_N,
                cublasOperation_t::CUBLAS_OP_N,
                n as i32,
                m as i32,
                k as i32,
                &1.0f32 as *const f32,
                *b.data.device_ptr() as *const f32,
                n as i32,
                *a.data.device_ptr() as *const f32,
                k as i32,
                &0.0f32 as *const f32,
                *out.data.device_ptr_mut() as *mut f32,
                n as i32,
            )
        }
        .map_err(|e| CudaError::Cublas(e.to_string()))?;

        Ok(out)
    }
    /// `gpu_sgd_update`：中文注释，说明函数用途、输入约束与输出语义。
    pub fn gpu_sgd_update(&self, param: &mut GpuTensor, grad: &GpuTensor, lr: f32) -> Result<(), CudaError> {
        assert_eq!(param.numel, grad.numel);
        let f = self.dev.get_func("elem", "sgd_kernel").unwrap();
        let cfg = launch_cfg(param.numel);
        unsafe { f.launch(cfg, (&mut param.data, &grad.data, lr, param.numel as u32)) }?;
        Ok(())
    }
    /// `gpu_accum`：中文注释，说明函数用途、输入约束与输出语义。
    pub fn gpu_accum(&self, a: &mut GpuTensor, b: &GpuTensor) -> Result<(), CudaError> {
        assert_eq!(a.numel, b.numel);
        let f = self.dev.get_func("elem", "accum_kernel").unwrap();
        let cfg = launch_cfg(a.numel);
        unsafe { f.launch(cfg, (&mut a.data, &b.data, a.numel as u32)) }?;
        Ok(())
    }

    // ============ 复合操作 (host-side orchestration + GPU kernels) ============
    /// `gpu_transpose`：中文注释，说明函数用途、输入约束与输出语义。
    pub fn gpu_transpose(&self, a: &GpuTensor) -> Result<GpuTensor, CudaError> {
        assert_eq!(a.shape.len(), 2);
        let (m, n) = (a.shape[0], a.shape[1]);
        let host = a.to_host(self)?;
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                out[j * m + i] = host[i * n + j];
            }
        }
        GpuTensor::from_host(self, &out, vec![n, m])
    }
    /// `gpu_broadcast_add`：中文注释，说明函数用途、输入约束与输出语义。
    pub fn gpu_broadcast_add(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor, CudaError> {
        let a_host = a.to_host(self)?;
        let b_host = b.to_host(self)?;
        let b_numel = b_host.len();
        let out: Vec<f32> = a_host
            .iter()
            .enumerate()
            .map(|(i, &v)| v + b_host[i % b_numel])
            .collect();
        GpuTensor::from_host(self, &out, a.shape.clone())
    }
    /// `gpu_softmax`：中文注释，说明函数用途、输入约束与输出语义。
    pub fn gpu_softmax(&self, a: &GpuTensor) -> Result<GpuTensor, CudaError> {
        let data = a.to_host(self)?;
        let shape = &a.shape;
        let out = if shape.len() == 1 {
            let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = data.iter().map(|x| (x - max_val).exp()).collect();
            let sum: f32 = exps.iter().sum();
            exps.iter().map(|e| e / sum).collect()
        } else {
            let (rows, cols) = (shape[0], shape[1]);
            let mut out = vec![0.0f32; rows * cols];
            for r in 0..rows {
                let off = r * cols;
                let max_val = (0..cols).map(|c| data[off + c]).fold(f32::NEG_INFINITY, f32::max);
                let sum: f32 = (0..cols).map(|c| (data[off + c] - max_val).exp()).sum();
                for c in 0..cols {
                    out[off + c] = (data[off + c] - max_val).exp() / sum;
                }
            }
            out
        };
        GpuTensor::from_host(self, &out, a.shape.clone())
    }
    /// `gpu_cross_entropy`：中文注释，说明函数用途、输入约束与输出语义。
    pub fn gpu_cross_entropy(&self, logits: &GpuTensor, targets: &[usize]) -> Result<(f32, GpuTensor), CudaError> {
        let data = logits.to_host(self)?;
        let shape = &logits.shape;
        let (seq_len, vocab) = (shape[0], shape[1]);
        assert_eq!(targets.len(), seq_len);

        let mut total_loss = 0.0f32;
        let mut sm = vec![0.0f32; seq_len * vocab];
        for r in 0..seq_len {
            let off = r * vocab;
            let max_val = (0..vocab).map(|c| data[off + c]).fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = (0..vocab).map(|c| (data[off + c] - max_val).exp()).sum();
            let log_sum = sum.ln();
            total_loss += -(data[off + targets[r]] - max_val - log_sum);
            for c in 0..vocab {
                sm[off + c] = (data[off + c] - max_val).exp() / sum;
            }
        }
        let loss = total_loss / seq_len as f32;
        let sm_gpu = GpuTensor::from_host(self, &sm, vec![seq_len, vocab])?;
        Ok((loss, sm_gpu))
    }
    /// `gpu_cross_entropy_backward`：中文注释，说明函数用途、输入约束与输出语义。
    pub fn gpu_cross_entropy_backward(&self, sm: &GpuTensor, targets: &[usize]) -> Result<GpuTensor, CudaError> {
        let mut grad = sm.to_host(self)?;
        let (seq_len, vocab) = (sm.shape[0], sm.shape[1]);
        for (r, &t) in targets.iter().enumerate() {
            grad[r * vocab + t] -= 1.0;
        }
        let scale = 1.0 / seq_len as f32;
        for v in grad.iter_mut() {
            *v *= scale;
        }
        GpuTensor::from_host(self, &grad, sm.shape.clone())
    }
    /// `gpu_embedding`：中文注释，说明函数用途、输入约束与输出语义。
    pub fn gpu_embedding(&self, weight: &GpuTensor, indices: &[usize]) -> Result<GpuTensor, CudaError> {
        let w = weight.to_host(self)?;
        let dim = weight.shape[1];
        let seq_len = indices.len();
        let mut out = vec![0.0f32; seq_len * dim];
        for (i, &idx) in indices.iter().enumerate() {
            out[i * dim..(i + 1) * dim].copy_from_slice(&w[idx * dim..(idx + 1) * dim]);
        }
        GpuTensor::from_host(self, &out, vec![seq_len, dim])
    }
    /// `gpu_embedding_backward`：中文注释，说明函数用途、输入约束与输出语义。
    pub fn gpu_embedding_backward(
        &self,
        grad: &GpuTensor,
        indices: &[usize],
        num_emb: usize,
        dim: usize,
    ) -> Result<GpuTensor, CudaError> {
        let g = grad.to_host(self)?;
        let mut dw = vec![0.0f32; num_emb * dim];
        for (i, &idx) in indices.iter().enumerate() {
            for d in 0..dim {
                dw[idx * dim + d] += g[i * dim + d];
            }
        }
        GpuTensor::from_host(self, &dw, vec![num_emb, dim])
    }
    /// `gpu_masked_fill`：中文注释，说明函数用途、输入约束与输出语义。
    pub fn gpu_masked_fill(&self, a: &GpuTensor, mask: &[bool], fill_value: f32) -> Result<GpuTensor, CudaError> {
        let mut data = a.to_host(self)?;
        for (i, &m) in mask.iter().enumerate() {
            if m {
                data[i] = fill_value;
            }
        }
        GpuTensor::from_host(self, &data, a.shape.clone())
    }
    /// `gpu_reshape`：中文注释，说明函数用途、输入约束与输出语义。
    pub fn gpu_reshape(&self, a: &GpuTensor, new_shape: Vec<usize>) -> Result<GpuTensor, CudaError> {
        let new_numel: usize = new_shape.iter().product();
        assert_eq!(a.numel, new_numel);
        // Clone the GPU data slice by copying through host (safe but not optimal)
        let host = a.to_host(self)?;
        GpuTensor::from_host(self, &host, new_shape)
    }
    /// `gpu_sum`：中文注释，说明函数用途、输入约束与输出语义。
    pub fn gpu_sum(&self, a: &GpuTensor) -> Result<f32, CudaError> {
        let host = a.to_host(self)?;
        Ok(host.iter().sum())
    }
    /// `gpu_layer_norm`：中文注释，说明函数用途、输入约束与输出语义。
    pub fn gpu_layer_norm(
        &self,
        input: &GpuTensor,
        gamma: &GpuTensor,
        beta: &GpuTensor,
        eps: f32,
    ) -> Result<GpuTensor, CudaError> {
        let data = input.to_host(self)?;
        let g = gamma.to_host(self)?;
        let b = beta.to_host(self)?;
        let dim = g.len();
        let leading = data.len() / dim;
        let mut out = vec![0.0f32; data.len()];
        for batch in 0..leading {
            let off = batch * dim;
            let mean: f32 = (0..dim).map(|i| data[off + i]).sum::<f32>() / dim as f32;
            let var: f32 = (0..dim).map(|i| (data[off + i] - mean).powi(2)).sum::<f32>() / dim as f32;
            let inv_std = 1.0 / (var + eps).sqrt();
            for i in 0..dim {
                out[off + i] = g[i] * (data[off + i] - mean) * inv_std + b[i];
            }
        }
        GpuTensor::from_host(self, &out, input.shape.clone())
    }
    /// `gpu_batch_matmul`：中文注释，说明函数用途、输入约束与输出语义。
    pub fn gpu_batch_matmul(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor, CudaError> {
        let (batch, m, k) = (a.shape[0], a.shape[1], a.shape[2]);
        let n = b.shape[2];
        // Use per-batch sgemm calls
        let a_host = a.to_host(self)?;
        let b_host = b.to_host(self)?;
        let mut out_host = vec![0.0f32; batch * m * n];
        for bi in 0..batch {
            let a_slice = &a_host[bi * m * k..(bi + 1) * m * k];
            let b_slice = &b_host[bi * k * n..(bi + 1) * k * n];
            let ga = GpuTensor::from_host(self, a_slice, vec![m, k])?;
            let gb = GpuTensor::from_host(self, b_slice, vec![k, n])?;
            let gc = self.gpu_matmul(&ga, &gb)?;
            let c_host = gc.to_host(self)?;
            out_host[bi * m * n..(bi + 1) * m * n].copy_from_slice(&c_host);
        }
        GpuTensor::from_host(self, &out_host, vec![batch, m, n])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // 中文注释：关键逻辑说明。
    fn setup() -> CudaBackend {
        let backend = CudaBackend::new(0).expect("Failed to create CUDA backend");
        backend.load_kernels().expect("Failed to load kernels");
        backend
    }

    // 中文注释：关键逻辑说明。
    #[test]
    fn test_host_roundtrip() {
        let b = setup();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let gpu = GpuTensor::from_host(&b, &data, vec![5]).unwrap();
        let host = gpu.to_host(&b).unwrap();
        assert_eq!(host, data);
    }

    // 中文注释：关键逻辑说明。
    #[test]
    fn test_gpu_add() {
        let b = setup();
        let a = GpuTensor::from_host(&b, &[1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let c = GpuTensor::from_host(&b, &[5.0, 6.0, 7.0, 8.0], vec![4]).unwrap();
        let out = b.gpu_add(&a, &c).unwrap();
        assert_eq!(out.to_host(&b).unwrap(), vec![6.0, 8.0, 10.0, 12.0]);
    }

    // 中文注释：关键逻辑说明。
    #[test]
    fn test_gpu_mul() {
        let b = setup();
        let a = GpuTensor::from_host(&b, &[2.0, 3.0, 4.0, 5.0], vec![4]).unwrap();
        let c = GpuTensor::from_host(&b, &[3.0, 4.0, 5.0, 6.0], vec![4]).unwrap();
        let out = b.gpu_mul(&a, &c).unwrap();
        assert_eq!(out.to_host(&b).unwrap(), vec![6.0, 12.0, 20.0, 30.0]);
    }

    // 中文注释：关键逻辑说明。
    #[test]
    fn test_gpu_neg() {
        let b = setup();
        let a = GpuTensor::from_host(&b, &[1.0, -2.0, 3.0], vec![3]).unwrap();
        let out = b.gpu_neg(&a).unwrap();
        assert_eq!(out.to_host(&b).unwrap(), vec![-1.0, 2.0, -3.0]);
    }

    // 中文注释：关键逻辑说明。
    #[test]
    fn test_gpu_scale() {
        let b = setup();
        let a = GpuTensor::from_host(&b, &[1.0, 2.0, 3.0], vec![3]).unwrap();
        let out = b.gpu_scale(&a, 2.5).unwrap();
        assert_eq!(out.to_host(&b).unwrap(), vec![2.5, 5.0, 7.5]);
    }

    // 中文注释：关键逻辑说明。
    #[test]
    fn test_gpu_exp() {
        let b = setup();
        let a = GpuTensor::from_host(&b, &[0.0, 1.0], vec![2]).unwrap();
        let out = b.gpu_exp(&a).unwrap();
        let r = out.to_host(&b).unwrap();
        assert!((r[0] - 1.0).abs() < 1e-6);
        assert!((r[1] - std::f32::consts::E).abs() < 1e-5);
    }

    // 中文注释：关键逻辑说明。
    #[test]
    fn test_gpu_log() {
        let b = setup();
        let a = GpuTensor::from_host(&b, &[1.0, std::f32::consts::E], vec![2]).unwrap();
        let out = b.gpu_log(&a).unwrap();
        let r = out.to_host(&b).unwrap();
        assert!((r[0] - 0.0).abs() < 1e-6);
        assert!((r[1] - 1.0).abs() < 1e-5);
    }

    // 中文注释：关键逻辑说明。
    #[test]
    fn test_gpu_gelu() {
        let b = setup();
        let a = GpuTensor::from_host(&b, &[-1.0, 0.0, 1.0, 2.0], vec![4]).unwrap();
        let out = b.gpu_gelu(&a).unwrap();
        let r = out.to_host(&b).unwrap();
        assert!((r[1] - 0.0).abs() < 1e-6); // gelu(0) = 0
        assert!(r[0] < 0.0); // gelu(-1) < 0
        assert!(r[2] > 0.8); // gelu(1) ~ 0.841
    }

    // 中文注释：关键逻辑说明。
    #[test]
    fn test_gpu_relu() {
        let b = setup();
        let a = GpuTensor::from_host(&b, &[-1.0, 0.0, 1.0, 2.0], vec![4]).unwrap();
        let out = b.gpu_relu(&a).unwrap();
        assert_eq!(out.to_host(&b).unwrap(), vec![0.0, 0.0, 1.0, 2.0]);
    }

    // 中文注释：关键逻辑说明。
    #[test]
    fn test_gpu_matmul() {
        let b = setup();
        let a = GpuTensor::from_host(&b, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let c = GpuTensor::from_host(&b, &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]).unwrap();
        let out = b.gpu_matmul(&a, &c).unwrap();
        assert_eq!(out.to_host(&b).unwrap(), vec![58.0, 64.0, 139.0, 154.0]);
    }

    // 中文注释：关键逻辑说明。
    #[test]
    fn test_gpu_matmul_square() {
        let b = setup();
        let a = GpuTensor::from_host(&b, &[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let c = GpuTensor::from_host(&b, &[5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
        let out = b.gpu_matmul(&a, &c).unwrap();
        assert_eq!(out.to_host(&b).unwrap(), vec![19.0, 22.0, 43.0, 50.0]);
    }

    // 中文注释：关键逻辑说明。
    #[test]
    fn test_gpu_vs_cpu_matmul_large() {
        let b = setup();
        let m = 64;
        let k = 48;
        let n = 32;
        let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.01).collect();

        let ga = GpuTensor::from_host(&b, &a_data, vec![m, k]).unwrap();
        let gb = GpuTensor::from_host(&b, &b_data, vec![k, n]).unwrap();
        let gpu_out = b.gpu_matmul(&ga, &gb).unwrap().to_host(&b).unwrap();

        // CPU reference
        let mut cpu_out = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                for p in 0..k {
                    cpu_out[i * n + j] += a_data[i * k + p] * b_data[p * n + j];
                }
            }
        }

        for i in 0..m * n {
            assert!(
                (gpu_out[i] - cpu_out[i]).abs() < 0.1,
                "mismatch at {}: gpu={} cpu={}",
                i,
                gpu_out[i],
                cpu_out[i]
            );
        }
    }
}

// ============ BackendDispatch implementation ============

use sptorch_core_tensor::{register_backend, BackendDispatch, Device};
use std::sync::OnceLock;

static CUDA_BACKEND: OnceLock<Option<CudaBackend>> = OnceLock::new();

// 中文注释：关键逻辑说明。
fn get_or_init_cuda() -> Option<&'static CudaBackend> {
    CUDA_BACKEND
        .get_or_init(|| match CudaBackend::new(0) {
            Ok(b) => {
                if b.load_kernels().is_ok() {
                    Some(b)
                } else {
                    None
                }
            }
            Err(_) => None,
        })
        .as_ref()
}
/// `register_cuda_backend`：中文注释，说明函数用途、输入约束与输出语义。
pub fn register_cuda_backend() {
    if get_or_init_cuda().is_some() {
        let dispatch = Arc::new(CudaDispatch);
        register_backend(Device::Cuda(0), dispatch);
        eprintln!("[sptorch] CUDA backend registered for autograd dispatch");
    }
}

struct CudaDispatch;

impl BackendDispatch for CudaDispatch {
    // 中文注释：关键逻辑说明。
    fn add_f32(&self, a: &[f32], b: &[f32], out: &mut [f32]) {
        let backend = get_or_init_cuda().unwrap();
        let ga = GpuTensor::from_host(backend, a, vec![a.len()]).unwrap();
        let gb = GpuTensor::from_host(backend, b, vec![b.len()]).unwrap();
        let gc = backend.gpu_add(&ga, &gb).unwrap();
        out.copy_from_slice(&gc.to_host(backend).unwrap());
    }

    // 中文注释：关键逻辑说明。
    fn mul_f32(&self, a: &[f32], b: &[f32], out: &mut [f32]) {
        let backend = get_or_init_cuda().unwrap();
        let ga = GpuTensor::from_host(backend, a, vec![a.len()]).unwrap();
        let gb = GpuTensor::from_host(backend, b, vec![b.len()]).unwrap();
        let gc = backend.gpu_mul(&ga, &gb).unwrap();
        out.copy_from_slice(&gc.to_host(backend).unwrap());
    }

    // 中文注释：关键逻辑说明。
    fn neg_f32(&self, a: &[f32], out: &mut [f32]) {
        let backend = get_or_init_cuda().unwrap();
        let ga = GpuTensor::from_host(backend, a, vec![a.len()]).unwrap();
        let gc = backend.gpu_neg(&ga).unwrap();
        out.copy_from_slice(&gc.to_host(backend).unwrap());
    }

    // 中文注释：关键逻辑说明。
    fn exp_f32(&self, a: &[f32], out: &mut [f32]) {
        let backend = get_or_init_cuda().unwrap();
        let ga = GpuTensor::from_host(backend, a, vec![a.len()]).unwrap();
        let gc = backend.gpu_exp(&ga).unwrap();
        out.copy_from_slice(&gc.to_host(backend).unwrap());
    }

    // 中文注释：关键逻辑说明。
    fn log_f32(&self, a: &[f32], out: &mut [f32]) {
        let backend = get_or_init_cuda().unwrap();
        let ga = GpuTensor::from_host(backend, a, vec![a.len()]).unwrap();
        let gc = backend.gpu_log(&ga).unwrap();
        out.copy_from_slice(&gc.to_host(backend).unwrap());
    }

    // 中文注释：关键逻辑说明。
    fn relu_f32(&self, a: &[f32], out: &mut [f32]) {
        let backend = get_or_init_cuda().unwrap();
        let ga = GpuTensor::from_host(backend, a, vec![a.len()]).unwrap();
        let gc = backend.gpu_relu(&ga).unwrap();
        out.copy_from_slice(&gc.to_host(backend).unwrap());
    }

    // 中文注释：关键逻辑说明。
    fn gelu_f32(&self, a: &[f32], out: &mut [f32]) {
        let backend = get_or_init_cuda().unwrap();
        let ga = GpuTensor::from_host(backend, a, vec![a.len()]).unwrap();
        let gc = backend.gpu_gelu(&ga).unwrap();
        out.copy_from_slice(&gc.to_host(backend).unwrap());
    }

    // 中文注释：关键逻辑说明。
    fn scale_f32(&self, a: &[f32], scalar: f32, out: &mut [f32]) {
        let backend = get_or_init_cuda().unwrap();
        let ga = GpuTensor::from_host(backend, a, vec![a.len()]).unwrap();
        let gc = backend.gpu_scale(&ga, scalar).unwrap();
        out.copy_from_slice(&gc.to_host(backend).unwrap());
    }

    // 中文注释：关键逻辑说明。
    fn matmul_f32(&self, a: &[f32], b: &[f32], out: &mut [f32], m: usize, k: usize, n: usize) {
        let backend = get_or_init_cuda().unwrap();
        let ga = GpuTensor::from_host(backend, a, vec![m, k]).unwrap();
        let gb = GpuTensor::from_host(backend, b, vec![k, n]).unwrap();
        let gc = backend.gpu_matmul(&ga, &gb).unwrap();
        out.copy_from_slice(&gc.to_host(backend).unwrap());
    }
}

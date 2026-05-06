//! Hardware Abstraction Layer for sptorch.
//!
//! Defines `Backend` and `KernelProvider` traits (20 kernel methods).
//! Any hardware (CPU, GPU, NPU) implements these traits to be plugged in.
//! Includes `CpuBackend` as the reference implementation.

pub mod topology;

use sptorch_core_tensor::DType;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DeviceId {
    pub backend: String,
    pub ordinal: usize,
}

impl DeviceId {
    /// `cpu`：中文注释，说明函数用途、输入约束与输出语义。
    pub fn cpu() -> Self {
        DeviceId {
            backend: "cpu".into(),
            ordinal: 0,
        }
    }
    /// `cuda`：中文注释，说明函数用途、输入约束与输出语义。
    pub fn cuda(ordinal: usize) -> Self {
        DeviceId {
            backend: "cuda".into(),
            ordinal,
        }
    }
    /// `tank9k`：中文注释，说明函数用途、输入约束与输出语义。
    pub fn tank9k(ordinal: usize) -> Self {
        DeviceId {
            backend: "tank9k".into(),
            ordinal,
        }
    }
}

impl fmt::Display for DeviceId {
    // 中文注释：关键逻辑说明。
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.backend, self.ordinal)
    }
}

pub struct RawBuffer {
    pub data: Vec<u8>,
    pub device: DeviceId,
}

impl fmt::Debug for RawBuffer {
    // 中文注释：关键逻辑说明。
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RawBuffer")
            .field("len", &self.data.len())
            .field("device", &self.device)
            .finish()
    }
}

#[derive(Debug)]
pub enum HalError {
    DeviceNotFound(DeviceId),
    AllocationFailed { size: usize, reason: String },
    DeviceMismatch { expected: DeviceId, got: DeviceId },
    DTypeMismatch { expected: DType, got: DType },
    Unsupported(String),
}

impl fmt::Display for HalError {
    // 中文注释：关键逻辑说明。
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HalError::DeviceNotFound(id) => write!(f, "device not found: {}", id),
            HalError::AllocationFailed { size, reason } => {
                write!(f, "allocation of {} bytes failed: {}", size, reason)
            }
            HalError::DeviceMismatch { expected, got } => {
                write!(f, "device mismatch: expected {}, got {}", expected, got)
            }
            HalError::DTypeMismatch { expected, got } => {
                write!(f, "dtype mismatch: expected {:?}, got {:?}", expected, got)
            }
            HalError::Unsupported(msg) => write!(f, "unsupported: {}", msg),
        }
    }
}

impl std::error::Error for HalError {}

pub type HalResult<T> = Result<T, HalError>;

pub trait Backend: Send + Sync + 'static {
    // 中文注释：关键逻辑说明。
    fn name(&self) -> &str;
    // 中文注释：关键逻辑说明。
    fn device_id(&self) -> DeviceId;
    // 中文注释：关键逻辑说明。
    fn allocate(&self, size: usize) -> HalResult<RawBuffer>;
    // 中文注释：关键逻辑说明。
    fn copy_to_host(&self, buf: &RawBuffer, dst: &mut [u8]) -> HalResult<()>;
    // 中文注释：关键逻辑说明。
    fn copy_from_host(&self, src: &[u8], buf: &mut RawBuffer) -> HalResult<()>;
    // 中文注释：关键逻辑说明。
    fn synchronize(&self) -> HalResult<()>;
}

pub trait KernelProvider: Backend {
    // ---- element-wise binary ----
    fn add_f32(&self, a: &[f32], b: &[f32], out: &mut [f32]);
    // 中文注释：关键逻辑说明。
    fn mul_f32(&self, a: &[f32], b: &[f32], out: &mut [f32]);

    // ---- element-wise unary ----
    fn neg_f32(&self, a: &[f32], out: &mut [f32]);
    // 中文注释：关键逻辑说明。
    fn exp_f32(&self, a: &[f32], out: &mut [f32]);
    // 中文注释：关键逻辑说明。
    fn log_f32(&self, a: &[f32], out: &mut [f32]);
    // 中文注释：关键逻辑说明。
    fn relu_f32(&self, a: &[f32], out: &mut [f32]);
    // 中文注释：关键逻辑说明。
    fn gelu_f32(&self, a: &[f32], out: &mut [f32]);
    // 中文注释：关键逻辑说明。
    fn scale_f32(&self, a: &[f32], scalar: f32, out: &mut [f32]);

    // ---- matmul ----
    fn matmul_f32(&self, a: &[f32], b: &[f32], out: &mut [f32], m: usize, k: usize, n: usize);
    // 中文注释：关键逻辑说明。
    #[allow(clippy::too_many_arguments)]
    fn batch_matmul_f32(&self, a: &[f32], b: &[f32], out: &mut [f32], batch: usize, m: usize, k: usize, n: usize);

    // ---- reduction ----
    fn sum_f32(&self, a: &[f32]) -> f32;
    // 中文注释：关键逻辑说明。
    fn softmax_f32(&self, a: &[f32], out: &mut [f32], rows: usize, cols: usize);

    // ---- misc ----
    fn masked_fill_f32(&self, a: &[f32], mask: &[bool], fill_value: f32, out: &mut [f32]);
    // 中文注释：关键逻辑说明。
    fn broadcast_add_f32(&self, a: &[f32], b: &[f32], out: &mut [f32], a_len: usize, b_len: usize);
    // 中文注释：关键逻辑说明。
    fn embedding_lookup_f32(&self, weight: &[f32], indices: &[usize], out: &mut [f32], vocab: usize, dim: usize);

    // ---- optimizer kernels ----
    fn sgd_update_f32(&self, params: &mut [f32], grad: &[f32], lr: f32);
    // 中文注释：关键逻辑说明。
    #[allow(clippy::too_many_arguments)]
    fn adam_update_f32(
        &self,
        params: &mut [f32],
        grad: &[f32],
        m: &mut [f32],
        v: &mut [f32],
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        bc1: f32,
        bc2: f32,
    );
}

pub struct CpuBackend;

impl Backend for CpuBackend {
    // 中文注释：关键逻辑说明。
    fn name(&self) -> &str {
        "cpu"
    }

    // 中文注释：关键逻辑说明。
    fn device_id(&self) -> DeviceId {
        DeviceId::cpu()
    }

    // 中文注释：关键逻辑说明。
    fn allocate(&self, size: usize) -> HalResult<RawBuffer> {
        Ok(RawBuffer {
            data: vec![0u8; size],
            device: DeviceId::cpu(),
        })
    }

    // 中文注释：关键逻辑说明。
    fn copy_to_host(&self, buf: &RawBuffer, dst: &mut [u8]) -> HalResult<()> {
        dst.copy_from_slice(&buf.data);
        Ok(())
    }

    // 中文注释：关键逻辑说明。
    fn copy_from_host(&self, src: &[u8], buf: &mut RawBuffer) -> HalResult<()> {
        buf.data.copy_from_slice(src);
        Ok(())
    }

    // 中文注释：关键逻辑说明。
    fn synchronize(&self) -> HalResult<()> {
        Ok(())
    }
}

impl KernelProvider for CpuBackend {
    // 中文注释：关键逻辑说明。
    fn add_f32(&self, a: &[f32], b: &[f32], out: &mut [f32]) {
        for i in 0..out.len() {
            out[i] = a[i] + b[i];
        }
    }

    // 中文注释：关键逻辑说明。
    fn mul_f32(&self, a: &[f32], b: &[f32], out: &mut [f32]) {
        for i in 0..out.len() {
            out[i] = a[i] * b[i];
        }
    }

    // 中文注释：关键逻辑说明。
    fn neg_f32(&self, a: &[f32], out: &mut [f32]) {
        for i in 0..out.len() {
            out[i] = -a[i];
        }
    }

    // 中文注释：关键逻辑说明。
    fn exp_f32(&self, a: &[f32], out: &mut [f32]) {
        for i in 0..out.len() {
            out[i] = a[i].exp();
        }
    }

    // 中文注释：关键逻辑说明。
    fn log_f32(&self, a: &[f32], out: &mut [f32]) {
        for i in 0..out.len() {
            out[i] = a[i].ln();
        }
    }

    // 中文注释：关键逻辑说明。
    fn relu_f32(&self, a: &[f32], out: &mut [f32]) {
        for i in 0..out.len() {
            out[i] = if a[i] > 0.0 { a[i] } else { 0.0 };
        }
    }

    // 中文注释：关键逻辑说明。
    fn gelu_f32(&self, a: &[f32], out: &mut [f32]) {
        for i in 0..out.len() {
            let x = a[i];
            out[i] = 0.5 * x * (1.0 + ((2.0_f32 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh());
        }
    }

    // 中文注释：关键逻辑说明。
    fn scale_f32(&self, a: &[f32], scalar: f32, out: &mut [f32]) {
        for i in 0..out.len() {
            out[i] = a[i] * scalar;
        }
    }

    // 中文注释：关键逻辑说明。
    fn matmul_f32(&self, a: &[f32], b: &[f32], out: &mut [f32], m: usize, k: usize, n: usize) {
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += a[i * k + p] * b[p * n + j];
                }
                out[i * n + j] = sum;
            }
        }
    }

    // 中文注释：关键逻辑说明。
    fn batch_matmul_f32(&self, a: &[f32], b: &[f32], out: &mut [f32], batch: usize, m: usize, k: usize, n: usize) {
        let a_stride = m * k;
        let b_stride = k * n;
        let o_stride = m * n;
        for bi in 0..batch {
            let a_off = bi * a_stride;
            let b_off = bi * b_stride;
            let o_off = bi * o_stride;
            self.matmul_f32(
                &a[a_off..a_off + a_stride],
                &b[b_off..b_off + b_stride],
                &mut out[o_off..o_off + o_stride],
                m,
                k,
                n,
            );
        }
    }

    // 中文注释：关键逻辑说明。
    fn sum_f32(&self, a: &[f32]) -> f32 {
        a.iter().sum()
    }

    // 中文注释：关键逻辑说明。
    fn softmax_f32(&self, a: &[f32], out: &mut [f32], rows: usize, cols: usize) {
        for r in 0..rows {
            let row = &a[r * cols..(r + 1) * cols];
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for c in 0..cols {
                let e = (row[c] - max).exp();
                out[r * cols + c] = e;
                sum += e;
            }
            for c in 0..cols {
                out[r * cols + c] /= sum;
            }
        }
    }

    // 中文注释：关键逻辑说明。
    fn masked_fill_f32(&self, a: &[f32], mask: &[bool], fill_value: f32, out: &mut [f32]) {
        for i in 0..a.len() {
            out[i] = if mask[i] { fill_value } else { a[i] };
        }
    }

    // 中文注释：关键逻辑说明。
    fn broadcast_add_f32(&self, a: &[f32], b: &[f32], out: &mut [f32], a_len: usize, b_len: usize) {
        for i in 0..a_len {
            out[i] = a[i] + b[i % b_len];
        }
    }

    // 中文注释：关键逻辑说明。
    fn embedding_lookup_f32(&self, weight: &[f32], indices: &[usize], out: &mut [f32], _vocab: usize, dim: usize) {
        for (i, &idx) in indices.iter().enumerate() {
            out[i * dim..(i + 1) * dim].copy_from_slice(&weight[idx * dim..(idx + 1) * dim]);
        }
    }

    // 中文注释：关键逻辑说明。
    fn sgd_update_f32(&self, params: &mut [f32], grad: &[f32], lr: f32) {
        for (w, g) in params.iter_mut().zip(grad.iter()) {
            *w -= lr * g;
        }
    }

    // 中文注释：关键逻辑说明。
    fn adam_update_f32(
        &self,
        params: &mut [f32],
        grad: &[f32],
        m: &mut [f32],
        v: &mut [f32],
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        bc1: f32,
        bc2: f32,
    ) {
        for j in 0..params.len() {
            if weight_decay != 0.0 {
                params[j] *= 1.0 - lr * weight_decay;
            }
            m[j] = beta1 * m[j] + (1.0 - beta1) * grad[j];
            v[j] = beta2 * v[j] + (1.0 - beta2) * grad[j] * grad[j];
            let m_hat = m[j] / bc1;
            let v_hat = v[j] / bc2;
            params[j] -= lr * m_hat / (v_hat.sqrt() + eps);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // 中文注释：关键逻辑说明。
    #[test]
    fn test_cpu_backend_add() {
        let backend = CpuBackend;
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![4.0f32, 5.0, 6.0];
        let mut out = vec![0.0f32; 3];
        backend.add_f32(&a, &b, &mut out);
        assert_eq!(out, vec![5.0, 7.0, 9.0]);
    }

    // 中文注释：关键逻辑说明。
    #[test]
    fn test_cpu_backend_mul() {
        let backend = CpuBackend;
        let a = vec![2.0f32, 3.0];
        let b = vec![4.0f32, 5.0];
        let mut out = vec![0.0f32; 2];
        backend.mul_f32(&a, &b, &mut out);
        assert_eq!(out, vec![8.0, 15.0]);
    }

    // 中文注释：关键逻辑说明。
    #[test]
    fn test_cpu_backend_matmul() {
        let backend = CpuBackend;
        // [2,3] @ [3,2] = [2,2]
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let mut out = vec![0.0f32; 4];
        backend.matmul_f32(&a, &b, &mut out, 2, 3, 2);
        assert_eq!(out, vec![58.0, 64.0, 139.0, 154.0]);
    }

    // 中文注释：关键逻辑说明。
    #[test]
    fn test_cpu_backend_allocate() {
        let backend = CpuBackend;
        let buf = backend.allocate(16).unwrap();
        assert_eq!(buf.data.len(), 16);
        assert_eq!(buf.device, DeviceId::cpu());
    }

    // 中文注释：关键逻辑说明。
    #[test]
    fn test_cpu_backend_neg() {
        let backend = CpuBackend;
        let a = vec![1.0f32, -2.0, 3.0];
        let mut out = vec![0.0f32; 3];
        backend.neg_f32(&a, &mut out);
        assert_eq!(out, vec![-1.0, 2.0, -3.0]);
    }

    // 中文注释：关键逻辑说明。
    #[test]
    fn test_cpu_backend_exp_log() {
        let backend = CpuBackend;
        let a = vec![0.0f32, 1.0, 2.0];
        let mut exp_out = vec![0.0f32; 3];
        backend.exp_f32(&a, &mut exp_out);
        assert!((exp_out[0] - 1.0).abs() < 1e-6);
        assert!((exp_out[1] - std::f32::consts::E).abs() < 1e-5);

        let mut log_out = vec![0.0f32; 3];
        backend.log_f32(&exp_out, &mut log_out);
        for i in 0..3 {
            assert!((log_out[i] - a[i]).abs() < 1e-5);
        }
    }

    // 中文注释：关键逻辑说明。
    #[test]
    fn test_cpu_backend_relu() {
        let backend = CpuBackend;
        let a = vec![-1.0f32, 0.0, 2.0, -0.5];
        let mut out = vec![0.0f32; 4];
        backend.relu_f32(&a, &mut out);
        assert_eq!(out, vec![0.0, 0.0, 2.0, 0.0]);
    }

    // 中文注释：关键逻辑说明。
    #[test]
    fn test_cpu_backend_gelu() {
        let backend = CpuBackend;
        let a = vec![0.0f32, 1.0, -1.0];
        let mut out = vec![0.0f32; 3];
        backend.gelu_f32(&a, &mut out);
        assert!((out[0] - 0.0).abs() < 1e-6);
        assert!((out[1] - 0.8413).abs() < 1e-3);
        assert!((out[2] - (-0.1587)).abs() < 1e-3);
    }

    // 中文注释：关键逻辑说明。
    #[test]
    fn test_cpu_backend_scale() {
        let backend = CpuBackend;
        let a = vec![1.0f32, 2.0, 3.0];
        let mut out = vec![0.0f32; 3];
        backend.scale_f32(&a, 0.5, &mut out);
        assert_eq!(out, vec![0.5, 1.0, 1.5]);
    }

    // 中文注释：关键逻辑说明。
    #[test]
    fn test_cpu_backend_softmax() {
        let backend = CpuBackend;
        let a = vec![1.0f32, 2.0, 3.0];
        let mut out = vec![0.0f32; 3];
        backend.softmax_f32(&a, &mut out, 1, 3);
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(out[2] > out[1] && out[1] > out[0]);
    }

    // 中文注释：关键逻辑说明。
    #[test]
    fn test_cpu_backend_batch_matmul() {
        let backend = CpuBackend;
        // batch=2, [2,2] @ [2,2]
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]; // identity
        let mut out = vec![0.0f32; 8];
        backend.batch_matmul_f32(&a, &b, &mut out, 2, 2, 2, 2);
        assert_eq!(out, a);
    }

    // 中文注释：关键逻辑说明。
    #[test]
    fn test_cpu_backend_sgd_update() {
        let backend = CpuBackend;
        let mut params = vec![1.0f32, 2.0];
        let grad = vec![0.1, 0.2];
        backend.sgd_update_f32(&mut params, &grad, 0.1);
        assert!((params[0] - 0.99).abs() < 1e-6);
        assert!((params[1] - 1.98).abs() < 1e-6);
    }

    // 中文注释：关键逻辑说明。
    #[test]
    fn test_cpu_backend_embedding() {
        let backend = CpuBackend;
        // vocab=3, dim=2
        let weight = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let indices = vec![2, 0, 1];
        let mut out = vec![0.0f32; 6];
        backend.embedding_lookup_f32(&weight, &indices, &mut out, 3, 2);
        assert_eq!(out, vec![0.5, 0.6, 0.1, 0.2, 0.3, 0.4]);
    }

    // 中文注释：关键逻辑说明。
    #[test]
    fn test_cpu_backend_masked_fill() {
        let backend = CpuBackend;
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let mask = vec![false, true, false, true];
        let mut out = vec![0.0f32; 4];
        backend.masked_fill_f32(&a, &mask, -999.0, &mut out);
        assert_eq!(out, vec![1.0, -999.0, 3.0, -999.0]);
    }

    // 中文注释：关键逻辑说明。
    #[test]
    fn test_cpu_backend_broadcast_add() {
        let backend = CpuBackend;
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![10.0, 20.0, 30.0];
        let mut out = vec![0.0f32; 6];
        backend.broadcast_add_f32(&a, &b, &mut out, 6, 3);
        assert_eq!(out, vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
    }
}

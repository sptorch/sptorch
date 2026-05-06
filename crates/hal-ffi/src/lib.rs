//! C FFI bridge for external hardware backends.
//!
//! Loads a shared library (.dll/.so) at runtime via `libloading`,
//! wrapping its `sptorch_*` C symbols into a `KernelProvider` implementation.
//! See `include/sptorch_hal.h` for the C API that vendors must implement.

use libloading::{Library, Symbol};
use sptorch_core_tensor::Device;
use sptorch_hal::{Backend, DeviceId, HalError, HalResult, KernelProvider, RawBuffer};
use std::ffi::CStr;
use std::path::Path;
use std::sync::Arc;

type InitFn = unsafe extern "C" fn() -> i32;
type ShutdownFn = unsafe extern "C" fn();
type NameFn = unsafe extern "C" fn() -> *const std::os::raw::c_char;
type AllocFn = unsafe extern "C" fn(n: usize) -> *mut std::ffi::c_void;
type FreeFn = unsafe extern "C" fn(handle: *mut std::ffi::c_void);
type CopyH2DFn = unsafe extern "C" fn(host: *const f32, device: *mut std::ffi::c_void, n: usize) -> i32;
type CopyD2HFn = unsafe extern "C" fn(device: *const std::ffi::c_void, host: *mut f32, n: usize) -> i32;
type SyncFn = unsafe extern "C" fn() -> i32;
type QueryRuntimeFn = unsafe extern "C" fn(queue_depth: *mut u32, online: *mut u32) -> i32;

type BinaryOpFn = unsafe extern "C" fn(
    a: *const std::ffi::c_void,
    b: *const std::ffi::c_void,
    out: *mut std::ffi::c_void,
    n: usize,
) -> i32;
type UnaryOpFn = unsafe extern "C" fn(a: *const std::ffi::c_void, out: *mut std::ffi::c_void, n: usize) -> i32;
type ScaleOpFn =
    unsafe extern "C" fn(a: *const std::ffi::c_void, scalar: f32, out: *mut std::ffi::c_void, n: usize) -> i32;
type MatmulFn = unsafe extern "C" fn(
    a: *const std::ffi::c_void,
    b: *const std::ffi::c_void,
    out: *mut std::ffi::c_void,
    m: usize,
    k: usize,
    n: usize,
) -> i32;
type BatchMatmulFn = unsafe extern "C" fn(
    a: *const std::ffi::c_void,
    b: *const std::ffi::c_void,
    out: *mut std::ffi::c_void,
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
) -> i32;
type SoftmaxFn =
    unsafe extern "C" fn(a: *const std::ffi::c_void, out: *mut std::ffi::c_void, rows: usize, cols: usize) -> i32;

/// Opaque device-side buffer handle returned by the external backend.
pub struct FfiDeviceBuffer {
    ptr: *mut std::ffi::c_void,
    len: usize,
    backend: Arc<FfiBackendInner>,
}

unsafe impl Send for FfiDeviceBuffer {}
unsafe impl Sync for FfiDeviceBuffer {}

impl std::fmt::Debug for FfiDeviceBuffer {
    // 中文注释：关键逻辑说明。
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FfiDeviceBuffer")
            .field("len", &self.len)
            .field("ptr", &self.ptr)
            .finish()
    }
}

impl sptorch_core_tensor::DeviceBuffer for FfiDeviceBuffer {
    // 中文注释：关键逻辑说明。
    fn device(&self) -> Device {
        Device::Custom(0)
    }

    // 中文注释：关键逻辑说明。
    fn len(&self) -> usize {
        self.len
    }

    // 中文注释：关键逻辑说明。
    fn to_host(&self) -> Vec<f32> {
        let mut host = vec![0.0f32; self.len];
        unsafe {
            (self.backend.copy_d2h)(self.ptr, host.as_mut_ptr(), self.len);
        }
        host
    }

    // 中文注释：关键逻辑说明。
    fn from_host(
        _data: &[f32],
        _device: Device,
    ) -> std::result::Result<Box<dyn sptorch_core_tensor::DeviceBuffer>, String> {
        Err("FfiDeviceBuffer::from_host requires a backend reference; use FfiBackend::upload() instead".into())
    }
}

impl Drop for FfiDeviceBuffer {
    // 中文注释：关键逻辑说明。
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                (self.backend.free)(self.ptr);
            }
        }
    }
}

struct FfiBackendInner {
    _lib: Library,
    shutdown: ShutdownFn,
    name: NameFn,
    alloc: AllocFn,
    free: FreeFn,
    copy_h2d: CopyH2DFn,
    copy_d2h: CopyD2HFn,
    sync: SyncFn,
    query_runtime: Option<QueryRuntimeFn>,
    add: BinaryOpFn,
    mul: BinaryOpFn,
    neg: UnaryOpFn,
    exp: UnaryOpFn,
    log: UnaryOpFn,
    relu: UnaryOpFn,
    gelu: UnaryOpFn,
    scale: ScaleOpFn,
    matmul: MatmulFn,
    batch_matmul: BatchMatmulFn,
    softmax: SoftmaxFn,
}

unsafe impl Send for FfiBackendInner {}
unsafe impl Sync for FfiBackendInner {}

impl Drop for FfiBackendInner {
    // 中文注释：关键逻辑说明。
    fn drop(&mut self) {
        unsafe {
            (self.shutdown)();
        }
    }
}

/// A HAL backend loaded from an external C shared library at runtime.
pub struct FfiBackend {
    inner: Arc<FfiBackendInner>,
}

macro_rules! load_sym {
    ($lib:expr, $name:expr) => {
        **$lib
            .get::<Symbol<_>>($name)
            .map_err(|e| format!("missing symbol {}: {}", String::from_utf8_lossy($name), e))?
    };
}

impl FfiBackend {
    /// Load an external backend from a shared library path.
    ///
    /// The library must export all `sptorch_*` symbols defined in `sptorch_hal.h`.
    pub fn load<P: AsRef<Path>>(path: P) -> std::result::Result<Self, String> {
        unsafe {
            let lib = Library::new(path.as_ref()).map_err(|e| format!("failed to load library: {}", e))?;

            let init: InitFn = load_sym!(lib, b"sptorch_backend_init\0");
            let rc = init();
            if rc != 0 {
                return Err(format!("sptorch_backend_init returned {}", rc));
            }

            let inner = FfiBackendInner {
                shutdown: load_sym!(lib, b"sptorch_backend_shutdown\0"),
                name: load_sym!(lib, b"sptorch_backend_name\0"),
                alloc: load_sym!(lib, b"sptorch_alloc\0"),
                free: load_sym!(lib, b"sptorch_free\0"),
                copy_h2d: load_sym!(lib, b"sptorch_copy_h2d\0"),
                copy_d2h: load_sym!(lib, b"sptorch_copy_d2h\0"),
                sync: load_sym!(lib, b"sptorch_sync\0"),
                query_runtime: match lib.get::<Symbol<QueryRuntimeFn>>(b"sptorch_query_runtime\0") {
                    Ok(sym) => Some(**sym),
                    Err(_) => None,
                },
                add: load_sym!(lib, b"sptorch_add_f32\0"),
                mul: load_sym!(lib, b"sptorch_mul_f32\0"),
                neg: load_sym!(lib, b"sptorch_neg_f32\0"),
                exp: load_sym!(lib, b"sptorch_exp_f32\0"),
                log: load_sym!(lib, b"sptorch_log_f32\0"),
                relu: load_sym!(lib, b"sptorch_relu_f32\0"),
                gelu: load_sym!(lib, b"sptorch_gelu_f32\0"),
                scale: load_sym!(lib, b"sptorch_scale_f32\0"),
                matmul: load_sym!(lib, b"sptorch_matmul_f32\0"),
                batch_matmul: load_sym!(lib, b"sptorch_batch_matmul_f32\0"),
                softmax: load_sym!(lib, b"sptorch_softmax_f32\0"),
                _lib: lib,
            };

            Ok(FfiBackend { inner: Arc::new(inner) })
        }
    }
    /// `upload`：中文注释，说明函数用途、输入约束与输出语义。
    pub fn upload(&self, data: &[f32]) -> std::result::Result<FfiDeviceBuffer, String> {
        let ptr = unsafe { (self.inner.alloc)(data.len()) };
        if ptr.is_null() {
            return Err("device allocation failed".into());
        }
        let rc = unsafe { (self.inner.copy_h2d)(data.as_ptr(), ptr, data.len()) };
        if rc != 0 {
            unsafe {
                (self.inner.free)(ptr);
            }
            return Err(format!("h2d copy failed with code {}", rc));
        }
        Ok(FfiDeviceBuffer {
            ptr,
            len: data.len(),
            backend: self.inner.clone(),
        })
    }

    // 中文注释：关键逻辑说明。
    fn run_unary(&self, a: &[f32], op: UnaryOpFn) -> Vec<f32> {
        let n = a.len();
        let a_buf = self.upload(a).expect("upload failed");
        let o_ptr = unsafe { (self.inner.alloc)(n) };
        unsafe {
            op(a_buf.ptr, o_ptr, n);
        }
        let mut out = vec![0.0f32; n];
        unsafe {
            (self.inner.copy_d2h)(o_ptr, out.as_mut_ptr(), n);
        }
        unsafe {
            (self.inner.free)(o_ptr);
        }
        out
    }

    // 中文注释：关键逻辑说明。
    fn run_binary(&self, a: &[f32], b: &[f32], op: BinaryOpFn) -> Vec<f32> {
        let n = a.len();
        let a_buf = self.upload(a).expect("upload failed");
        let b_buf = self.upload(b).expect("upload failed");
        let o_ptr = unsafe { (self.inner.alloc)(n) };
        unsafe {
            op(a_buf.ptr, b_buf.ptr, o_ptr, n);
        }
        let mut out = vec![0.0f32; n];
        unsafe {
            (self.inner.copy_d2h)(o_ptr, out.as_mut_ptr(), n);
        }
        unsafe {
            (self.inner.free)(o_ptr);
        }
        out
    }
    /// `query_runtime`：中文注释，说明函数用途、输入约束与输出语义。
    pub fn query_runtime(&self) -> Option<(u32, bool)> {
        let f = self.inner.query_runtime?;
        let mut queue_depth = 0u32;
        let mut online = 0u32;
        let rc = unsafe { f(&mut queue_depth as *mut u32, &mut online as *mut u32) };
        if rc != 0 {
            return None;
        }
        Some((queue_depth, online != 0))
    }
}

impl Backend for FfiBackend {
    // 中文注释：关键逻辑说明。
    fn name(&self) -> &str {
        unsafe {
            let ptr = (self.inner.name)();
            CStr::from_ptr(ptr).to_str().unwrap_or("unknown")
        }
    }

    // 中文注释：关键逻辑说明。
    fn device_id(&self) -> DeviceId {
        DeviceId {
            backend: self.name().to_string(),
            ordinal: 0,
        }
    }

    // 中文注释：关键逻辑说明。
    fn allocate(&self, size: usize) -> HalResult<RawBuffer> {
        Ok(RawBuffer {
            data: vec![0u8; size],
            device: self.device_id(),
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
        let rc = unsafe { (self.inner.sync)() };
        if rc != 0 {
            return Err(HalError::Unsupported(format!("sync failed: {}", rc)));
        }
        Ok(())
    }
}

impl KernelProvider for FfiBackend {
    // 中文注释：关键逻辑说明。
    fn add_f32(&self, a: &[f32], b: &[f32], out: &mut [f32]) {
        let result = self.run_binary(a, b, self.inner.add);
        out.copy_from_slice(&result);
    }

    // 中文注释：关键逻辑说明。
    fn mul_f32(&self, a: &[f32], b: &[f32], out: &mut [f32]) {
        let result = self.run_binary(a, b, self.inner.mul);
        out.copy_from_slice(&result);
    }

    // 中文注释：关键逻辑说明。
    fn neg_f32(&self, a: &[f32], out: &mut [f32]) {
        let result = self.run_unary(a, self.inner.neg);
        out.copy_from_slice(&result);
    }

    // 中文注释：关键逻辑说明。
    fn exp_f32(&self, a: &[f32], out: &mut [f32]) {
        let result = self.run_unary(a, self.inner.exp);
        out.copy_from_slice(&result);
    }

    // 中文注释：关键逻辑说明。
    fn log_f32(&self, a: &[f32], out: &mut [f32]) {
        let result = self.run_unary(a, self.inner.log);
        out.copy_from_slice(&result);
    }

    // 中文注释：关键逻辑说明。
    fn relu_f32(&self, a: &[f32], out: &mut [f32]) {
        let result = self.run_unary(a, self.inner.relu);
        out.copy_from_slice(&result);
    }

    // 中文注释：关键逻辑说明。
    fn gelu_f32(&self, a: &[f32], out: &mut [f32]) {
        let result = self.run_unary(a, self.inner.gelu);
        out.copy_from_slice(&result);
    }

    // 中文注释：关键逻辑说明。
    fn scale_f32(&self, a: &[f32], scalar: f32, out: &mut [f32]) {
        let n = a.len();
        let a_buf = self.upload(a).expect("upload failed");
        let o_ptr = unsafe { (self.inner.alloc)(n) };
        unsafe {
            (self.inner.scale)(a_buf.ptr, scalar, o_ptr, n);
        }
        unsafe {
            (self.inner.copy_d2h)(o_ptr, out.as_mut_ptr(), n);
        }
        unsafe {
            (self.inner.free)(o_ptr);
        }
    }

    // 中文注释：关键逻辑说明。
    fn matmul_f32(&self, a: &[f32], b: &[f32], out: &mut [f32], m: usize, k: usize, n: usize) {
        let a_buf = self.upload(a).expect("upload failed");
        let b_buf = self.upload(b).expect("upload failed");
        let o_ptr = unsafe { (self.inner.alloc)(m * n) };
        unsafe {
            (self.inner.matmul)(a_buf.ptr, b_buf.ptr, o_ptr, m, k, n);
        }
        unsafe {
            (self.inner.copy_d2h)(o_ptr, out.as_mut_ptr(), m * n);
        }
        unsafe {
            (self.inner.free)(o_ptr);
        }
    }

    // 中文注释：关键逻辑说明。
    fn batch_matmul_f32(&self, a: &[f32], b: &[f32], out: &mut [f32], batch: usize, m: usize, k: usize, n: usize) {
        let a_buf = self.upload(a).expect("upload failed");
        let b_buf = self.upload(b).expect("upload failed");
        let total = batch * m * n;
        let o_ptr = unsafe { (self.inner.alloc)(total) };
        unsafe {
            (self.inner.batch_matmul)(a_buf.ptr, b_buf.ptr, o_ptr, batch, m, k, n);
        }
        unsafe {
            (self.inner.copy_d2h)(o_ptr, out.as_mut_ptr(), total);
        }
        unsafe {
            (self.inner.free)(o_ptr);
        }
    }

    // 中文注释：关键逻辑说明。
    fn sum_f32(&self, a: &[f32]) -> f32 {
        a.iter().sum()
    }

    // 中文注释：关键逻辑说明。
    fn softmax_f32(&self, a: &[f32], out: &mut [f32], rows: usize, cols: usize) {
        let n = rows * cols;
        let a_buf = self.upload(a).expect("upload failed");
        let o_ptr = unsafe { (self.inner.alloc)(n) };
        unsafe {
            (self.inner.softmax)(a_buf.ptr, o_ptr, rows, cols);
        }
        unsafe {
            (self.inner.copy_d2h)(o_ptr, out.as_mut_ptr(), n);
        }
        unsafe {
            (self.inner.free)(o_ptr);
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

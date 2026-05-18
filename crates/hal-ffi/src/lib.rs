//! 外部硬件后端的 C FFI 桥接层。
//!
//! `hal-ffi` 在运行时通过 `libloading` 加载 `.dll/.so`，把供应商实现的
//! `sptorch_*` C 符号包装成 `KernelProvider`。这让 Tang9k、mock NPU 或未来
//! PCIe/串口后端可以在不侵入 core-ops 的情况下接入框架。C ABI 以
//! `include/sptorch_hal.h` 为准。

pub mod probe_record;
pub mod serial_backend;

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

/// 外部后端返回的不透明设备缓冲句柄。
///
/// Rust 侧只保存指针、长度和后端引用，不解释指针内部结构。释放必须回到同一个
/// C 后端，避免跨 allocator 或跨驱动释放造成未定义行为。
pub struct FfiDeviceBuffer {
    ptr: *mut std::ffi::c_void,
    len: usize,
    backend: Arc<FfiBackendInner>,
}

unsafe impl Send for FfiDeviceBuffer {}
unsafe impl Sync for FfiDeviceBuffer {}

impl std::fmt::Debug for FfiDeviceBuffer {
    // Debug 只打印指针和值长度，不尝试解引用外部设备内存。
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FfiDeviceBuffer")
            .field("len", &self.len)
            .field("ptr", &self.ptr)
            .finish()
    }
}

impl sptorch_core_tensor::DeviceBuffer for FfiDeviceBuffer {
    // FFI 缓冲目前统一暴露为 Custom 设备；具体后端名由 FfiBackend::name 提供。
    fn device(&self) -> Device {
        Device::Custom(0)
    }

    // 长度以 f32 元素数计，与 C ABI 的 `n` 参数保持一致。
    fn len(&self) -> usize {
        self.len
    }

    // DeviceBuffer trait 没有错误返回，这里只能按 ABI 约定执行 d2h 拷贝。
    fn to_host(&self) -> Vec<f32> {
        let mut host = vec![0.0f32; self.len];
        unsafe {
            (self.backend.copy_d2h)(self.ptr, host.as_mut_ptr(), self.len);
        }
        host
    }

    // 构造 FFI 缓冲必须知道具体后端库，所以禁止通过静态 trait 方法绕过 upload。
    fn from_host(
        _data: &[f32],
        _device: Device,
    ) -> std::result::Result<Box<dyn sptorch_core_tensor::DeviceBuffer>, String> {
        Err("FfiDeviceBuffer::from_host requires a backend reference; use FfiBackend::upload() instead".into())
    }
}

impl Drop for FfiDeviceBuffer {
    // 设备指针和动态库生命周期都由包装对象兜住，释放时回调供应商后端。
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
    // 设备指针和动态库生命周期都由包装对象兜住，释放时回调供应商后端。
    fn drop(&mut self) {
        unsafe {
            (self.shutdown)();
        }
    }
}

/// 从外部 C 动态库加载的 HAL 后端。
///
/// 加载成功后，所有 kernel 调用都会经过 C ABI。当前实现偏向验证链路清晰度，
/// 每次算子会显式 upload/download；真实高性能后端后续应复用设备缓冲和流。
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
    /// 从动态库路径加载外部后端。
    ///
    /// 动态库必须导出 `sptorch_hal.h` 中定义的必需符号；`sptorch_query_runtime`
    /// 是可选遥测钩子，缺失时不会导致加载失败。
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
    /// 将主机侧 f32 切片上传到外部后端管理的设备缓冲。
    ///
    /// 返回对象持有设备指针所有权；如果 H2D 拷贝失败，会立即释放刚分配的设备内存。
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

    // 一元算子的验证路径：上传输入、分配输出、调用 C kernel、再下载结果。
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

    // 二元算子保持与一元算子同样的显式生命周期，便于定位 FFI 边界错误。
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
    /// 查询后端运行时遥测。
    ///
    /// 返回 `(queue_depth, online)`；若后端未实现可选符号或返回错误码，则返回 `None`。
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
    // 后端名来自 C 字符串，无法解析 UTF-8 时降级为 unknown。
    fn name(&self) -> &str {
        unsafe {
            let ptr = (self.inner.name)();
            CStr::from_ptr(ptr).to_str().unwrap_or("unknown")
        }
    }

    // FFI 缓冲目前统一暴露为 Custom 设备；具体后端名由 FfiBackend::name 提供。
    fn device_id(&self) -> DeviceId {
        DeviceId {
            backend: self.name().to_string(),
            ordinal: 0,
        }
    }

    // Backend trait 的 RawBuffer 路径仍保留主机字节语义，kernel 路径走 C 设备指针。
    fn allocate(&self, size: usize) -> HalResult<RawBuffer> {
        Ok(RawBuffer {
            data: vec![0u8; size],
            device: self.device_id(),
        })
    }

    // RawBuffer 是 HAL 通用缓冲，不等同于 FfiDeviceBuffer 的外部设备指针。
    fn copy_to_host(&self, buf: &RawBuffer, dst: &mut [u8]) -> HalResult<()> {
        dst.copy_from_slice(&buf.data);
        Ok(())
    }

    // 这里保持 CPU 字节拷贝语义，供通用 HAL 测试和 fallback 路径使用。
    fn copy_from_host(&self, src: &[u8], buf: &mut RawBuffer) -> HalResult<()> {
        buf.data.copy_from_slice(src);
        Ok(())
    }

    // synchronize 是双缓冲 swap 和真实硬件队列 drain 的统一 fence 边界。
    fn synchronize(&self) -> HalResult<()> {
        let rc = unsafe { (self.inner.sync)() };
        if rc != 0 {
            return Err(HalError::Unsupported(format!("sync failed: {}", rc)));
        }
        Ok(())
    }
}

impl KernelProvider for FfiBackend {
    // 加法等逐元素 kernel 通过 C 后端执行，输出切片由调用方预先分配。
    fn add_f32(&self, a: &[f32], b: &[f32], out: &mut [f32]) {
        let result = self.run_binary(a, b, self.inner.add);
        out.copy_from_slice(&result);
    }

    // 乘法沿用显式 upload/download，确保 mock NPU 与真实插件走同一 ABI。
    fn mul_f32(&self, a: &[f32], b: &[f32], out: &mut [f32]) {
        let result = self.run_binary(a, b, self.inner.mul);
        out.copy_from_slice(&result);
    }

    // 取负是一元 kernel，常用于验证最小外设计算链路。
    fn neg_f32(&self, a: &[f32], out: &mut [f32]) {
        let result = self.run_unary(a, self.inner.neg);
        out.copy_from_slice(&result);
    }

    // exp/log 这类非线性函数用于检查后端数学库与 CPU 参考的一致性。
    fn exp_f32(&self, a: &[f32], out: &mut [f32]) {
        let result = self.run_unary(a, self.inner.exp);
        out.copy_from_slice(&result);
    }

    // 调用方负责输入定义域；FFI 层不额外修正非法数值。
    fn log_f32(&self, a: &[f32], out: &mut [f32]) {
        let result = self.run_unary(a, self.inner.log);
        out.copy_from_slice(&result);
    }

    // ReLU 是最小激活函数验收路径，硬件后端应与 CPU 截断语义一致。
    fn relu_f32(&self, a: &[f32], out: &mut [f32]) {
        let result = self.run_unary(a, self.inner.relu);
        out.copy_from_slice(&result);
    }

    // GELU 保留在 ABI 中，避免 Transformer 路径必须回退到 CPU。
    fn gelu_f32(&self, a: &[f32], out: &mut [f32]) {
        let result = self.run_unary(a, self.inner.gelu);
        out.copy_from_slice(&result);
    }

    // scale 需要额外标量参数，单独走 C ABI 的 scale 函数。
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

    // MatMul 是 Tang9k/ASIC 验证主路径，ABI 约定输入为 row-major 展平矩阵。
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

    // batch matmul 当前一次性上传展平数据，真实后端可在 C 侧自行分批调度。
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

    // sum 暂未进入 C ABI，保留 CPU 聚合以覆盖 KernelProvider 完整 trait。
    fn sum_f32(&self, a: &[f32]) -> f32 {
        a.iter().sum()
    }

    // softmax 以二维 rows/cols 形式传入，便于注意力分数按行归一化。
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

    // masked_fill 暂在 Rust 侧执行，避免 C ABI 过早绑定 bool mask 表示。
    fn masked_fill_f32(&self, a: &[f32], mask: &[bool], fill_value: f32, out: &mut [f32]) {
        for i in 0..a.len() {
            out[i] = if mask[i] { fill_value } else { a[i] };
        }
    }

    // broadcast_add 的最小语义是尾部循环广播，供旧 kernel 和测试夹具复用。
    fn broadcast_add_f32(&self, a: &[f32], b: &[f32], out: &mut [f32], a_len: usize, b_len: usize) {
        for i in 0..a_len {
            out[i] = a[i] + b[i % b_len];
        }
    }

    // embedding 查表保持 row-major `[vocab, dim]`，后续 DMA 可按行分块搬运。
    fn embedding_lookup_f32(&self, weight: &[f32], indices: &[usize], out: &mut [f32], _vocab: usize, dim: usize) {
        for (i, &idx) in indices.iter().enumerate() {
            out[i * dim..(i + 1) * dim].copy_from_slice(&weight[idx * dim..(idx + 1) * dim]);
        }
    }

    // 优化器更新暂在 Rust 侧执行，保证外部后端即使只实现前向 kernel 也可参与测试。
    fn sgd_update_f32(&self, params: &mut [f32], grad: &[f32], lr: f32) {
        for (w, g) in params.iter_mut().zip(grad.iter()) {
            *w -= lr * g;
        }
    }

    // AdamW 参考实现放在 FFI 包装层，避免供应商后端必须立即实现优化器状态机。
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

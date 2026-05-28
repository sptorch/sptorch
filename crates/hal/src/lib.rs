//! SPTorch 的硬件抽象层（HAL）。
//!
//! 这一层只描述“框架如何看待硬件”，不绑定具体 CUDA、NPU 或 Tank9k
//! 驱动实现。上层 Tensor/Op 通过 [`Backend`] 管理设备内存，通过
//! [`KernelProvider`] 调用数值内核；外部硬件只要实现这两个 trait，就能
//! 被框架以统一方式调度。
//!
//! [`CpuBackend`] 是正确性优先的参考实现：它不是性能目标，而是所有硬件
//! 后端对齐数学语义、边界行为和测试期望的标尺。

pub mod serial;
pub mod topology;

use sptorch_core_tensor::DType;
use std::fmt;

/// 逻辑设备标识。
///
/// `backend` 表示后端族，例如 `cpu`、`cuda`、`tank9k`；`ordinal` 表示该族
/// 内部的第几块设备。这里刻意不保存驱动句柄或物理地址，因为 HAL 需要跨
/// 进程、FFI 和 dry-run 计划复用同一套标识。
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DeviceId {
    pub backend: String,
    pub ordinal: usize,
}

impl DeviceId {
    /// 返回标准 CPU 设备。
    ///
    /// CPU 后端在框架里承担“永远可用的参考路径”角色，因此固定为
    /// `cpu:0`，避免测试和 fallback 逻辑因机器环境不同而漂移。
    pub fn cpu() -> Self {
        DeviceId {
            backend: "cpu".into(),
            ordinal: 0,
        }
    }

    /// 返回指定序号的 CUDA 逻辑设备。
    ///
    /// 这里不检查 CUDA runtime 是否真的存在；可用性检查属于具体 Backend
    /// 初始化阶段。这样上层可以先构造拓扑和计划，再决定是否落到真实硬件。
    pub fn cuda(ordinal: usize) -> Self {
        DeviceId {
            backend: "cuda".into(),
            ordinal,
        }
    }

    /// 返回指定序号的 Tank9k 逻辑板卡。
    ///
    /// Tank9k 多板互联会先通过 topology 做 dry-run 验证，所以这里的
    /// `ordinal` 是稳定的规划编号，不要求已经完成串口、DMA 或 bitstream
    /// 点亮。
    pub fn tank9k(ordinal: usize) -> Self {
        DeviceId {
            backend: "tank9k".into(),
            ordinal,
        }
    }
}

impl fmt::Display for DeviceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.backend, self.ordinal)
    }
}

/// HAL 边界上的原始字节缓冲。
///
/// 这里故意只保存 `Vec<u8>` 和设备标识，不保存 shape、stride、dtype 等
/// Tensor 语义。Tensor 语义由 `core-tensor` 管理，HAL 只负责“这段内存
/// 属于哪个设备、如何复制和同步”，从而让外部硬件实现保持尽可能薄。
pub struct RawBuffer {
    pub data: Vec<u8>,
    pub device: DeviceId,
}

impl fmt::Debug for RawBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RawBuffer")
            .field("len", &self.data.len())
            .field("device", &self.device)
            .finish()
    }
}

/// HAL 统一错误类型。
///
/// 上层不应该直接依赖某个驱动的错误码；具体后端需要把设备不存在、分配
/// 失败、dtype 不匹配等情况收敛到这些语义化错误，便于框架做 fallback、
/// 日志记录和硬件健康诊断。
#[derive(Debug)]
pub enum HalError {
    DeviceNotFound(DeviceId),
    AllocationFailed { size: usize, reason: String },
    DeviceMismatch { expected: DeviceId, got: DeviceId },
    DTypeMismatch { expected: DType, got: DType },
    Unsupported(String),
}

impl fmt::Display for HalError {
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

/// 设备侧 KV cache 的布局规格。
///
/// 这个类型刻意放在 HAL，而不是 `nn`：`nn` 关心注意力语义，HAL 关心设备内存
/// 需要多大、每个槽位如何定位、窗口滑动后元数据如何同步。真实 CUDA/Tank9k
/// 后端可以先按这份规格分配连续 K/V buffer，再逐步补增量 attention kernel。
#[derive(Debug, Clone, PartialEq)]
pub struct KvCacheBufferSpec {
    pub device: DeviceId,
    pub dtype: DType,
    pub n_layer: usize,
    pub max_context: usize,
    pub n_head: usize,
    pub head_dim: usize,
}

impl KvCacheBufferSpec {
    /// 创建一份设备 KV cache 规格。
    pub fn new(
        device: DeviceId,
        dtype: DType,
        n_layer: usize,
        max_context: usize,
        n_head: usize,
        head_dim: usize,
    ) -> Self {
        assert!(max_context > 0, "max_context must be positive");
        assert!(n_head > 0, "n_head must be positive");
        assert!(head_dim > 0, "head_dim must be positive");
        Self {
            device,
            dtype,
            n_layer,
            max_context,
            n_head,
            head_dim,
        }
    }

    /// 单个 K 或 V buffer 需要保存的元素数量。
    pub fn elements_per_kind(&self) -> usize {
        self.n_layer * self.max_context * self.n_head * self.head_dim
    }

    /// K/V 两个 buffer 合计的元素数量。
    pub fn total_elements(&self) -> usize {
        self.elements_per_kind() * 2
    }

    /// 按 dtype 计算设备侧 K/V 合计字节数。
    pub fn total_bytes(&self) -> usize {
        self.total_elements() * dtype_size_bytes(self.dtype)
    }

    /// 返回 `[layer, position, head, dim]` 在单个 K 或 V buffer 内的线性元素偏移。
    ///
    /// `position` 是窗口内槽位，不是绝对 token 位置。绝对位置由
    /// [`KvCacheWindowState`] 记录，后端可用它做 RoPE offset 或诊断。
    pub fn element_offset(&self, layer: usize, position: usize, head: usize, dim: usize) -> usize {
        assert!(layer < self.n_layer, "layer out of range");
        assert!(position < self.max_context, "position out of range");
        assert!(head < self.n_head, "head out of range");
        assert!(dim < self.head_dim, "dim out of range");
        (((layer * self.max_context + position) * self.n_head + head) * self.head_dim) + dim
    }
}

/// KV cache 当前窗口的设备侧元数据。
///
/// 真实设备 buffer 往往不会移动整段历史，而是维护一个 ring/write cursor。
/// 这里先不强迫所有后端采用 ring，只把“窗口从完整序列哪个绝对位置开始”
/// 和“当前有效 token 数”固定下来；CUDA、Tank9k、CPU dry-run 都能据此对齐。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvCacheWindowState {
    pub window_start: usize,
    pub cached_len: usize,
    pub max_context: usize,
}

impl KvCacheWindowState {
    /// 创建空窗口状态。
    pub fn new(max_context: usize) -> Self {
        assert!(max_context > 0, "max_context must be positive");
        Self {
            window_start: 0,
            cached_len: 0,
            max_context,
        }
    }

    /// 下一次 append 对应的绝对 token 位置。
    pub fn next_position(&self) -> usize {
        self.window_start + self.cached_len
    }

    /// 当前窗口最后一个 token 的绝对位置。
    pub fn last_position(&self) -> Option<usize> {
        if self.cached_len == 0 {
            None
        } else {
            Some(self.window_start + self.cached_len - 1)
        }
    }

    /// 用一段 prefill 窗口覆盖当前状态。
    pub fn prefill(&mut self, start_position: usize, len: usize) {
        assert!(len <= self.max_context, "prefill len cannot exceed max_context");
        self.window_start = start_position;
        self.cached_len = len;
    }

    /// 追加 `len` 个 token，并在超出窗口时前移 `window_start`。
    pub fn append(&mut self, len: usize) {
        let total = self.cached_len + len;
        let dropped = total.saturating_sub(self.max_context);
        self.window_start += dropped;
        self.cached_len = total.min(self.max_context);
    }
}

/// 设备 KV cache buffer 的最小抽象。
///
/// 这不是 attention kernel 接口，而是“内存和窗口元数据”的底座。后续 CUDA/HAL
/// 可以实现真正的 K/V 写入、ring buffer 和 fence 同步；上层只需要依赖这份
/// 规格即可知道 buffer 是否足够、窗口位置是否正确。
pub trait KvCacheBuffer: Send + Sync + fmt::Debug {
    fn spec(&self) -> &KvCacheBufferSpec;
    fn window_state(&self) -> KvCacheWindowState;
    fn prefill_window(&mut self, start_position: usize, len: usize) -> HalResult<()>;
    fn append_window(&mut self, len: usize) -> HalResult<()>;
}

/// CPU 参考 KV cache buffer。
///
/// 它只分配连续字节并维护窗口元数据，不计算 attention。这个实现的价值是把
/// dtype、容量和滑动窗口规则先跑通，避免 CUDA/Tank9k 后端各自解释一套。
#[derive(Debug)]
pub struct CpuKvCacheBuffer {
    spec: KvCacheBufferSpec,
    storage: RawBuffer,
    window: KvCacheWindowState,
}

impl CpuKvCacheBuffer {
    pub fn new(spec: KvCacheBufferSpec) -> HalResult<Self> {
        let backend = CpuBackend;
        let storage = backend.allocate(spec.total_bytes())?;
        let window = KvCacheWindowState::new(spec.max_context);
        Ok(Self { spec, storage, window })
    }

    /// 返回底层字节容量，用于测试和后端规划校验。
    pub fn storage_len(&self) -> usize {
        self.storage.data.len()
    }
}

impl KvCacheBuffer for CpuKvCacheBuffer {
    fn spec(&self) -> &KvCacheBufferSpec {
        &self.spec
    }

    fn window_state(&self) -> KvCacheWindowState {
        self.window
    }

    fn prefill_window(&mut self, start_position: usize, len: usize) -> HalResult<()> {
        if len > self.spec.max_context {
            return Err(HalError::Unsupported(format!(
                "prefill len {len} exceeds kv cache max_context {}",
                self.spec.max_context
            )));
        }
        self.window.prefill(start_position, len);
        Ok(())
    }

    fn append_window(&mut self, len: usize) -> HalResult<()> {
        self.window.append(len);
        Ok(())
    }
}

fn dtype_size_bytes(dtype: DType) -> usize {
    match dtype {
        DType::F32 => 4,
        DType::F16 | DType::BF16 => 2,
    }
}

/// Minimal hardware fence snapshot shared by HAL, planners and Studio.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FenceState {
    pub phase: String,
    pub queue_depth: u32,
    pub synced: bool,
    pub message: String,
}

/// Minimal queue telemetry for heterogeneous backends.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueueState {
    pub depth: u32,
    pub capacity: u32,
    pub draining: bool,
}

/// HAL 操作的标准返回类型。
pub type HalResult<T> = Result<T, HalError>;

/// 设备后端的最低契约。
///
/// `Backend` 只关心内存生命周期和同步边界，不关心具体算子。这样的拆分
/// 允许一个硬件先实现分配/拷贝/栅栏同步，再逐步补齐 kernel；也方便
/// Tank9k 这类新硬件先跑连通性和 DMA 验证。
pub trait Backend: Send + Sync + 'static {
    /// 返回后端族名称，应与 [`DeviceId::backend`] 保持一致。
    fn name(&self) -> &str;

    /// 返回当前 Backend 管理的逻辑设备。
    fn device_id(&self) -> DeviceId;

    /// 在设备侧分配一段原始缓冲。
    ///
    /// 调用者传入字节数而不是元素数；dtype 到字节数的换算必须在
    /// `core-tensor` 或 op dispatch 层完成，避免 HAL 混入 Tensor 语义。
    fn allocate(&self, size: usize) -> HalResult<RawBuffer>;

    /// 将设备缓冲复制回主机切片。
    ///
    /// 目标切片长度必须与缓冲长度匹配；参考 CPU 后端会让切片复制自然
    /// panic，真实后端应优先返回结构化错误，避免跨 FFI 传播未定义状态。
    fn copy_to_host(&self, buf: &RawBuffer, dst: &mut [u8]) -> HalResult<()>;

    /// 将主机切片写入设备缓冲。
    ///
    /// 这个方法是 host 到 device 的唯一通用入口，未来接 DMA 时也应在这里
    /// 保持“写入完成后由 synchronize 建立可见性”的约束。
    fn copy_from_host(&self, src: &[u8], buf: &mut RawBuffer) -> HalResult<()>;

    /// 等待当前设备队列中已提交任务完成。
    ///
    /// CPU 后端可以是空操作；异构设备必须把它实现为真实 fence 或队列
    /// drain，否则双缓冲 swap 可能在 kernel 尚未读完旧指针时发生。
    fn synchronize(&self) -> HalResult<()>;
}

/// 框架当前要求硬件后端提供的 F32 数值内核集合。
///
/// 这些方法以切片形式暴露，是为了把调度、shape 检查和 Tensor view 处理
/// 留在上层；具体 Backend 只需要执行已经展平后的连续数据。所有切片长度
/// 约束都由调用者保证，CPU 实现作为参考语义，不在热路径重复检查。
pub trait KernelProvider: Backend {
    // ---- element-wise binary ----
    /// 逐元素加法：`out[i] = a[i] + b[i]`。
    fn add_f32(&self, a: &[f32], b: &[f32], out: &mut [f32]);

    /// 逐元素乘法：`out[i] = a[i] * b[i]`。
    fn mul_f32(&self, a: &[f32], b: &[f32], out: &mut [f32]);

    // ---- element-wise unary ----
    /// 逐元素取负：`out[i] = -a[i]`。
    fn neg_f32(&self, a: &[f32], out: &mut [f32]);

    /// 逐元素指数函数。
    fn exp_f32(&self, a: &[f32], out: &mut [f32]);

    /// 逐元素自然对数；调用者需要保证输入位于有效定义域。
    fn log_f32(&self, a: &[f32], out: &mut [f32]);

    /// ReLU 激活：负数截断为 0，非负值保持不变。
    fn relu_f32(&self, a: &[f32], out: &mut [f32]);

    /// GELU 近似实现，使用 tanh 形式以匹配常见 Transformer 推理路径。
    fn gelu_f32(&self, a: &[f32], out: &mut [f32]);

    /// 标量缩放，常用于梯度累积后的平均化或混合精度 scale 调整。
    fn scale_f32(&self, a: &[f32], scalar: f32, out: &mut [f32]);

    // ---- matmul ----
    /// 矩阵乘法：`[m, k] @ [k, n] -> [m, n]`，输入均为 row-major 连续布局。
    fn matmul_f32(&self, a: &[f32], b: &[f32], out: &mut [f32], m: usize, k: usize, n: usize);

    /// 批量矩阵乘法，每个 batch 独立执行 `[m, k] @ [k, n]`。
    ///
    /// 这里先复用单个 matmul 的语义，未来硬件后端可以替换成真正的 batched
    /// kernel，但输出布局仍必须保持 `batch * m * n` 的连续 row-major 约定。
    #[allow(clippy::too_many_arguments)]
    fn batch_matmul_f32(&self, a: &[f32], b: &[f32], out: &mut [f32], batch: usize, m: usize, k: usize, n: usize);

    // ---- reduction ----
    /// 对连续切片求和；并行后端需要注意归约顺序会带来轻微浮点误差差异。
    fn sum_f32(&self, a: &[f32]) -> f32;

    /// 按行 softmax，`rows * cols` 必须覆盖完整输入输出。
    ///
    /// CPU 参考实现使用减最大值技巧保证数值稳定，硬件后端也应保留这个
    /// 语义，否则大 logits 在推理时容易溢出。
    fn softmax_f32(&self, a: &[f32], out: &mut [f32], rows: usize, cols: usize);

    // ---- misc ----
    /// 按布尔 mask 写入填充值，常用于 attention mask。
    fn masked_fill_f32(&self, a: &[f32], mask: &[bool], fill_value: f32, out: &mut [f32]);

    /// 简化版 broadcast add：`b` 沿 `a` 的线性维度周期性复用。
    fn broadcast_add_f32(&self, a: &[f32], b: &[f32], out: &mut [f32], a_len: usize, b_len: usize);

    /// Embedding 查表，`weight` 按 `[vocab, dim]` row-major 排布。
    ///
    /// `_vocab` 当前只保留为 ABI/语义提示；调用者必须保证 indices 不越界。
    fn embedding_lookup_f32(&self, weight: &[f32], indices: &[usize], out: &mut [f32], vocab: usize, dim: usize);

    // ---- optimizer kernels ----
    /// 原地 SGD 更新：`param -= lr * grad`。
    fn sgd_update_f32(&self, params: &mut [f32], grad: &[f32], lr: f32);

    /// 原地 AdamW 更新。
    ///
    /// `bc1` / `bc2` 是调用者预先计算好的 bias correction 分母，避免后端
    /// 需要理解 optimizer step 计数。`weight_decay` 采用 decoupled 形式，
    /// 先对参数做衰减，再应用 Adam 动量更新。
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

/// CPU 参考后端。
///
/// 该实现使用普通 Rust 切片和循环，目标是语义透明、便于测试。任何新硬件
/// 后端都应该先跑过这些测试，再比较允许范围内的浮点误差。
pub struct CpuBackend;

impl Backend for CpuBackend {
    fn name(&self) -> &str {
        "cpu"
    }

    fn device_id(&self) -> DeviceId {
        DeviceId::cpu()
    }

    fn allocate(&self, size: usize) -> HalResult<RawBuffer> {
        Ok(RawBuffer {
            data: vec![0u8; size],
            device: DeviceId::cpu(),
        })
    }

    fn copy_to_host(&self, buf: &RawBuffer, dst: &mut [u8]) -> HalResult<()> {
        dst.copy_from_slice(&buf.data);
        Ok(())
    }

    fn copy_from_host(&self, src: &[u8], buf: &mut RawBuffer) -> HalResult<()> {
        buf.data.copy_from_slice(src);
        Ok(())
    }

    fn synchronize(&self) -> HalResult<()> {
        Ok(())
    }
}

impl KernelProvider for CpuBackend {
    fn add_f32(&self, a: &[f32], b: &[f32], out: &mut [f32]) {
        for i in 0..out.len() {
            out[i] = a[i] + b[i];
        }
    }

    fn mul_f32(&self, a: &[f32], b: &[f32], out: &mut [f32]) {
        for i in 0..out.len() {
            out[i] = a[i] * b[i];
        }
    }

    fn neg_f32(&self, a: &[f32], out: &mut [f32]) {
        for i in 0..out.len() {
            out[i] = -a[i];
        }
    }

    fn exp_f32(&self, a: &[f32], out: &mut [f32]) {
        for i in 0..out.len() {
            out[i] = a[i].exp();
        }
    }

    fn log_f32(&self, a: &[f32], out: &mut [f32]) {
        for i in 0..out.len() {
            out[i] = a[i].ln();
        }
    }

    fn relu_f32(&self, a: &[f32], out: &mut [f32]) {
        for i in 0..out.len() {
            out[i] = if a[i] > 0.0 { a[i] } else { 0.0 };
        }
    }

    fn gelu_f32(&self, a: &[f32], out: &mut [f32]) {
        for i in 0..out.len() {
            let x = a[i];
            out[i] = 0.5 * x * (1.0 + ((2.0_f32 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh());
        }
    }

    fn scale_f32(&self, a: &[f32], scalar: f32, out: &mut [f32]) {
        for i in 0..out.len() {
            out[i] = a[i] * scalar;
        }
    }

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

    fn sum_f32(&self, a: &[f32]) -> f32 {
        a.iter().sum()
    }

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

    fn masked_fill_f32(&self, a: &[f32], mask: &[bool], fill_value: f32, out: &mut [f32]) {
        for i in 0..a.len() {
            out[i] = if mask[i] { fill_value } else { a[i] };
        }
    }

    fn broadcast_add_f32(&self, a: &[f32], b: &[f32], out: &mut [f32], a_len: usize, b_len: usize) {
        for i in 0..a_len {
            out[i] = a[i] + b[i % b_len];
        }
    }

    fn embedding_lookup_f32(&self, weight: &[f32], indices: &[usize], out: &mut [f32], _vocab: usize, dim: usize) {
        for (i, &idx) in indices.iter().enumerate() {
            out[i * dim..(i + 1) * dim].copy_from_slice(&weight[idx * dim..(idx + 1) * dim]);
        }
    }

    fn sgd_update_f32(&self, params: &mut [f32], grad: &[f32], lr: f32) {
        for (w, g) in params.iter_mut().zip(grad.iter()) {
            *w -= lr * g;
        }
    }

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

    // 这组 CPU 测试定义了 HAL 的“金标准”数值语义，外部硬件后端应以它为对照。

    #[test]
    fn test_fence_and_queue_state_types() {
        let fence = FenceState {
            phase: "WaitFence".into(),
            queue_depth: 7,
            synced: false,
            message: "draining kernel queue".into(),
        };
        let queue = QueueState {
            depth: 7,
            capacity: 16,
            draining: true,
        };
        assert_eq!(fence.phase, "WaitFence");
        assert_eq!(queue.depth, fence.queue_depth);
        assert!(queue.draining);
    }

    #[test]
    fn test_kv_cache_buffer_spec_layout_and_capacity() {
        let spec = KvCacheBufferSpec::new(DeviceId::cuda(0), DType::BF16, 2, 4, 3, 8);

        assert_eq!(spec.elements_per_kind(), 2 * 4 * 3 * 8);
        assert_eq!(spec.total_elements(), 2 * 2 * 4 * 3 * 8);
        assert_eq!(spec.total_bytes(), 2 * 2 * 4 * 3 * 8 * 2);
        assert_eq!(spec.element_offset(1, 2, 1, 3), (((4 + 2) * 3 + 1) * 8) + 3);
    }

    #[test]
    fn test_kv_cache_window_state_prefill_and_append() {
        let mut window = KvCacheWindowState::new(3);
        assert_eq!(window.next_position(), 0);
        assert_eq!(window.last_position(), None);

        window.prefill(5, 2);
        assert_eq!(window.window_start, 5);
        assert_eq!(window.cached_len, 2);
        assert_eq!(window.last_position(), Some(6));
        assert_eq!(window.next_position(), 7);

        window.append(2);
        assert_eq!(window.window_start, 6);
        assert_eq!(window.cached_len, 3);
        assert_eq!(window.last_position(), Some(8));
        assert_eq!(window.next_position(), 9);
    }

    #[test]
    fn test_cpu_kv_cache_buffer_tracks_capacity_and_window() {
        let spec = KvCacheBufferSpec::new(DeviceId::cpu(), DType::F32, 1, 3, 2, 4);
        let mut buffer = CpuKvCacheBuffer::new(spec.clone()).unwrap();

        assert_eq!(buffer.spec(), &spec);
        assert_eq!(buffer.storage_len(), spec.total_bytes());
        assert_eq!(buffer.window_state(), KvCacheWindowState::new(3));

        buffer.prefill_window(10, 3).unwrap();
        buffer.append_window(1).unwrap();
        assert_eq!(
            buffer.window_state(),
            KvCacheWindowState {
                window_start: 11,
                cached_len: 3,
                max_context: 3,
            }
        );
    }

    #[test]
    fn test_cpu_backend_add() {
        let backend = CpuBackend;
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![4.0f32, 5.0, 6.0];
        let mut out = vec![0.0f32; 3];
        backend.add_f32(&a, &b, &mut out);
        assert_eq!(out, vec![5.0, 7.0, 9.0]);
    }

    // 乘法是 optimizer 和激活函数组合中的基础路径，先锁定逐元素行为。
    #[test]
    fn test_cpu_backend_mul() {
        let backend = CpuBackend;
        let a = vec![2.0f32, 3.0];
        let b = vec![4.0f32, 5.0];
        let mut out = vec![0.0f32; 2];
        backend.mul_f32(&a, &b, &mut out);
        assert_eq!(out, vec![8.0, 15.0]);
    }

    // MatMul 的 row-major 下标约定必须稳定，否则 GPU/Tank9k kernel 会和 CPU fallback 对不上。
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

    // 分配测试确保 RawBuffer 不偷偷携带 Tensor 语义，只记录字节长度和设备归属。
    #[test]
    fn test_cpu_backend_allocate() {
        let backend = CpuBackend;
        let buf = backend.allocate(16).unwrap();
        assert_eq!(buf.data.len(), 16);
        assert_eq!(buf.device, DeviceId::cpu());
    }

    // unary kernel 的符号行为会影响 autograd 的反向校验。
    #[test]
    fn test_cpu_backend_neg() {
        let backend = CpuBackend;
        let a = vec![1.0f32, -2.0, 3.0];
        let mut out = vec![0.0f32; 3];
        backend.neg_f32(&a, &mut out);
        assert_eq!(out, vec![-1.0, 2.0, -3.0]);
    }

    // exp/log 的往返误差给后续设备实现一个可接受的数值精度参照。
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

    // ReLU 的 0 点语义需要固定，避免不同后端对负零或边界值处理不一致。
    #[test]
    fn test_cpu_backend_relu() {
        let backend = CpuBackend;
        let a = vec![-1.0f32, 0.0, 2.0, -0.5];
        let mut out = vec![0.0f32; 4];
        backend.relu_f32(&a, &mut out);
        assert_eq!(out, vec![0.0, 0.0, 2.0, 0.0]);
    }

    // GELU 使用 tanh 近似，这个测试避免后端误换成 erf 版本导致数值基线漂移。
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

    // scale 是梯度累积和混合精度监控会频繁使用的小 kernel，语义必须简单可预测。
    #[test]
    fn test_cpu_backend_scale() {
        let backend = CpuBackend;
        let a = vec![1.0f32, 2.0, 3.0];
        let mut out = vec![0.0f32; 3];
        backend.scale_f32(&a, 0.5, &mut out);
        assert_eq!(out, vec![0.5, 1.0, 1.5]);
    }

    // softmax 测试锁定“每行归一化为概率分布”的语义，而不是只检查不崩溃。
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

    // batch matmul 先验证逐 batch 切片边界，真实 batched kernel 必须保持同样布局。
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

    // SGD 更新是最小 optimizer kernel，后续设备端优化器要和这里逐元素一致。
    #[test]
    fn test_cpu_backend_sgd_update() {
        let backend = CpuBackend;
        let mut params = vec![1.0f32, 2.0];
        let grad = vec![0.1, 0.2];
        backend.sgd_update_f32(&mut params, &grad, 0.1);
        assert!((params[0] - 0.99).abs() < 1e-6);
        assert!((params[1] - 1.98).abs() < 1e-6);
    }

    // Embedding 查表固定 row-major `[vocab, dim]` 约定，便于后续 DMA 分块读取。
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

    // masked_fill 是 attention mask 的基础语义，true 表示写入 fill_value。
    #[test]
    fn test_cpu_backend_masked_fill() {
        let backend = CpuBackend;
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let mask = vec![false, true, false, true];
        let mut out = vec![0.0f32; 4];
        backend.masked_fill_f32(&a, &mask, -999.0, &mut out);
        assert_eq!(out, vec![1.0, -999.0, 3.0, -999.0]);
    }

    // 当前 broadcast_add 采用一维周期复用，测试防止未来误解成完整 numpy broadcast。
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

//! SPTorch 的核心张量库。
//!
//! 这里定义框架最底层的 [`Tensor`]：它同时携带存储位置、shape/stride 视图、
//! dtype 信息和 autograd 计算图边。上层算子、优化器、序列化、HAL 和在线
//! 进化都围绕这个类型展开，所以本 crate 更关注“语义稳定”和“边界清晰”，
//! 而不是把所有高性能 kernel 都塞进来。
//!
//! 主要职责：
//! - 统一 CPU / Device 存储抽象。
//! - 表达 shape、strides、offset，支持非连续 view 被拉平成连续数据。
//! - 提供 F32/F16/BF16 的轻量转换工具。
//! - 维护最小 autograd 图，并用迭代队列执行反向传播。
//! - 提供全局后端注册表，让算子可以按设备分发到外部后端。

use std::cell::Cell;
use std::collections::HashMap;
use std::collections::VecDeque;
use std::fmt;
use std::sync::OnceLock;
use std::sync::{Arc, RwLock};

use thiserror::Error;

thread_local! {
    static GRAD_ENABLED: Cell<bool> = const { Cell::new(true) };
}

/// 返回当前线程是否允许新算子挂接 autograd 计算图。
///
/// 这个开关只影响“新产生的 Tensor 是否记录 creator”，不会修改已经存在的
/// `requires_grad` 标记。这样可以模拟 PyTorch 的 `no_grad` 语义：参数仍然是
/// 可训练参数，但在推理或权重检查路径里不会额外挂图。
pub fn is_grad_enabled() -> bool {
    GRAD_ENABLED.with(Cell::get)
}

/// 临时切换当前线程的梯度记录模式，离开作用域时自动恢复。
///
/// 这是底层守卫；普通调用者优先使用 [`no_grad`]，需要手动跨多个语句控制时
/// 再持有该 guard。使用线程局部状态是为了避免一个训练 worker 的推理检查影响
/// 其他 worker 的 autograd 行为。
pub struct GradModeGuard {
    previous: bool,
}

impl Drop for GradModeGuard {
    fn drop(&mut self) {
        GRAD_ENABLED.with(|enabled| enabled.set(self.previous));
    }
}

/// 设置当前线程的梯度记录模式，并返回可恢复旧状态的 guard。
pub fn set_grad_enabled(enabled: bool) -> GradModeGuard {
    let previous = GRAD_ENABLED.with(|state| {
        let previous = state.get();
        state.set(enabled);
        previous
    });
    GradModeGuard { previous }
}

/// 在不记录新 autograd 节点的模式下执行闭包。
///
/// 这不是“关闭参数训练”，而是“本段计算不参与反向图”。它适合验证集推理、
/// 指标计算、权重导出前的探针计算等场景。
pub fn no_grad<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    let _guard = set_grad_enabled(false);
    f()
}

// ============ Global Backend Registry ============

/// 注册到 `core-tensor` 的设备侧计算后端。
///
/// 这个 trait 是早期轻量 dispatch 面，供 `core-ops` 在发现张量位于非 CPU
/// 设备时调用。它只暴露当前核心算子需要的连续 F32 kernel；更完整的硬件
/// 抽象位于 `sptorch-hal`，两者会在后续硬件主线中逐步收敛。
pub trait BackendDispatch: Send + Sync {
    /// 逐元素加法 kernel。
    fn add_f32(&self, a: &[f32], b: &[f32], out: &mut [f32]);
    /// 逐元素乘法 kernel。
    fn mul_f32(&self, a: &[f32], b: &[f32], out: &mut [f32]);
    /// 逐元素取负 kernel。
    fn neg_f32(&self, a: &[f32], out: &mut [f32]);
    /// 逐元素指数 kernel。
    fn exp_f32(&self, a: &[f32], out: &mut [f32]);
    /// 逐元素自然对数 kernel。
    fn log_f32(&self, a: &[f32], out: &mut [f32]);
    /// ReLU 激活 kernel。
    fn relu_f32(&self, a: &[f32], out: &mut [f32]);
    /// GELU 激活 kernel。
    fn gelu_f32(&self, a: &[f32], out: &mut [f32]);
    /// 标量缩放 kernel。
    fn scale_f32(&self, a: &[f32], scalar: f32, out: &mut [f32]);
    /// Row-major 矩阵乘法 kernel：`[m, k] @ [k, n] -> [m, n]`。
    fn matmul_f32(&self, a: &[f32], b: &[f32], out: &mut [f32], m: usize, k: usize, n: usize);
}

static BACKEND_REGISTRY: OnceLock<RwLock<HashMap<Device, Arc<dyn BackendDispatch>>>> = OnceLock::new();

/// 懒初始化全局后端注册表。
///
/// 注册表本身很小，用 `RwLock` 足够：注册发生在启动或测试阶段，热路径主要
/// 是读。返回 `Arc<dyn BackendDispatch>` 可以让算子在释放注册表读锁后继续
/// 调用后端，避免把全局锁带进 kernel 执行过程。
fn registry() -> &'static RwLock<HashMap<Device, Arc<dyn BackendDispatch>>> {
    BACKEND_REGISTRY.get_or_init(|| RwLock::new(HashMap::new()))
}

/// 为指定逻辑设备注册计算后端。
///
/// 后注册的实现会覆盖同一 [`Device`] 的旧实现，这让测试可以替换 mock
/// backend，也让运行时可以在硬件重新初始化后刷新 dispatch 目标。
pub fn register_backend(device: Device, backend: Arc<dyn BackendDispatch>) {
    registry().write().unwrap().insert(device, backend);
}

/// 查询指定设备的已注册后端。
///
/// 未注册时返回 `None`，调用方通常回退到 CPU 实现。这里克隆的是 `Arc`，
/// 不是后端对象本身。
pub fn get_backend(device: &Device) -> Option<Arc<dyn BackendDispatch>> {
    registry().read().unwrap().get(device).cloned()
}

/// Tensor 层统一错误。
#[derive(Error, Debug)]
pub enum TensorError {
    #[error("shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch { expected: Vec<usize>, got: Vec<usize> },

    #[error("device mismatch: expected {expected:?}, got {got:?}")]
    DeviceMismatch { expected: Device, got: Device },

    #[error("dtype mismatch: expected {expected:?}, got {got:?}")]
    DTypeMismatch { expected: DType, got: DType },

    #[error("invalid shape: {0}")]
    InvalidShape(String),

    #[error("lock poisoned")]
    LockPoisoned,

    #[error("device storage error: {0}")]
    DeviceError(String),
}

pub type Result<T> = std::result::Result<T, TensorError>;

/// Autograd 算子的反向传播接口。
///
/// `grad_output` 是当前节点输出收到的上游梯度，返回值按输入顺序给出每个
/// 输入的梯度。`None` 表示该输入不需要梯度或该路径不可导。
pub trait Op: std::fmt::Debug + Send + Sync {
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>>;
}

/// 计算图节点。
///
/// 每个由可导算子产生的 Tensor 会持有一个 `Node`，其中 `op` 保存局部反向
/// 公式，`inputs` 保存图边。这里用 `Arc<Node>` 共享节点，避免 Tensor clone
/// 时复制整张计算图。
#[derive(Debug)]
pub struct Node {
    pub op: Box<dyn Op>,
    pub inputs: Vec<Tensor>,
}

/// 张量所在的逻辑设备。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Device {
    CPU,
    Cuda(usize),
    Custom(u16),
}

/// 张量数值类型。
///
/// 当前存储仍以 `Vec<f32>` 承载，`F16/BF16` 通过 round-trip 模拟精度损失。
/// 这让 dtype 语义先进入框架 API，真实半精度设备存储可在后续 HAL 路径中
/// 逐步替换。
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum DType {
    #[default]
    F32,
    F16,
    BF16,
}

// ============ F16/BF16 conversion utilities ============

/// 将 `f32` 量化为 IEEE754 half (`f16`) 的位表示。
///
/// 这是软件转换工具，用于测试和 dtype round-trip，不追求完全覆盖硬件
/// 指令的舍入模式。Inf/NaN、溢出和下溢按常见 half 语义处理。
pub fn f32_to_f16(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7FFFFF;

    if exp == 255 {
        // Inf/NaN
        return (sign | 0x7C00 | (mant >> 13).min(1)) as u16;
    }

    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        return (sign | 0x7C00) as u16; // overflow -> Inf
    }
    if new_exp <= 0 {
        if new_exp < -10 {
            return sign as u16; // underflow -> 0
        }
        let m = (mant | 0x800000) >> (1 - new_exp);
        return (sign | (m >> 13)) as u16;
    }

    (sign | ((new_exp as u32) << 10) | (mant >> 13)) as u16
}

/// 将 IEEE754 half (`f16`) 位表示还原为 `f32`。
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;

    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign << 31);
        }
        let mut m = mant;
        let mut e = 0i32;
        while (m & 0x400) == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x3FF;
        let f32_exp = (127 - 15 + 1 + e) as u32;
        return f32::from_bits((sign << 31) | (f32_exp << 23) | (m << 13));
    }
    if exp == 31 {
        return f32::from_bits((sign << 31) | (0xFF << 23) | (mant << 13));
    }

    let f32_exp = exp + 127 - 15;
    f32::from_bits((sign << 31) | (f32_exp << 23) | (mant << 13))
}

/// 将 `f32` 转换为 `bf16` 位表示。
///
/// BF16 保留高 16 位，因此指数范围接近 F32，但尾数精度更低。当前实现不做
/// 额外舍入，适合表达“精度截断”而非严格硬件数值复现。
pub fn f32_to_bf16(val: f32) -> u16 {
    (val.to_bits() >> 16) as u16
}

/// 将 `bf16` 位表示还原为 `f32`。
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// 不透明设备内存缓冲。
///
/// Tensor 层不理解 GPU/NPU/Tank9k 的真实指针，只要求设备缓冲能报告设备、
/// 元素数量，并能在需要时拷回主机。这是 `core-tensor` 和硬件后端之间的
/// 最小边界。
pub trait DeviceBuffer: Send + Sync + fmt::Debug {
    /// 返回缓冲所属逻辑设备。
    fn device(&self) -> Device;

    /// 返回缓冲中 F32 元素数量。
    fn len(&self) -> usize;

    /// 判断缓冲是否为空。
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// 将设备侧数据拷贝回主机。
    fn to_host(&self) -> Vec<f32>;

    /// 从主机数据创建指定设备上的缓冲。
    fn from_host(data: &[f32], device: Device) -> std::result::Result<Box<dyn DeviceBuffer>, String>
    where
        Self: Sized;
}

/// 张量底层存储。
///
/// `Storage` 只描述数据在哪里，不描述 shape/stride；同一段存储可以被不同
/// Tensor view 以不同 shape/offset 解释。
pub enum Storage {
    Cpu(Vec<f32>),
    Device(Box<dyn DeviceBuffer>),
}

impl Storage {
    /// 用主机内存构造 CPU 存储。
    pub fn cpu(data: Vec<f32>) -> Self {
        Storage::Cpu(data)
    }

    /// 判断当前存储是否位于 CPU。
    pub fn is_cpu(&self) -> bool {
        matches!(self, Storage::Cpu(_))
    }

    /// 以不可变切片借用 CPU 存储；设备存储会 panic。
    ///
    /// 这是内部热路径便捷接口。对外或不确定存储位置时，优先使用
    /// [`Self::try_as_cpu_slice`] 或 [`Self::to_cpu_vec`]。
    pub fn as_cpu_slice(&self) -> &[f32] {
        match self {
            Storage::Cpu(v) => v,
            Storage::Device(_) => panic!("cannot borrow device storage as CPU slice; call to_cpu_vec() first"),
        }
    }

    /// 尝试以不可变切片借用 CPU 存储；设备存储返回错误。
    pub fn try_as_cpu_slice(&self) -> Result<&[f32]> {
        match self {
            Storage::Cpu(v) => Ok(v),
            Storage::Device(_) => Err(TensorError::DeviceError(
                "cannot borrow device storage as CPU slice; call to_cpu_vec() first".into(),
            )),
        }
    }

    /// 以可变切片借用 CPU 存储；设备存储会 panic。
    pub fn as_cpu_slice_mut(&mut self) -> &mut [f32] {
        match self {
            Storage::Cpu(v) => v,
            Storage::Device(_) => panic!("cannot mutably borrow device storage as CPU slice"),
        }
    }

    /// 尝试以可变切片借用 CPU 存储；设备存储返回错误。
    pub fn try_as_cpu_slice_mut(&mut self) -> Result<&mut [f32]> {
        match self {
            Storage::Cpu(v) => Ok(v),
            Storage::Device(_) => Err(TensorError::DeviceError(
                "cannot mutably borrow device storage as CPU slice".into(),
            )),
        }
    }

    /// 将任意存储导出为 CPU `Vec<f32>`。
    ///
    /// 对 CPU 存储是 clone；对设备存储会触发 host transfer。调用方如果在热
    /// 路径频繁调用，需要意识到这可能隐藏一次昂贵的数据搬运。
    pub fn to_cpu_vec(&self) -> Vec<f32> {
        match self {
            Storage::Cpu(v) => v.clone(),
            Storage::Device(buf) => buf.to_host(),
        }
    }

    /// 返回存储元素数量。
    pub fn len(&self) -> usize {
        match self {
            Storage::Cpu(v) => v.len(),
            Storage::Device(buf) => buf.len(),
        }
    }

    /// 判断存储是否为空。
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// 返回存储所在逻辑设备。
    pub fn device(&self) -> Device {
        match self {
            Storage::Cpu(_) => Device::CPU,
            Storage::Device(buf) => buf.device(),
        }
    }
}

impl fmt::Debug for Storage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Storage::Cpu(v) => f.debug_tuple("Cpu").field(v).finish(),
            Storage::Device(buf) => f.debug_tuple("Device").field(buf).finish(),
        }
    }
}

/// Tensor 的可变内部状态。
///
/// 外层 [`Tensor`] 用 `Arc<RwLock<TensorInner>>` 包裹它，使 clone 后的 Tensor
/// 共享同一份元数据和梯度槽。`storage` 也单独放在 `Arc<RwLock<_>>` 中，
/// 方便 view 或双缓冲场景在不复制数据的情况下共享底层存储。
#[derive(Debug)]
pub struct TensorInner {
    pub storage: Arc<RwLock<Storage>>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub offset: usize,
    pub dtype: DType,
    pub device: Device,
    pub requires_grad: bool,
    pub grad: Option<Tensor>,
    pub creator: Option<Arc<Node>>,
}

/// 框架核心张量类型。
///
/// `Tensor` clone 是浅拷贝：它复制 `Arc`，不复制底层数据。需要真实连续数据
/// 副本时，应显式调用 [`Tensor::contiguous_data`]。
#[derive(Clone)]
pub struct Tensor(pub Arc<RwLock<TensorInner>>);

impl Tensor {
    /// 根据数据和形状创建 F32 CPU Tensor，默认不追踪梯度。
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let strides = compute_strides(&shape);
        let inner = TensorInner {
            storage: Arc::new(RwLock::new(Storage::cpu(data))),
            shape,
            strides,
            offset: 0,
            dtype: DType::F32,
            device: Device::CPU,
            requires_grad: false,
            grad: None,
            creator: None,
        };
        Tensor(Arc::new(RwLock::new(inner)))
    }

    /// 创建张量并显式设置 `requires_grad`。
    pub fn with_grad(data: Vec<f32>, shape: Vec<usize>, requires_grad: bool) -> Self {
        let t = Self::new(data, shape);
        t.0.write().unwrap().requires_grad = requires_grad;
        t
    }

    /// 返回张量数据的 CPU 拷贝。
    ///
    /// 该方法会忽略 stride 视图语义，直接导出底层 storage；需要逻辑顺序数据
    /// 时应使用 [`Self::contiguous_data`]。历史上很多测试用的是连续张量，
    /// 因此保留这个轻量接口。
    pub fn data(&self) -> Vec<f32> {
        self.0.read().unwrap().storage.read().unwrap().to_cpu_vec()
    }

    /// 返回张量形状。
    pub fn shape(&self) -> Vec<usize> {
        self.0.read().unwrap().shape.clone()
    }

    /// 返回张量 stride。
    pub fn strides(&self) -> Vec<usize> {
        self.0.read().unwrap().strides.clone()
    }

    /// 返回张量视图在底层 storage 中的逻辑起始偏移。
    pub fn offset(&self) -> usize {
        self.0.read().unwrap().offset
    }

    /// 返回逻辑元素总数，即 shape 各维乘积。
    pub fn numel(&self) -> usize {
        self.shape().iter().product()
    }

    /// 返回张量秩，也就是 shape 维度数量。
    pub fn rank(&self) -> usize {
        self.0.read().unwrap().shape.len()
    }

    /// 标量在本框架里用 `[1]` 表示；后续若引入零维张量，这里可以兼容扩展。
    pub fn is_scalar(&self) -> bool {
        self.numel() == 1
    }

    /// 判断两个张量是否具有完全相同的 shape。
    pub fn same_shape(&self, other: &Tensor) -> bool {
        self.shape() == other.shape()
    }

    /// 返回张量所在逻辑设备。
    pub fn device(&self) -> Device {
        self.0.read().unwrap().device
    }

    /// 返回张量 dtype。
    pub fn dtype(&self) -> DType {
        self.0.read().unwrap().dtype
    }

    /// 转换到目标 dtype，shape 保持不变。
    ///
    /// 目标 dtype 与当前一致时返回浅 clone；否则会先取逻辑连续数据，再做
    /// F16/BF16 round-trip 以模拟精度变化。
    pub fn to_dtype(&self, target: DType) -> Tensor {
        let current = self.dtype();
        if current == target {
            return self.clone();
        }
        let f32_data = self.contiguous_data();
        let shape = self.shape();

        let stored: Vec<f32> = match target {
            DType::F32 => f32_data,
            DType::F16 => {
                // 通过 f16 往返模拟半精度存储带来的精度损失。
                f32_data.iter().map(|&v| f16_to_f32(f32_to_f16(v))).collect()
            }
            DType::BF16 => f32_data.iter().map(|&v| bf16_to_f32(f32_to_bf16(v))).collect(),
        };

        let t = Tensor::new(stored, shape);
        t.0.write().unwrap().dtype = target;
        t
    }

    /// 转换为 `F16`。
    pub fn half(&self) -> Tensor {
        self.to_dtype(DType::F16)
    }

    /// 转换为 `BF16`。
    pub fn bfloat16(&self) -> Tensor {
        self.to_dtype(DType::BF16)
    }

    /// 转换为 `F32`。
    pub fn float(&self) -> Tensor {
        self.to_dtype(DType::F32)
    }

    /// 迁移到目标逻辑设备。
    ///
    /// 当前实现是“主机拷贝语义”：先取连续 CPU 数据，再创建新 Tensor 并标记
    /// device。真正的设备分配与数据上传会由后续 HAL/Backend 路径接管。
    pub fn to_device(&self, target: Device) -> Tensor {
        let current = self.device();
        if current == target {
            return self.clone();
        }
        let data = self.contiguous_data();
        let shape = self.shape();
        let t = Tensor::new(data, shape);
        {
            let mut inner = t.0.write().unwrap();
            inner.device = target;
            inner.requires_grad = self.requires_grad();
        }
        t
    }

    /// 迁移到默认 CUDA 设备 `Cuda(0)`。
    pub fn cuda(&self) -> Tensor {
        self.to_device(Device::Cuda(0))
    }

    /// 迁移到 CPU。
    pub fn cpu(&self) -> Tensor {
        self.to_device(Device::CPU)
    }

    /// 判断张量是否为 row-major 连续布局。
    pub fn is_contiguous(&self) -> bool {
        let inner = self.0.read().unwrap();
        inner.strides == compute_strides(&inner.shape)
    }

    /// 按逻辑索引顺序提取连续数据。
    ///
    /// 对连续张量直接导出 storage；对非连续 view，会根据 `offset + Σ index_i *
    /// stride_i` 计算物理位置并重新排列。这是算子进入 kernel 前最重要的
    /// 数据规范化入口。
    pub fn contiguous_data(&self) -> Vec<f32> {
        let inner = self.0.read().unwrap();
        let storage = inner.storage.read().unwrap();
        let expected_strides = compute_strides(&inner.shape);
        if inner.offset == 0 && inner.strides == expected_strides {
            return storage.to_cpu_vec();
        }
        let cpu_data = storage.to_cpu_vec();
        let numel: usize = inner.shape.iter().product();
        let ndim = inner.shape.len();
        let mut result = Vec::with_capacity(numel);
        let mut indices = vec![0usize; ndim];
        for _ in 0..numel {
            let physical: usize = inner.offset
                + indices
                    .iter()
                    .zip(inner.strides.iter())
                    .map(|(i, s)| i * s)
                    .sum::<usize>();
            result.push(cpu_data[physical]);
            for d in (0..ndim).rev() {
                indices[d] += 1;
                if indices[d] < inner.shape[d] {
                    break;
                }
                indices[d] = 0;
            }
        }
        result
    }

    /// 返回是否追踪梯度。
    pub fn requires_grad(&self) -> bool {
        self.0.read().unwrap().requires_grad
    }

    /// 显式切换当前张量是否接收梯度。
    ///
    /// 关闭时会同时清空累计梯度和 creator，避免调用者误把旧图上的中间 Tensor
    /// 当作新的叶子参数继续训练。开启时只允许未来的算子把它作为可训练输入，
    /// 不会恢复已经丢弃的历史计算图。
    pub fn set_requires_grad(&self, enabled: bool) {
        let mut inner = self.0.write().unwrap();
        inner.requires_grad = enabled;
        if !enabled {
            inner.grad = None;
            inner.creator = None;
        }
    }

    /// 返回一个共享数据但与当前计算图断开的 Tensor。
    ///
    /// detach 后的 Tensor 保留 shape、stride、offset、dtype、device 与 storage，
    /// 但 `requires_grad=false` 且没有 creator。共享 storage 可以避免大权重复制；
    /// 如果后续需要写时隔离，应在更高层引入 clone/copy 语义。
    pub fn detach(&self) -> Tensor {
        let inner = self.0.read().unwrap();
        Tensor(Arc::new(RwLock::new(TensorInner {
            storage: inner.storage.clone(),
            shape: inner.shape.clone(),
            strides: inner.strides.clone(),
            offset: inner.offset,
            dtype: inner.dtype,
            device: inner.device,
            requires_grad: false,
            grad: None,
            creator: None,
        })))
    }

    /// 清空当前张量的累计梯度。
    pub fn zero_grad(&self) {
        self.0.write().unwrap().grad = None;
    }

    /// 将传入梯度累加到当前张量的梯度槽。
    ///
    /// 同一个叶子节点可能通过多条路径影响 loss，因此这里必须累加而不是
    /// 覆盖。传入梯度会先按逻辑连续顺序展开，避免 view 的 stride 影响梯度
    /// 存储格式。
    pub fn accum_grad(&self, grad_tensor: &Tensor) {
        let mut inner = self.0.write().unwrap();
        if !inner.requires_grad {
            return;
        }

        let expected_shape = inner.shape.clone();
        let incoming_shape = grad_tensor.shape();
        assert_eq!(
            incoming_shape, expected_shape,
            "grad shape mismatch: expected {:?}, got {:?}",
            expected_shape, incoming_shape
        );

        let incoming_data = grad_tensor.contiguous_data();

        if let Some(existing_grad) = &inner.grad {
            let g_inner = existing_grad.0.write().unwrap();
            let mut g_storage = g_inner.storage.write().unwrap();
            let g_slice = g_storage.as_cpu_slice_mut();
            for (g, &inc) in g_slice.iter_mut().zip(incoming_data.iter()) {
                *g += inc;
            }
        } else {
            inner.grad = Some(Tensor::new(incoming_data, inner.shape.clone()));
        }
    }

    /// 返回当前累计梯度。
    pub fn grad(&self) -> Option<Vec<f32>> {
        let inner = self.0.read().unwrap();
        inner.grad.as_ref().map(|g| g.data())
    }

    /// 从当前张量出发执行反向传播。
    ///
    /// 这是 `backward_with_grad` 的便捷形式，只允许标量输出。它会为当前输出
    /// 构造一个全 1 的上游梯度，因此适合 loss 这类单值张量；如果输出不是
    /// 标量，调用者应显式使用 [`Tensor::backward_with_grad`] 提供 seed。
    pub fn backward(&self) {
        assert!(
            self.is_scalar(),
            "backward() requires a scalar tensor; use backward_with_grad() for non-scalar outputs"
        );
        let shape = self.shape();
        let size: usize = shape.iter().product();
        let seed = Tensor::new(vec![1.0; size], shape);
        self.backward_with_grad(&seed);
    }

    /// 从当前张量出发执行反向传播，并显式给出上游梯度 seed。
    ///
    /// 这相当于 PyTorch 中对非标量输出传入 `grad_tensors`。seed 的 shape 必须
    /// 与当前输出完全一致，否则反向传播将失去明确的 VJP 语义。
    pub fn backward_with_grad(&self, grad_output: &Tensor) {
        let expected_shape = self.shape();
        let incoming_shape = grad_output.shape();
        assert_eq!(
            incoming_shape, expected_shape,
            "backward seed shape mismatch: expected {:?}, got {:?}",
            expected_shape, incoming_shape
        );

        self.accum_grad(grad_output);

        let mut queue: VecDeque<(Tensor, Tensor)> = VecDeque::new();
        queue.push_back((self.clone(), grad_output.clone()));

        while let Some((tensor, grad_output)) = queue.pop_front() {
            let creator = {
                let inner = tensor.0.read().unwrap();
                inner.creator.clone()
            };

            if let Some(node) = creator {
                let input_grads = node.op.backward(&grad_output);

                for (input, maybe_grad) in node.inputs.iter().zip(input_grads) {
                    if let Some(grad) = maybe_grad {
                        input.accum_grad(&grad);

                        let has_creator = input.0.read().unwrap().creator.is_some();
                        if has_creator {
                            queue.push_back((input.clone(), grad));
                        }
                    }
                }
            }
        }
    }
}

/// 按 row-major 连续布局计算每个维度的 stride。
fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    if shape.is_empty() {
        return strides;
    }
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// 计算两个 shape 的 PyTorch/Numpy 风格广播结果。
///
/// 规则从尾维开始对齐：相等维度直接保留，任一侧为 `1` 时扩展到另一侧；
/// 其他组合不可广播。SPTorch 当前仍把标量表达为 `[1]`，因此该函数不额外
/// 引入零维张量语义。
pub fn broadcast_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
    let rank = a.len().max(b.len());
    let mut out = vec![1; rank];
    for i in 0..rank {
        let a_dim = a.get(a.len().wrapping_sub(1 + i)).copied().unwrap_or(1);
        let b_dim = b.get(b.len().wrapping_sub(1 + i)).copied().unwrap_or(1);
        out[rank - 1 - i] = if a_dim == b_dim {
            a_dim
        } else if a_dim == 1 {
            b_dim
        } else if b_dim == 1 {
            a_dim
        } else {
            return Err(TensorError::ShapeMismatch {
                expected: a.to_vec(),
                got: b.to_vec(),
            });
        };
    }
    Ok(out)
}

/// 判断两个 shape 是否可广播。
pub fn can_broadcast(a: &[usize], b: &[usize]) -> bool {
    broadcast_shape(a, b).is_ok()
}

/// 把已广播输出上的梯度归约回某个输入原始 shape。
///
/// 广播反向传播的核心是不沿着扩展维复制梯度，而是把这些维度求和。例如
/// `[2, 3] + [3]` 中第二个输入在 batch 维被复用两次，所以它的梯度要按
/// 第 0 维累加回 `[3]`。
pub fn unbroadcast_grad(grad: &Tensor, target_shape: &[usize]) -> Tensor {
    let grad_shape = grad.shape();
    let grad_data = grad.contiguous_data();
    let out_numel: usize = target_shape.iter().product();
    let mut out = vec![0.0f32; out_numel];
    if out_numel == 0 {
        return Tensor::new(out, target_shape.to_vec());
    }

    let grad_rank = grad_shape.len();
    let target_rank = target_shape.len();
    let rank_offset = grad_rank.saturating_sub(target_rank);
    let target_strides = compute_strides(target_shape);

    for (flat_idx, value) in grad_data.iter().enumerate() {
        let grad_indices = unravel_index(flat_idx, &grad_shape);
        let mut target_flat = 0usize;
        for target_dim in 0..target_rank {
            let grad_dim = target_dim + rank_offset;
            let coord = if target_shape[target_dim] == 1 {
                0
            } else {
                grad_indices[grad_dim]
            };
            target_flat += coord * target_strides[target_dim];
        }
        out[target_flat] += value;
    }

    Tensor::new(out, target_shape.to_vec())
}

fn unravel_index(mut flat_idx: usize, shape: &[usize]) -> Vec<usize> {
    let strides = compute_strides(shape);
    let mut indices = vec![0; shape.len()];
    for (dim, stride) in strides.iter().enumerate() {
        if *stride == 0 {
            continue;
        }
        indices[dim] = flat_idx / stride;
        flat_idx %= stride;
    }
    indices
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let inner = self.0.read().unwrap();
        f.debug_struct("Tensor")
            .field("data", &inner.storage)
            .field("shape", &inner.shape)
            .field("requires_grad", &inner.requires_grad)
            .field("device", &inner.device)
            .field("dtype", &inner.dtype)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[derive(Debug)]
    struct PassThroughOp;

    impl Op for PassThroughOp {
        fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
            vec![Some(grad_output.clone())]
        }
    }

    fn passthrough(input: &Tensor) -> Tensor {
        let out = Tensor::new(input.contiguous_data(), input.shape());
        if input.requires_grad() {
            let mut inner = out.0.write().unwrap();
            inner.requires_grad = true;
            inner.creator = Some(Arc::new(Node {
                op: Box::new(PassThroughOp),
                inputs: vec![input.clone()],
            }));
        }
        out
    }

    // F16 往返需要在典型范围内保持可接受误差，避免 dtype 模拟完全失真。
    #[test]
    fn test_f32_f16_roundtrip() {
        let vals = [0.0f32, 1.0, -1.0, 0.5, 65504.0, -65504.0, 1e-4];
        for &v in &vals {
            let h = f32_to_f16(v);
            let back = f16_to_f32(h);
            assert!(
                (back - v).abs() / (v.abs() + 1e-10) < 0.01,
                "f16 roundtrip failed for {}: got {}",
                v,
                back
            );
        }
    }

    // BF16 保留指数范围但降低尾数精度，这里锁定基础 round-trip 语义。
    #[test]
    fn test_f32_bf16_roundtrip() {
        let vals = [0.0f32, 1.0, -1.0, 3.14, 1e10, -1e10];
        for &v in &vals {
            let b = f32_to_bf16(v);
            let back = bf16_to_f32(b);
            assert!(
                (back - v).abs() / (v.abs() + 1e-10) < 0.01,
                "bf16 roundtrip failed for {}: got {}",
                v,
                back
            );
        }
    }

    // 新建张量默认使用 F32，这是绝大多数算子和测试的基线 dtype。
    #[test]
    fn test_tensor_dtype() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(t.dtype(), DType::F32);
    }

    // half() 应保留 shape，同时体现 F16 量化后的近似值。
    #[test]
    fn test_tensor_half() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let h = t.half();
        assert_eq!(h.dtype(), DType::F16);
        assert_eq!(h.shape(), vec![3]);
        let d = h.data();
        assert!((d[0] - 1.0).abs() < 1e-3);
        assert!((d[1] - 2.0).abs() < 1e-3);
        assert!((d[2] - 3.0).abs() < 1e-3);
    }

    // bfloat16() 的误差容忍略大于 F32，但 shape 和 dtype 元数据必须正确。
    #[test]
    fn test_tensor_bfloat16() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let b = t.bfloat16();
        assert_eq!(b.dtype(), DType::BF16);
        let d = b.data();
        assert!((d[0] - 1.0).abs() < 0.02);
        assert!((d[1] - 2.0).abs() < 0.02);
        assert!((d[2] - 3.0).abs() < 0.02);
    }

    // F16 再转回 F32 不会恢复丢失精度，但 dtype 元数据应回到 F32。
    #[test]
    fn test_tensor_float_roundtrip() {
        let t = Tensor::new(vec![1.5, 2.5, 3.5], vec![3]);
        let h = t.half();
        let back = h.float();
        assert_eq!(back.dtype(), DType::F32);
        let d = back.data();
        assert!((d[0] - 1.5).abs() < 1e-3);
    }

    // 转换到相同 dtype 时应走浅 clone，避免不必要的数据复制。
    #[test]
    fn test_tensor_to_dtype_noop() {
        let t = Tensor::new(vec![1.0], vec![1]);
        let same = t.to_dtype(DType::F32);
        assert_eq!(same.data(), t.data());
    }

    // F16 只有约 3 位十进制有效精度，这里验证精度损失处于预期范围。
    #[test]
    fn test_f16_precision_loss() {
        let t = Tensor::new(vec![1.001], vec![1]);
        let h = t.half();
        let d = h.data();
        assert!((d[0] - 1.001).abs() < 0.002);
    }

    #[test]
    fn test_grad_mode_no_grad_restores_state() {
        assert!(is_grad_enabled());
        let result = no_grad(|| {
            assert!(!is_grad_enabled());
            42
        });
        assert_eq!(result, 42);
        assert!(is_grad_enabled());
    }

    #[test]
    fn test_detach_keeps_data_but_clears_autograd_state() {
        let t = Tensor::with_grad(vec![1.0, 2.0], vec![2], true);
        t.accum_grad(&Tensor::new(vec![3.0, 4.0], vec![2]));

        let detached = t.detach();
        assert_eq!(detached.data(), vec![1.0, 2.0]);
        assert_eq!(detached.shape(), vec![2]);
        assert!(!detached.requires_grad());
        assert!(detached.grad().is_none());
    }

    #[test]
    fn test_backward_with_grad_supports_non_scalar_output() {
        let t = Tensor::with_grad(vec![1.0, 2.0], vec![2], true);
        let out = passthrough(&t);

        out.backward_with_grad(&Tensor::new(vec![10.0, 20.0], vec![2]));

        assert_eq!(t.grad().unwrap(), vec![10.0, 20.0]);
    }

    #[test]
    #[should_panic(expected = "backward() requires a scalar tensor")]
    fn test_backward_rejects_non_scalar_output() {
        let t = Tensor::with_grad(vec![1.0, 2.0], vec![2], true);
        passthrough(&t).backward();
    }

    #[test]
    #[should_panic(expected = "backward seed shape mismatch")]
    fn test_backward_with_grad_rejects_seed_shape_mismatch() {
        let t = Tensor::with_grad(vec![1.0, 2.0], vec![2], true);
        passthrough(&t).backward_with_grad(&Tensor::new(vec![1.0], vec![1]));
    }

    #[test]
    fn test_set_requires_grad_false_clears_grad() {
        let t = Tensor::with_grad(vec![1.0], vec![1], true);
        t.accum_grad(&Tensor::new(vec![2.0], vec![1]));
        t.set_requires_grad(false);
        assert!(!t.requires_grad());
        assert!(t.grad().is_none());
    }

    #[test]
    fn test_shape_helpers() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        assert_eq!(t.rank(), 2);
        assert_eq!(t.numel(), 6);
        assert!(!t.is_scalar());
        assert!(t.same_shape(&Tensor::new(vec![0.0; 6], vec![2, 3])));
    }

    #[test]
    fn test_broadcast_shape_scalar_tail_and_batch() {
        assert_eq!(broadcast_shape(&[2, 3], &[1]).unwrap(), vec![2, 3]);
        assert_eq!(broadcast_shape(&[2, 3], &[3]).unwrap(), vec![2, 3]);
        assert_eq!(broadcast_shape(&[4, 2, 3], &[1, 3]).unwrap(), vec![4, 2, 3]);
        assert!(broadcast_shape(&[2, 3], &[2]).is_err());
    }

    #[test]
    fn test_unbroadcast_grad_reduces_expanded_axes() {
        let grad = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let reduced = unbroadcast_grad(&grad, &[3]);
        assert_eq!(reduced.data(), vec![5.0, 7.0, 9.0]);

        let scalar = unbroadcast_grad(&grad, &[1]);
        assert_eq!(scalar.data(), vec![21.0]);
    }
}

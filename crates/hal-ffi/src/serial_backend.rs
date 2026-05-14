//! Tang9k 串行后端的 dry-run 接入层。
//!
//! 这一层不是“假装已经有真实硬件”，而是把 Week 3 需要的注册与调度边界先固定下来：
//! 上层通过 `core-tensor` 的 `BackendDispatch` 选择 `Device::Custom(n)`，MatMul 请求会被转换成
//! `sptorch-hal::serial` 定义的 32×32 tile 指令帧，然后通过 loopback 严格校验帧格式。数值结果
//! 暂时由 CPU 参考路径计算，用来保持训练/测试可继续跑；真正 UART/DMA 接入后，只需要替换发送层。

use sptorch_core_tensor::{register_backend, BackendDispatch, Device};
use sptorch_hal::serial::{
    plan_matmul32x32_commands, LoopbackSerialTransport, Matmul32x32Plan, MatmulMemoryLayout, SerialFrame,
    SerialProtocolError,
};
use std::sync::{Arc, Mutex};

pub use sptorch_hal::serial::{
    MATMUL32X32_FLAG_ACCUMULATE, MATMUL32X32_FLAG_CLEAR_OUTPUT, MATMUL32X32_FLAG_LAST_K_TILE,
};

/// dry-run 后端默认注册到 `Device::Custom(9)`，用数字 9 对应 Tang9k 主线。
pub const DEFAULT_TANG9K_SERIAL_DEVICE: Device = Device::Custom(9);

/// 单次 serial MatMul 调度留下的可观测状态。
///
/// 这里保存编码前的 plan 和编码后的 frame，方便测试、日志或未来 Studio 展示“到底发了哪些
/// tile 指令”。真实串口 backend 也应该保留类似轻量遥测，否则硬件 bring-up 失败时很难定位
/// 是 shape/tiling 错，还是链路/bitstream 错。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tang9kSerialTrace {
    pub plan: Matmul32x32Plan,
    pub frames: Vec<SerialFrame>,
    pub queue_depth_after_submit: u32,
}

/// Tang9k serial backend 的传输层边界。
///
/// dry-run 默认用 loopback，但真实 UART/DMA 也应该实现这个 trait：调用方给出已经编码好的
/// [`SerialFrame`]，传输层负责发送、等待响应并返回对端帧。把这个边界抽出来后，调度层可以继续
/// 复用 tile planner、sequence 管理和 trace 记录，而不用知道底层是内存回环、串口还是 DMA。
pub trait Tang9kSerialTransport: Send + Sync + std::fmt::Debug {
    fn exchange(&self, frame: &SerialFrame) -> Result<SerialFrame, SerialProtocolError>;
}

/// 默认 loopback 传输层，用于 CI、dry-run 和没有板卡时的 bring-up 演练。
#[derive(Debug, Default)]
pub struct LoopbackTang9kTransport {
    loopback: Mutex<LoopbackSerialTransport>,
}

impl LoopbackTang9kTransport {
    pub fn frames_seen(&self) -> usize {
        self.loopback.lock().unwrap().frames_seen()
    }
}

impl Tang9kSerialTransport for LoopbackTang9kTransport {
    fn exchange(&self, frame: &SerialFrame) -> Result<SerialFrame, SerialProtocolError> {
        let mut loopback = self.loopback.lock().unwrap();
        let echoed = loopback.exchange(&frame.encode()?)?;
        SerialFrame::decode(&echoed)
    }
}

/// Tang9k serial backend 的纯 Rust dry-run 实现。
///
/// elementwise kernel 先保持 CPU 语义；MatMul 会额外生成并 loopback 校验 Tang9k 指令帧。
/// `next_sequence` 用于模拟硬件提交队列的递增序号，`last_trace` 则是最近一次 MatMul 的调试快照。
#[derive(Debug)]
pub struct Tang9kSerialDryRunBackend {
    device: Device,
    transport: Arc<dyn Tang9kSerialTransport>,
    next_sequence: Mutex<u32>,
    last_trace: Mutex<Option<Tang9kSerialTrace>>,
}

impl Default for Tang9kSerialDryRunBackend {
    fn default() -> Self {
        Self::new(DEFAULT_TANG9K_SERIAL_DEVICE)
    }
}

impl Tang9kSerialDryRunBackend {
    pub fn new(device: Device) -> Self {
        Self::with_transport(device, Arc::new(LoopbackTang9kTransport::default()))
    }

    pub fn with_transport(device: Device, transport: Arc<dyn Tang9kSerialTransport>) -> Self {
        Self {
            device,
            transport,
            next_sequence: Mutex::new(0),
            last_trace: Mutex::new(None),
        }
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// 返回最近一次 MatMul 调度产生的 trace。
    pub fn last_trace(&self) -> Option<Tang9kSerialTrace> {
        self.last_trace.lock().unwrap().clone()
    }

    /// 返回模拟队列深度；dry-run 在 loopback 完成后会 drain 到 0。
    pub fn queue_depth(&self) -> u32 {
        self.last_trace
            .lock()
            .unwrap()
            .as_ref()
            .map(|trace| trace.queue_depth_after_submit)
            .unwrap_or(0)
    }

    /// 将 dry-run 后端注册到 core-tensor 的全局 dispatch 表。
    pub fn register(self: &Arc<Self>) {
        register_backend(self.device, self.clone());
    }

    fn record_serial_plan(&self, m: usize, k: usize, n: usize) -> Result<(), SerialProtocolError> {
        let layout = MatmulMemoryLayout::row_major(m, k, n, 0, (m * k * 4) as u64, ((m * k + k * n) * 4) as u64, 4);
        let plan = plan_matmul32x32_commands(m, k, n, layout)?;
        let first_sequence = {
            let mut next = self.next_sequence.lock().unwrap();
            let current = *next;
            *next = next.wrapping_add(plan.command_count() as u32);
            current
        };
        let frames = plan.frames(first_sequence);

        for frame in &frames {
            let echoed = self.transport.exchange(frame)?;
            if echoed.sequence != frame.sequence || echoed.opcode != frame.opcode {
                return Err(SerialProtocolError::InvalidMatmulLayout {
                    reason: format!(
                        "serial transport echoed mismatched frame: sent {:?}/{} got {:?}/{}",
                        frame.opcode, frame.sequence, echoed.opcode, echoed.sequence
                    ),
                });
            }
        }

        *self.last_trace.lock().unwrap() = Some(Tang9kSerialTrace {
            plan,
            frames,
            queue_depth_after_submit: 0,
        });
        Ok(())
    }
}

impl BackendDispatch for Tang9kSerialDryRunBackend {
    fn add_f32(&self, a: &[f32], b: &[f32], out: &mut [f32]) {
        for ((dst, lhs), rhs) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
            *dst = lhs + rhs;
        }
    }

    fn mul_f32(&self, a: &[f32], b: &[f32], out: &mut [f32]) {
        for ((dst, lhs), rhs) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
            *dst = lhs * rhs;
        }
    }

    fn neg_f32(&self, a: &[f32], out: &mut [f32]) {
        for (dst, value) in out.iter_mut().zip(a.iter()) {
            *dst = -*value;
        }
    }

    fn exp_f32(&self, a: &[f32], out: &mut [f32]) {
        for (dst, value) in out.iter_mut().zip(a.iter()) {
            *dst = value.exp();
        }
    }

    fn log_f32(&self, a: &[f32], out: &mut [f32]) {
        for (dst, value) in out.iter_mut().zip(a.iter()) {
            *dst = value.ln();
        }
    }

    fn relu_f32(&self, a: &[f32], out: &mut [f32]) {
        for (dst, value) in out.iter_mut().zip(a.iter()) {
            *dst = value.max(0.0);
        }
    }

    fn gelu_f32(&self, a: &[f32], out: &mut [f32]) {
        for (dst, value) in out.iter_mut().zip(a.iter()) {
            *dst = 0.5
                * value
                * (1.0 + ((2.0_f32 / std::f32::consts::PI).sqrt() * (value + 0.044715 * value * value * value)).tanh());
        }
    }

    fn scale_f32(&self, a: &[f32], scalar: f32, out: &mut [f32]) {
        for (dst, value) in out.iter_mut().zip(a.iter()) {
            *dst = value * scalar;
        }
    }

    fn matmul_f32(&self, a: &[f32], b: &[f32], out: &mut [f32], m: usize, k: usize, n: usize) {
        self.record_serial_plan(m, k, n)
            .expect("Tang9k serial dry-run failed to submit MatMul frames");

        for row in 0..m {
            for col in 0..n {
                let mut sum = 0.0f32;
                for kk in 0..k {
                    sum += a[row * k + kk] * b[kk * n + col];
                }
                out[row * n + col] = sum;
            }
        }
    }
}

/// 注册默认 Tang9k serial dry-run backend，并返回可观测句柄。
pub fn register_tang9k_serial_dry_run_backend() -> Arc<Tang9kSerialDryRunBackend> {
    let backend = Arc::new(Tang9kSerialDryRunBackend::default());
    backend.register();
    backend
}

/// 注册指定逻辑设备号的 Tang9k serial dry-run backend。
pub fn register_tang9k_serial_dry_run_backend_for(device: Device) -> Arc<Tang9kSerialDryRunBackend> {
    let backend = Arc::new(Tang9kSerialDryRunBackend::new(device));
    backend.register();
    backend
}

/// 使用自定义传输层注册 Tang9k serial dry-run backend。
///
/// 这个入口是未来接真实串口的最小替换点：先把 UART/DMA 实现包成 [`Tang9kSerialTransport`]，
/// 再通过这里注册，`core-ops::matmul` 的上层 API 不需要变化。
pub fn register_tang9k_serial_dry_run_backend_with_transport(
    device: Device,
    transport: Arc<dyn Tang9kSerialTransport>,
) -> Arc<Tang9kSerialDryRunBackend> {
    let backend = Arc::new(Tang9kSerialDryRunBackend::with_transport(device, transport));
    backend.register();
    backend
}

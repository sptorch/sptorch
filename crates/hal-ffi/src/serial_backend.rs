//! Tang9k 串行后端的 dry-run 接入层。
//!
//! 这一层不是“假装已经有真实硬件”，而是把 Week 3 需要的注册与调度边界先固定下来：
//! 上层通过 `core-tensor` 的 `BackendDispatch` 选择 `Device::Custom(n)`，MatMul 请求会被转换成
//! `sptorch-hal::serial` 定义的 32×32 tile 指令帧，然后通过 loopback 严格校验帧格式。数值结果
//! 暂时由 CPU 参考路径计算，用来保持训练/测试可继续跑；真正 UART/DMA 接入后，只需要替换发送层。

use sptorch_core_tensor::{register_backend, BackendDispatch, Device};
use sptorch_hal::serial::{
    plan_matmul32x32_commands, validate_result_value_response, validate_scratch_value_response,
    validate_status_response, LoopbackSerialTransport, Matmul32x32Command, Matmul32x32Plan, MatmulMemoryLayout,
    ResultRead32Command, ScratchRead32Command, ScratchWrite32Command, SerialFrame, SerialOpcode, SerialProtocolError,
    SerialStatusCode, SerialStatusPayload, SerialStreamDecoder, SerialSubmitQueue, SerialSubmitReport,
    TANG9K_RESULT_WINDOW_WORD_STRIDE_BYTES,
};
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

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
    pub reports: Vec<SerialSubmitReport>,
    pub queue_high_watermark: u32,
    pub queue_depth_after_submit: u32,
}

/// Tang9k serial backend 的传输层边界。
///
/// dry-run 默认用 loopback，但真实 UART/DMA 也应该实现这个 trait：调用方给出已经编码好的
/// [`SerialFrame`]，传输层负责发送、等待响应并返回对端状态帧。v1 要求返回 `Ack/Ok`
/// 才表示命令被目标接收；`Busy`、`Error` 或 sequence 不一致都会被调度层视为提交失败。
/// 把这个边界抽出来后，调度层可以继续复用 tile planner、sequence 管理和 trace 记录，
/// 而不用知道底层是内存回环、串口还是 DMA。
pub trait Tang9kSerialTransport: Send + Sync + std::fmt::Debug {
    fn exchange(&self, frame: &SerialFrame) -> Result<SerialFrame, SerialProtocolError>;
}

/// 一次真实 UART 交换留下的字节级证据。
///
/// 真实板卡 bring-up 里，`SerialFrame` 只能说明“最终解码出了什么”；而 checksum mismatch、
/// padding 污染、旧响应残留这类问题必须回到原始字节才能定位。因此 trace 同时保留 host 写出的
/// request bytes 和设备侧读回的 raw bytes，CLI 可以直接打印，测试和后续 Studio 也能复用。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UartTang9kExchangeTrace {
    pub request_bytes: Vec<u8>,
    pub raw_response_bytes: Vec<u8>,
    pub response: SerialFrame,
}

/// 带原始字节上下文的 UART 交换错误。
///
/// 内层仍然使用协议层统一的 [`SerialProtocolError`]，这里额外携带字节上下文，避免硬件调试时
/// 只看到“checksum mismatch”却看不到 FPGA 实际发回了什么。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UartTang9kExchangeError {
    pub error: SerialProtocolError,
    pub request_bytes: Vec<u8>,
    pub raw_response_bytes: Vec<u8>,
}

impl UartTang9kExchangeError {
    fn new(error: SerialProtocolError, request_bytes: Vec<u8>, raw_response_bytes: Vec<u8>) -> Self {
        Self {
            error,
            request_bytes,
            raw_response_bytes,
        }
    }
}

impl fmt::Display for UartTang9kExchangeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.error.fmt(f)
    }
}

impl std::error::Error for UartTang9kExchangeError {}

/// Windows/Linux/macOS 串口传输层。
///
/// 这个实现只负责字节流 I/O：写入一帧、持续读取，直到 `SerialStreamDecoder`
/// 切出第一帧响应或超时。它不解释 `Ack/Busy/Error` 是否成功，成功语义仍交给
/// `SerialSubmitQueue` / `validate_status_response`，这样 UART、DMA 和 loopback 的上层行为一致。
#[derive(Debug, Clone)]
pub struct UartTang9kTransport {
    port_name: String,
    baud_rate: u32,
    timeout: Duration,
}

impl UartTang9kTransport {
    pub fn new(port_name: impl Into<String>, baud_rate: u32, timeout: Duration) -> Self {
        Self {
            port_name: port_name.into(),
            baud_rate,
            timeout,
        }
    }

    pub fn port_name(&self) -> &str {
        &self.port_name
    }

    pub fn baud_rate(&self) -> u32 {
        self.baud_rate
    }

    pub fn timeout(&self) -> Duration {
        self.timeout
    }

    /// 发送一帧并返回字节级 trace。
    ///
    /// 打开串口后会短暂等待并清理输入缓冲，这是为了避免上一轮 probe 的残留响应污染当前命令。
    /// Tang9k 当前 responder 是简单同步状态机，20ms 已远大于 115200 baud 下一帧 32 字节响应所需时间。
    pub fn exchange_with_trace(&self, frame: &SerialFrame) -> Result<UartTang9kExchangeTrace, UartTang9kExchangeError> {
        let encoded = frame
            .encode()
            .map_err(|error| UartTang9kExchangeError::new(error, Vec::new(), Vec::new()))?;
        let mut port = serialport::new(&self.port_name, self.baud_rate)
            .timeout(Duration::from_millis(20))
            .open()
            .map_err(|err| SerialProtocolError::TransportIo {
                reason: format!("open {} failed: {err}", self.port_name),
            })
            .map_err(|error| UartTang9kExchangeError::new(error, encoded.clone(), Vec::new()))?;

        std::thread::sleep(Duration::from_millis(20));
        let _ = port.clear(serialport::ClearBuffer::Input);
        port.write_all(&encoded)
            .map_err(|err| SerialProtocolError::TransportIo {
                reason: format!("write {} failed: {err}", self.port_name),
            })
            .map_err(|error| UartTang9kExchangeError::new(error, encoded.clone(), Vec::new()))?;
        port.flush()
            .map_err(|err| SerialProtocolError::TransportIo {
                reason: format!("flush {} failed: {err}", self.port_name),
            })
            .map_err(|error| UartTang9kExchangeError::new(error, encoded.clone(), Vec::new()))?;

        let deadline = Instant::now() + self.timeout;
        let mut decoder = SerialStreamDecoder::new();
        let mut buf = [0u8; 256];
        let mut raw_response_bytes = Vec::new();

        loop {
            match port.read(&mut buf) {
                Ok(0) => {}
                Ok(n) => {
                    raw_response_bytes.extend_from_slice(&buf[..n]);
                    let frames = decoder.push_bytes(&buf[..n]).map_err(|error| {
                        UartTang9kExchangeError::new(error, encoded.clone(), raw_response_bytes.clone())
                    })?;
                    if let Some(response) = frames.into_iter().next() {
                        return Ok(UartTang9kExchangeTrace {
                            request_bytes: encoded,
                            raw_response_bytes,
                            response,
                        });
                    }
                }
                Err(err) if err.kind() == std::io::ErrorKind::TimedOut => {}
                Err(err) => {
                    let error = SerialProtocolError::TransportIo {
                        reason: format!("read {} failed: {err}", self.port_name),
                    };
                    return Err(UartTang9kExchangeError::new(error, encoded, raw_response_bytes));
                }
            }

            if Instant::now() >= deadline {
                let error = SerialProtocolError::ResponseTimeout {
                    timeout_ms: self.timeout.as_millis() as u64,
                };
                return Err(UartTang9kExchangeError::new(error, encoded, raw_response_bytes));
            }
        }
    }
}

impl Tang9kSerialTransport for UartTang9kTransport {
    fn exchange(&self, frame: &SerialFrame) -> Result<SerialFrame, SerialProtocolError> {
        self.exchange_with_trace(frame)
            .map(|trace| trace.response)
            .map_err(|err| err.error)
    }
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
        let echoed = SerialFrame::decode(&echoed)?;
        if echoed != *frame {
            return Err(SerialProtocolError::InvalidMatmulLayout {
                reason: "loopback transport returned a frame that differs from the submitted command".into(),
            });
        }
        Ok(SerialFrame::ack(frame.sequence, SerialStatusPayload::ok()))
    }
}

/// 枚举当前主机可见串口。
///
/// probe 工具和用户代码都走这个入口，避免把 `serialport` crate 的类型泄漏为框架 API。
pub fn list_tang9k_serial_ports() -> Result<Vec<String>, SerialProtocolError> {
    serialport::available_ports()
        .map(|ports| {
            ports
                .into_iter()
                .map(|port| match port.port_type {
                    serialport::SerialPortType::UsbPort(info) => {
                        format!(
                            "{} USB vid={:04x} pid={:04x} serial={}",
                            port.port_name,
                            info.vid,
                            info.pid,
                            info.serial_number.unwrap_or_else(|| "unknown".into())
                        )
                    }
                    other => format!("{} {:?}", port.port_name, other),
                })
                .collect()
        })
        .map_err(|err| SerialProtocolError::TransportIo {
            reason: format!("list serial ports failed: {err}"),
        })
}

/// 对真实串口发送一帧 `Ping`，用于板卡上电后的最小链路验收。
///
/// 目标固件若还没实现 `Pong`，也可以返回 `Ack/Ok`，便于先验证 host->device->host
/// 的控制通路。其它响应都会保留为协议错误，而不是吞掉。
pub fn probe_tang9k_ping(
    port_name: &str,
    baud_rate: u32,
    timeout: Duration,
) -> Result<SerialFrame, SerialProtocolError> {
    probe_tang9k_ping_with_trace(port_name, baud_rate, timeout)
        .map(|trace| trace.response)
        .map_err(|err| err.error)
}

pub fn probe_tang9k_ping_with_trace(
    port_name: &str,
    baud_rate: u32,
    timeout: Duration,
) -> Result<UartTang9kExchangeTrace, UartTang9kExchangeError> {
    let transport = UartTang9kTransport::new(port_name, baud_rate, timeout);
    let command = SerialFrame::new(SerialOpcode::Ping, 0, b"sptorch-ping".to_vec());
    let trace = transport.exchange_with_trace(&command)?;
    if trace.response.sequence != command.sequence {
        let error = SerialProtocolError::SequenceMismatch {
            expected: command.sequence,
            actual: trace.response.sequence,
        };
        return Err(UartTang9kExchangeError::new(
            error,
            trace.request_bytes,
            trace.raw_response_bytes,
        ));
    }

    match trace.response.opcode {
        SerialOpcode::Pong => Ok(trace),
        SerialOpcode::Ack => {
            if let Err(error) = sptorch_hal::serial::validate_status_response(&command, &trace.response) {
                return Err(UartTang9kExchangeError::new(
                    error,
                    trace.request_bytes,
                    trace.raw_response_bytes,
                ));
            }
            Ok(trace)
        }
        actual => Err(UartTang9kExchangeError::new(
            SerialProtocolError::UnexpectedResponseOpcode { actual },
            trace.request_bytes,
            trace.raw_response_bytes,
        )),
    }
}

/// 构造真实板卡 smoke test 使用的最小 MatMul 控制帧。
///
/// 地址只是协议层面的设备侧偏移，当前 responder 不会真正访问这些位置。把构造逻辑单独拆出来，
/// 是为了让 CLI、测试和未来批量 bring-up 工具使用同一条命令样本，避免“host 发的不是我们以为的
/// MatMul 帧”这种很难看的硬件调试分叉。
pub fn tang9k_matmul_smoke_frame(sequence: u32) -> SerialFrame {
    Matmul32x32Command {
        tile_id: 0,
        a_offset: 0,
        b_offset: (Matmul32x32Command::M * Matmul32x32Command::K * std::mem::size_of::<f32>()) as u64,
        out_offset: ((Matmul32x32Command::M * Matmul32x32Command::K + Matmul32x32Command::K * Matmul32x32Command::N)
            * std::mem::size_of::<f32>()) as u64,
        flags: MATMUL32X32_FLAG_CLEAR_OUTPUT | MATMUL32X32_FLAG_LAST_K_TILE,
    }
    .into_frame(sequence)
}

/// 向真实 Tang9k 串口发送一条最小 `Matmul32x32` 控制帧，并要求板端返回 `Ack/Ok`。
///
/// 这个探针不验证矩阵计算结果；它只验证板端已经能接收标准命令帧、校验 payload/checksum/padding，
/// 并按 v1 规则回显 sequence 与状态载荷。换句话说，这是进入真实 PE 阵列之前的“命令提交生命周期”
/// 烟测，能把 UART 物理链路问题和后续计算数据通路问题拆开定位。
pub fn probe_tang9k_matmul_smoke(
    port_name: &str,
    baud_rate: u32,
    timeout: Duration,
) -> Result<SerialFrame, SerialProtocolError> {
    probe_tang9k_matmul_smoke_with_trace(port_name, baud_rate, timeout)
        .map(|trace| trace.response)
        .map_err(|err| err.error)
}

pub fn probe_tang9k_matmul_smoke_with_trace(
    port_name: &str,
    baud_rate: u32,
    timeout: Duration,
) -> Result<UartTang9kExchangeTrace, UartTang9kExchangeError> {
    let transport = UartTang9kTransport::new(port_name, baud_rate, timeout);
    let command = tang9k_matmul_smoke_frame(1);
    let trace = transport.exchange_with_trace(&command)?;
    if let Err(error) = validate_status_response(&command, &trace.response) {
        return Err(UartTang9kExchangeError::new(
            error,
            trace.request_bytes,
            trace.raw_response_bytes,
        ));
    }
    Ok(trace)
}

/// 向 Tang9k 发送一个最小 scratch 写入帧，然后再读回同一位置。
///
/// 这条路径不代表最终 DMA/共享内存协议，只是先验证“非 ACK 载荷”的数据面回环能力：host 能写，
/// target 能记住，host 能再把 value 原样读回来。后续 MatMul 结果回读会复用同样的控制面接口。
pub fn probe_tang9k_scratch_smoke_with_trace(
    port_name: &str,
    baud_rate: u32,
    timeout: Duration,
) -> Result<(UartTang9kExchangeTrace, UartTang9kExchangeTrace), UartTang9kExchangeError> {
    let transport = UartTang9kTransport::new(port_name, baud_rate, timeout);
    let write_command = ScratchWrite32Command::new(0x44, 0x1122_3344).into_frame(2);
    let write_trace = transport.exchange_with_trace(&write_command)?;
    if let Err(error) = validate_status_response(&write_command, &write_trace.response) {
        return Err(UartTang9kExchangeError::new(
            error,
            write_trace.request_bytes,
            write_trace.raw_response_bytes,
        ));
    }

    let read_command = ScratchRead32Command::new(0x44).into_frame(3);
    let read_trace = transport.exchange_with_trace(&read_command)?;
    let read_value = match validate_scratch_value_response(&read_command, &read_trace.response) {
        Ok(value) => value,
        Err(error) => {
            return Err(UartTang9kExchangeError::new(
                error,
                read_trace.request_bytes.clone(),
                read_trace.raw_response_bytes.clone(),
            ));
        }
    };

    if read_value.offset != 0x44 || read_value.value != 0x1122_3344 {
        return Err(UartTang9kExchangeError::new(
            SerialProtocolError::ScratchValueMismatch {
                expected_offset: 0x44,
                expected_value: 0x1122_3344,
                actual_offset: read_value.offset,
                actual_value: read_value.value,
            },
            read_trace.request_bytes,
            read_trace.raw_response_bytes,
        ));
    }

    Ok((write_trace, read_trace))
}

/// 发送一条最小 MatMul 控制帧，再从结果窗口读回一个 32-bit 摘要值。
///
/// 这条路径仍然不是“真正矩阵乘法完成确认”，而是把板端的结果窗口语义先练稳：
/// MatMul 命令负责更新结果槽，随后 `ResultRead32` 负责把这个槽读回。这样 host 以后就
/// 能把“命令生命周期”与“结果回读”分成两个稳定步骤。
pub fn probe_tang9k_result_smoke_with_trace(
    port_name: &str,
    baud_rate: u32,
    timeout: Duration,
) -> Result<(UartTang9kExchangeTrace, UartTang9kExchangeTrace), UartTang9kExchangeError> {
    let transport = UartTang9kTransport::new(port_name, baud_rate, timeout);
    let matmul_command = tang9k_matmul_smoke_frame(4);
    let matmul_trace = transport.exchange_with_trace(&matmul_command)?;
    if let Err(error) = validate_status_response(&matmul_command, &matmul_trace.response) {
        return Err(UartTang9kExchangeError::new(
            error,
            matmul_trace.request_bytes,
            matmul_trace.raw_response_bytes,
        ));
    }

    let decoded_matmul = match Matmul32x32Command::decode_payload(&matmul_command.payload) {
        Ok(command) => command,
        Err(error) => {
            return Err(UartTang9kExchangeError::new(
                error,
                matmul_trace.request_bytes.clone(),
                matmul_trace.raw_response_bytes.clone(),
            ));
        }
    };
    let expected_result = decoded_matmul.smoke_result_summary();
    let read_command = ResultRead32Command::new(expected_result.offset).into_frame(5);
    let read_trace = transport.exchange_with_trace(&read_command)?;
    let read_value = match validate_result_value_response(&read_command, &read_trace.response) {
        Ok(value) => value,
        Err(error) => {
            return Err(UartTang9kExchangeError::new(
                error,
                read_trace.request_bytes.clone(),
                read_trace.raw_response_bytes.clone(),
            ));
        }
    };

    if read_value != expected_result {
        return Err(UartTang9kExchangeError::new(
            SerialProtocolError::ResultValueMismatch {
                expected_offset: expected_result.offset,
                expected_value: expected_result.value,
                actual_offset: read_value.offset,
                actual_value: read_value.value,
            },
            read_trace.request_bytes,
            read_trace.raw_response_bytes,
        ));
    }

    Ok((matmul_trace, read_trace))
}

/// 发送 MatMul 控制帧后连续读回 4 个 result-window word。
///
/// 单槽摘要只能证明“有一个结果寄存器会变”；这个 helper 多走三次 `ResultRead32`，用不同 offset
/// 逼迫 RTL 做真正的窗口选择。它仍然不代表 PE 阵列已经算出矩阵，只是在完整 DMA/result buffer
/// 之前，把 host/RTL 的读回协议地基夯实。
pub fn probe_tang9k_result_window_smoke_with_trace(
    port_name: &str,
    baud_rate: u32,
    timeout: Duration,
) -> Result<Vec<UartTang9kExchangeTrace>, UartTang9kExchangeError> {
    let transport = UartTang9kTransport::new(port_name, baud_rate, timeout);
    let matmul_command = tang9k_matmul_smoke_frame(4);
    let matmul_trace = transport.exchange_with_trace(&matmul_command)?;
    if let Err(error) = validate_status_response(&matmul_command, &matmul_trace.response) {
        return Err(UartTang9kExchangeError::new(
            error,
            matmul_trace.request_bytes,
            matmul_trace.raw_response_bytes,
        ));
    }

    let decoded_matmul = match Matmul32x32Command::decode_payload(&matmul_command.payload) {
        Ok(command) => command,
        Err(error) => {
            return Err(UartTang9kExchangeError::new(
                error,
                matmul_trace.request_bytes.clone(),
                matmul_trace.raw_response_bytes.clone(),
            ));
        }
    };
    let expected_window = decoded_matmul.smoke_result_window();
    let mut traces = Vec::with_capacity(1 + expected_window.len());
    traces.push(matmul_trace);

    for (idx, expected_result) in expected_window.iter().enumerate() {
        let read_sequence = 5 + idx as u32;
        let read_command = ResultRead32Command::new(expected_result.offset).into_frame(read_sequence);
        let read_trace = transport.exchange_with_trace(&read_command)?;
        let read_value = match validate_result_value_response(&read_command, &read_trace.response) {
            Ok(value) => value,
            Err(error) => {
                return Err(UartTang9kExchangeError::new(
                    error,
                    read_trace.request_bytes.clone(),
                    read_trace.raw_response_bytes.clone(),
                ));
            }
        };

        if read_value != *expected_result {
            return Err(UartTang9kExchangeError::new(
                SerialProtocolError::ResultValueMismatch {
                    expected_offset: expected_result.offset,
                    expected_value: expected_result.value,
                    actual_offset: read_value.offset,
                    actual_value: read_value.value,
                },
                read_trace.request_bytes,
                read_trace.raw_response_bytes,
            ));
        }

        traces.push(read_trace);
    }

    Ok(traces)
}

/// 验证 result-window 的越界读会被板端明确拒绝。
///
/// 这条负向探针非常适合在接入真实 BRAM/FIFO 前保底：如果越界 offset 也返回 `ResultValue32`，
/// host 后续就无法区分“真正结果 word”和“RTL 默认值/旧值泄漏”。因此这里要求板端返回
/// `Error + HardwareFault`，并把 detail 写成被拒绝的 offset。
pub fn probe_tang9k_result_oob_smoke_with_trace(
    port_name: &str,
    baud_rate: u32,
    timeout: Duration,
) -> Result<(Vec<UartTang9kExchangeTrace>, UartTang9kExchangeTrace), UartTang9kExchangeError> {
    let transport = UartTang9kTransport::new(port_name, baud_rate, timeout);
    let matmul_command = tang9k_matmul_smoke_frame(4);
    let matmul_trace = transport.exchange_with_trace(&matmul_command)?;
    if let Err(error) = validate_status_response(&matmul_command, &matmul_trace.response) {
        return Err(UartTang9kExchangeError::new(
            error,
            matmul_trace.request_bytes,
            matmul_trace.raw_response_bytes,
        ));
    }

    let decoded_matmul = match Matmul32x32Command::decode_payload(&matmul_command.payload) {
        Ok(command) => command,
        Err(error) => {
            return Err(UartTang9kExchangeError::new(
                error,
                matmul_trace.request_bytes.clone(),
                matmul_trace.raw_response_bytes.clone(),
            ));
        }
    };
    let expected_window = decoded_matmul.smoke_result_window();
    let mut setup_traces = Vec::with_capacity(1 + expected_window.len());
    setup_traces.push(matmul_trace);

    for (idx, expected_result) in expected_window.iter().enumerate() {
        let read_sequence = 5 + idx as u32;
        let read_command = ResultRead32Command::new(expected_result.offset).into_frame(read_sequence);
        let read_trace = transport.exchange_with_trace(&read_command)?;
        let read_value = match validate_result_value_response(&read_command, &read_trace.response) {
            Ok(value) => value,
            Err(error) => {
                return Err(UartTang9kExchangeError::new(
                    error,
                    read_trace.request_bytes.clone(),
                    read_trace.raw_response_bytes.clone(),
                ));
            }
        };

        if read_value != *expected_result {
            return Err(UartTang9kExchangeError::new(
                SerialProtocolError::ResultValueMismatch {
                    expected_offset: expected_result.offset,
                    expected_value: expected_result.value,
                    actual_offset: read_value.offset,
                    actual_value: read_value.value,
                },
                read_trace.request_bytes,
                read_trace.raw_response_bytes,
            ));
        }

        setup_traces.push(read_trace);
    }

    let last_word = expected_window
        .last()
        .expect("fixed Tang9k result smoke window is non-empty");
    let oob_offset = last_word.offset.wrapping_add(TANG9K_RESULT_WINDOW_WORD_STRIDE_BYTES);
    let oob_command = ResultRead32Command::new(oob_offset).into_frame(9);
    let oob_trace = transport.exchange_with_trace(&oob_command)?;
    match validate_status_response(&oob_command, &oob_trace.response) {
        Err(SerialProtocolError::CommandRejected {
            command_opcode: SerialOpcode::ResultRead32,
            response_opcode: SerialOpcode::Error,
            sequence: 9,
            status: SerialStatusCode::HardwareFault,
            detail,
        }) if detail == oob_offset => Ok((setup_traces, oob_trace)),
        Ok(_) => Err(UartTang9kExchangeError::new(
            SerialProtocolError::UnexpectedResponseOpcode {
                actual: oob_trace.response.opcode,
            },
            oob_trace.request_bytes,
            oob_trace.raw_response_bytes,
        )),
        Err(error) => Err(UartTang9kExchangeError::new(
            error,
            oob_trace.request_bytes,
            oob_trace.raw_response_bytes,
        )),
    }
}

/// Tang9k serial backend 的纯 Rust dry-run 实现。
///
/// elementwise kernel 先保持 CPU 语义；MatMul 会额外生成并 loopback 校验 Tang9k 指令帧。
/// `submit_queue` 用于模拟硬件提交窗口、sequence 分配和 ACK 生命周期，`last_trace`
/// 则是最近一次 MatMul 的调试快照。
#[derive(Debug)]
pub struct Tang9kSerialDryRunBackend {
    device: Device,
    transport: Arc<dyn Tang9kSerialTransport>,
    submit_queue: Mutex<SerialSubmitQueue>,
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
            submit_queue: Mutex::new(SerialSubmitQueue::default()),
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
        self.submit_queue.lock().unwrap().queue_depth()
    }

    /// 将 dry-run 后端注册到 core-tensor 的全局 dispatch 表。
    pub fn register(self: &Arc<Self>) {
        register_backend(self.device, self.clone());
    }

    fn record_serial_plan(&self, m: usize, k: usize, n: usize) -> Result<(), SerialProtocolError> {
        let layout = MatmulMemoryLayout::row_major(m, k, n, 0, (m * k * 4) as u64, ((m * k + k * n) * 4) as u64, 4);
        let plan = plan_matmul32x32_commands(m, k, n, layout)?;
        let mut submit_queue = self.submit_queue.lock().unwrap();
        let first_sequence = submit_queue.reserve_sequences(plan.command_count());
        let frames = plan.frames(first_sequence);
        let mut reports = Vec::with_capacity(frames.len());

        for frame in &frames {
            reports.push(submit_queue.submit(frame, |frame| self.transport.exchange(frame))?);
        }
        let queue_high_watermark = submit_queue.high_watermark();
        let queue_depth_after_submit = submit_queue.queue_depth();
        drop(submit_queue);

        *self.last_trace.lock().unwrap() = Some(Tang9kSerialTrace {
            plan,
            frames,
            reports,
            queue_high_watermark,
            queue_depth_after_submit,
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

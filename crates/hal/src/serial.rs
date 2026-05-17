//! Tang9k 串行链路的最小协议层。
//!
//! 这个模块故意放在 `sptorch-hal` 内，而不是直接塞进 `hal-ffi`：HAL 需要先拥有一份
//! 与具体串口库无关的“字节协议契约”，这样 Verilog testbench、Rust loopback、后续
//! `serial_backend` 以及 Studio 遥测都能围绕同一份帧格式做验证。真正的 UART、USB-CDC、
//! DMA 或 libloading 接入仍然属于下一层实现。
//!
//! v1 协议只追求三件事：
//! - 帧边界稳定：固定 16 字节头，显式 payload 长度，整帧 8 字节对齐。
//! - 错误可定位：magic、version、opcode、长度、checksum、padding 都分别校验。
//! - 指令可演进：先定义 32x32 MatMul tile 指令，后续再增加可变 shape 或 DMA 描述符。

use std::fmt;

/// Tang9k serial v1 帧魔数，固定为 ASCII `"SP"`。
pub const SERIAL_MAGIC: [u8; 2] = [b'S', b'P'];
/// 当前线协议版本。版本号进入帧头，Verilog 与 host backend 必须显式校验。
pub const SERIAL_VERSION: u8 = 1;
/// 固定帧头长度，单位字节。
pub const SERIAL_HEADER_LEN: usize = 16;
/// checksum 字段长度，单位字节。
pub const SERIAL_CHECKSUM_LEN: usize = 4;
/// 整帧对齐粒度，单位字节。v1 选择 8 字节是为了照顾 64-bit DMA/FIFO 读取。
pub const SERIAL_ALIGNMENT: usize = 8;
/// v1 最大 payload 长度。控制指令应远小于该值，大 payload 应走 DMA 数据区。
pub const SERIAL_MAX_PAYLOAD_LEN: usize = 64 * 1024;
/// host 侧 dry-run 默认命令队列容量。
///
/// 这个值不是 Tang9k 硬件承诺，而是给 Rust 层先建立“提交窗口”概念：真实 UART/DMA
/// 接入后可以把它替换成板端 FIFO 深度或运行时查询值。
pub const SERIAL_DEFAULT_QUEUE_CAPACITY: u32 = 64;

/// Tang9k 串行协议 v1 的 opcode。
///
/// opcode 的数值会直接进入线协议，所以这里不要因为 Rust 侧枚举顺序好看而随意调整。
/// 新增指令时优先追加新值，避免已经烧进 FPGA testbench 的解码表失效。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SerialOpcode {
    Ping = 0x01,
    Pong = 0x02,
    Matmul32x32 = 0x10,
    ScratchWrite32 = 0x20,
    ScratchRead32 = 0x21,
    ScratchValue32 = 0x22,
    ResultRead32 = 0x30,
    ResultValue32 = 0x31,
    Ack = 0x7e,
    Error = 0x7f,
}

impl SerialOpcode {
    /// 从线协议中的单字节 opcode 恢复 Rust 枚举。
    pub fn from_u8(value: u8) -> Result<Self, SerialProtocolError> {
        match value {
            0x01 => Ok(Self::Ping),
            0x02 => Ok(Self::Pong),
            0x10 => Ok(Self::Matmul32x32),
            0x20 => Ok(Self::ScratchWrite32),
            0x21 => Ok(Self::ScratchRead32),
            0x22 => Ok(Self::ScratchValue32),
            0x30 => Ok(Self::ResultRead32),
            0x31 => Ok(Self::ResultValue32),
            0x7e => Ok(Self::Ack),
            0x7f => Ok(Self::Error),
            other => Err(SerialProtocolError::UnknownOpcode(other)),
        }
    }

    fn as_u8(self) -> u8 {
        self as u8
    }
}

/// Tang9k serial v1 的标准状态码。
///
/// 状态码用于 `Ack` / `Error` payload，也可被硬件日志直接记录。解析错误仍然由 Rust
/// 侧的 [`SerialProtocolError`] 表达；状态码描述的是“对端理解了帧以后”的执行结果。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum SerialStatusCode {
    Ok = 0x0000,
    BadFrame = 0x0001,
    UnsupportedOpcode = 0x0002,
    InvalidPayload = 0x0003,
    Busy = 0x0004,
    HardwareFault = 0x0005,
}

impl SerialStatusCode {
    pub fn from_u16(value: u16) -> Result<Self, SerialProtocolError> {
        match value {
            0x0000 => Ok(Self::Ok),
            0x0001 => Ok(Self::BadFrame),
            0x0002 => Ok(Self::UnsupportedOpcode),
            0x0003 => Ok(Self::InvalidPayload),
            0x0004 => Ok(Self::Busy),
            0x0005 => Ok(Self::HardwareFault),
            other => Err(SerialProtocolError::UnknownStatusCode(other)),
        }
    }
}

/// `Ack` / `Error` 帧的标准 payload。
///
/// 布局固定为 8 字节：
/// - `0..2`: [`SerialStatusCode`]，little-endian `u16`
/// - `2..4`: reserved，必须为 0
/// - `4..8`: detail，little-endian `u32`
///
/// `detail` 的语义由 opcode 决定：例如 MatMul 可以写 tile id，链路层错误可以写硬件错误寄存器。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SerialStatusPayload {
    pub code: SerialStatusCode,
    pub detail: u32,
}

impl SerialStatusPayload {
    pub const LEN: usize = 8;

    pub fn ok() -> Self {
        Self {
            code: SerialStatusCode::Ok,
            detail: 0,
        }
    }

    pub fn encode(self) -> [u8; Self::LEN] {
        let mut payload = [0u8; Self::LEN];
        payload[0..2].copy_from_slice(&(self.code as u16).to_le_bytes());
        payload[4..8].copy_from_slice(&self.detail.to_le_bytes());
        payload
    }

    pub fn decode(payload: &[u8]) -> Result<Self, SerialProtocolError> {
        if payload.len() != Self::LEN {
            return Err(SerialProtocolError::InvalidStatusPayloadLen { actual: payload.len() });
        }
        let reserved = u16::from_le_bytes(payload[2..4].try_into().expect("payload length checked"));
        if reserved != 0 {
            return Err(SerialProtocolError::NonZeroReservedField {
                field: "status_payload.reserved",
            });
        }
        Ok(Self {
            code: SerialStatusCode::from_u16(u16::from_le_bytes(
                payload[0..2].try_into().expect("payload length checked"),
            ))?,
            detail: u32::from_le_bytes(payload[4..8].try_into().expect("payload length checked")),
        })
    }
}

/// 串行帧解析和编码阶段可能暴露的协议错误。
///
/// 这些错误是“主机侧可以立刻判断”的问题；真实硬件执行失败，例如 PE 阵列溢出、
/// DDR 地址不可达，后续应通过 `SerialOpcode::Error` 的 payload 返回，而不是混进解析错误。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SerialProtocolError {
    FrameTooShort {
        actual: usize,
    },
    InvalidAlignment {
        actual: usize,
        alignment: usize,
    },
    BadMagic {
        found: [u8; 2],
    },
    UnsupportedVersion(u8),
    UnknownOpcode(u8),
    UnknownStatusCode(u16),
    PayloadTooLarge {
        actual: usize,
        max: usize,
    },
    LengthMismatch {
        expected: usize,
        actual: usize,
    },
    ChecksumMismatch {
        expected: u32,
        actual: u32,
    },
    NonZeroPadding,
    IncompleteFrame {
        needed: usize,
        actual: usize,
    },
    InvalidMatmulPayloadLen {
        actual: usize,
    },
    InvalidScratchPayloadLen {
        opcode: SerialOpcode,
        actual: usize,
        expected: usize,
    },
    InvalidReadbackPayloadLen {
        opcode: SerialOpcode,
        actual: usize,
        expected: usize,
    },
    InvalidStatusPayloadLen {
        actual: usize,
    },
    NonZeroReservedField {
        field: &'static str,
    },
    InvalidMatmulShape {
        m: usize,
        k: usize,
        n: usize,
    },
    InvalidMatmulLayout {
        reason: String,
    },
    MatmulAddressOverflow {
        base: u64,
        element_index: usize,
        elem_size_bytes: u64,
    },
    QueueFull {
        capacity: u32,
    },
    UnexpectedResponseOpcode {
        actual: SerialOpcode,
    },
    SequenceMismatch {
        expected: u32,
        actual: u32,
    },
    CommandRejected {
        command_opcode: SerialOpcode,
        response_opcode: SerialOpcode,
        sequence: u32,
        status: SerialStatusCode,
        detail: u32,
    },
    ScratchValueMismatch {
        expected_offset: u32,
        expected_value: u32,
        actual_offset: u32,
        actual_value: u32,
    },
    ResultValueMismatch {
        expected_offset: u32,
        expected_value: u32,
        actual_offset: u32,
        actual_value: u32,
    },
    TransportIo {
        reason: String,
    },
    ResponseTimeout {
        timeout_ms: u64,
    },
}

impl fmt::Display for SerialProtocolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FrameTooShort { actual } => write!(f, "serial frame too short: {actual} bytes"),
            Self::InvalidAlignment { actual, alignment } => {
                write!(f, "serial frame length {actual} is not aligned to {alignment} bytes")
            }
            Self::BadMagic { found } => write!(f, "bad serial magic: {:02x?}", found),
            Self::UnsupportedVersion(version) => write!(f, "unsupported serial protocol version: {version}"),
            Self::UnknownOpcode(opcode) => write!(f, "unknown serial opcode: 0x{opcode:02x}"),
            Self::UnknownStatusCode(code) => write!(f, "unknown serial status code: 0x{code:04x}"),
            Self::PayloadTooLarge { actual, max } => {
                write!(f, "serial payload too large: {actual} bytes, max {max}")
            }
            Self::LengthMismatch { expected, actual } => {
                write!(f, "serial frame length mismatch: expected {expected}, got {actual}")
            }
            Self::ChecksumMismatch { expected, actual } => {
                write!(
                    f,
                    "serial checksum mismatch: expected 0x{expected:08x}, got 0x{actual:08x}"
                )
            }
            Self::NonZeroPadding => f.write_str("serial frame padding must be zero-filled"),
            Self::IncompleteFrame { needed, actual } => {
                write!(f, "serial stream needs {needed} bytes for a full frame, got {actual}")
            }
            Self::InvalidMatmulPayloadLen { actual } => {
                write!(f, "MatMul32x32 payload must be 32 bytes, got {actual}")
            }
            Self::InvalidScratchPayloadLen {
                opcode,
                actual,
                expected,
            } => {
                write!(f, "{opcode:?} payload must be {expected} bytes, got {actual}")
            }
            Self::InvalidReadbackPayloadLen {
                opcode,
                actual,
                expected,
            } => {
                write!(f, "{opcode:?} readback payload must be {expected} bytes, got {actual}")
            }
            Self::InvalidStatusPayloadLen { actual } => {
                write!(f, "serial status payload must be 8 bytes, got {actual}")
            }
            Self::NonZeroReservedField { field } => write!(f, "reserved serial field must be zero: {field}"),
            Self::InvalidMatmulShape { m, k, n } => {
                write!(f, "MatMul32x32 shape must be non-zero and divisible by 32, got [{m}, {k}, {n}]")
            }
            Self::InvalidMatmulLayout { reason } => write!(f, "invalid MatMul32x32 memory layout: {reason}"),
            Self::MatmulAddressOverflow {
                base,
                element_index,
                elem_size_bytes,
            } => write!(
                f,
                "MatMul32x32 address overflow: base=0x{base:016x}, element_index={element_index}, elem_size_bytes={elem_size_bytes}"
            ),
            Self::QueueFull { capacity } => write!(f, "serial submit queue is full, capacity={capacity}"),
            Self::UnexpectedResponseOpcode { actual } => {
                write!(f, "serial response must be Ack or Error, got {actual:?}")
            }
            Self::SequenceMismatch { expected, actual } => {
                write!(f, "serial response sequence mismatch: expected {expected}, got {actual}")
            }
            Self::CommandRejected {
                command_opcode,
                response_opcode,
                sequence,
                status,
                detail,
            } => write!(
                f,
                "serial command {command_opcode:?} seq={sequence} rejected by {response_opcode:?}: status={status:?}, detail=0x{detail:08x}"
            ),
            Self::ScratchValueMismatch {
                expected_offset,
                expected_value,
                actual_offset,
                actual_value,
            } => write!(
                f,
                "serial scratch readback mismatch: expected offset=0x{expected_offset:08x} value=0x{expected_value:08x}, got offset=0x{actual_offset:08x} value=0x{actual_value:08x}"
            ),
            Self::ResultValueMismatch {
                expected_offset,
                expected_value,
                actual_offset,
                actual_value,
            } => write!(
                f,
                "serial result readback mismatch: expected offset=0x{expected_offset:08x} value=0x{expected_value:08x}, got offset=0x{actual_offset:08x} value=0x{actual_value:08x}"
            ),
            Self::TransportIo { reason } => write!(f, "serial transport I/O failed: {reason}"),
            Self::ResponseTimeout { timeout_ms } => {
                write!(f, "serial response timed out after {timeout_ms} ms")
            }
        }
    }
}

impl std::error::Error for SerialProtocolError {}

/// 一帧完整的 Tang9k 主机侧协议消息。
///
/// 帧布局固定为：
///
/// ```text
/// 0..2    magic = "SP"
/// 2       version = 1
/// 3       opcode
/// 4..8    sequence (little-endian u32)
/// 8..12   payload_len (little-endian u32)
/// 12..14  flags (little-endian u16)
/// 14..16  reserved = 0
/// 16..N   payload
/// N..N+4  checksum(header + payload)
/// ...     zero padding until total length is 8-byte aligned
/// ```
///
/// checksum 不覆盖 padding。这样接收端能先按 header 找到 payload 和 checksum，再单独检查
/// padding 是否为零，硬件状态机也不用把补齐字节喂进校验器。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SerialFrame {
    pub opcode: SerialOpcode,
    pub sequence: u32,
    pub flags: u16,
    pub payload: Vec<u8>,
}

impl SerialFrame {
    /// 构造一个不带 flags 的协议帧。
    pub fn new(opcode: SerialOpcode, sequence: u32, payload: impl Into<Vec<u8>>) -> Self {
        Self {
            opcode,
            sequence,
            flags: 0,
            payload: payload.into(),
        }
    }

    /// 构造一个带 flags 的协议帧。
    pub fn with_flags(opcode: SerialOpcode, sequence: u32, flags: u16, payload: impl Into<Vec<u8>>) -> Self {
        Self {
            opcode,
            sequence,
            flags,
            payload: payload.into(),
        }
    }

    /// 构造标准 `Ack` 响应帧。
    ///
    /// 真实硬件和 dry-run 都应该用相同的状态载荷布局。把这个构造器放在协议层，
    /// 可以避免每个 backend 自己手写 `SerialStatusPayload::encode()` 时漏掉 reserved 规则。
    pub fn ack(sequence: u32, status: SerialStatusPayload) -> Self {
        Self::new(SerialOpcode::Ack, sequence, status.encode())
    }

    /// 构造标准 `Error` 响应帧。
    pub fn error(sequence: u32, status: SerialStatusPayload) -> Self {
        Self::new(SerialOpcode::Error, sequence, status.encode())
    }

    /// 将帧编码为可直接写入串行链路的字节序列。
    pub fn encode(&self) -> Result<Vec<u8>, SerialProtocolError> {
        validate_payload_len(self.payload.len())?;

        let body_len = SERIAL_HEADER_LEN + self.payload.len() + SERIAL_CHECKSUM_LEN;
        let padded_len = align_up(body_len, SERIAL_ALIGNMENT);
        let mut encoded = Vec::with_capacity(padded_len);

        encoded.extend_from_slice(&SERIAL_MAGIC);
        encoded.push(SERIAL_VERSION);
        encoded.push(self.opcode.as_u8());
        encoded.extend_from_slice(&self.sequence.to_le_bytes());
        encoded.extend_from_slice(&(self.payload.len() as u32).to_le_bytes());
        encoded.extend_from_slice(&self.flags.to_le_bytes());
        encoded.extend_from_slice(&0u16.to_le_bytes());
        encoded.extend_from_slice(&self.payload);

        let checksum = checksum32(&encoded);
        encoded.extend_from_slice(&checksum.to_le_bytes());
        encoded.resize(padded_len, 0);
        Ok(encoded)
    }

    /// 从一段完整帧字节中解析协议帧。
    ///
    /// 这里要求传入的是“一帧”的完整字节，而不是任意长的 stream buffer。后续真正接 UART 时，
    /// stream framing 可以先按 header 中的长度切帧，再调用这个函数做严格校验。
    pub fn decode(bytes: &[u8]) -> Result<Self, SerialProtocolError> {
        if bytes.len() < SERIAL_HEADER_LEN + SERIAL_CHECKSUM_LEN {
            return Err(SerialProtocolError::FrameTooShort { actual: bytes.len() });
        }
        if bytes.len() % SERIAL_ALIGNMENT != 0 {
            return Err(SerialProtocolError::InvalidAlignment {
                actual: bytes.len(),
                alignment: SERIAL_ALIGNMENT,
            });
        }

        let found_magic = [bytes[0], bytes[1]];
        if found_magic != SERIAL_MAGIC {
            return Err(SerialProtocolError::BadMagic { found: found_magic });
        }
        if bytes[2] != SERIAL_VERSION {
            return Err(SerialProtocolError::UnsupportedVersion(bytes[2]));
        }

        let opcode = SerialOpcode::from_u8(bytes[3])?;
        let sequence = u32::from_le_bytes(bytes[4..8].try_into().expect("slice length checked by header"));
        let payload_len = u32::from_le_bytes(bytes[8..12].try_into().expect("slice length checked by header")) as usize;
        validate_payload_len(payload_len)?;
        let flags = u16::from_le_bytes(bytes[12..14].try_into().expect("slice length checked by header"));

        let reserved = u16::from_le_bytes(bytes[14..16].try_into().expect("slice length checked by header"));
        if reserved != 0 {
            return Err(SerialProtocolError::NonZeroReservedField {
                field: "frame_header.reserved",
            });
        }

        let body_len = SERIAL_HEADER_LEN + payload_len + SERIAL_CHECKSUM_LEN;
        let padded_len = align_up(body_len, SERIAL_ALIGNMENT);
        if bytes.len() != padded_len {
            return Err(SerialProtocolError::LengthMismatch {
                expected: padded_len,
                actual: bytes.len(),
            });
        }

        let checksum_offset = SERIAL_HEADER_LEN + payload_len;
        let actual_checksum = u32::from_le_bytes(
            bytes[checksum_offset..checksum_offset + SERIAL_CHECKSUM_LEN]
                .try_into()
                .expect("checksum slice length checked by frame length"),
        );
        let expected_checksum = checksum32(&bytes[..checksum_offset]);
        if actual_checksum != expected_checksum {
            return Err(SerialProtocolError::ChecksumMismatch {
                expected: expected_checksum,
                actual: actual_checksum,
            });
        }

        if bytes[checksum_offset + SERIAL_CHECKSUM_LEN..]
            .iter()
            .any(|&byte| byte != 0)
        {
            return Err(SerialProtocolError::NonZeroPadding);
        }

        Ok(Self {
            opcode,
            sequence,
            flags,
            payload: bytes[SERIAL_HEADER_LEN..checksum_offset].to_vec(),
        })
    }
}

/// 验证设备侧对一条 host 命令的响应。
///
/// v1 暂不定义异步完成队列，因此响应必须回显原命令的 sequence；成功只能表示为
/// `Ack + SerialStatusCode::Ok`。`Ack + Busy` 也会被视为拒绝，因为它代表目标队列没有接收
/// 当前命令，host 必须重试或退避，不能把它当作已经提交成功。
pub fn validate_status_response(
    command: &SerialFrame,
    response: &SerialFrame,
) -> Result<SerialStatusPayload, SerialProtocolError> {
    if response.sequence != command.sequence {
        return Err(SerialProtocolError::SequenceMismatch {
            expected: command.sequence,
            actual: response.sequence,
        });
    }

    match response.opcode {
        SerialOpcode::Ack | SerialOpcode::Error => {
            let status = SerialStatusPayload::decode(&response.payload)?;
            if response.opcode == SerialOpcode::Ack && status.code == SerialStatusCode::Ok {
                Ok(status)
            } else {
                Err(SerialProtocolError::CommandRejected {
                    command_opcode: command.opcode,
                    response_opcode: response.opcode,
                    sequence: command.sequence,
                    status: status.code,
                    detail: status.detail,
                })
            }
        }
        actual => Err(SerialProtocolError::UnexpectedResponseOpcode { actual }),
    }
}

/// 一次 host 命令提交的可观测结果。
///
/// 这里保留 command/response 和队列深度，是为了让测试、日志、Studio 或未来硬件 bring-up
/// 能回答一个很实际的问题：命令是在发出前被背压挡住、发出后被板端拒绝，还是正常进入队列并完成。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SerialSubmitReport {
    pub command: SerialFrame,
    pub response: SerialFrame,
    pub status: SerialStatusPayload,
    pub queue_depth_before: u32,
    pub queue_depth_after_enqueue: u32,
    pub queue_depth_after_submit: u32,
}

/// host 侧同步提交队列模型。
///
/// 第一版 UART/DMA 接入前，我们先用这个小队列固定三个不变量：
/// - sequence 由 host 单调分配，允许 `u32` 自然回绕；
/// - 超过容量时必须显式返回 `Busy/QueueFull` 语义，不能静默丢帧；
/// - 无论响应成功、拒绝还是传输错误，队列深度都必须恢复，避免 dry-run 掩盖资源泄漏。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SerialSubmitQueue {
    next_sequence: u32,
    queue_depth: u32,
    capacity: u32,
    high_watermark: u32,
}

impl Default for SerialSubmitQueue {
    fn default() -> Self {
        Self::new(SERIAL_DEFAULT_QUEUE_CAPACITY)
    }
}

impl SerialSubmitQueue {
    pub fn new(capacity: u32) -> Self {
        Self {
            next_sequence: 0,
            queue_depth: 0,
            capacity,
            high_watermark: 0,
        }
    }

    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    pub fn next_sequence(&self) -> u32 {
        self.next_sequence
    }

    pub fn queue_depth(&self) -> u32 {
        self.queue_depth
    }

    pub fn high_watermark(&self) -> u32 {
        self.high_watermark
    }

    /// 为一批即将发送的帧预留连续 sequence。
    ///
    /// 预留 sequence 不等于提交命令；它只决定帧号。真正的队列深度变化发生在 [`Self::submit`]。
    pub fn reserve_sequences(&mut self, count: usize) -> u32 {
        let first = self.next_sequence;
        self.next_sequence = self.next_sequence.wrapping_add(count as u32);
        first
    }

    /// 同步提交一条命令并校验响应。
    pub fn submit<F>(&mut self, frame: &SerialFrame, exchange: F) -> Result<SerialSubmitReport, SerialProtocolError>
    where
        F: FnOnce(&SerialFrame) -> Result<SerialFrame, SerialProtocolError>,
    {
        if self.queue_depth >= self.capacity {
            return Err(SerialProtocolError::QueueFull {
                capacity: self.capacity,
            });
        }

        let queue_depth_before = self.queue_depth;
        self.queue_depth += 1;
        self.high_watermark = self.high_watermark.max(self.queue_depth);
        let queue_depth_after_enqueue = self.queue_depth;

        let response = exchange(frame);
        self.queue_depth -= 1;
        let queue_depth_after_submit = self.queue_depth;

        let response = response?;
        let status = validate_status_response(frame, &response)?;

        Ok(SerialSubmitReport {
            command: frame.clone(),
            response,
            status,
            queue_depth_before,
            queue_depth_after_enqueue,
            queue_depth_after_submit,
        })
    }
}

/// 面向 UART/USB-CDC 的增量帧切分器。
///
/// 串口读取通常只保证“字节顺序正确”，不保证一次 read 正好是一帧。这个解码器维护一个内部
/// 缓冲区：先寻找 `SERIAL_MAGIC`，再根据 header 中的 payload 长度计算完整帧长度，等字节到齐后
/// 调用 [`SerialFrame::decode`] 做严格校验。magic 前的噪声会被丢弃，便于设备复位或日志串扰后恢复。
#[derive(Debug, Default)]
pub struct SerialStreamDecoder {
    buffer: Vec<u8>,
}

impl SerialStreamDecoder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn buffered_len(&self) -> usize {
        self.buffer.len()
    }

    /// 追加一段串口字节，并尽可能解析出完整帧。
    pub fn push_bytes(&mut self, bytes: &[u8]) -> Result<Vec<SerialFrame>, SerialProtocolError> {
        self.buffer.extend_from_slice(bytes);
        let mut frames = Vec::new();

        loop {
            self.discard_until_magic();
            if self.buffer.len() < SERIAL_HEADER_LEN {
                break;
            }

            let payload_len =
                u32::from_le_bytes(self.buffer[8..12].try_into().expect("header length checked")) as usize;
            validate_payload_len(payload_len)?;
            let frame_len = encoded_frame_len(payload_len);
            if self.buffer.len() < frame_len {
                break;
            }

            let frame_bytes: Vec<u8> = self.buffer.drain(..frame_len).collect();
            frames.push(SerialFrame::decode(&frame_bytes)?);
        }

        Ok(frames)
    }

    /// 当前缓冲区已经看到 header 但还缺完整帧时，返回缺口信息。
    pub fn pending_need(&mut self) -> Result<Option<(usize, usize)>, SerialProtocolError> {
        self.discard_until_magic();
        if self.buffer.is_empty() {
            return Ok(None);
        }
        if self.buffer.len() < SERIAL_HEADER_LEN {
            return Ok(Some((SERIAL_HEADER_LEN, self.buffer.len())));
        }
        let payload_len = u32::from_le_bytes(self.buffer[8..12].try_into().expect("header length checked")) as usize;
        validate_payload_len(payload_len)?;
        let needed = encoded_frame_len(payload_len);
        if self.buffer.len() >= needed {
            Ok(None)
        } else {
            Ok(Some((needed, self.buffer.len())))
        }
    }

    /// 丢弃所有未完成的缓冲字节。
    pub fn clear(&mut self) {
        self.buffer.clear();
    }

    fn discard_until_magic(&mut self) {
        if self.buffer.len() < SERIAL_MAGIC.len() {
            return;
        }
        if let Some(pos) = self
            .buffer
            .windows(SERIAL_MAGIC.len())
            .position(|window| window == SERIAL_MAGIC)
        {
            if pos > 0 {
                self.buffer.drain(..pos);
            }
        } else {
            let keep = self
                .buffer
                .last()
                .copied()
                .filter(|byte| *byte == SERIAL_MAGIC[0])
                .into_iter()
                .collect::<Vec<_>>();
            self.buffer = keep;
        }
    }
}

/// 验证 `ScratchRead32` 的回包是否至少是一个结构正确的 `ScratchValue32`。
///
/// 这个 helper 只负责协议层形态校验，不替调用者猜测“这次读回该得到什么值”。
/// 调用方通常会在这里拿到 `offset/value`，再和自己写入过的期望值做比对，这样
/// 协议层和业务层的职责就不会搅在一起。
pub fn validate_scratch_value_response(
    command: &SerialFrame,
    response: &SerialFrame,
) -> Result<ScratchValue32Payload, SerialProtocolError> {
    if response.sequence != command.sequence {
        return Err(SerialProtocolError::SequenceMismatch {
            expected: command.sequence,
            actual: response.sequence,
        });
    }

    match response.opcode {
        SerialOpcode::ScratchValue32 => ScratchValue32Payload::decode_payload(&response.payload),
        actual => Err(SerialProtocolError::UnexpectedResponseOpcode { actual }),
    }
}

/// 验证 `ResultRead32` 的回包是否至少是一个结构正确的 `ResultValue32`。
///
/// `Result*` 和 `Scratch*` 的 payload 形状相同，但语义不同：Scratch 是任意调试槽，
/// Result 是 kernel 或 MatMul 完成后暴露给 host 的结果窗口。分成两个 opcode 能让
/// 后续硬件实现把调试寄存器和计算结果映射到不同 BRAM/FIFO，而 host 不必猜测来源。
pub fn validate_result_value_response(
    command: &SerialFrame,
    response: &SerialFrame,
) -> Result<ResultValue32Payload, SerialProtocolError> {
    if response.sequence != command.sequence {
        return Err(SerialProtocolError::SequenceMismatch {
            expected: command.sequence,
            actual: response.sequence,
        });
    }

    match response.opcode {
        SerialOpcode::ResultValue32 => ResultValue32Payload::decode_payload(&response.payload),
        actual => Err(SerialProtocolError::UnexpectedResponseOpcode { actual }),
    }
}

/// Tang9k 第一轮硬件验证使用的 32x32 MatMul tile 指令。
///
/// 指令只携带三段设备侧偏移与 tile 编号，不携带矩阵内容本身。这样它更接近真实硬件路径：
/// 主机先通过 DMA/上传接口把 A、B 放进板端内存，再发一条短控制指令让 PE 阵列计算，
/// 最后从 `out_offset` 读取结果。v1 固定 32x32，是为了让 Verilog 与 Rust 参考实现先对齐。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Matmul32x32Command {
    pub tile_id: u32,
    pub a_offset: u64,
    pub b_offset: u64,
    pub out_offset: u64,
    pub flags: u32,
}

/// 首个 K tile 需要清空输出 tile，再写入部分和。
pub const MATMUL32X32_FLAG_CLEAR_OUTPUT: u32 = 0x0000_0001;
/// 非首个 K tile 在同一个输出 tile 上累加部分和。
pub const MATMUL32X32_FLAG_ACCUMULATE: u32 = 0x0000_0002;
/// 当前指令是该输出 tile 的最后一个 K 分片，可触发硬件侧 done/flush。
pub const MATMUL32X32_FLAG_LAST_K_TILE: u32 = 0x0000_0004;

impl Matmul32x32Command {
    pub const M: usize = 32;
    pub const K: usize = 32;
    pub const N: usize = 32;
    pub const PAYLOAD_LEN: usize = 32;

    /// 创建一条默认 flags 为 0 的 32x32 tile 指令。
    pub fn new(tile_id: u32, a_offset: u64, b_offset: u64, out_offset: u64) -> Self {
        Self {
            tile_id,
            a_offset,
            b_offset,
            out_offset,
            flags: 0,
        }
    }

    /// 返回该指令固定描述的矩阵形状。
    pub fn shape(&self) -> (usize, usize, usize) {
        (Self::M, Self::K, Self::N)
    }

    /// 编码为 `SerialOpcode::Matmul32x32` 的 payload。
    pub fn encode_payload(&self) -> [u8; Self::PAYLOAD_LEN] {
        let mut payload = [0u8; Self::PAYLOAD_LEN];
        payload[0..4].copy_from_slice(&self.tile_id.to_le_bytes());
        payload[4..12].copy_from_slice(&self.a_offset.to_le_bytes());
        payload[12..20].copy_from_slice(&self.b_offset.to_le_bytes());
        payload[20..28].copy_from_slice(&self.out_offset.to_le_bytes());
        payload[28..32].copy_from_slice(&self.flags.to_le_bytes());
        payload
    }

    /// 从 MatMul payload 解码控制指令。
    pub fn decode_payload(payload: &[u8]) -> Result<Self, SerialProtocolError> {
        if payload.len() != Self::PAYLOAD_LEN {
            return Err(SerialProtocolError::InvalidMatmulPayloadLen { actual: payload.len() });
        }
        Ok(Self {
            tile_id: u32::from_le_bytes(payload[0..4].try_into().expect("payload length checked")),
            a_offset: u64::from_le_bytes(payload[4..12].try_into().expect("payload length checked")),
            b_offset: u64::from_le_bytes(payload[12..20].try_into().expect("payload length checked")),
            out_offset: u64::from_le_bytes(payload[20..28].try_into().expect("payload length checked")),
            flags: u32::from_le_bytes(payload[28..32].try_into().expect("payload length checked")),
        })
    }

    /// 包装成完整串行帧，便于调用侧直接进入发送队列。
    pub fn into_frame(self, sequence: u32) -> SerialFrame {
        SerialFrame::new(SerialOpcode::Matmul32x32, sequence, self.encode_payload())
    }

    /// 由 MatMul 命令字段稳定派生出的结果窗口摘要。
    ///
    /// v1 的 Tang9k 结果窗口还不是完整矩阵回读，所以这里先把 command 字段折叠成一个
    /// 32-bit 摘要，作为板端结果槽的 smoke 值。这样 host 和 RTL 都能用同一条语义链路
    /// 证明“MatMul 命令确实触发了一个可读回结果窗口”。
    pub fn smoke_result_summary(&self) -> ResultValue32Payload {
        let value = self.tile_id
            ^ self.flags
            ^ self.a_offset as u32
            ^ (self.a_offset >> 32) as u32
            ^ self.b_offset as u32
            ^ (self.b_offset >> 32) as u32
            ^ self.out_offset as u32
            ^ (self.out_offset >> 32) as u32;
        ResultValue32Payload::new(self.out_offset as u32, value)
    }
}

/// 最小数据面烟测使用的 32-bit scratch 写指令。
///
/// 这条指令不代表最终 DMA 方案，只是把“host 写入设备侧状态，再从设备侧读回”的路径先练稳。
/// 地址字段保留下来，是为了让同一套 payload 将来能自然扩展到小块寄存器窗口或 BRAM scratch 区。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScratchWrite32Command {
    pub offset: u32,
    pub value: u32,
}

impl ScratchWrite32Command {
    pub const PAYLOAD_LEN: usize = 8;

    pub fn new(offset: u32, value: u32) -> Self {
        Self { offset, value }
    }

    pub fn encode_payload(&self) -> [u8; Self::PAYLOAD_LEN] {
        let mut payload = [0u8; Self::PAYLOAD_LEN];
        payload[0..4].copy_from_slice(&self.offset.to_le_bytes());
        payload[4..8].copy_from_slice(&self.value.to_le_bytes());
        payload
    }

    pub fn decode_payload(payload: &[u8]) -> Result<Self, SerialProtocolError> {
        if payload.len() != Self::PAYLOAD_LEN {
            return Err(SerialProtocolError::InvalidScratchPayloadLen {
                opcode: SerialOpcode::ScratchWrite32,
                actual: payload.len(),
                expected: Self::PAYLOAD_LEN,
            });
        }
        Ok(Self {
            offset: u32::from_le_bytes(payload[0..4].try_into().expect("payload length checked")),
            value: u32::from_le_bytes(payload[4..8].try_into().expect("payload length checked")),
        })
    }

    pub fn into_frame(self, sequence: u32) -> SerialFrame {
        SerialFrame::new(SerialOpcode::ScratchWrite32, sequence, self.encode_payload())
    }
}

/// 最小数据面烟测使用的 32-bit scratch 读指令。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScratchRead32Command {
    pub offset: u32,
}

impl ScratchRead32Command {
    pub const PAYLOAD_LEN: usize = 4;

    pub fn new(offset: u32) -> Self {
        Self { offset }
    }

    pub fn encode_payload(&self) -> [u8; Self::PAYLOAD_LEN] {
        self.offset.to_le_bytes()
    }

    pub fn decode_payload(payload: &[u8]) -> Result<Self, SerialProtocolError> {
        if payload.len() != Self::PAYLOAD_LEN {
            return Err(SerialProtocolError::InvalidScratchPayloadLen {
                opcode: SerialOpcode::ScratchRead32,
                actual: payload.len(),
                expected: Self::PAYLOAD_LEN,
            });
        }
        Ok(Self {
            offset: u32::from_le_bytes(payload[0..4].try_into().expect("payload length checked")),
        })
    }

    pub fn into_frame(self, sequence: u32) -> SerialFrame {
        SerialFrame::new(SerialOpcode::ScratchRead32, sequence, self.encode_payload())
    }
}

/// `ScratchRead32` 的标准响应 payload。
///
/// 响应独立使用 `ScratchValue32` opcode，而不是把 value 塞进 `Ack.detail`，是为了保持 ACK 只表达
/// “命令是否被接收”，真实数据返回走显式数据帧。这条边界后面会复用到 MatMul 结果回读。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScratchValue32Payload {
    pub offset: u32,
    pub value: u32,
}

impl ScratchValue32Payload {
    pub const PAYLOAD_LEN: usize = 8;

    pub fn new(offset: u32, value: u32) -> Self {
        Self { offset, value }
    }

    pub fn encode_payload(&self) -> [u8; Self::PAYLOAD_LEN] {
        let mut payload = [0u8; Self::PAYLOAD_LEN];
        payload[0..4].copy_from_slice(&self.offset.to_le_bytes());
        payload[4..8].copy_from_slice(&self.value.to_le_bytes());
        payload
    }

    pub fn decode_payload(payload: &[u8]) -> Result<Self, SerialProtocolError> {
        if payload.len() != Self::PAYLOAD_LEN {
            return Err(SerialProtocolError::InvalidScratchPayloadLen {
                opcode: SerialOpcode::ScratchValue32,
                actual: payload.len(),
                expected: Self::PAYLOAD_LEN,
            });
        }
        Ok(Self {
            offset: u32::from_le_bytes(payload[0..4].try_into().expect("payload length checked")),
            value: u32::from_le_bytes(payload[4..8].try_into().expect("payload length checked")),
        })
    }

    pub fn into_frame(self, sequence: u32) -> SerialFrame {
        SerialFrame::new(SerialOpcode::ScratchValue32, sequence, self.encode_payload())
    }
}

/// `ResultRead32` 的标准命令 payload。
///
/// 这条命令是给“结果窗口”准备的最小读回接口。它和 scratch 读写共用同样的
/// 32-bit offset/value 形状，但语义上更接近 MatMul / Kernel 结果槽，而不是任意调试寄存器。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResultRead32Command {
    pub offset: u32,
}

impl ResultRead32Command {
    pub const PAYLOAD_LEN: usize = 4;

    pub fn new(offset: u32) -> Self {
        Self { offset }
    }

    pub fn encode_payload(&self) -> [u8; Self::PAYLOAD_LEN] {
        self.offset.to_le_bytes()
    }

    pub fn decode_payload(payload: &[u8]) -> Result<Self, SerialProtocolError> {
        if payload.len() != Self::PAYLOAD_LEN {
            return Err(SerialProtocolError::InvalidReadbackPayloadLen {
                opcode: SerialOpcode::ResultRead32,
                actual: payload.len(),
                expected: Self::PAYLOAD_LEN,
            });
        }
        Ok(Self {
            offset: u32::from_le_bytes(payload[0..4].try_into().expect("payload length checked")),
        })
    }

    pub fn into_frame(self, sequence: u32) -> SerialFrame {
        SerialFrame::new(SerialOpcode::ResultRead32, sequence, self.encode_payload())
    }
}

/// `ResultRead32` 的标准响应 payload。
///
/// 第一版只返回一个 32-bit 摘要值，用于证明 MatMul 命令可以触发板端结果窗口更新。
/// 真正 PE 阵列接入后，这个 offset/value 可以继续作为小块 BRAM 窗口或结果摘要寄存器。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResultValue32Payload {
    pub offset: u32,
    pub value: u32,
}

impl ResultValue32Payload {
    pub const PAYLOAD_LEN: usize = 8;

    pub fn new(offset: u32, value: u32) -> Self {
        Self { offset, value }
    }

    pub fn encode_payload(&self) -> [u8; Self::PAYLOAD_LEN] {
        let mut payload = [0u8; Self::PAYLOAD_LEN];
        payload[0..4].copy_from_slice(&self.offset.to_le_bytes());
        payload[4..8].copy_from_slice(&self.value.to_le_bytes());
        payload
    }

    pub fn decode_payload(payload: &[u8]) -> Result<Self, SerialProtocolError> {
        if payload.len() != Self::PAYLOAD_LEN {
            return Err(SerialProtocolError::InvalidReadbackPayloadLen {
                opcode: SerialOpcode::ResultValue32,
                actual: payload.len(),
                expected: Self::PAYLOAD_LEN,
            });
        }
        Ok(Self {
            offset: u32::from_le_bytes(payload[0..4].try_into().expect("payload length checked")),
            value: u32::from_le_bytes(payload[4..8].try_into().expect("payload length checked")),
        })
    }

    pub fn into_frame(self, sequence: u32) -> SerialFrame {
        SerialFrame::new(SerialOpcode::ResultValue32, sequence, self.encode_payload())
    }
}

/// 主机对 Tang9k 板端 MatMul 缓冲区的 row-major 视图。
///
/// `a_*`、`b_*`、`out_*` 都是设备侧字节偏移，不是主机虚拟地址。stride 使用“元素数”
/// 而不是字节数，是为了和 Tensor/矩阵 shape 的语义保持一致，最终乘以 `elem_size_bytes`
/// 的动作集中在协议规划层完成。第一轮只接受 row-major 连续或带行 padding 的布局。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatmulMemoryLayout {
    pub a_base: u64,
    pub b_base: u64,
    pub out_base: u64,
    pub a_row_stride: usize,
    pub b_row_stride: usize,
    pub out_row_stride: usize,
    pub elem_size_bytes: u64,
}

impl MatmulMemoryLayout {
    /// 构造标准 row-major 连续布局。
    pub fn row_major(
        m: usize,
        k: usize,
        n: usize,
        a_base: u64,
        b_base: u64,
        out_base: u64,
        elem_size_bytes: u64,
    ) -> Self {
        let _ = m;
        Self {
            a_base,
            b_base,
            out_base,
            a_row_stride: k,
            b_row_stride: n,
            out_row_stride: n,
            elem_size_bytes,
        }
    }

    fn validate(&self, k: usize, n: usize) -> Result<(), SerialProtocolError> {
        if self.elem_size_bytes == 0 {
            return Err(SerialProtocolError::InvalidMatmulLayout {
                reason: "elem_size_bytes must be greater than zero".into(),
            });
        }
        if self.a_row_stride < k {
            return Err(SerialProtocolError::InvalidMatmulLayout {
                reason: format!("a_row_stride {} is smaller than k {}", self.a_row_stride, k),
            });
        }
        if self.b_row_stride < n {
            return Err(SerialProtocolError::InvalidMatmulLayout {
                reason: format!("b_row_stride {} is smaller than n {}", self.b_row_stride, n),
            });
        }
        if self.out_row_stride < n {
            return Err(SerialProtocolError::InvalidMatmulLayout {
                reason: format!("out_row_stride {} is smaller than n {}", self.out_row_stride, n),
            });
        }
        Ok(())
    }
}

/// 一组可以顺序发送到 Tang9k 的 32x32 MatMul 指令。
///
/// 指令顺序采用 `m_tile -> n_tile -> k_tile`：先锁定一个输出 tile，再遍历所有 K 分片。
/// 这样硬件侧可以用 flags 判断“清零输出、累加部分和、最后 flush”，不需要额外维护复杂的
/// 输出 tile 生命周期表。它不是最终高性能调度器，但足够支撑 Week 2 的端到端协议验收。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Matmul32x32Plan {
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub tiles_m: usize,
    pub tiles_k: usize,
    pub tiles_n: usize,
    pub commands: Vec<Matmul32x32Command>,
}

impl Matmul32x32Plan {
    pub fn command_count(&self) -> usize {
        self.commands.len()
    }

    /// 将指令序列包装成连续 sequence 的串行帧。
    ///
    /// sequence 使用 `wrapping_add`，和多数硬件队列计数器一样允许自然回绕；真实 backend
    /// 如果需要窗口确认，可以在发送层追加更严格的 outstanding 限制。
    pub fn frames(&self, first_sequence: u32) -> Vec<SerialFrame> {
        self.commands
            .iter()
            .enumerate()
            .map(|(idx, command)| command.into_frame(first_sequence.wrapping_add(idx as u32)))
            .collect()
    }
}

/// 为 row-major MatMul 生成 Tang9k 32x32 tile 指令序列。
///
/// v1 要求 `m`、`k`、`n` 都是 32 的倍数。这个限制看起来保守，但它能让 Verilog PE 阵列、
/// Rust CPU baseline 和串口协议先共享最小确定语义；边缘 tile padding/裁剪应在下一轮扩展，
/// 避免第一版硬件 bring-up 同时背上 shape 边界和链路稳定性两个风险。
pub fn plan_matmul32x32_commands(
    m: usize,
    k: usize,
    n: usize,
    layout: MatmulMemoryLayout,
) -> Result<Matmul32x32Plan, SerialProtocolError> {
    if m == 0
        || k == 0
        || n == 0
        || m % Matmul32x32Command::M != 0
        || k % Matmul32x32Command::K != 0
        || n % Matmul32x32Command::N != 0
    {
        return Err(SerialProtocolError::InvalidMatmulShape { m, k, n });
    }
    layout.validate(k, n)?;

    let tiles_m = m / Matmul32x32Command::M;
    let tiles_k = k / Matmul32x32Command::K;
    let tiles_n = n / Matmul32x32Command::N;
    let mut commands = Vec::with_capacity(tiles_m * tiles_k * tiles_n);
    let mut tile_id = 0u32;

    for m_tile in 0..tiles_m {
        let row_start = m_tile * Matmul32x32Command::M;
        for n_tile in 0..tiles_n {
            let col_start = n_tile * Matmul32x32Command::N;
            for k_tile in 0..tiles_k {
                let k_start = k_tile * Matmul32x32Command::K;
                let mut flags = if k_tile == 0 {
                    MATMUL32X32_FLAG_CLEAR_OUTPUT
                } else {
                    MATMUL32X32_FLAG_ACCUMULATE
                };
                if k_tile + 1 == tiles_k {
                    flags |= MATMUL32X32_FLAG_LAST_K_TILE;
                }

                let a_element = row_start
                    .checked_mul(layout.a_row_stride)
                    .and_then(|base| base.checked_add(k_start))
                    .ok_or_else(|| SerialProtocolError::InvalidMatmulLayout {
                        reason: "A tile element index overflow".into(),
                    })?;
                let b_element = k_start
                    .checked_mul(layout.b_row_stride)
                    .and_then(|base| base.checked_add(col_start))
                    .ok_or_else(|| SerialProtocolError::InvalidMatmulLayout {
                        reason: "B tile element index overflow".into(),
                    })?;
                let out_element = row_start
                    .checked_mul(layout.out_row_stride)
                    .and_then(|base| base.checked_add(col_start))
                    .ok_or_else(|| SerialProtocolError::InvalidMatmulLayout {
                        reason: "output tile element index overflow".into(),
                    })?;

                commands.push(Matmul32x32Command {
                    tile_id,
                    a_offset: checked_device_offset(layout.a_base, a_element, layout.elem_size_bytes)?,
                    b_offset: checked_device_offset(layout.b_base, b_element, layout.elem_size_bytes)?,
                    out_offset: checked_device_offset(layout.out_base, out_element, layout.elem_size_bytes)?,
                    flags,
                });
                tile_id = tile_id.wrapping_add(1);
            }
        }
    }

    Ok(Matmul32x32Plan {
        m,
        k,
        n,
        tiles_m,
        tiles_k,
        tiles_n,
        commands,
    })
}

/// 纯内存 loopback 传输器，用来在没有 Tang9k 板卡时压测帧收发稳定性。
#[derive(Debug, Default)]
pub struct LoopbackSerialTransport {
    frames_seen: usize,
}

impl LoopbackSerialTransport {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn frames_seen(&self) -> usize {
        self.frames_seen
    }

    /// 模拟“收到一帧、严格解析、原样回发”的最小硬件链路。
    pub fn exchange(&mut self, encoded: &[u8]) -> Result<Vec<u8>, SerialProtocolError> {
        let frame = SerialFrame::decode(encoded)?;
        self.frames_seen += 1;
        frame.encode()
    }
}

fn validate_payload_len(len: usize) -> Result<(), SerialProtocolError> {
    if len > SERIAL_MAX_PAYLOAD_LEN {
        return Err(SerialProtocolError::PayloadTooLarge {
            actual: len,
            max: SERIAL_MAX_PAYLOAD_LEN,
        });
    }
    Ok(())
}

fn encoded_frame_len(payload_len: usize) -> usize {
    align_up(SERIAL_HEADER_LEN + payload_len + SERIAL_CHECKSUM_LEN, SERIAL_ALIGNMENT)
}

fn checked_device_offset(base: u64, element_index: usize, elem_size_bytes: u64) -> Result<u64, SerialProtocolError> {
    let element_index_u64 = u64::try_from(element_index).map_err(|_| SerialProtocolError::MatmulAddressOverflow {
        base,
        element_index,
        elem_size_bytes,
    })?;
    let byte_offset =
        element_index_u64
            .checked_mul(elem_size_bytes)
            .ok_or(SerialProtocolError::MatmulAddressOverflow {
                base,
                element_index,
                elem_size_bytes,
            })?;
    base.checked_add(byte_offset)
        .ok_or(SerialProtocolError::MatmulAddressOverflow {
            base,
            element_index,
            elem_size_bytes,
        })
}

fn align_up(value: usize, alignment: usize) -> usize {
    debug_assert!(alignment.is_power_of_two());
    (value + alignment - 1) & !(alignment - 1)
}

fn checksum32(bytes: &[u8]) -> u32 {
    // FNV-1a 足够小，FPGA 侧也容易照着实现；这里用它做帧损坏哨兵，不把它包装成安全承诺。
    let mut hash = 0x811c_9dc5u32;
    for &byte in bytes {
        hash ^= byte as u32;
        hash = hash.wrapping_mul(0x0100_0193);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serial_frame_roundtrips_metadata_and_payload() {
        let frame = SerialFrame::with_flags(SerialOpcode::Ping, 42, 0x0003, b"hello".to_vec());
        let encoded = frame.encode().unwrap();
        let decoded = SerialFrame::decode(&encoded).unwrap();
        assert_eq!(decoded, frame);
    }

    #[test]
    fn serial_frame_is_eight_byte_aligned() {
        let frame = SerialFrame::new(SerialOpcode::Ack, 7, vec![1, 2, 3]);
        let encoded = frame.encode().unwrap();
        assert_eq!(encoded.len() % SERIAL_ALIGNMENT, 0);
    }

    #[test]
    fn serial_frame_rejects_length_mismatch() {
        let frame = SerialFrame::new(SerialOpcode::Ping, 1, b"abc".to_vec());
        let mut encoded = frame.encode().unwrap();
        encoded.extend_from_slice(&[0u8; SERIAL_ALIGNMENT]);
        let err = SerialFrame::decode(&encoded).unwrap_err();
        assert!(matches!(err, SerialProtocolError::LengthMismatch { .. }));
    }

    #[test]
    fn serial_frame_rejects_checksum_corruption() {
        let frame = SerialFrame::new(SerialOpcode::Ping, 1, b"abc".to_vec());
        let mut encoded = frame.encode().unwrap();
        encoded[SERIAL_HEADER_LEN] ^= 0x55;
        let err = SerialFrame::decode(&encoded).unwrap_err();
        assert!(matches!(err, SerialProtocolError::ChecksumMismatch { .. }));
    }

    #[test]
    fn serial_frame_rejects_non_zero_padding() {
        let frame = SerialFrame::new(SerialOpcode::Ping, 1, b"a".to_vec());
        let mut encoded = frame.encode().unwrap();
        let last = encoded.last_mut().unwrap();
        *last = 1;
        let err = SerialFrame::decode(&encoded).unwrap_err();
        assert_eq!(err, SerialProtocolError::NonZeroPadding);
    }

    #[test]
    fn serial_frame_rejects_non_zero_header_reserved() {
        let frame = SerialFrame::new(SerialOpcode::Ping, 1, b"abc".to_vec());
        let mut encoded = frame.encode().unwrap();
        encoded[14] = 1;
        let err = SerialFrame::decode(&encoded).unwrap_err();
        assert_eq!(
            err,
            SerialProtocolError::NonZeroReservedField {
                field: "frame_header.reserved"
            }
        );
    }

    #[test]
    fn serial_status_payload_roundtrips() {
        let payload = SerialStatusPayload {
            code: SerialStatusCode::Busy,
            detail: 42,
        };
        let encoded = payload.encode();
        assert_eq!(SerialStatusPayload::decode(&encoded).unwrap(), payload);
    }

    #[test]
    fn serial_status_payload_rejects_reserved_bits() {
        let mut encoded = SerialStatusPayload::ok().encode();
        encoded[2] = 1;
        let err = SerialStatusPayload::decode(&encoded).unwrap_err();
        assert_eq!(
            err,
            SerialProtocolError::NonZeroReservedField {
                field: "status_payload.reserved"
            }
        );
    }

    #[test]
    fn serial_status_payload_rejects_unknown_status_code() {
        let mut encoded = [0u8; SerialStatusPayload::LEN];
        encoded[0..2].copy_from_slice(&0xffffu16.to_le_bytes());
        let err = SerialStatusPayload::decode(&encoded).unwrap_err();
        assert_eq!(err, SerialProtocolError::UnknownStatusCode(0xffff));
    }

    #[test]
    fn status_response_accepts_ack_ok_only() {
        let command = SerialFrame::new(SerialOpcode::Matmul32x32, 7, vec![0; Matmul32x32Command::PAYLOAD_LEN]);
        let response = SerialFrame::ack(7, SerialStatusPayload::ok());

        let status = validate_status_response(&command, &response).unwrap();
        assert_eq!(status.code, SerialStatusCode::Ok);
    }

    #[test]
    fn status_response_rejects_busy_ack() {
        let command = SerialFrame::new(SerialOpcode::Matmul32x32, 7, vec![0; Matmul32x32Command::PAYLOAD_LEN]);
        let response = SerialFrame::ack(
            7,
            SerialStatusPayload {
                code: SerialStatusCode::Busy,
                detail: 0x1234,
            },
        );

        let err = validate_status_response(&command, &response).unwrap_err();
        assert_eq!(
            err,
            SerialProtocolError::CommandRejected {
                command_opcode: SerialOpcode::Matmul32x32,
                response_opcode: SerialOpcode::Ack,
                sequence: 7,
                status: SerialStatusCode::Busy,
                detail: 0x1234,
            }
        );
    }

    #[test]
    fn submit_queue_tracks_depth_and_high_watermark() {
        let mut queue = SerialSubmitQueue::new(2);
        let first_sequence = queue.reserve_sequences(2);
        let frames = [
            SerialFrame::new(SerialOpcode::Ping, first_sequence, Vec::new()),
            SerialFrame::new(SerialOpcode::Ping, first_sequence.wrapping_add(1), Vec::new()),
        ];

        let first = queue
            .submit(&frames[0], |frame| {
                Ok(SerialFrame::ack(frame.sequence, SerialStatusPayload::ok()))
            })
            .unwrap();
        let second = queue
            .submit(&frames[1], |frame| {
                Ok(SerialFrame::ack(frame.sequence, SerialStatusPayload::ok()))
            })
            .unwrap();

        assert_eq!(first.command.sequence, 0);
        assert_eq!(second.command.sequence, 1);
        assert_eq!(queue.next_sequence(), 2);
        assert_eq!(queue.queue_depth(), 0);
        assert_eq!(queue.high_watermark(), 1);
        assert_eq!(first.queue_depth_before, 0);
        assert_eq!(first.queue_depth_after_enqueue, 1);
        assert_eq!(first.queue_depth_after_submit, 0);
    }

    #[test]
    fn submit_queue_drains_after_transport_error() {
        let mut queue = SerialSubmitQueue::new(1);
        let frame = SerialFrame::new(SerialOpcode::Ping, 0, Vec::new());

        let err = queue
            .submit(&frame, |_| Err(SerialProtocolError::BadMagic { found: [0x00, 0x00] }))
            .unwrap_err();

        assert_eq!(err, SerialProtocolError::BadMagic { found: [0x00, 0x00] });
        assert_eq!(queue.queue_depth(), 0);
        assert_eq!(queue.high_watermark(), 1);
    }

    #[test]
    fn submit_queue_rejects_full_capacity() {
        let mut queue = SerialSubmitQueue::new(0);
        let frame = SerialFrame::new(SerialOpcode::Ping, 0, Vec::new());

        let err = queue
            .submit(&frame, |_| Ok(SerialFrame::ack(0, SerialStatusPayload::ok())))
            .unwrap_err();

        assert_eq!(err, SerialProtocolError::QueueFull { capacity: 0 });
        assert_eq!(queue.queue_depth(), 0);
    }

    #[test]
    fn serial_stream_decoder_waits_for_fragmented_frame() {
        let frame = SerialFrame::new(SerialOpcode::Ping, 7, b"fragmented".to_vec());
        let encoded = frame.encode().unwrap();
        let split = 5;
        let mut decoder = SerialStreamDecoder::new();

        assert!(decoder.push_bytes(&encoded[..split]).unwrap().is_empty());
        assert_eq!(decoder.pending_need().unwrap().unwrap().1, split);

        let frames = decoder.push_bytes(&encoded[split..]).unwrap();
        assert_eq!(frames, vec![frame]);
        assert_eq!(decoder.buffered_len(), 0);
    }

    #[test]
    fn serial_stream_decoder_recovers_after_noise() {
        let frame = SerialFrame::new(SerialOpcode::Pong, 8, b"clean".to_vec());
        let mut stream = vec![0xaa, 0xbb, b'S'];
        stream.extend_from_slice(&[0x00, 0x11]);
        stream.extend_from_slice(&frame.encode().unwrap());

        let mut decoder = SerialStreamDecoder::new();
        let frames = decoder.push_bytes(&stream).unwrap();
        assert_eq!(frames, vec![frame]);
        assert_eq!(decoder.buffered_len(), 0);
    }

    #[test]
    fn serial_stream_decoder_parses_multiple_frames() {
        let first = SerialFrame::new(SerialOpcode::Ping, 1, b"a".to_vec());
        let second = SerialFrame::new(SerialOpcode::Ack, 2, SerialStatusPayload::ok().encode());
        let mut stream = first.encode().unwrap();
        stream.extend_from_slice(&second.encode().unwrap());

        let mut decoder = SerialStreamDecoder::new();
        let frames = decoder.push_bytes(&stream).unwrap();
        assert_eq!(frames, vec![first, second]);
    }

    #[test]
    fn serial_stream_decoder_preserves_split_magic_prefix() {
        let frame = SerialFrame::new(SerialOpcode::Ping, 3, Vec::new());
        let mut decoder = SerialStreamDecoder::new();

        assert!(decoder.push_bytes(&[0xff, b'S']).unwrap().is_empty());
        assert_eq!(decoder.buffered_len(), 1);

        let encoded = frame.encode().unwrap();
        let frames = decoder.push_bytes(&encoded[1..]).unwrap();
        assert_eq!(frames, vec![frame]);
    }

    #[test]
    fn serial_stream_decoder_rejects_oversized_declared_payload() {
        let mut header = vec![0u8; SERIAL_HEADER_LEN];
        header[0..2].copy_from_slice(&SERIAL_MAGIC);
        header[2] = SERIAL_VERSION;
        header[3] = SerialOpcode::Ping as u8;
        header[8..12].copy_from_slice(&((SERIAL_MAX_PAYLOAD_LEN as u32) + 1).to_le_bytes());

        let mut decoder = SerialStreamDecoder::new();
        let err = decoder.push_bytes(&header).unwrap_err();
        assert_eq!(
            err,
            SerialProtocolError::PayloadTooLarge {
                actual: SERIAL_MAX_PAYLOAD_LEN + 1,
                max: SERIAL_MAX_PAYLOAD_LEN
            }
        );
    }

    #[test]
    fn matmul32x32_command_payload_roundtrips() {
        let mut command = Matmul32x32Command::new(9, 0x1000, 0x2000, 0x3000);
        command.flags = 0x5a5a_0001;
        let payload = command.encode_payload();
        let decoded = Matmul32x32Command::decode_payload(&payload).unwrap();
        assert_eq!(decoded, command);
        assert_eq!(decoded.shape(), (32, 32, 32));
    }

    #[test]
    fn matmul32x32_command_wraps_into_frame() {
        let command = Matmul32x32Command::new(2, 128, 4096, 8192);
        let frame = command.into_frame(99);
        let decoded_frame = SerialFrame::decode(&frame.encode().unwrap()).unwrap();
        let decoded_command = Matmul32x32Command::decode_payload(&decoded_frame.payload).unwrap();
        assert_eq!(decoded_frame.opcode, SerialOpcode::Matmul32x32);
        assert_eq!(decoded_frame.sequence, 99);
        assert_eq!(decoded_command, command);
    }

    #[test]
    fn matmul32x32_smoke_result_summary_is_stable() {
        let mut command = Matmul32x32Command::new(
            0x0102_0304,
            0x1112_1314_1516_1718,
            0x2122_2324_2526_2728,
            0x3132_3334_3536_3738,
        );
        command.flags = MATMUL32X32_FLAG_CLEAR_OUTPUT | MATMUL32X32_FLAG_LAST_K_TILE;

        assert_eq!(
            command.smoke_result_summary(),
            ResultValue32Payload::new(0x3536_3738, 0x0506_070d)
        );
    }

    #[test]
    fn scratch_write32_command_roundtrips() {
        let command = ScratchWrite32Command::new(0x44, 0x1122_3344);
        let frame = command.into_frame(77);
        let decoded_frame = SerialFrame::decode(&frame.encode().unwrap()).unwrap();
        let decoded_command = ScratchWrite32Command::decode_payload(&decoded_frame.payload).unwrap();
        assert_eq!(decoded_frame.opcode, SerialOpcode::ScratchWrite32);
        assert_eq!(decoded_frame.sequence, 77);
        assert_eq!(decoded_command, command);
    }

    #[test]
    fn scratch_read32_command_roundtrips() {
        let command = ScratchRead32Command::new(0x44);
        let frame = command.into_frame(78);
        let decoded_frame = SerialFrame::decode(&frame.encode().unwrap()).unwrap();
        let decoded_command = ScratchRead32Command::decode_payload(&decoded_frame.payload).unwrap();
        assert_eq!(decoded_frame.opcode, SerialOpcode::ScratchRead32);
        assert_eq!(decoded_frame.sequence, 78);
        assert_eq!(decoded_command, command);
    }

    #[test]
    fn scratch_value32_payload_roundtrips() {
        let payload = ScratchValue32Payload::new(0x44, 0x1122_3344);
        let frame = payload.into_frame(79);
        let decoded_frame = SerialFrame::decode(&frame.encode().unwrap()).unwrap();
        let decoded_payload = ScratchValue32Payload::decode_payload(&decoded_frame.payload).unwrap();
        assert_eq!(decoded_frame.opcode, SerialOpcode::ScratchValue32);
        assert_eq!(decoded_frame.sequence, 79);
        assert_eq!(decoded_payload, payload);
    }

    #[test]
    fn result32_payloads_roundtrip() {
        let command = ResultRead32Command::new(0x88);
        let frame = command.into_frame(80);
        let decoded_frame = SerialFrame::decode(&frame.encode().unwrap()).unwrap();
        let decoded_command = ResultRead32Command::decode_payload(&decoded_frame.payload).unwrap();
        assert_eq!(decoded_frame.opcode, SerialOpcode::ResultRead32);
        assert_eq!(decoded_frame.sequence, 80);
        assert_eq!(decoded_command, command);

        let payload = ResultValue32Payload::new(0x88, 0xdead_beef);
        let frame = payload.into_frame(81);
        let decoded_frame = SerialFrame::decode(&frame.encode().unwrap()).unwrap();
        let decoded_payload = ResultValue32Payload::decode_payload(&decoded_frame.payload).unwrap();
        assert_eq!(decoded_frame.opcode, SerialOpcode::ResultValue32);
        assert_eq!(decoded_frame.sequence, 81);
        assert_eq!(decoded_payload, payload);
    }

    #[test]
    fn matmul32x32_plan_tiles_64_cube_in_stable_order() {
        let layout = MatmulMemoryLayout::row_major(64, 64, 64, 0x1000, 0x2000, 0x3000, 4);
        let plan = plan_matmul32x32_commands(64, 64, 64, layout).unwrap();

        assert_eq!(plan.tiles_m, 2);
        assert_eq!(plan.tiles_k, 2);
        assert_eq!(plan.tiles_n, 2);
        assert_eq!(plan.command_count(), 8);

        let first = plan.commands[0];
        assert_eq!(first.tile_id, 0);
        assert_eq!(first.a_offset, 0x1000);
        assert_eq!(first.b_offset, 0x2000);
        assert_eq!(first.out_offset, 0x3000);
        assert_eq!(first.flags, MATMUL32X32_FLAG_CLEAR_OUTPUT);

        let second_k = plan.commands[1];
        assert_eq!(second_k.a_offset, 0x1000 + 32 * 4);
        assert_eq!(second_k.b_offset, 0x2000 + 32 * 64 * 4);
        assert_eq!(second_k.out_offset, 0x3000);
        assert_eq!(
            second_k.flags,
            MATMUL32X32_FLAG_ACCUMULATE | MATMUL32X32_FLAG_LAST_K_TILE
        );

        let next_n_tile = plan.commands[2];
        assert_eq!(next_n_tile.b_offset, 0x2000 + 32 * 4);
        assert_eq!(next_n_tile.out_offset, 0x3000 + 32 * 4);

        let next_m_tile = plan.commands[4];
        assert_eq!(next_m_tile.a_offset, 0x1000 + 32 * 64 * 4);
        assert_eq!(next_m_tile.out_offset, 0x3000 + 32 * 64 * 4);
    }

    #[test]
    fn matmul32x32_plan_frames_are_sequenced_and_decodable() {
        let layout = MatmulMemoryLayout::row_major(32, 32, 64, 0x1000, 0x2000, 0x3000, 4);
        let plan = plan_matmul32x32_commands(32, 32, 64, layout).unwrap();
        let frames = plan.frames(500);

        assert_eq!(frames.len(), 2);
        for (idx, frame) in frames.iter().enumerate() {
            let decoded = SerialFrame::decode(&frame.encode().unwrap()).unwrap();
            let command = Matmul32x32Command::decode_payload(&decoded.payload).unwrap();
            assert_eq!(decoded.sequence, 500 + idx as u32);
            assert_eq!(decoded.opcode, SerialOpcode::Matmul32x32);
            assert_eq!(command.tile_id, idx as u32);
        }
    }

    #[test]
    fn matmul32x32_plan_rejects_non_tile_aligned_shape() {
        let layout = MatmulMemoryLayout::row_major(33, 32, 32, 0, 0, 0, 4);
        let err = plan_matmul32x32_commands(33, 32, 32, layout).unwrap_err();
        assert!(matches!(
            err,
            SerialProtocolError::InvalidMatmulShape { m: 33, k: 32, n: 32 }
        ));
    }

    #[test]
    fn matmul32x32_plan_supports_row_padding_strides() {
        let layout = MatmulMemoryLayout {
            a_base: 0x1000,
            b_base: 0x2000,
            out_base: 0x3000,
            a_row_stride: 80,
            b_row_stride: 96,
            out_row_stride: 128,
            elem_size_bytes: 4,
        };
        let plan = plan_matmul32x32_commands(64, 64, 64, layout).unwrap();

        assert_eq!(plan.commands[1].a_offset, 0x1000 + 32 * 4);
        assert_eq!(plan.commands[1].b_offset, 0x2000 + 32 * 96 * 4);
        assert_eq!(plan.commands[4].a_offset, 0x1000 + 32 * 80 * 4);
        assert_eq!(plan.commands[4].out_offset, 0x3000 + 32 * 128 * 4);
    }

    #[test]
    fn matmul32x32_plan_rejects_address_overflow() {
        let layout = MatmulMemoryLayout::row_major(64, 32, 32, u64::MAX - 4, 0, 0, 4);
        let err = plan_matmul32x32_commands(64, 32, 32, layout).unwrap_err();
        assert!(matches!(err, SerialProtocolError::MatmulAddressOverflow { .. }));
    }

    #[test]
    fn loopback_transport_survives_10k_frames() {
        let mut transport = LoopbackSerialTransport::new();

        for seq in 0..10_000u32 {
            let frame = SerialFrame::new(SerialOpcode::Ping, seq, seq.to_le_bytes());
            let echoed = transport.exchange(&frame.encode().unwrap()).unwrap();
            let decoded = SerialFrame::decode(&echoed).unwrap();
            assert_eq!(decoded.sequence, seq);
            assert_eq!(decoded.payload, seq.to_le_bytes());
        }

        assert_eq!(transport.frames_seen(), 10_000);
    }
}

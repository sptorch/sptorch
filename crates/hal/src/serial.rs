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

const MAGIC: [u8; 2] = [b'S', b'P'];
const VERSION: u8 = 1;
const HEADER_LEN: usize = 16;
const CHECKSUM_LEN: usize = 4;
const ALIGNMENT: usize = 8;
const MAX_PAYLOAD_LEN: usize = 64 * 1024;

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
            0x7e => Ok(Self::Ack),
            0x7f => Ok(Self::Error),
            other => Err(SerialProtocolError::UnknownOpcode(other)),
        }
    }

    fn as_u8(self) -> u8 {
        self as u8
    }
}

/// 串行帧解析和编码阶段可能暴露的协议错误。
///
/// 这些错误是“主机侧可以立刻判断”的问题；真实硬件执行失败，例如 PE 阵列溢出、
/// DDR 地址不可达，后续应通过 `SerialOpcode::Error` 的 payload 返回，而不是混进解析错误。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SerialProtocolError {
    FrameTooShort { actual: usize },
    InvalidAlignment { actual: usize, alignment: usize },
    BadMagic { found: [u8; 2] },
    UnsupportedVersion(u8),
    UnknownOpcode(u8),
    PayloadTooLarge { actual: usize, max: usize },
    LengthMismatch { expected: usize, actual: usize },
    ChecksumMismatch { expected: u32, actual: u32 },
    NonZeroPadding,
    InvalidMatmulPayloadLen { actual: usize },
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
            Self::InvalidMatmulPayloadLen { actual } => {
                write!(f, "MatMul32x32 payload must be 32 bytes, got {actual}")
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

    /// 将帧编码为可直接写入串行链路的字节序列。
    pub fn encode(&self) -> Result<Vec<u8>, SerialProtocolError> {
        validate_payload_len(self.payload.len())?;

        let body_len = HEADER_LEN + self.payload.len() + CHECKSUM_LEN;
        let padded_len = align_up(body_len, ALIGNMENT);
        let mut encoded = Vec::with_capacity(padded_len);

        encoded.extend_from_slice(&MAGIC);
        encoded.push(VERSION);
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
        if bytes.len() < HEADER_LEN + CHECKSUM_LEN {
            return Err(SerialProtocolError::FrameTooShort { actual: bytes.len() });
        }
        if bytes.len() % ALIGNMENT != 0 {
            return Err(SerialProtocolError::InvalidAlignment {
                actual: bytes.len(),
                alignment: ALIGNMENT,
            });
        }

        let found_magic = [bytes[0], bytes[1]];
        if found_magic != MAGIC {
            return Err(SerialProtocolError::BadMagic { found: found_magic });
        }
        if bytes[2] != VERSION {
            return Err(SerialProtocolError::UnsupportedVersion(bytes[2]));
        }

        let opcode = SerialOpcode::from_u8(bytes[3])?;
        let sequence = u32::from_le_bytes(bytes[4..8].try_into().expect("slice length checked by header"));
        let payload_len = u32::from_le_bytes(bytes[8..12].try_into().expect("slice length checked by header")) as usize;
        validate_payload_len(payload_len)?;
        let flags = u16::from_le_bytes(bytes[12..14].try_into().expect("slice length checked by header"));

        let body_len = HEADER_LEN + payload_len + CHECKSUM_LEN;
        let padded_len = align_up(body_len, ALIGNMENT);
        if bytes.len() != padded_len {
            return Err(SerialProtocolError::LengthMismatch {
                expected: padded_len,
                actual: bytes.len(),
            });
        }

        let checksum_offset = HEADER_LEN + payload_len;
        let actual_checksum = u32::from_le_bytes(
            bytes[checksum_offset..checksum_offset + CHECKSUM_LEN]
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

        if bytes[checksum_offset + CHECKSUM_LEN..].iter().any(|&byte| byte != 0) {
            return Err(SerialProtocolError::NonZeroPadding);
        }

        Ok(Self {
            opcode,
            sequence,
            flags,
            payload: bytes[HEADER_LEN..checksum_offset].to_vec(),
        })
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
    if len > MAX_PAYLOAD_LEN {
        return Err(SerialProtocolError::PayloadTooLarge {
            actual: len,
            max: MAX_PAYLOAD_LEN,
        });
    }
    Ok(())
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
        assert_eq!(encoded.len() % ALIGNMENT, 0);
    }

    #[test]
    fn serial_frame_rejects_length_mismatch() {
        let frame = SerialFrame::new(SerialOpcode::Ping, 1, b"abc".to_vec());
        let mut encoded = frame.encode().unwrap();
        encoded.extend_from_slice(&[0u8; ALIGNMENT]);
        let err = SerialFrame::decode(&encoded).unwrap_err();
        assert!(matches!(err, SerialProtocolError::LengthMismatch { .. }));
    }

    #[test]
    fn serial_frame_rejects_checksum_corruption() {
        let frame = SerialFrame::new(SerialOpcode::Ping, 1, b"abc".to_vec());
        let mut encoded = frame.encode().unwrap();
        encoded[HEADER_LEN] ^= 0x55;
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

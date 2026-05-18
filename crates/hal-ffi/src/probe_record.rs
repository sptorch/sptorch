//! Tang9k probe 的机器可读验收记录。
//!
//! CLI 面向人，JSON record 面向复现、归档和后续 Studio/CI 消费。把这层放在库里，
//! 是为了避免每个硬件工具都重新发明一套 raw bytes、payload 解码和错误记录格式。

use crate::serial_backend::{Tang9kBringupSuiteTrace, UartTang9kExchangeError, UartTang9kExchangeTrace};
use serde::{Deserialize, Serialize};
use sptorch_hal::serial::{
    DeviceInfoPayload, ResultValue32Payload, ResultWindowStatusPayload, ScratchValue32Payload, SerialOpcode,
    SerialStatusPayload, TANG9K_CAP_MATMUL32X32, TANG9K_CAP_RESULT_WINDOW, TANG9K_CAP_RESULT_WINDOW_STATUS,
    TANG9K_CAP_SCRATCH32, TANG9K_DEVICE_KIND_UART_RESPONDER, TANG9K_RESULT_WINDOW_SMOKE_WORDS,
    TANG9K_RESULT_WINDOW_WORD_STRIDE_BYTES, TANG9K_UART_RESPONDER_BAUD, TANG9K_UART_RESPONDER_BUILD_ID,
    TANG9K_UART_RESPONDER_CLK_HZ,
};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

/// Tang9k probe JSON 记录的 schema 名称。
///
/// 只要字段语义不兼容，就应该升级这里的版本号；追加字段可以保持 v1。
pub const TANG9K_PROBE_RECORD_SCHEMA: &str = "sptorch.tang9k.probe.v1";

/// 一次 probe 的上下文元数据。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProbeRecordMetadata {
    pub command: String,
    pub port: String,
    pub baud: u32,
    pub timeout_ms: u64,
}

impl ProbeRecordMetadata {
    pub fn new(command: impl Into<String>, port: impl Into<String>, baud: u32, timeout_ms: u64) -> Self {
        Self {
            command: command.into(),
            port: port.into(),
            baud,
            timeout_ms,
        }
    }
}

/// 可直接写入磁盘或作为 Studio/CI artifact 的 probe 记录。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProbeRecord {
    pub schema: String,
    pub timestamp_unix_ms: u128,
    pub command: String,
    pub port: String,
    pub baud: u32,
    pub timeout_ms: u64,
    pub status: String,
    pub traces: Vec<TraceRecord>,
    pub error: Option<ErrorRecord>,
}

impl ProbeRecord {
    /// 构造成功记录。时间戳在这里生成，调用方只需保证 metadata 来自同一次命令。
    pub fn ok(metadata: &ProbeRecordMetadata, traces: Vec<TraceRecord>) -> Self {
        Self {
            schema: TANG9K_PROBE_RECORD_SCHEMA.to_string(),
            timestamp_unix_ms: timestamp_unix_ms(),
            command: metadata.command.clone(),
            port: metadata.port.clone(),
            baud: metadata.baud,
            timeout_ms: metadata.timeout_ms,
            status: "ok".to_string(),
            traces,
            error: None,
        }
    }

    /// 构造失败记录。失败时也保留 raw bytes，方便定位是无响应、半帧、checksum 还是 payload 漂移。
    pub fn error(metadata: &ProbeRecordMetadata, err: &UartTang9kExchangeError) -> Self {
        Self {
            schema: TANG9K_PROBE_RECORD_SCHEMA.to_string(),
            timestamp_unix_ms: timestamp_unix_ms(),
            command: metadata.command.clone(),
            port: metadata.port.clone(),
            baud: metadata.baud,
            timeout_ms: metadata.timeout_ms,
            status: "error".to_string(),
            traces: Vec::new(),
            error: Some(ErrorRecord {
                message: err.to_string(),
                request_raw_len: err.request_bytes.len(),
                request_raw_hex: format_probe_bytes_hex(&err.request_bytes),
                response_raw_len: err.raw_response_bytes.len(),
                response_raw_hex: format_probe_bytes_hex(&err.raw_response_bytes),
            }),
        }
    }

    /// 以 pretty JSON 写入文件。父目录不存在时会自动创建。
    pub fn write_pretty_json<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let json = serde_json::to_string_pretty(self).map_err(std::io::Error::other)?;
        std::fs::write(path, json)
    }

    /// 从 JSON 文件读回 probe record。
    ///
    /// 读回能力是归档体系的另一半：写 JSON 只解决“留下证据”，读回并校验才能让
    /// CI、Studio 或多板脚本机械判断证据是否满足当前验收门槛。
    pub fn read_json<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        serde_json::from_str(&json).map_err(std::io::Error::other)
    }

    /// 按 Tang9k 当前 bring-up 规则做基础验收校验。
    pub fn validate_tang9k_acceptance(&self) -> Result<Tang9kAcceptanceSummary, ProbeRecordValidationError> {
        validate_tang9k_acceptance(self)
    }
}

/// 单条 request/response 的结构化证据。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceRecord {
    pub label: String,
    pub opcode: String,
    pub sequence: u32,
    pub payload_len: usize,
    pub request_raw_len: usize,
    pub request_raw_hex: String,
    pub response_raw_len: usize,
    pub response_raw_hex: String,
    pub decoded: TraceDecodedRecord,
}

impl TraceRecord {
    pub fn from_trace(label: &str, trace: &UartTang9kExchangeTrace) -> Self {
        Self {
            label: label.to_string(),
            opcode: format!("{:?}", trace.response.opcode),
            sequence: trace.response.sequence,
            payload_len: trace.response.payload.len(),
            request_raw_len: trace.request_bytes.len(),
            request_raw_hex: format_probe_bytes_hex(&trace.request_bytes),
            response_raw_len: trace.raw_response_bytes.len(),
            response_raw_hex: format_probe_bytes_hex(&trace.raw_response_bytes),
            decoded: decode_trace_payload(trace),
        }
    }
}

/// 已知 Tang9k payload 的结构化解码；未知或坏 payload 保留十六进制。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "record_type")]
pub enum TraceDecodedRecord {
    Generic {
        payload_hex: String,
    },
    Status {
        status: String,
        detail: u32,
    },
    ScratchValue32 {
        offset: u32,
        value: u32,
    },
    ResultValue32 {
        offset: u32,
        value: u32,
    },
    ResultWindowStatus {
        valid: bool,
        words: u8,
        stride: u16,
        base: u32,
        last_sequence: u32,
    },
    DeviceInfo {
        protocol: u8,
        kind: u8,
        responder_version: u16,
        capabilities: u32,
        clk_hz: u32,
        baud: u32,
        result_words: u8,
        result_stride: u8,
        build_id: u32,
    },
}

/// 失败路径的最小可复现上下文。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ErrorRecord {
    pub message: String,
    pub request_raw_len: usize,
    pub request_raw_hex: String,
    pub response_raw_len: usize,
    pub response_raw_hex: String,
}

/// 通过 JSON 记录能确认的 Tang9k 验收摘要。
///
/// 注意：这不是“物理真板证明”。它只说明给定 record 内含有当前规则要求的证据。
/// 物理真实性仍取决于 record 的来源、烧录日志和实验流程。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tang9kAcceptanceSummary {
    pub schema: String,
    pub command: String,
    pub port: String,
    pub trace_count: usize,
    pub device_info_seen: bool,
    pub ping_seen: bool,
    pub matmul_ack_seen: bool,
    pub scratch_seen: bool,
    pub result_window_status_seen: bool,
    pub result_window_words_seen: usize,
    pub oob_rejection_seen: bool,
}

/// JSON 记录不满足当前 Tang9k 验收规则时返回的错误。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProbeRecordValidationError {
    SchemaMismatch { expected: &'static str, actual: String },
    StatusNotOk { status: String },
    MissingTrace { label_hint: &'static str },
    DeviceInfoMismatch { reason: String },
    ResultWindowStatusMismatch { reason: String },
    ResultWindowWordCountMismatch { expected: usize, actual: usize },
    OobRejectionMissing,
}

impl std::fmt::Display for ProbeRecordValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SchemaMismatch { expected, actual } => {
                write!(f, "probe record schema mismatch: expected {expected}, got {actual}")
            }
            Self::StatusNotOk { status } => write!(f, "probe record status must be ok, got {status}"),
            Self::MissingTrace { label_hint } => write!(f, "probe record is missing trace: {label_hint}"),
            Self::DeviceInfoMismatch { reason } => write!(f, "probe record device-info mismatch: {reason}"),
            Self::ResultWindowStatusMismatch { reason } => {
                write!(f, "probe record result-window status mismatch: {reason}")
            }
            Self::ResultWindowWordCountMismatch { expected, actual } => write!(
                f,
                "probe record result-window word count mismatch: expected {expected}, got {actual}"
            ),
            Self::OobRejectionMissing => f.write_str("probe record is missing OOB HardwareFault rejection"),
        }
    }
}

impl std::error::Error for ProbeRecordValidationError {}

/// 对 Tang9k 当前验收 JSON 做机械校验。
///
/// `BringupSuite` 必须覆盖全部关键路径；单条 `DeviceInfo` 记录只校验身份字段，便于烧录后先
/// 做最小 bitstream 身份确认。
pub fn validate_tang9k_acceptance(record: &ProbeRecord) -> Result<Tang9kAcceptanceSummary, ProbeRecordValidationError> {
    if record.schema != TANG9K_PROBE_RECORD_SCHEMA {
        return Err(ProbeRecordValidationError::SchemaMismatch {
            expected: TANG9K_PROBE_RECORD_SCHEMA,
            actual: record.schema.clone(),
        });
    }
    if record.status != "ok" {
        return Err(ProbeRecordValidationError::StatusNotOk {
            status: record.status.clone(),
        });
    }

    let device_info_seen = record
        .traces
        .iter()
        .any(|trace| validate_device_info_trace(trace).is_ok());
    let ping_seen = record
        .traces
        .iter()
        .any(|trace| trace.opcode == "Pong" || trace.label.contains("ping"));
    let matmul_ack_seen = record.traces.iter().any(|trace| {
        trace.label.contains("matmul")
            && matches!(
                trace.decoded,
                TraceDecodedRecord::Status {
                    ref status,
                    detail: 0
                } if status == "Ok"
            )
    });
    let scratch_seen = record
        .traces
        .iter()
        .any(|trace| matches!(trace.decoded, TraceDecodedRecord::ScratchValue32 { .. }));
    let result_window_status_seen = record
        .traces
        .iter()
        .any(|trace| validate_result_window_status_trace(trace).is_ok());
    let result_window_words_seen = count_expected_result_window_words(&record.traces);
    let oob_rejection_seen = record.traces.iter().any(|trace| {
        trace.label.contains("oob")
            && trace.opcode == "Error"
            && matches!(
                trace.decoded,
                TraceDecodedRecord::Status {
                    ref status,
                    detail
                } if status == "HardwareFault" && detail == expected_oob_offset()
            )
    });

    if !device_info_seen {
        return Err(ProbeRecordValidationError::MissingTrace {
            label_hint: "DeviceInfo",
        });
    }

    match record.command.as_str() {
        "DeviceInfo" => {}
        "BringupSuite" => {
            if !ping_seen {
                return Err(ProbeRecordValidationError::MissingTrace {
                    label_hint: "Ping/Pong",
                });
            }
            if !matmul_ack_seen {
                return Err(ProbeRecordValidationError::MissingTrace {
                    label_hint: "Matmul32x32 Ack/Ok",
                });
            }
            if !scratch_seen {
                return Err(ProbeRecordValidationError::MissingTrace {
                    label_hint: "ScratchValue32",
                });
            }
            if !result_window_status_seen {
                return Err(ProbeRecordValidationError::MissingTrace {
                    label_hint: "ResultWindowStatus",
                });
            }
            if result_window_words_seen != TANG9K_RESULT_WINDOW_SMOKE_WORDS {
                return Err(ProbeRecordValidationError::ResultWindowWordCountMismatch {
                    expected: TANG9K_RESULT_WINDOW_SMOKE_WORDS,
                    actual: result_window_words_seen,
                });
            }
            if !oob_rejection_seen {
                return Err(ProbeRecordValidationError::OobRejectionMissing);
            }
        }
        _ => {}
    }

    Ok(Tang9kAcceptanceSummary {
        schema: record.schema.clone(),
        command: record.command.clone(),
        port: record.port.clone(),
        trace_count: record.traces.len(),
        device_info_seen,
        ping_seen,
        matmul_ack_seen,
        scratch_seen,
        result_window_status_seen,
        result_window_words_seen,
        oob_rejection_seen,
    })
}

/// 将 bring-up suite 展平成稳定顺序的 trace records。
pub fn suite_trace_records(suite: &Tang9kBringupSuiteTrace) -> Vec<TraceRecord> {
    let mut records = vec![
        TraceRecord::from_trace("suite device info", &suite.device_info),
        TraceRecord::from_trace("suite ping", &suite.ping),
        TraceRecord::from_trace("suite matmul", &suite.matmul),
        TraceRecord::from_trace("suite scratch write", &suite.scratch_write),
        TraceRecord::from_trace("suite scratch read", &suite.scratch_read),
        TraceRecord::from_trace("suite status matmul", &suite.result_window_status_matmul),
        TraceRecord::from_trace("suite status", &suite.result_window_status),
    ];
    for (idx, trace) in suite.result_window.iter().enumerate() {
        let label = if idx == 0 {
            "suite result window matmul".to_string()
        } else {
            format!("suite result window read {}", idx - 1)
        };
        records.push(TraceRecord::from_trace(&label, trace));
    }
    for (idx, trace) in suite.result_oob_setup.iter().enumerate() {
        let label = if idx == 0 {
            "suite oob matmul".to_string()
        } else {
            format!("suite oob setup read {}", idx - 1)
        };
        records.push(TraceRecord::from_trace(&label, trace));
    }
    records.push(TraceRecord::from_trace(
        "suite oob rejected read",
        &suite.result_oob_rejected_read,
    ));
    records
}

/// 与 CLI raw 输出保持一致的十六进制格式。
pub fn format_probe_bytes_hex(bytes: &[u8]) -> String {
    if bytes.is_empty() {
        return "<empty>".into();
    }

    bytes
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<Vec<_>>()
        .join(" ")
}

fn validate_device_info_trace(trace: &TraceRecord) -> Result<(), ProbeRecordValidationError> {
    match &trace.decoded {
        TraceDecodedRecord::DeviceInfo {
            protocol,
            kind,
            capabilities,
            clk_hz,
            baud,
            result_words,
            result_stride,
            build_id,
            ..
        } => {
            if *protocol != sptorch_hal::serial::SERIAL_VERSION {
                return Err(ProbeRecordValidationError::DeviceInfoMismatch {
                    reason: format!(
                        "protocol expected {}, got {}",
                        sptorch_hal::serial::SERIAL_VERSION,
                        protocol
                    ),
                });
            }
            if *kind != TANG9K_DEVICE_KIND_UART_RESPONDER {
                return Err(ProbeRecordValidationError::DeviceInfoMismatch {
                    reason: format!("kind expected {}, got {}", TANG9K_DEVICE_KIND_UART_RESPONDER, kind),
                });
            }
            for capability in [
                TANG9K_CAP_MATMUL32X32,
                TANG9K_CAP_SCRATCH32,
                TANG9K_CAP_RESULT_WINDOW,
                TANG9K_CAP_RESULT_WINDOW_STATUS,
            ] {
                if capabilities & capability == 0 {
                    return Err(ProbeRecordValidationError::DeviceInfoMismatch {
                        reason: format!("missing capability 0x{capability:08x}"),
                    });
                }
            }
            if *clk_hz != TANG9K_UART_RESPONDER_CLK_HZ {
                return Err(ProbeRecordValidationError::DeviceInfoMismatch {
                    reason: format!("clk_hz expected {}, got {}", TANG9K_UART_RESPONDER_CLK_HZ, clk_hz),
                });
            }
            if *baud != TANG9K_UART_RESPONDER_BAUD {
                return Err(ProbeRecordValidationError::DeviceInfoMismatch {
                    reason: format!("baud expected {}, got {}", TANG9K_UART_RESPONDER_BAUD, baud),
                });
            }
            if usize::from(*result_words) != TANG9K_RESULT_WINDOW_SMOKE_WORDS {
                return Err(ProbeRecordValidationError::DeviceInfoMismatch {
                    reason: format!(
                        "result_words expected {}, got {}",
                        TANG9K_RESULT_WINDOW_SMOKE_WORDS, result_words
                    ),
                });
            }
            if u32::from(*result_stride) != TANG9K_RESULT_WINDOW_WORD_STRIDE_BYTES {
                return Err(ProbeRecordValidationError::DeviceInfoMismatch {
                    reason: format!(
                        "result_stride expected {}, got {}",
                        TANG9K_RESULT_WINDOW_WORD_STRIDE_BYTES, result_stride
                    ),
                });
            }
            if *build_id != TANG9K_UART_RESPONDER_BUILD_ID {
                return Err(ProbeRecordValidationError::DeviceInfoMismatch {
                    reason: format!(
                        "build_id expected 0x{:08x}, got 0x{:08x}",
                        TANG9K_UART_RESPONDER_BUILD_ID, build_id
                    ),
                });
            }
            Ok(())
        }
        _ => Err(ProbeRecordValidationError::MissingTrace {
            label_hint: "DeviceInfo",
        }),
    }
}

fn validate_result_window_status_trace(trace: &TraceRecord) -> Result<(), ProbeRecordValidationError> {
    match &trace.decoded {
        TraceDecodedRecord::ResultWindowStatus {
            valid,
            words,
            stride,
            base,
            ..
        } => {
            if !*valid {
                return Err(ProbeRecordValidationError::ResultWindowStatusMismatch {
                    reason: "valid flag is false".into(),
                });
            }
            if usize::from(*words) != TANG9K_RESULT_WINDOW_SMOKE_WORDS {
                return Err(ProbeRecordValidationError::ResultWindowStatusMismatch {
                    reason: format!("words expected {}, got {}", TANG9K_RESULT_WINDOW_SMOKE_WORDS, words),
                });
            }
            if u32::from(*stride) != TANG9K_RESULT_WINDOW_WORD_STRIDE_BYTES {
                return Err(ProbeRecordValidationError::ResultWindowStatusMismatch {
                    reason: format!(
                        "stride expected {}, got {}",
                        TANG9K_RESULT_WINDOW_WORD_STRIDE_BYTES, stride
                    ),
                });
            }
            if *base != expected_result_window_base() {
                return Err(ProbeRecordValidationError::ResultWindowStatusMismatch {
                    reason: format!(
                        "base expected 0x{:08x}, got 0x{:08x}",
                        expected_result_window_base(),
                        base
                    ),
                });
            }
            Ok(())
        }
        _ => Err(ProbeRecordValidationError::MissingTrace {
            label_hint: "ResultWindowStatus",
        }),
    }
}

fn count_expected_result_window_words(traces: &[TraceRecord]) -> usize {
    expected_result_window_values()
        .iter()
        .filter(|(expected_offset, expected_value)| {
            traces.iter().any(|trace| {
                matches!(
                    trace.decoded,
                    TraceDecodedRecord::ResultValue32 { offset, value }
                        if offset == *expected_offset && value == *expected_value
                )
            })
        })
        .count()
}

fn expected_result_window_values() -> [(u32, u32); TANG9K_RESULT_WINDOW_SMOKE_WORDS] {
    let command = crate::serial_backend::tang9k_matmul_smoke_frame(4);
    let decoded = sptorch_hal::serial::Matmul32x32Command::decode_payload(&command.payload)
        .expect("Tang9k smoke frame is generated by the protocol layer");
    decoded
        .smoke_result_window()
        .map(|payload| (payload.offset, payload.value))
}

fn expected_result_window_base() -> u32 {
    expected_result_window_values()[0].0
}

fn expected_oob_offset() -> u32 {
    expected_result_window_values()[TANG9K_RESULT_WINDOW_SMOKE_WORDS - 1]
        .0
        .wrapping_add(TANG9K_RESULT_WINDOW_WORD_STRIDE_BYTES)
}

fn decode_trace_payload(trace: &UartTang9kExchangeTrace) -> TraceDecodedRecord {
    match trace.response.opcode {
        SerialOpcode::ScratchValue32 => ScratchValue32Payload::decode_payload(&trace.response.payload)
            .map(|payload| TraceDecodedRecord::ScratchValue32 {
                offset: payload.offset,
                value: payload.value,
            })
            .unwrap_or_else(|_| generic_decoded_payload(&trace.response.payload)),
        SerialOpcode::ResultValue32 => ResultValue32Payload::decode_payload(&trace.response.payload)
            .map(|payload| TraceDecodedRecord::ResultValue32 {
                offset: payload.offset,
                value: payload.value,
            })
            .unwrap_or_else(|_| generic_decoded_payload(&trace.response.payload)),
        SerialOpcode::ResultWindowStatus => ResultWindowStatusPayload::decode_payload(&trace.response.payload)
            .map(|payload| TraceDecodedRecord::ResultWindowStatus {
                valid: payload.valid(),
                words: payload.word_count,
                stride: payload.stride_bytes,
                base: payload.base_offset,
                last_sequence: payload.last_sequence,
            })
            .unwrap_or_else(|_| generic_decoded_payload(&trace.response.payload)),
        SerialOpcode::DeviceInfo => DeviceInfoPayload::decode_payload(&trace.response.payload)
            .map(|payload| TraceDecodedRecord::DeviceInfo {
                protocol: payload.protocol_version,
                kind: payload.device_kind,
                responder_version: payload.responder_version,
                capabilities: payload.capabilities,
                clk_hz: payload.clk_hz,
                baud: payload.baud,
                result_words: payload.result_window_words,
                result_stride: payload.result_window_stride_bytes,
                build_id: payload.build_id,
            })
            .unwrap_or_else(|_| generic_decoded_payload(&trace.response.payload)),
        SerialOpcode::Ack | SerialOpcode::Error => SerialStatusPayload::decode(&trace.response.payload)
            .map(|payload| TraceDecodedRecord::Status {
                status: format!("{:?}", payload.code),
                detail: payload.detail,
            })
            .unwrap_or_else(|_| generic_decoded_payload(&trace.response.payload)),
        _ => generic_decoded_payload(&trace.response.payload),
    }
}

fn generic_decoded_payload(payload: &[u8]) -> TraceDecodedRecord {
    TraceDecodedRecord::Generic {
        payload_hex: format_probe_bytes_hex(payload),
    }
}

fn timestamp_unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

#[cfg(test)]
mod tests {
    use super::*;
    use sptorch_hal::serial::{
        DeviceInfoReadCommand, Matmul32x32Command, ResultRead32Command, ResultWindowStatusPayload,
        ScratchRead32Command, ScratchValue32Payload, SerialFrame, SerialStatusPayload, TANG9K_UART_RESPONDER_BUILD_ID,
    };

    #[test]
    fn format_hex_keeps_byte_boundaries_visible() {
        assert_eq!(format_probe_bytes_hex(&[]), "<empty>");
        assert_eq!(format_probe_bytes_hex(&[0x53, 0x50, 0x01, 0x7e]), "53 50 01 7e");
    }

    #[test]
    fn writes_probe_record_json_for_single_trace() {
        let output = std::env::temp_dir().join(format!(
            "sptorch-tang9k-probe-record-{}-{}.json",
            std::process::id(),
            timestamp_unix_ms()
        ));
        let metadata = ProbeRecordMetadata::new("DeviceInfo", "COM3", 115_200, 1_000);
        let payload = DeviceInfoPayload::tang9k_uart_responder();
        let frame = payload.into_frame(11);
        let trace = UartTang9kExchangeTrace {
            request_bytes: DeviceInfoReadCommand.into_frame(11).encode().unwrap(),
            raw_response_bytes: frame.encode().unwrap(),
            response: frame,
        };

        let record = ProbeRecord::ok(&metadata, vec![TraceRecord::from_trace("device info", &trace)]);
        record.write_pretty_json(&output).unwrap();
        let roundtrip = ProbeRecord::read_json(&output).unwrap();

        let json = std::fs::read_to_string(&output).unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(value["schema"], TANG9K_PROBE_RECORD_SCHEMA);
        assert_eq!(value["status"], "ok");
        assert_eq!(value["command"], "DeviceInfo");
        assert_eq!(value["traces"][0]["decoded"]["record_type"], "DeviceInfo");
        assert_eq!(
            value["traces"][0]["decoded"]["build_id"],
            TANG9K_UART_RESPONDER_BUILD_ID
        );
        assert_eq!(roundtrip.validate_tang9k_acceptance().unwrap().device_info_seen, true);

        let _ = std::fs::remove_file(output);
    }

    #[test]
    fn generic_trace_payload_survives_unknown_opcode_decoding_path() {
        let frame = SerialFrame::new(SerialOpcode::Pong, 7, vec![0xaa, 0xbb]);
        let trace = UartTang9kExchangeTrace {
            request_bytes: Vec::new(),
            raw_response_bytes: frame.encode().unwrap(),
            response: frame,
        };

        let record = TraceRecord::from_trace("pong", &trace);
        assert_eq!(
            record.decoded,
            TraceDecodedRecord::Generic {
                payload_hex: "aa bb".into()
            }
        );
    }

    #[test]
    fn rejects_schema_mismatch() {
        let metadata = ProbeRecordMetadata::new("DeviceInfo", "COM3", 115_200, 1_000);
        let mut record = ProbeRecord::ok(
            &metadata,
            vec![TraceRecord::from_trace("device info", &device_info_trace())],
        );
        record.schema = "sptorch.tang9k.probe.v0".into();

        let err = record.validate_tang9k_acceptance().unwrap_err();
        assert!(matches!(err, ProbeRecordValidationError::SchemaMismatch { .. }));
    }

    #[test]
    fn accepts_full_bringup_suite_record() {
        let metadata = ProbeRecordMetadata::new("BringupSuite", "COM3", 115_200, 1_000);
        let record = ProbeRecord::ok(&metadata, full_bringup_records());

        let summary = record.validate_tang9k_acceptance().unwrap();
        assert_eq!(summary.command, "BringupSuite");
        assert!(summary.device_info_seen);
        assert!(summary.ping_seen);
        assert!(summary.matmul_ack_seen);
        assert!(summary.scratch_seen);
        assert!(summary.result_window_status_seen);
        assert_eq!(summary.result_window_words_seen, TANG9K_RESULT_WINDOW_SMOKE_WORDS);
        assert!(summary.oob_rejection_seen);
    }

    #[test]
    fn rejects_bringup_suite_without_result_window_words() {
        let metadata = ProbeRecordMetadata::new("BringupSuite", "COM3", 115_200, 1_000);
        let mut records = full_bringup_records();
        records.retain(|trace| !matches!(trace.decoded, TraceDecodedRecord::ResultValue32 { .. }));
        let record = ProbeRecord::ok(&metadata, records);

        let err = record.validate_tang9k_acceptance().unwrap_err();
        assert_eq!(
            err,
            ProbeRecordValidationError::ResultWindowWordCountMismatch {
                expected: TANG9K_RESULT_WINDOW_SMOKE_WORDS,
                actual: 0
            }
        );
    }

    fn full_bringup_records() -> Vec<TraceRecord> {
        let mut records = vec![
            TraceRecord::from_trace("suite device info", &device_info_trace()),
            TraceRecord::from_trace(
                "suite ping",
                &simple_trace(SerialOpcode::Ping, SerialFrame::new(SerialOpcode::Pong, 0, Vec::new())),
            ),
            TraceRecord::from_trace(
                "suite matmul",
                &simple_trace(
                    SerialOpcode::Matmul32x32,
                    SerialFrame::ack(4, SerialStatusPayload::ok()),
                ),
            ),
            TraceRecord::from_trace(
                "suite scratch read",
                &simple_trace(
                    SerialOpcode::ScratchRead32,
                    ScratchValue32Payload::new(0x44, 0x1122_3344).into_frame(3),
                ),
            ),
            TraceRecord::from_trace(
                "suite status",
                &simple_trace(
                    SerialOpcode::ResultWindowStatusRead,
                    ResultWindowStatusPayload::new(true, 4, 4, expected_result_window_base(), 4).into_frame(10),
                ),
            ),
        ];
        for (idx, (offset, value)) in expected_result_window_values().iter().enumerate() {
            records.push(TraceRecord::from_trace(
                &format!("suite result window read {idx}"),
                &simple_trace(
                    SerialOpcode::ResultRead32,
                    ResultValue32Payload::new(*offset, *value).into_frame(5 + idx as u32),
                ),
            ));
        }
        records.push(TraceRecord::from_trace(
            "suite oob rejected read",
            &simple_trace(
                SerialOpcode::ResultRead32,
                SerialFrame::error(
                    9,
                    SerialStatusPayload {
                        code: sptorch_hal::serial::SerialStatusCode::HardwareFault,
                        detail: expected_oob_offset(),
                    },
                ),
            ),
        ));
        records
    }

    fn device_info_trace() -> UartTang9kExchangeTrace {
        let frame = DeviceInfoPayload::tang9k_uart_responder().into_frame(11);
        UartTang9kExchangeTrace {
            request_bytes: DeviceInfoReadCommand.into_frame(11).encode().unwrap(),
            raw_response_bytes: frame.encode().unwrap(),
            response: frame,
        }
    }

    fn simple_trace(request_opcode: SerialOpcode, response: SerialFrame) -> UartTang9kExchangeTrace {
        let request = SerialFrame::new(request_opcode, response.sequence, request_payload_for(request_opcode));
        UartTang9kExchangeTrace {
            request_bytes: request.encode().unwrap(),
            raw_response_bytes: response.encode().unwrap(),
            response,
        }
    }

    fn request_payload_for(opcode: SerialOpcode) -> Vec<u8> {
        match opcode {
            SerialOpcode::Matmul32x32 => Matmul32x32Command::new(0, 0, 4096, 8192).encode_payload().to_vec(),
            SerialOpcode::ScratchRead32 => ScratchRead32Command::new(0x44).encode_payload().to_vec(),
            SerialOpcode::ResultRead32 => ResultRead32Command::new(expected_oob_offset())
                .encode_payload()
                .to_vec(),
            _ => Vec::new(),
        }
    }
}

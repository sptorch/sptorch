//! Tang9k probe 的机器可读验收记录。
//!
//! CLI 面向人，JSON record 面向复现、归档和后续 Studio/CI 消费。把这层放在库里，
//! 是为了避免每个硬件工具都重新发明一套 raw bytes、payload 解码和错误记录格式。

use crate::serial_backend::{Tang9kBringupSuiteTrace, UartTang9kExchangeError, UartTang9kExchangeTrace};
use serde::Serialize;
use sptorch_hal::serial::{
    DeviceInfoPayload, ResultValue32Payload, ResultWindowStatusPayload, ScratchValue32Payload, SerialOpcode,
    SerialStatusPayload,
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
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ProbeRecord {
    pub schema: &'static str,
    pub timestamp_unix_ms: u128,
    pub command: String,
    pub port: String,
    pub baud: u32,
    pub timeout_ms: u64,
    pub status: &'static str,
    pub traces: Vec<TraceRecord>,
    pub error: Option<ErrorRecord>,
}

impl ProbeRecord {
    /// 构造成功记录。时间戳在这里生成，调用方只需保证 metadata 来自同一次命令。
    pub fn ok(metadata: &ProbeRecordMetadata, traces: Vec<TraceRecord>) -> Self {
        Self {
            schema: TANG9K_PROBE_RECORD_SCHEMA,
            timestamp_unix_ms: timestamp_unix_ms(),
            command: metadata.command.clone(),
            port: metadata.port.clone(),
            baud: metadata.baud,
            timeout_ms: metadata.timeout_ms,
            status: "ok",
            traces,
            error: None,
        }
    }

    /// 构造失败记录。失败时也保留 raw bytes，方便定位是无响应、半帧、checksum 还是 payload 漂移。
    pub fn error(metadata: &ProbeRecordMetadata, err: &UartTang9kExchangeError) -> Self {
        Self {
            schema: TANG9K_PROBE_RECORD_SCHEMA,
            timestamp_unix_ms: timestamp_unix_ms(),
            command: metadata.command.clone(),
            port: metadata.port.clone(),
            baud: metadata.baud,
            timeout_ms: metadata.timeout_ms,
            status: "error",
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
}

/// 单条 request/response 的结构化证据。
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
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
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
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
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ErrorRecord {
    pub message: String,
    pub request_raw_len: usize,
    pub request_raw_hex: String,
    pub response_raw_len: usize,
    pub response_raw_hex: String,
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
    use sptorch_hal::serial::{DeviceInfoReadCommand, SerialFrame, TANG9K_UART_RESPONDER_BUILD_ID};

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
}

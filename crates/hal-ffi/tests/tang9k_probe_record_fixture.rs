use sptorch_hal::serial::{
    DeviceInfoPayload, DeviceInfoReadCommand, Matmul32x32Command, ResultRead32Command, ResultValue32Payload,
    ResultWindowStatusPayload, ResultWindowStatusReadCommand, ScratchRead32Command, ScratchValue32Payload,
    ScratchWrite32Command, SerialFrame, SerialOpcode, SerialStatusCode, SerialStatusPayload,
    TANG9K_RESULT_WINDOW_WORD_STRIDE_BYTES,
};
use sptorch_hal_ffi::probe_record::{ProbeRecord, ProbeRecordMetadata, TraceRecord};
use sptorch_hal_ffi::serial_backend::tang9k_matmul_smoke_frame;
use std::path::{Path, PathBuf};
use std::process::Command;

fn fixture_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("tang9k_bringup_suite.json")
}

#[test]
fn regenerate_fixture_when_requested() {
    if std::env::var_os("GENERATE_TANG9K_FIXTURE").is_none() {
        return;
    }

    let record = build_bringup_suite_record();
    record.write_pretty_json(fixture_path()).unwrap();
}

#[test]
fn fixture_validates_with_library() {
    let record = ProbeRecord::read_json(fixture_path()).expect("fixture JSON should exist");
    let summary = record.validate_tang9k_acceptance().expect("fixture should be accepted");

    assert_eq!(summary.schema, "sptorch.tang9k.probe.v1");
    assert_eq!(summary.command, "BringupSuite");
    assert_eq!(summary.port, "COM3");
    assert!(summary.device_info_seen);
    assert!(summary.ping_seen);
    assert!(summary.matmul_ack_seen);
    assert!(summary.scratch_seen);
    assert!(summary.result_window_status_seen);
    assert_eq!(summary.result_window_words_seen, 4);
    assert!(summary.oob_rejection_seen);
}

#[test]
fn fixture_validates_with_cli() {
    let output = Command::new(env!("CARGO_BIN_EXE_tang9k_probe"))
        .arg("--validate-record")
        .arg(fixture_path())
        .output()
        .expect("tang9k_probe should run");

    assert!(
        output.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("OK: probe record accepted"));
    assert!(stdout.contains("command=BringupSuite"));
}

fn build_bringup_suite_record() -> ProbeRecord {
    let metadata = ProbeRecordMetadata::new("BringupSuite", "COM3", 115_200, 1_000);
    ProbeRecord::ok(&metadata, build_bringup_suite_traces())
}

fn build_bringup_suite_traces() -> Vec<TraceRecord> {
    let mut traces = vec![
        TraceRecord::from_trace("suite device info", &device_info_trace()),
        TraceRecord::from_trace("suite ping", &ping_trace()),
        TraceRecord::from_trace("suite matmul", &matmul_ack_trace(1)),
        TraceRecord::from_trace("suite scratch write", &scratch_write_trace()),
        TraceRecord::from_trace("suite scratch read", &scratch_read_trace()),
        TraceRecord::from_trace("suite status matmul", &matmul_ack_trace(4)),
        TraceRecord::from_trace("suite status", &result_window_status_trace(10)),
    ];

    for (idx, (offset, value)) in expected_result_window_values().iter().enumerate() {
        traces.push(TraceRecord::from_trace(
            &format!("suite result window read {}", idx),
            &result_value_trace(5 + idx as u32, *offset, *value),
        ));
    }

    traces.push(TraceRecord::from_trace(
        "suite oob rejected read",
        &oob_rejected_read_trace(),
    ));
    traces
}

fn device_info_trace() -> sptorch_hal_ffi::serial_backend::UartTang9kExchangeTrace {
    let response = DeviceInfoPayload::tang9k_uart_responder().into_frame(11);
    sptorch_hal_ffi::serial_backend::UartTang9kExchangeTrace {
        request_bytes: DeviceInfoReadCommand.into_frame(11).encode().unwrap(),
        raw_response_bytes: response.encode().unwrap(),
        response,
    }
}

fn ping_trace() -> sptorch_hal_ffi::serial_backend::UartTang9kExchangeTrace {
    let response = SerialFrame::new(SerialOpcode::Pong, 0, Vec::new());
    sptorch_hal_ffi::serial_backend::UartTang9kExchangeTrace {
        request_bytes: SerialFrame::new(SerialOpcode::Ping, 0, b"sptorch-ping".to_vec())
            .encode()
            .unwrap(),
        raw_response_bytes: response.encode().unwrap(),
        response,
    }
}

fn matmul_ack_trace(sequence: u32) -> sptorch_hal_ffi::serial_backend::UartTang9kExchangeTrace {
    let response = SerialFrame::ack(sequence, SerialStatusPayload::ok());
    sptorch_hal_ffi::serial_backend::UartTang9kExchangeTrace {
        request_bytes: tang9k_matmul_smoke_frame(sequence).encode().unwrap(),
        raw_response_bytes: response.encode().unwrap(),
        response,
    }
}

fn scratch_write_trace() -> sptorch_hal_ffi::serial_backend::UartTang9kExchangeTrace {
    let request = ScratchWrite32Command::new(0x44, 0x1122_3344).into_frame(2);
    let response = SerialFrame::ack(2, SerialStatusPayload::ok());
    sptorch_hal_ffi::serial_backend::UartTang9kExchangeTrace {
        request_bytes: request.encode().unwrap(),
        raw_response_bytes: response.encode().unwrap(),
        response,
    }
}

fn scratch_read_trace() -> sptorch_hal_ffi::serial_backend::UartTang9kExchangeTrace {
    let response = ScratchValue32Payload::new(0x44, 0x1122_3344).into_frame(3);
    sptorch_hal_ffi::serial_backend::UartTang9kExchangeTrace {
        request_bytes: ScratchRead32Command::new(0x44).into_frame(3).encode().unwrap(),
        raw_response_bytes: response.encode().unwrap(),
        response,
    }
}

fn result_window_status_trace(sequence: u32) -> sptorch_hal_ffi::serial_backend::UartTang9kExchangeTrace {
    let expected = expected_result_window_values();
    let response = ResultWindowStatusPayload::new(
        true,
        expected.len() as u8,
        TANG9K_RESULT_WINDOW_WORD_STRIDE_BYTES as u16,
        expected[0].0,
        4,
    )
    .into_frame(sequence);
    sptorch_hal_ffi::serial_backend::UartTang9kExchangeTrace {
        request_bytes: ResultWindowStatusReadCommand.into_frame(sequence).encode().unwrap(),
        raw_response_bytes: response.encode().unwrap(),
        response,
    }
}

fn result_value_trace(
    sequence: u32,
    offset: u32,
    value: u32,
) -> sptorch_hal_ffi::serial_backend::UartTang9kExchangeTrace {
    let response = ResultValue32Payload::new(offset, value).into_frame(sequence);
    sptorch_hal_ffi::serial_backend::UartTang9kExchangeTrace {
        request_bytes: ResultRead32Command::new(offset).into_frame(sequence).encode().unwrap(),
        raw_response_bytes: response.encode().unwrap(),
        response,
    }
}

fn oob_rejected_read_trace() -> sptorch_hal_ffi::serial_backend::UartTang9kExchangeTrace {
    let expected = expected_result_window_values();
    let offset = expected
        .last()
        .unwrap()
        .0
        .wrapping_add(TANG9K_RESULT_WINDOW_WORD_STRIDE_BYTES);
    let response = SerialFrame::error(
        9,
        SerialStatusPayload {
            code: SerialStatusCode::HardwareFault,
            detail: offset,
        },
    );
    sptorch_hal_ffi::serial_backend::UartTang9kExchangeTrace {
        request_bytes: ResultRead32Command::new(offset).into_frame(9).encode().unwrap(),
        raw_response_bytes: response.encode().unwrap(),
        response,
    }
}

fn expected_result_window_values() -> [(u32, u32); 4] {
    let frame = tang9k_matmul_smoke_frame(4);
    let command = Matmul32x32Command::decode_payload(&frame.payload).unwrap();
    let window = command.smoke_result_window();
    [
        (window[0].offset, window[0].value),
        (window[1].offset, window[1].value),
        (window[2].offset, window[2].value),
        (window[3].offset, window[3].value),
    ]
}

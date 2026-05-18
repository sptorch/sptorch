use sptorch_hal::serial::{
    DeviceInfoPayload, ResultValue32Payload, ResultWindowStatusPayload, ScratchValue32Payload, SerialOpcode,
    SerialStatusPayload,
};
use sptorch_hal_ffi::probe_record::{
    format_probe_bytes_hex, suite_trace_records, ProbeRecord, ProbeRecordMetadata, TraceRecord,
};
use sptorch_hal_ffi::serial_backend::{
    list_tang9k_serial_ports, probe_tang9k_bringup_suite_with_trace, probe_tang9k_device_info_with_trace,
    probe_tang9k_matmul_smoke_with_trace, probe_tang9k_ping_with_trace, probe_tang9k_result_oob_smoke_with_trace,
    probe_tang9k_result_smoke_with_trace, probe_tang9k_result_window_smoke_with_trace,
    probe_tang9k_result_window_status_smoke_with_trace, probe_tang9k_scratch_smoke_with_trace,
};
use std::path::PathBuf;
use std::time::Duration;

#[derive(Debug)]
struct Args {
    port: Option<String>,
    baud: u32,
    timeout_ms: u64,
    command: ProbeCommand,
    dump_raw: bool,
    record_json: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProbeCommand {
    Ping,
    DeviceInfo,
    BringupSuite,
    MatmulSmoke,
    ResultSmoke,
    ResultWindowSmoke,
    ResultWindowStatusSmoke,
    ResultOobSmoke,
    ScratchSmoke,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            port: None,
            baud: 115_200,
            timeout_ms: 1_000,
            command: ProbeCommand::Ping,
            dump_raw: false,
            record_json: None,
        }
    }
}

impl std::fmt::Display for ProbeCommand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

fn main() {
    let args = parse_args(std::env::args().skip(1).collect()).unwrap_or_else(|err| {
        eprintln!("{err}");
        print_usage();
        std::process::exit(2);
    });

    if let Some(port) = args.port.clone() {
        println!(
            "Probing Tang9k on {port} at {} baud, timeout={}ms, command={:?}",
            args.baud, args.timeout_ms, args.command
        );
        match args.command {
            ProbeCommand::Ping => {
                let result = probe_tang9k_ping_with_trace(&port, args.baud, Duration::from_millis(args.timeout_ms));
                match result {
                    Ok(trace) => {
                        print_trace_summary("response", &trace);
                        if args.dump_raw {
                            print_raw_exchange("response", &trace.request_bytes, &trace.raw_response_bytes);
                        }
                        write_single_record_if_requested(&args, &port, TraceRecord::from_trace("response", &trace));
                    }
                    Err(err) => {
                        write_error_record_if_requested(&args, &port, &err);
                        print_probe_error(args.dump_raw, err);
                    }
                }
            }
            ProbeCommand::DeviceInfo => {
                let result =
                    probe_tang9k_device_info_with_trace(&port, args.baud, Duration::from_millis(args.timeout_ms));
                match result {
                    Ok(trace) => {
                        print_trace_summary("device info", &trace);
                        if args.dump_raw {
                            print_raw_exchange("device info", &trace.request_bytes, &trace.raw_response_bytes);
                        }
                        write_single_record_if_requested(&args, &port, TraceRecord::from_trace("device info", &trace));
                    }
                    Err(err) => {
                        write_error_record_if_requested(&args, &port, &err);
                        print_probe_error(args.dump_raw, err);
                    }
                }
            }
            ProbeCommand::BringupSuite => {
                let result =
                    probe_tang9k_bringup_suite_with_trace(&port, args.baud, Duration::from_millis(args.timeout_ms));
                match result {
                    Ok(suite) => {
                        println!("OK: bringup suite completed sequentially");
                        print_trace_summary("suite device info", &suite.device_info);
                        print_trace_summary("suite ping", &suite.ping);
                        print_trace_summary("suite matmul", &suite.matmul);
                        print_trace_summary("suite scratch write", &suite.scratch_write);
                        print_trace_summary("suite scratch read", &suite.scratch_read);
                        print_trace_summary("suite status matmul", &suite.result_window_status_matmul);
                        print_trace_summary("suite status", &suite.result_window_status);
                        for (idx, trace) in suite.result_window.iter().enumerate() {
                            let label = if idx == 0 {
                                "suite result window matmul".to_string()
                            } else {
                                format!("suite result window read {}", idx - 1)
                            };
                            print_trace_summary(&label, trace);
                        }
                        for (idx, trace) in suite.result_oob_setup.iter().enumerate() {
                            let label = if idx == 0 {
                                "suite oob matmul".to_string()
                            } else {
                                format!("suite oob setup read {}", idx - 1)
                            };
                            print_trace_summary(&label, trace);
                        }
                        print_trace_summary("suite oob rejected read", &suite.result_oob_rejected_read);

                        if args.dump_raw {
                            print_raw_exchange(
                                "suite device info",
                                &suite.device_info.request_bytes,
                                &suite.device_info.raw_response_bytes,
                            );
                            print_raw_exchange("suite ping", &suite.ping.request_bytes, &suite.ping.raw_response_bytes);
                            print_raw_exchange(
                                "suite matmul",
                                &suite.matmul.request_bytes,
                                &suite.matmul.raw_response_bytes,
                            );
                            print_raw_exchange(
                                "suite scratch write",
                                &suite.scratch_write.request_bytes,
                                &suite.scratch_write.raw_response_bytes,
                            );
                            print_raw_exchange(
                                "suite scratch read",
                                &suite.scratch_read.request_bytes,
                                &suite.scratch_read.raw_response_bytes,
                            );
                            print_raw_exchange(
                                "suite status matmul",
                                &suite.result_window_status_matmul.request_bytes,
                                &suite.result_window_status_matmul.raw_response_bytes,
                            );
                            print_raw_exchange(
                                "suite status",
                                &suite.result_window_status.request_bytes,
                                &suite.result_window_status.raw_response_bytes,
                            );
                            for (idx, trace) in suite.result_window.iter().enumerate() {
                                let label = if idx == 0 {
                                    "suite result window matmul".to_string()
                                } else {
                                    format!("suite result window read {}", idx - 1)
                                };
                                print_raw_exchange(&label, &trace.request_bytes, &trace.raw_response_bytes);
                            }
                            for (idx, trace) in suite.result_oob_setup.iter().enumerate() {
                                let label = if idx == 0 {
                                    "suite oob matmul".to_string()
                                } else {
                                    format!("suite oob setup read {}", idx - 1)
                                };
                                print_raw_exchange(&label, &trace.request_bytes, &trace.raw_response_bytes);
                            }
                            print_raw_exchange(
                                "suite oob rejected read",
                                &suite.result_oob_rejected_read.request_bytes,
                                &suite.result_oob_rejected_read.raw_response_bytes,
                            );
                        }
                        let records = suite_trace_records(&suite);
                        write_suite_record_if_requested(&args, &port, records);
                    }
                    Err(err) => {
                        write_error_record_if_requested(&args, &port, &err);
                        print_probe_error(args.dump_raw, err);
                    }
                }
            }
            ProbeCommand::MatmulSmoke => {
                let result =
                    probe_tang9k_matmul_smoke_with_trace(&port, args.baud, Duration::from_millis(args.timeout_ms));
                match result {
                    Ok(trace) => {
                        print_trace_summary("matmul", &trace);
                        if args.dump_raw {
                            print_raw_exchange("matmul", &trace.request_bytes, &trace.raw_response_bytes);
                        }
                        write_single_record_if_requested(&args, &port, TraceRecord::from_trace("matmul", &trace));
                    }
                    Err(err) => {
                        write_error_record_if_requested(&args, &port, &err);
                        print_probe_error(args.dump_raw, err);
                    }
                }
            }
            ProbeCommand::ResultSmoke => {
                let result =
                    probe_tang9k_result_smoke_with_trace(&port, args.baud, Duration::from_millis(args.timeout_ms));
                match result {
                    Ok((matmul_trace, read_trace)) => {
                        print_trace_summary("result matmul", &matmul_trace);
                        print_trace_summary("result read", &read_trace);
                        if args.dump_raw {
                            print_raw_exchange(
                                "result matmul",
                                &matmul_trace.request_bytes,
                                &matmul_trace.raw_response_bytes,
                            );
                            print_raw_exchange(
                                "result read",
                                &read_trace.request_bytes,
                                &read_trace.raw_response_bytes,
                            );
                        }
                        write_suite_record_if_requested(
                            &args,
                            &port,
                            vec![
                                TraceRecord::from_trace("result matmul", &matmul_trace),
                                TraceRecord::from_trace("result read", &read_trace),
                            ],
                        );
                    }
                    Err(err) => {
                        write_error_record_if_requested(&args, &port, &err);
                        print_probe_error(args.dump_raw, err);
                    }
                }
            }
            ProbeCommand::ResultWindowSmoke => {
                let result = probe_tang9k_result_window_smoke_with_trace(
                    &port,
                    args.baud,
                    Duration::from_millis(args.timeout_ms),
                );
                match result {
                    Ok(traces) => {
                        let mut records = Vec::with_capacity(traces.len());
                        for (idx, trace) in traces.iter().enumerate() {
                            let label = if idx == 0 {
                                "result window matmul".to_string()
                            } else {
                                format!("result window read {}", idx - 1)
                            };
                            print_trace_summary(&label, trace);
                            if args.dump_raw {
                                print_raw_exchange(&label, &trace.request_bytes, &trace.raw_response_bytes);
                            }
                            records.push(TraceRecord::from_trace(&label, trace));
                        }
                        write_suite_record_if_requested(&args, &port, records);
                    }
                    Err(err) => {
                        write_error_record_if_requested(&args, &port, &err);
                        print_probe_error(args.dump_raw, err);
                    }
                }
            }
            ProbeCommand::ResultWindowStatusSmoke => {
                let result = probe_tang9k_result_window_status_smoke_with_trace(
                    &port,
                    args.baud,
                    Duration::from_millis(args.timeout_ms),
                );
                match result {
                    Ok((matmul_trace, status_trace)) => {
                        print_trace_summary("result window status matmul", &matmul_trace);
                        print_trace_summary("result window status", &status_trace);
                        if args.dump_raw {
                            print_raw_exchange(
                                "result window status matmul",
                                &matmul_trace.request_bytes,
                                &matmul_trace.raw_response_bytes,
                            );
                            print_raw_exchange(
                                "result window status",
                                &status_trace.request_bytes,
                                &status_trace.raw_response_bytes,
                            );
                        }
                        write_suite_record_if_requested(
                            &args,
                            &port,
                            vec![
                                TraceRecord::from_trace("result window status matmul", &matmul_trace),
                                TraceRecord::from_trace("result window status", &status_trace),
                            ],
                        );
                    }
                    Err(err) => {
                        write_error_record_if_requested(&args, &port, &err);
                        print_probe_error(args.dump_raw, err);
                    }
                }
            }
            ProbeCommand::ResultOobSmoke => {
                let result =
                    probe_tang9k_result_oob_smoke_with_trace(&port, args.baud, Duration::from_millis(args.timeout_ms));
                match result {
                    Ok((setup_traces, oob_trace)) => {
                        let mut records = Vec::with_capacity(setup_traces.len() + 1);
                        for (idx, trace) in setup_traces.iter().enumerate() {
                            let label = if idx == 0 {
                                "result oob matmul".to_string()
                            } else {
                                format!("result oob setup read {}", idx - 1)
                            };
                            print_trace_summary(&label, trace);
                            if args.dump_raw {
                                print_raw_exchange(&label, &trace.request_bytes, &trace.raw_response_bytes);
                            }
                            records.push(TraceRecord::from_trace(&label, trace));
                        }
                        print_trace_summary("result oob rejected read", &oob_trace);
                        if args.dump_raw {
                            print_raw_exchange(
                                "result oob rejected read",
                                &oob_trace.request_bytes,
                                &oob_trace.raw_response_bytes,
                            );
                        }
                        records.push(TraceRecord::from_trace("result oob rejected read", &oob_trace));
                        write_suite_record_if_requested(&args, &port, records);
                    }
                    Err(err) => {
                        write_error_record_if_requested(&args, &port, &err);
                        print_probe_error(args.dump_raw, err);
                    }
                }
            }
            ProbeCommand::ScratchSmoke => {
                let result =
                    probe_tang9k_scratch_smoke_with_trace(&port, args.baud, Duration::from_millis(args.timeout_ms));
                match result {
                    Ok((write_trace, read_trace)) => {
                        print_trace_summary("scratch write", &write_trace);
                        print_trace_summary("scratch read", &read_trace);
                        if args.dump_raw {
                            print_raw_exchange(
                                "scratch write",
                                &write_trace.request_bytes,
                                &write_trace.raw_response_bytes,
                            );
                            print_raw_exchange(
                                "scratch read",
                                &read_trace.request_bytes,
                                &read_trace.raw_response_bytes,
                            );
                        }
                        write_suite_record_if_requested(
                            &args,
                            &port,
                            vec![
                                TraceRecord::from_trace("scratch write", &write_trace),
                                TraceRecord::from_trace("scratch read", &read_trace),
                            ],
                        );
                    }
                    Err(err) => {
                        write_error_record_if_requested(&args, &port, &err);
                        print_probe_error(args.dump_raw, err);
                    }
                }
            }
        }
    } else {
        println!("Visible serial ports:");
        match list_tang9k_serial_ports() {
            Ok(ports) if ports.is_empty() => {
                println!("  <none>");
                println!("No USB-UART COM port is visible yet. Check cable mode, driver, and board power.");
            }
            Ok(ports) => {
                for port in ports {
                    println!("  {port}");
                }
                println!();
                println!("Use --port COMx to send a protocol Ping after confirming the Tang9k port.");
            }
            Err(err) => {
                eprintln!("Port listing failed: {err}");
                std::process::exit(1);
            }
        }
    }
}

fn parse_args(raw: Vec<String>) -> Result<Args, String> {
    let mut args = Args::default();
    let mut idx = 0;

    while idx < raw.len() {
        match raw[idx].as_str() {
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            "--list" => {
                args.port = None;
                idx += 1;
            }
            "--port" => {
                let value = raw.get(idx + 1).ok_or("--port requires a value like COM3")?;
                args.port = Some(value.clone());
                idx += 2;
            }
            "--baud" => {
                let value = raw.get(idx + 1).ok_or("--baud requires a numeric value")?;
                args.baud = value.parse().map_err(|_| format!("invalid --baud value: {value}"))?;
                idx += 2;
            }
            "--timeout-ms" => {
                let value = raw.get(idx + 1).ok_or("--timeout-ms requires a numeric value")?;
                args.timeout_ms = value
                    .parse()
                    .map_err(|_| format!("invalid --timeout-ms value: {value}"))?;
                idx += 2;
            }
            "--matmul-smoke" => {
                args.command = ProbeCommand::MatmulSmoke;
                idx += 1;
            }
            "--device-info" => {
                args.command = ProbeCommand::DeviceInfo;
                idx += 1;
            }
            "--bringup-suite" => {
                args.command = ProbeCommand::BringupSuite;
                idx += 1;
            }
            "--result-smoke" => {
                args.command = ProbeCommand::ResultSmoke;
                idx += 1;
            }
            "--result-window-smoke" => {
                args.command = ProbeCommand::ResultWindowSmoke;
                idx += 1;
            }
            "--result-window-status-smoke" => {
                args.command = ProbeCommand::ResultWindowStatusSmoke;
                idx += 1;
            }
            "--result-oob-smoke" => {
                args.command = ProbeCommand::ResultOobSmoke;
                idx += 1;
            }
            "--scratch-smoke" => {
                args.command = ProbeCommand::ScratchSmoke;
                idx += 1;
            }
            "--dump-raw" => {
                args.dump_raw = true;
                idx += 1;
            }
            "--record-json" => {
                let value = raw.get(idx + 1).ok_or("--record-json requires an output path")?;
                args.record_json = Some(PathBuf::from(value));
                idx += 2;
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }

    Ok(args)
}

fn write_single_record_if_requested(args: &Args, port: &str, trace: TraceRecord) {
    write_suite_record_if_requested(args, port, vec![trace]);
}

fn write_suite_record_if_requested(args: &Args, port: &str, traces: Vec<TraceRecord>) {
    if let Some(path) = &args.record_json {
        let metadata = ProbeRecordMetadata::new(args.command.to_string(), port, args.baud, args.timeout_ms);
        let record = ProbeRecord::ok(&metadata, traces);
        write_record_json_or_exit(path, &record);
        println!("record_json={}", path.display());
    }
}

fn write_error_record_if_requested(
    args: &Args,
    port: &str,
    err: &sptorch_hal_ffi::serial_backend::UartTang9kExchangeError,
) {
    if let Some(path) = &args.record_json {
        let metadata = ProbeRecordMetadata::new(args.command.to_string(), port, args.baud, args.timeout_ms);
        let record = ProbeRecord::error(&metadata, err);
        write_record_json_or_exit(path, &record);
        eprintln!("record_json={}", path.display());
    }
}

fn write_record_json_or_exit(path: &PathBuf, record: &ProbeRecord) {
    record.write_pretty_json(path).unwrap_or_else(|err| {
        eprintln!("failed to write probe record {}: {err}", path.display());
        std::process::exit(1);
    });
}

fn print_trace_summary(label: &str, trace: &sptorch_hal_ffi::serial_backend::UartTang9kExchangeTrace) {
    match trace.response.opcode {
        SerialOpcode::ScratchValue32 => match ScratchValue32Payload::decode_payload(&trace.response.payload) {
            Ok(payload) => {
                println!(
                    "OK: {label} opcode={:?}, sequence={}, offset=0x{:08x}, value=0x{:08x}",
                    trace.response.opcode, trace.response.sequence, payload.offset, payload.value
                );
            }
            Err(_) => {
                println!(
                    "OK: {label} opcode={:?}, sequence={}, payload_len={}",
                    trace.response.opcode,
                    trace.response.sequence,
                    trace.response.payload.len()
                );
            }
        },
        SerialOpcode::ResultValue32 => match ResultValue32Payload::decode_payload(&trace.response.payload) {
            Ok(payload) => {
                println!(
                    "OK: {label} opcode={:?}, sequence={}, offset=0x{:08x}, value=0x{:08x}",
                    trace.response.opcode, trace.response.sequence, payload.offset, payload.value
                );
            }
            Err(_) => {
                println!(
                    "OK: {label} opcode={:?}, sequence={}, payload_len={}",
                    trace.response.opcode,
                    trace.response.sequence,
                    trace.response.payload.len()
                );
            }
        },
        SerialOpcode::ResultWindowStatus => match ResultWindowStatusPayload::decode_payload(&trace.response.payload) {
            Ok(payload) => {
                println!(
                    "OK: {label} opcode={:?}, sequence={}, valid={}, words={}, stride={}, base=0x{:08x}, last_sequence={}",
                    trace.response.opcode,
                    trace.response.sequence,
                    payload.valid(),
                    payload.word_count,
                    payload.stride_bytes,
                    payload.base_offset,
                    payload.last_sequence
                );
            }
            Err(_) => {
                println!(
                    "OK: {label} opcode={:?}, sequence={}, payload_len={}",
                    trace.response.opcode,
                    trace.response.sequence,
                    trace.response.payload.len()
                );
            }
        },
        SerialOpcode::DeviceInfo => match DeviceInfoPayload::decode_payload(&trace.response.payload) {
            Ok(payload) => {
                println!(
                    "OK: {label} opcode={:?}, sequence={}, protocol={}, kind={}, responder_version={}, capabilities=0x{:08x}, clk_hz={}, baud={}, result_words={}, result_stride={}, build_id=0x{:08x}",
                    trace.response.opcode,
                    trace.response.sequence,
                    payload.protocol_version,
                    payload.device_kind,
                    payload.responder_version,
                    payload.capabilities,
                    payload.clk_hz,
                    payload.baud,
                    payload.result_window_words,
                    payload.result_window_stride_bytes,
                    payload.build_id
                );
            }
            Err(_) => {
                println!(
                    "OK: {label} opcode={:?}, sequence={}, payload_len={}",
                    trace.response.opcode,
                    trace.response.sequence,
                    trace.response.payload.len()
                );
            }
        },
        SerialOpcode::Ack | SerialOpcode::Error => match SerialStatusPayload::decode(&trace.response.payload) {
            Ok(payload) => {
                println!(
                    "OK: {label} opcode={:?}, sequence={}, status={:?}, detail=0x{:08x}",
                    trace.response.opcode, trace.response.sequence, payload.code, payload.detail
                );
            }
            Err(_) => {
                println!(
                    "OK: {label} opcode={:?}, sequence={}, payload_len={}",
                    trace.response.opcode,
                    trace.response.sequence,
                    trace.response.payload.len()
                );
            }
        },
        _ => {
            println!(
                "OK: {label} opcode={:?}, sequence={}, payload_len={}",
                trace.response.opcode,
                trace.response.sequence,
                trace.response.payload.len()
            );
        }
    }
}

fn print_probe_error(dump_raw: bool, err: sptorch_hal_ffi::serial_backend::UartTang9kExchangeError) -> ! {
    eprintln!("Probe failed: {err}");
    if dump_raw {
        print_raw_exchange("error", &err.request_bytes, &err.raw_response_bytes);
    }
    std::process::exit(1);
}

fn print_raw_exchange(label: &str, request_bytes: &[u8], response_bytes: &[u8]) {
    println!("{label}_request_raw_len={}", request_bytes.len());
    println!("{label}_request_raw={}", format_hex(request_bytes));
    println!("{label}_response_raw_len={}", response_bytes.len());
    println!("{label}_response_raw={}", format_hex(response_bytes));
}

fn format_hex(bytes: &[u8]) -> String {
    format_probe_bytes_hex(bytes)
}

fn print_usage() {
    println!("Usage:");
    println!("  cargo run -p sptorch-hal-ffi --bin tang9k_probe -- --list");
    println!("  cargo run -p sptorch-hal-ffi --bin tang9k_probe -- --port COM3 --baud 115200 --timeout-ms 1000");
    println!(
        "  cargo run -p sptorch-hal-ffi --bin tang9k_probe -- --port COM3 --device-info --baud 115200 --timeout-ms 1000"
    );
    println!(
        "  cargo run -p sptorch-hal-ffi --bin tang9k_probe -- --port COM3 --bringup-suite --baud 115200 --timeout-ms 1000"
    );
    println!(
        "  cargo run -p sptorch-hal-ffi --bin tang9k_probe -- --port COM3 --matmul-smoke --baud 115200 --timeout-ms 1000"
    );
    println!(
        "  cargo run -p sptorch-hal-ffi --bin tang9k_probe -- --port COM3 --result-window-smoke --baud 115200 --timeout-ms 1000"
    );
    println!(
        "  cargo run -p sptorch-hal-ffi --bin tang9k_probe -- --port COM3 --result-window-status-smoke --baud 115200 --timeout-ms 1000"
    );
    println!(
        "  cargo run -p sptorch-hal-ffi --bin tang9k_probe -- --port COM3 --result-oob-smoke --baud 115200 --timeout-ms 1000"
    );
    println!(
        "  cargo run -p sptorch-hal-ffi --bin tang9k_probe -- --port COM3 --scratch-smoke --baud 115200 --timeout-ms 1000"
    );
    println!("  add --dump-raw to print request/response bytes for board bring-up diagnostics");
    println!("  add --record-json <path> to write a machine-readable acceptance record");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_defaults_to_list_mode() {
        let args = parse_args(Vec::new()).unwrap();
        assert_eq!(args.port, None);
        assert_eq!(args.baud, 115_200);
        assert_eq!(args.timeout_ms, 1_000);
        assert_eq!(args.command, ProbeCommand::Ping);
        assert!(!args.dump_raw);
        assert_eq!(args.record_json, None);
    }

    #[test]
    fn parse_probe_arguments() {
        let args = parse_args(vec![
            "--port".into(),
            "COM7".into(),
            "--baud".into(),
            "921600".into(),
            "--timeout-ms".into(),
            "250".into(),
        ])
        .unwrap();

        assert_eq!(args.port.as_deref(), Some("COM7"));
        assert_eq!(args.baud, 921_600);
        assert_eq!(args.timeout_ms, 250);
        assert_eq!(args.command, ProbeCommand::Ping);
        assert!(!args.dump_raw);
    }

    #[test]
    fn parse_matmul_smoke_and_dump_raw_arguments() {
        let args = parse_args(vec![
            "--port".into(),
            "COM7".into(),
            "--matmul-smoke".into(),
            "--dump-raw".into(),
            "--timeout-ms".into(),
            "250".into(),
        ])
        .unwrap();

        assert_eq!(args.port.as_deref(), Some("COM7"));
        assert_eq!(args.timeout_ms, 250);
        assert_eq!(args.command, ProbeCommand::MatmulSmoke);
        assert!(args.dump_raw);
    }

    #[test]
    fn parse_record_json_argument() {
        let args = parse_args(vec![
            "--port".into(),
            "COM3".into(),
            "--device-info".into(),
            "--record-json".into(),
            "target/tang9k/device-info.json".into(),
        ])
        .unwrap();

        assert_eq!(args.port.as_deref(), Some("COM3"));
        assert_eq!(args.command, ProbeCommand::DeviceInfo);
        assert_eq!(
            args.record_json.as_deref(),
            Some(std::path::Path::new("target/tang9k/device-info.json"))
        );
    }

    #[test]
    fn parse_bringup_suite_arguments() {
        let args = parse_args(vec!["--port".into(), "COM3".into(), "--bringup-suite".into()]).unwrap();

        assert_eq!(args.port.as_deref(), Some("COM3"));
        assert_eq!(args.command, ProbeCommand::BringupSuite);
    }

    #[test]
    fn parse_device_info_arguments() {
        let args = parse_args(vec!["--port".into(), "COM3".into(), "--device-info".into()]).unwrap();

        assert_eq!(args.port.as_deref(), Some("COM3"));
        assert_eq!(args.command, ProbeCommand::DeviceInfo);
    }

    #[test]
    fn parse_result_smoke_arguments() {
        let args = parse_args(vec!["--port".into(), "COM3".into(), "--result-smoke".into()]).unwrap();

        assert_eq!(args.port.as_deref(), Some("COM3"));
        assert_eq!(args.command, ProbeCommand::ResultSmoke);
    }

    #[test]
    fn parse_result_window_smoke_arguments() {
        let args = parse_args(vec!["--port".into(), "COM3".into(), "--result-window-smoke".into()]).unwrap();

        assert_eq!(args.port.as_deref(), Some("COM3"));
        assert_eq!(args.command, ProbeCommand::ResultWindowSmoke);
    }

    #[test]
    fn parse_result_window_status_smoke_arguments() {
        let args = parse_args(vec![
            "--port".into(),
            "COM3".into(),
            "--result-window-status-smoke".into(),
        ])
        .unwrap();

        assert_eq!(args.port.as_deref(), Some("COM3"));
        assert_eq!(args.command, ProbeCommand::ResultWindowStatusSmoke);
    }

    #[test]
    fn parse_result_oob_smoke_arguments() {
        let args = parse_args(vec!["--port".into(), "COM3".into(), "--result-oob-smoke".into()]).unwrap();

        assert_eq!(args.port.as_deref(), Some("COM3"));
        assert_eq!(args.command, ProbeCommand::ResultOobSmoke);
    }

    #[test]
    fn parse_scratch_smoke_arguments() {
        let args = parse_args(vec!["--port".into(), "COM3".into(), "--scratch-smoke".into()]).unwrap();

        assert_eq!(args.port.as_deref(), Some("COM3"));
        assert_eq!(args.command, ProbeCommand::ScratchSmoke);
    }

    #[test]
    fn format_hex_keeps_byte_boundaries_visible() {
        assert_eq!(format_hex(&[]), "<empty>");
        assert_eq!(format_hex(&[0x53, 0x50, 0x01, 0x7e]), "53 50 01 7e");
    }

    #[test]
    fn parse_rejects_unknown_arguments() {
        let err = parse_args(vec!["--wat".into()]).unwrap_err();
        assert!(err.contains("unknown argument"));
    }
}

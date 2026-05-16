use sptorch_hal_ffi::serial_backend::{
    list_tang9k_serial_ports, probe_tang9k_matmul_smoke_with_trace, probe_tang9k_ping_with_trace,
};
use std::time::Duration;

#[derive(Debug)]
struct Args {
    port: Option<String>,
    baud: u32,
    timeout_ms: u64,
    command: ProbeCommand,
    dump_raw: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProbeCommand {
    Ping,
    MatmulSmoke,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            port: None,
            baud: 115_200,
            timeout_ms: 1_000,
            command: ProbeCommand::Ping,
            dump_raw: false,
        }
    }
}

fn main() {
    let args = parse_args(std::env::args().skip(1).collect()).unwrap_or_else(|err| {
        eprintln!("{err}");
        print_usage();
        std::process::exit(2);
    });

    if let Some(port) = args.port {
        println!(
            "Probing Tang9k on {port} at {} baud, timeout={}ms, command={:?}",
            args.baud, args.timeout_ms, args.command
        );
        let result = match args.command {
            ProbeCommand::Ping => {
                probe_tang9k_ping_with_trace(&port, args.baud, Duration::from_millis(args.timeout_ms))
            }
            ProbeCommand::MatmulSmoke => {
                probe_tang9k_matmul_smoke_with_trace(&port, args.baud, Duration::from_millis(args.timeout_ms))
            }
        };

        match result {
            Ok(trace) => {
                println!(
                    "OK: response opcode={:?}, sequence={}, payload_len={}",
                    trace.response.opcode,
                    trace.response.sequence,
                    trace.response.payload.len()
                );
                if args.dump_raw {
                    print_raw_exchange(&trace.request_bytes, &trace.raw_response_bytes);
                }
            }
            Err(err) => {
                eprintln!("Probe failed: {err}");
                if args.dump_raw {
                    print_raw_exchange(&err.request_bytes, &err.raw_response_bytes);
                }
                std::process::exit(1);
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
            "--dump-raw" => {
                args.dump_raw = true;
                idx += 1;
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }

    Ok(args)
}

fn print_raw_exchange(request_bytes: &[u8], response_bytes: &[u8]) {
    println!("request_raw_len={}", request_bytes.len());
    println!("request_raw={}", format_hex(request_bytes));
    println!("response_raw_len={}", response_bytes.len());
    println!("response_raw={}", format_hex(response_bytes));
}

fn format_hex(bytes: &[u8]) -> String {
    if bytes.is_empty() {
        return "<empty>".into();
    }

    bytes
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<Vec<_>>()
        .join(" ")
}

fn print_usage() {
    println!("Usage:");
    println!("  cargo run -p sptorch-hal-ffi --bin tang9k_probe -- --list");
    println!("  cargo run -p sptorch-hal-ffi --bin tang9k_probe -- --port COM3 --baud 115200 --timeout-ms 1000");
    println!(
        "  cargo run -p sptorch-hal-ffi --bin tang9k_probe -- --port COM3 --matmul-smoke --baud 115200 --timeout-ms 1000"
    );
    println!("  add --dump-raw to print request/response bytes for board bring-up diagnostics");
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

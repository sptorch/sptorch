use sptorch_hal_ffi::serial_backend::{list_tang9k_serial_ports, probe_tang9k_ping};
use std::time::Duration;

#[derive(Debug)]
struct Args {
    port: Option<String>,
    baud: u32,
    timeout_ms: u64,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            port: None,
            baud: 115_200,
            timeout_ms: 1_000,
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
            "Probing Tang9k on {port} at {} baud, timeout={}ms",
            args.baud, args.timeout_ms
        );
        match probe_tang9k_ping(&port, args.baud, Duration::from_millis(args.timeout_ms)) {
            Ok(response) => {
                println!(
                    "OK: response opcode={:?}, sequence={}, payload_len={}",
                    response.opcode,
                    response.sequence,
                    response.payload.len()
                );
            }
            Err(err) => {
                eprintln!("Probe failed: {err}");
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
            other => return Err(format!("unknown argument: {other}")),
        }
    }

    Ok(args)
}

fn print_usage() {
    println!("Usage:");
    println!("  cargo run -p sptorch-hal-ffi --bin tang9k_probe -- --list");
    println!("  cargo run -p sptorch-hal-ffi --bin tang9k_probe -- --port COM3 --baud 115200 --timeout-ms 1000");
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
    }

    #[test]
    fn parse_rejects_unknown_arguments() {
        let err = parse_args(vec!["--wat".into()]).unwrap_err();
        assert!(err.contains("unknown argument"));
    }
}

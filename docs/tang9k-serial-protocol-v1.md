# Tang9k Serial Protocol v1

This document is the wire-level contract for the Tang9k bring-up path. It is intentionally small: the goal is to make the host, Verilog testbench, dry-run backend, and future UART/DMA transport agree on bytes before optimizing performance.

## Scope

- Owner crate: `sptorch-hal::serial`.
- Dispatch bridge: `sptorch-hal-ffi::serial_backend`.
- Current operation scope: control frames, 32x32 F32 MatMul tile commands, scratch read/write smoke frames, and a 32-bit result-window smoke readback.
- Out of scope for v1: large tensor payload transfer, arbitrary tile sizes, real UART timing, DMA descriptor rings, and full matrix-result buffer transfer.

## Frame Format

All integer fields are little-endian.

```text
0..2    magic        [u8; 2] = ASCII "SP"
2       version      u8 = 1
3       opcode       u8
4..8    sequence     u32
8..12   payload_len  u32
12..14  flags        u16
14..16  reserved     u16 = 0
16..N   payload      [u8; payload_len]
N..N+4  checksum     u32 = FNV-1a(header + payload), excludes padding
...     padding      zero bytes until total frame length is 8-byte aligned
```

Normative rules:

- `magic` must be `0x53 0x50`.
- `version` must be `1`.
- `payload_len` must be at most `64 KiB`.
- `reserved` must be zero.
- Padding bytes must be zero.
- The encoded frame length must be aligned to 8 bytes.
- The checksum covers the 16-byte header and payload only.

## Opcodes

| Name | Value | Direction | Payload |
| --- | ---: | --- | --- |
| `Ping` | `0x01` | host -> device | implementation-defined or empty |
| `Pong` | `0x02` | device -> host | implementation-defined or empty |
| `Matmul32x32` | `0x10` | host -> device | `Matmul32x32Command` |
| `ScratchWrite32` | `0x20` | host -> device | `ScratchWrite32Command` |
| `ScratchRead32` | `0x21` | host -> device | `ScratchRead32Command` |
| `ScratchValue32` | `0x22` | device -> host | `ScratchValue32Payload` |
| `ResultRead32` | `0x30` | host -> device | `ResultRead32Command` |
| `ResultValue32` | `0x31` | device -> host | `ResultValue32Payload` |
| `Ack` | `0x7e` | device -> host | `SerialStatusPayload` |
| `Error` | `0x7f` | device -> host | `SerialStatusPayload` |

Unknown opcodes must be rejected before execution.

## Status Payload

`Ack` and `Error` frames use a fixed 8-byte payload.

```text
0..2  status_code  u16
2..4  reserved     u16 = 0
4..8  detail       u32
```

Status codes:

| Name | Value | Meaning |
| --- | ---: | --- |
| `Ok` | `0x0000` | Command accepted or completed |
| `BadFrame` | `0x0001` | Frame failed structural validation |
| `UnsupportedOpcode` | `0x0002` | Opcode is known to host spec but unsupported by target |
| `InvalidPayload` | `0x0003` | Payload length or content is invalid |
| `Busy` | `0x0004` | Target queue cannot accept more work |
| `HardwareFault` | `0x0005` | Target reported PE/DMA/FIFO fault |

`detail` is opcode-specific. For MatMul it should prefer `tile_id`; for hardware faults it may contain a device status register snapshot.

## Scratch Data-Plane Smoke Frames

`ScratchWrite32` and `ScratchRead32` are the first strict data-plane extension on top of the command lifecycle.
They are intentionally tiny: the goal is to verify that the board can store and return one 32-bit word before
real DMA, queue depth exposure, or matrix-result readback exists.

Payloads:

```text
ScratchWrite32Command
0..4    offset  u32
4..8    value   u32

ScratchRead32Command
0..4    offset  u32

ScratchValue32Payload
0..4    offset  u32
4..8    value   u32
```

Rules:

- `ScratchWrite32` must reply with `Ack/Ok` after accepting the write.
- `ScratchRead32` must reply with `ScratchValue32`, not `Ack`, when the stored offset matches the read offset.
- `ScratchValue32` is the host-visible readback payload; it is not a status frame.
- A read before any write may return `Error/HardwareFault`.
- The board-side responder currently stores one 32-bit slot and uses it as the scratch smoke baseline.

## Result Window Smoke Frames

`ResultRead32` and `ResultValue32` are the first MatMul-adjacent readback contract. They do not transfer the
full 32x32 output tile yet. Instead, the target stores a 32-bit result summary after accepting a `Matmul32x32`
command, and the host reads that summary from a small result window.

Payloads:

```text
ResultRead32Command
0..4    offset  u32

ResultValue32Payload
0..4    offset  u32
4..8    value   u32
```

Rules:

- `ResultRead32` must reply with `ResultValue32`, not `Ack`, when the requested offset matches a valid result slot.
- A result read before any MatMul command may return `Error/HardwareFault`.
- The current responder writes `offset = Matmul32x32Command.out_offset[31:0]`.
- The current responder writes `value = tile_id ^ flags ^ a_offset_low ^ a_offset_high ^ b_offset_low ^ b_offset_high ^ out_offset_low ^ out_offset_high`.
- The Rust host computes the same smoke value through `Matmul32x32Command::smoke_result_summary`.
- This path only proves command-triggered result-window visibility. Real PE output and larger buffer readback remain future work.

## Matmul32x32 Command

The v1 MatMul command payload is fixed at 32 bytes.

```text
0..4    tile_id     u32
4..12   a_offset    u64
12..20  b_offset    u64
20..28  out_offset  u64
28..32  flags       u32
```

Semantics:

- The tile shape is fixed: `M=32`, `K=32`, `N=32`.
- Offsets are device-side byte offsets, not host virtual addresses.
- Matrix memory layout is row-major.
- Host-side planner must reject `m`, `k`, or `n` values that are zero or not divisible by 32.
- Host-side planner must reject row strides smaller than the logical row length.

Flags:

| Name | Value | Meaning |
| --- | ---: | --- |
| `CLEAR_OUTPUT` | `0x0000_0001` | Clear output tile before writing partial sum |
| `ACCUMULATE` | `0x0000_0002` | Accumulate into existing output tile |
| `LAST_K_TILE` | `0x0000_0004` | Last K tile for this output tile; target may flush/done |

The standard command order is:

```text
m_tile -> n_tile -> k_tile
```

This lets the target keep one output tile lifecycle at a time: clear on the first K tile, accumulate on middle K tiles, and flush on the last K tile.

## Sequence Rules

- Sequence numbers are `u32`.
- Host-side dry-run uses wrapping increment.
- A response frame must echo the command sequence unless a future extension explicitly defines asynchronous completion queues.
- A command is accepted only by `Ack + SerialStatusCode::Ok`.
- `Ack + Busy` means the target queue did not accept the command; host code must retry or apply backpressure instead of treating it as success.
- `Error` frames always represent command rejection, even when the embedded status code is `Ok`.
- Transport layers must not silently accept mismatched opcode or sequence responses.

## Host Submit Queue

The Rust host model uses `SerialSubmitQueue` as the normative dry-run lifecycle before real UART/DMA exists.

Rules:

- The queue owns host sequence allocation and may wrap naturally at `u32::MAX`.
- Queue capacity is explicit; a full queue must fail with `QueueFull` rather than dropping frames.
- Submission depth increases only while a frame is outstanding.
- Submission depth must return to the previous value after success, command rejection, or transport failure.
- Host trace records should include the command frame, response frame, decoded status, queue depth before enqueue, queue depth after enqueue, and queue depth after submit.

This keeps dry-run behavior honest: a loopback backend cannot accidentally pass by echoing command frames, and a real transport can be swapped in without changing the dispatch contract.

## Stream Framing

UART and USB-CDC transports expose a byte stream, not frame boundaries. Host transports should use `SerialStreamDecoder` before interpreting a response.

Rules:

- The decoder may receive arbitrary byte chunks.
- Bytes before the first `magic` sequence are treated as noise and discarded.
- A partial `magic` prefix at the end of a chunk must be preserved for the next chunk.
- A complete header is required before calculating the expected frame length.
- `payload_len` must be validated before allocating or waiting for a full payload.
- Once enough bytes are buffered, the full frame must still pass `SerialFrame::decode`.

This split keeps recovery behavior consistent across loopback, UART, and future DMA-backed transports.

## UART Bring-Up

The first real-board host path is implemented by `sptorch-hal-ffi::serial_backend::UartTang9kTransport`.
The first minimal target-side smoke-test bitstream lives in `hardware/tang9k/uart_responder`: it validates serial-v1 `Ping` frames and returns an empty `Pong` with the same sequence. The command-lifecycle version also accepts a `Matmul32x32` frame and returns `Ack/Ok` after validating payload length, checksum, and padding; the scratch smoke version stores one 32-bit word on `ScratchWrite32` and returns it via `ScratchValue32` on `ScratchRead32`; the result smoke version records a deterministic MatMul summary and returns it through `ResultValue32`.

Safe bring-up sequence:

```powershell
& 'C:\Gowin\Gowin_V1.9.12.02_SP2_x64\IDE\bin\gw_sh.exe' hardware\tang9k\uart_responder\build.tcl
$fs = (Resolve-Path hardware\tang9k\uart_responder\impl\pnr\tang9k_uart_responder.fs).Path
& 'C:\Gowin\Gowin_V1.9.12.02_SP2_x64\Programmer\bin\programmer_cli.exe' --device GW1NR-9C --operation_index 2 --fsFile $fs --cable "USB Debugger A" --frequency 2.5MHz
cargo run -p sptorch-hal-ffi --bin tang9k_probe -- --list
cargo run -p sptorch-hal-ffi --bin tang9k_probe -- --port COM3 --baud 115200 --timeout-ms 1000
cargo run -p sptorch-hal-ffi --bin tang9k_probe -- --port COM3 --matmul-smoke --baud 115200 --timeout-ms 1000
cargo run -p sptorch-hal-ffi --bin tang9k_probe -- --port COM3 --matmul-smoke --baud 115200 --timeout-ms 1000 --dump-raw
cargo run -p sptorch-hal-ffi --bin tang9k_probe -- --port COM3 --result-smoke --baud 115200 --timeout-ms 1000
cargo run -p sptorch-hal-ffi --bin tang9k_probe -- --port COM3 --result-smoke --baud 115200 --timeout-ms 1000 --dump-raw
cargo run -p sptorch-hal-ffi --bin tang9k_probe -- --port COM3 --scratch-smoke --baud 115200 --timeout-ms 1000
cargo run -p sptorch-hal-ffi --bin tang9k_probe -- --port COM3 --scratch-smoke --baud 115200 --timeout-ms 1000 --dump-raw
```

Rules:

- `--list` must be used first because it does not write to any device.
- `--port` sends exactly one `Ping` frame with sequence `0` and payload `sptorch-ping`.
- `--matmul-smoke` sends exactly one `Matmul32x32` command frame with sequence `1`.
- `--result-smoke` sends one `Matmul32x32` command frame with sequence `4`, then one `ResultRead32` frame with sequence `5`, and checks the returned `ResultValue32`.
- `--scratch-smoke` sends one `ScratchWrite32` followed by one `ScratchRead32`, then checks the returned `ScratchValue32`.
- `--dump-raw` prints host request bytes and target response bytes; keep it enabled when debugging checksum drift, stale serial data, or RTL framing changes.
- A target may answer with `Pong`, `Ack/Ok`, `ScratchValue32`, or `ResultValue32` during early bring-up.
- `Busy`, `Error`, bad sequence, malformed frames, transport I/O errors, and timeouts are all reported explicitly.
- If Windows only shows `COM1 Unknown`, treat it as suspicious unless the board documentation confirms Tang9k is mapped there; many built-in ACPI serial ports appear this way.

Real-board acceptance recorded on 2026-05-18:

- JTAG cable: `USB Debugger A`.
- Device scan: `GW1NR-9C`, ID `0x1100481B`.
- SRAM Program status: `0x0003F020`.
- UART port: `COM3`, `115200 8N1`.
- Ping host result: `OK: response opcode=Pong, sequence=0, payload_len=0`.
- Matmul smoke host result: `OK: response opcode=Ack, sequence=1, payload_len=8`.
- Matmul smoke ACK response bytes:

```text
53 50 01 7e 01 00 00 00 08 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 b8 59 20 24 00 00 00 00
```

- Scratch smoke host result: `OK: scratch write opcode=Ack, sequence=2, payload_len=8`.
- Scratch smoke host result: `OK: scratch read opcode=ScratchValue32, sequence=3, offset=0x00000044, value=0x11223344`.
- Scratch smoke write raw response bytes:

```text
53 50 01 7e 02 00 00 00 08 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 c1 fc 65 21 00 00 00 00
```

- Scratch smoke read raw response bytes:

```text
53 50 01 22 03 00 00 00 08 00 00 00 00 00 00 00 44 00 00 00 44 33 22 11 86 82 c3 d0 00 00 00 00
```

- Result smoke host result: `OK: result matmul opcode=Ack, sequence=4, payload_len=8`.
- Result smoke host result: `OK: result read opcode=ResultValue32, sequence=5, offset=0x00002000, value=0x00003005`.
- Result smoke read raw response bytes:

```text
53 50 01 31 05 00 00 00 08 00 00 00 00 00 00 00 00 20 00 00 05 30 00 00 d4 d0 6f 06 00 00 00 00
```

## Implementation Standards

Host implementations:

- Must use `sptorch-hal::serial` constants instead of duplicating magic/version/header lengths.
- Must use `SerialStreamDecoder` or equivalent behavior for byte-stream transports.
- Must parse a full frame with `SerialFrame::decode` after stream framing.
- Must keep large tensor data out of control payloads; use offsets into device memory instead.
- Must preserve traceability: record at least tile plan, generated frames, and queue depth around submit.
- Must validate `Ack/Ok` with `validate_status_response` or equivalent logic before reporting command success.

Target implementations:

- Must validate `magic`, `version`, `opcode`, `payload_len`, reserved fields, checksum, and padding before execution.
- Must reply with `Ack`, `Error`, or the dedicated data-plane response opcode when a new readback path is defined.
- Must return `Busy` when the command FIFO cannot accept work; it must not accept and silently drop the frame.
- Must not reinterpret reserved bytes until a new protocol version is defined.
- Should compute response checksums in a timing-safe way. On GW1NR-9C, chaining 24 FNV-1a feeds in one combinational function produced ACK checksum drift; the responder computes one checksum byte feed per clock before transmitting.

Compatibility:

- v1 is strict by default. Any non-zero reserved field is an error.
- New opcodes should be appended, not renumbered.
- New payload fields require either a new opcode or a new protocol version.

## Golden Vectors

The repository includes byte-level conformance vectors in `crates/hal/tests/tang9k_serial_golden.rs`.

Golden coverage:

- `Ping` frame with flags, payload, checksum, and padding.
- `Ack` frame using `SerialStatusPayload { Busy, detail = 0x11223344 }`.
- `Matmul32x32Command` payload and full frame.
- `ScratchWrite32`, `ScratchRead32`, `ScratchValue32`, `ResultRead32`, and `ResultValue32` payloads and full frames.
- Stream decoder behavior with leading noise and multiple golden frames in one byte stream.

Any non-Rust implementation should reproduce these exact bytes before being treated as protocol-compatible.

## Current Acceptance Tests

- `cargo test -p sptorch-hal`
- `cargo test -p sptorch-hal-ffi --test test_serial_backend`
- `cargo test -p sptorch-hal --test tang9k_serial_golden`
- 10k-frame loopback stability.
- 32x32 and 64x64 MatMul tile planning.
- Frame corruption rejection: checksum, padding, reserved fields, unknown status code.
- Stream framing: fragmented frames, noise recovery, split magic prefixes, multiple frames per chunk, oversized declared payload rejection.
- Golden vectors: byte-for-byte frame, payload, checksum, padding, and stream framing compatibility.
- Submit lifecycle: host queue sequence allocation, ACK/OK validation, Busy rejection, depth drain after transport failure, and queue high-watermark telemetry.
- Dispatch dry-run: `core-ops::matmul` -> `Tang9kSerialDryRunBackend` -> `SerialSubmitQueue` -> `Tang9kSerialTransport`.

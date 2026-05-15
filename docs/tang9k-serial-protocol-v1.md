# Tang9k Serial Protocol v1

This document is the wire-level contract for the Tang9k bring-up path. It is intentionally small: the goal is to make the host, Verilog testbench, dry-run backend, and future UART/DMA transport agree on bytes before optimizing performance.

## Scope

- Owner crate: `sptorch-hal::serial`.
- Dispatch bridge: `sptorch-hal-ffi::serial_backend`.
- Current operation scope: control frames and 32x32 F32 MatMul tile commands.
- Out of scope for v1: large tensor payload transfer, arbitrary tile sizes, real UART timing, DMA descriptor rings, and result readback protocol.

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
- Transport layers must not silently accept mismatched opcode or sequence echoes.

## Implementation Standards

Host implementations:

- Must use `sptorch-hal::serial` constants instead of duplicating magic/version/header lengths.
- Must parse a full frame with `SerialFrame::decode` after stream framing.
- Must keep large tensor data out of control payloads; use offsets into device memory instead.
- Must preserve traceability: record at least tile plan, generated frames, and queue depth around submit.

Target implementations:

- Must validate `magic`, `version`, `opcode`, `payload_len`, reserved fields, checksum, and padding before execution.
- Must reply with `Ack` or `Error` using `SerialStatusPayload`.
- Must not reinterpret reserved bytes until a new protocol version is defined.

Compatibility:

- v1 is strict by default. Any non-zero reserved field is an error.
- New opcodes should be appended, not renumbered.
- New payload fields require either a new opcode or a new protocol version.

## Current Acceptance Tests

- `cargo test -p sptorch-hal`
- `cargo test -p sptorch-hal-ffi --test test_serial_backend`
- 10k-frame loopback stability.
- 32x32 and 64x64 MatMul tile planning.
- Frame corruption rejection: checksum, padding, reserved fields, unknown status code.
- Dispatch dry-run: `core-ops::matmul` -> `Tang9kSerialDryRunBackend` -> `Tang9kSerialTransport`.

# SPTorch - Rust Industrial Heterogeneous AI Framework

SPTorch is the framework/base repository of the ecosystem: tensor runtime, autograd, ops, HAL, distributed training, live evolution, serialization, and the stable `sptorch::v1` facade. Products and IDEs are intentionally kept in separate repositories.

## Architecture Positioning

- **Framework base**: this repo, `sptorch`, owns the reusable engine, protocols, hardware abstraction, and publishing pipeline.
- **IDE control center**: `SPTorch Studio` lives in `../sptorch-studio` and consumes framework crates through Git dependencies.
- **Product layer**: industrial `Text2SQL` lives in `../text2sql` and consumes the framework through `sptorch = { git = "https://github.com/hjd92215202/sptorch.git", branch = "main" }`.
- **Boundary rule**: framework code must not depend on product or IDE code. Products should prefer the stable `sptorch::v1` facade instead of internal crates.

## Core Vision

- **Compute sovereignty**: low-cost distributed training and heterogeneous execution.
- **Hardware sovereignty**: HAL + C FFI, so custom CUDA/NPU/FPGA backends can be plugged in cleanly.
- **Live evolution**: double-buffered parameters, EWC, monitoring, rollback, and versioned tensor protocol.
- **Developer sovereignty**: Studio is an independent IDE/control center over the framework ecosystem.

## Framework Workspace

```text
crates/
  sptorch/          Stable public facade crate for external consumers
  core-tensor/      Tensor, Shape, DType, Storage, strides, basic backward support
  core-autograd/    Computation graph and backward scheduling
  core-ops/         Differentiable operators and backend dispatch
  hal/              Hardware Abstraction Layer: Backend + KernelProvider + multi-board topology
  hal-ffi/          C FFI bridge for external hardware plugins
  mock-npu/         Mock NPU cdylib for FFI chain validation
  nn/               Module trait, Linear, LoRA, Embedding, LayerNorm, MHA, Transformer, GPT
  optim/            SGD, AdamW, schedulers, gradient clipping
  data/             Tokenizers, TextDataset, DataLoader
  serialize/        Checkpoint and safetensors support
  runtime-cuda/     CUDA backend kernels and cuBLAS matmul
  distributed/      gRPC coordinator/worker, AllReduce, Barrier, hardware-aware parallel plans
  live-evolution/   Double-buffer parameters, EWC, online monitoring and rollback
  versioning/       Versioned tensor protocol shared with Studio
  benchmarks/       Internal Criterion performance baselines (publish = false)
```

External ecosystem repositories:

```text
../text2sql/          Production Text2SQL product workspace
../sptorch-studio/    Tauri + React Studio IDE workspace
```


## Hardware Roadmap Focus

- Tank9k/Tang 9k bring-up is now treated as a framework capability, not a product feature.
- `sptorch-hal::topology` models multi-board nodes, serial/PCIe/Ethernet links, connectivity validation, ring allreduce estimates, and matmul shard plans.
- `sptorch-hal::serial` provides the first Tang9k protocol scaffold: aligned frames, checksum validation, loopback testing, 32x32 MatMul command payloads, `ScratchWrite32`/`ScratchRead32` data-plane smoke frames, and a 4-word `ResultRead32`/`ResultValue32` result-window smoke path.
- `sptorch-hal::serial::SerialSubmitQueue` models host-side sequence allocation, ACK/OK response validation, queue depth, and Busy backpressure before real UART/DMA is wired in.
- `sptorch-hal::serial::plan_matmul32x32_commands` turns row-major board memory layouts into deterministic Tang9k tile command streams.
- `sptorch-hal-ffi::serial_backend` registers a Tang9k serial dry-run backend into core dispatch, so MatMul can exercise serial frames and ACK/Error lifecycle before real UART/DMA is connected.
- `Tang9kSerialTransport` isolates the send/receive boundary, letting loopback, UART, or DMA transports plug into the same dispatch path.
- `UartTang9kTransport` and `tang9k_probe` provide the first real serial bring-up path: list visible COM ports first, then send protocol `Ping`, `Matmul32x32`, scratch write/read, single result, or 4-word result-window probes after confirming the Tang9k port.
- `hardware/tang9k/uart_responder` contains the first minimal Gowin bitstream project for real Tang9k smoke testing: it receives serial-v1 `Ping` over USB-UART and returns a checksum-valid `Pong`; it accepts `Matmul32x32` and returns checksum-valid `Ack/Ok`; it stores/reads back one 32-bit scratch value; it also records a deterministic 4-word MatMul summary window for result readback.
- `sptorch-hal::serial::SerialStreamDecoder` standardizes byte-stream framing for UART/USB-CDC transports before strict frame decoding.
- Tang9k serial v1 is now documented as a strict wire contract: [docs/tang9k-serial-protocol-v1.md](docs/tang9k-serial-protocol-v1.md).
- Tang9k protocol conformance is guarded by byte-level golden vectors in `crates/hal/tests/tang9k_serial_golden.rs`.
- `sptorch-distributed::hardware_parallel` turns a hardware topology into dry-run validation plans for multi-board matmul + allreduce before real serial/PCIe DMA is wired in.

```bash
# Safe: lists serial ports without writing to the board.
cargo run -p sptorch-hal-ffi --bin tang9k_probe -- --list

# Sends one Tang9k protocol Ping after you have confirmed the COM port.
cargo run -p sptorch-hal-ffi --bin tang9k_probe -- --port COM3 --baud 115200 --timeout-ms 1000

# Sends one command-lifecycle Matmul32x32 smoke frame; requires the newer responder bitstream.
cargo run -p sptorch-hal-ffi --bin tang9k_probe -- --port COM3 --matmul-smoke --baud 115200 --timeout-ms 1000

# Sends Matmul32x32, then reads back the 32-bit result-window summary.
cargo run -p sptorch-hal-ffi --bin tang9k_probe -- --port COM3 --result-smoke --baud 115200 --timeout-ms 1000

# Sends Matmul32x32, then reads back all 4 smoke words from the result window.
cargo run -p sptorch-hal-ffi --bin tang9k_probe -- --port COM3 --result-window-smoke --baud 115200 --timeout-ms 1000

# Writes and reads back one 32-bit scratch value on the board.
cargo run -p sptorch-hal-ffi --bin tang9k_probe -- --port COM3 --scratch-smoke --baud 115200 --timeout-ms 1000

# Add --dump-raw when debugging board bytes, checksum drift, or stale serial data.
cargo run -p sptorch-hal-ffi --bin tang9k_probe -- --port COM3 --result-window-smoke --baud 115200 --timeout-ms 1000 --dump-raw
```

Current real-board smoke test:

```text
Gowin SRAM Program: USB Debugger A -> GW1NR-9C, Status Code 0x0003F020
Host probe: COM3 @ 115200 -> OK: response opcode=Pong, sequence=0, payload_len=0
Host command lifecycle: COM3 @ 115200 -> OK: response opcode=Ack, sequence=1, payload_len=8
Host scratch data-plane: COM3 @ 115200 -> OK: ScratchValue32 offset=0x00000044, value=0x11223344
Host result window: COM3 @ 115200 -> OK: ResultValue32 offset=0x00002000, value=0x00003005
Host result window 4-word smoke: COM3 @ 115200 -> [0x00003005, 0x9e3749bc, 0x3c6ec377, 0xdaa65d2e]
Ack raw response: 53 50 01 7e 01 00 00 00 08 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 b8 59 20 24 00 00 00 00
```

## Quick Start

```bash
# Framework tests
cargo test --workspace

# Framework checks
cargo check --workspace

# Performance baselines used by CI as a trend sentinel
cargo bench -p sptorch-benchmarks
```

Product and IDE commands are owned by their independent repositories:

```bash
cd ../text2sql && cargo test --workspace
cd ../sptorch-studio && npm run test
```

Engineering tool commands (local training/demo) are owned by the independent tools repository:

```bash
cd ../sptorch-tools && cargo check --workspace
```

## Release Notes

- Publishing and version strategy: [docs/docs.release-strategy.md](docs/docs.release-strategy.md).
- crates.io publish order: [docs/docs.publish-order.md](docs/docs.publish-order.md).
- Current roadmap and acceptance record: [docs/roadmap.md](docs/roadmap.md).

## Current Release Checklist

- [x] `sptorch` facade metadata is filled in.
- [x] Non-published framework target is marked `publish = false`: `sptorch-mock-npu`.
- [x] `sptorch-core-tensor`, `sptorch-data`, and `sptorch-versioning` have package metadata and README baselines.
- [ ] `cargo package -p sptorch` still requires publishing internal dependency crates first.

## Studio / Product Status

- `SPTorch Studio` has moved to `../sptorch-studio`; it remains the ecosystem control center for versioned tensors, live-evolution metrics, memory snapshots, autograd graph visualization, and hardware fence monitoring.
- `Text2SQL` has moved to `../text2sql`; it remains the first production-grade sample product and validates the framework from training to inference to delivery.
- This framework repo stays clean: no product source, no IDE source, no product CI, no frontend CI.

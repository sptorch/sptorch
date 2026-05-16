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
- `sptorch-hal::serial` provides the first Tang9k protocol scaffold: aligned frames, checksum validation, loopback testing, and 32x32 MatMul command payloads.
- `sptorch-hal::serial::SerialSubmitQueue` models host-side sequence allocation, ACK/OK response validation, queue depth, and Busy backpressure before real UART/DMA is wired in.
- `sptorch-hal::serial::plan_matmul32x32_commands` turns row-major board memory layouts into deterministic Tang9k tile command streams.
- `sptorch-hal-ffi::serial_backend` registers a Tang9k serial dry-run backend into core dispatch, so MatMul can exercise serial frames and ACK/Error lifecycle before real UART/DMA is connected.
- `Tang9kSerialTransport` isolates the send/receive boundary, letting loopback, UART, or DMA transports plug into the same dispatch path.
- `sptorch-hal::serial::SerialStreamDecoder` standardizes byte-stream framing for UART/USB-CDC transports before strict frame decoding.
- Tang9k serial v1 is now documented as a strict wire contract: [docs/tang9k-serial-protocol-v1.md](docs/tang9k-serial-protocol-v1.md).
- Tang9k protocol conformance is guarded by byte-level golden vectors in `crates/hal/tests/tang9k_serial_golden.rs`.
- `sptorch-distributed::hardware_parallel` turns a hardware topology into dry-run validation plans for multi-board matmul + allreduce before real serial/PCIe DMA is wired in.

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

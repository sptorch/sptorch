# SPTorch Three-Repository Release Strategy (v2)

This document defines the repository and release boundary: framework is framework, product is product, and IDE is IDE. The current ecosystem uses three independent repositories.

## 1. Repository Boundaries

- Framework repo: `C:\Users\brace\Documents\work\ai\sptorch`, root Cargo workspace, `crates/*` only.
- Product repo: `C:\Users\brace\Documents\work\ai\text2sql`, `text2sql/` plus `cli-text2sql/`.
- IDE repo: `C:\Users\brace\Documents\work\ai\sptorch-studio`, Tauri 2 backend, React frontend, Studio tests.
- The framework repo must not depend on product or IDE code. Product and IDE consume the framework through Git or versioned dependencies.

## 2. Stable API Contract

External product code should prefer the `crates/sptorch` facade namespaces:

- `sptorch::v1::nn`
- `sptorch::v1::optim`
- `sptorch::v1::ops`
- `sptorch::v1::checkpoint`
- `sptorch::v1::prelude`

Text2SQL should depend on the `sptorch` facade. Studio may also depend on protocol/runtime crates such as `sptorch-versioning`, `sptorch-live-evolution`, and `sptorch-hal` because it is the ecosystem control center.

## 3. Transition Dependency Policy

Before the full framework is published to crates.io, external repositories use Git dependencies:

```toml
sptorch = { git = "https://github.com/hjd92215202/sptorch.git", branch = "main" }
```

Studio uses the same Git source for framework subcrates:

```toml
sptorch-versioning = { git = "https://github.com/hjd92215202/sptorch.git", branch = "main" }
sptorch-live-evolution = { git = "https://github.com/hjd92215202/sptorch.git", branch = "main" }
```

After framework publishing stabilizes, product and IDE repos should move to crates.io versions or fixed Git `rev` pins.

## 4. Release Strategy

1. Publish foundational framework crates first, for example `sptorch-core-tensor`, `sptorch-data`, and `sptorch-versioning`.
2. Publish dependent framework crates, for example `sptorch-core-ops`, `sptorch-nn`, `sptorch-optim`, and `sptorch-live-evolution`.
3. Publish the facade crate `sptorch`.
4. Update Text2SQL and Studio dependencies in their own repositories.

## 5. CI Boundaries

- Framework CI: Rust framework checks only, such as `cargo fmt`, `cargo clippy`, and `cargo test --workspace`.
- Text2SQL CI: `cargo metadata --no-deps`, `cargo check --workspace`, and `cargo test --workspace`.
- Studio CI: Tauri/Rust checks plus frontend `npm ci` and `npm run test`.

## 6. Repository Roles

- `sptorch`: reusable framework, protocol, HAL, runtime, and release pipeline.
- `text2sql`: production product built from SPTorch capabilities.
- `sptorch-studio`: ecosystem IDE/control center built on SPTorch protocols and telemetry.

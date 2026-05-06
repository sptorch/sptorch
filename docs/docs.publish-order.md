# SPTorch crates.io 发布顺序（v1）

目标：发布 `sptorch` 门面 crate，使产品仓可通过 `sptorch = "0.1.x"` 依赖框架。

Naming convention: crates.io package names use `sptorch-*`; Rust library crate identifiers use `sptorch_*`.

## A. 预检查

1. 全部框架 crate 的内部依赖均使用 `path + version`（已执行）。
2. 非发布目标 crate 设置 `publish = false`（已执行）。
3. 使用 `cargo package -p <crate> --no-verify` 做本地打包检查。

## B. 建议发布序列（按依赖拓扑）

1. `sptorch-core-tensor`
2. `sptorch-data`
3. `sptorch-versioning`
4. `sptorch-optim`（依赖 core-tensor）
5. `sptorch-serialize`（依赖 core-tensor）
6. `sptorch-hal`（依赖 core-tensor）
7. `sptorch-core-autograd`（依赖 core-tensor）
8. `sptorch-runtime-cuda`（依赖 core-tensor）
9. `sptorch-core-ops`（依赖 core-tensor，optional sptorch-runtime-cuda）
10. `sptorch-nn`（依赖 core-tensor + core-ops）
11. `sptorch-hal-ffi`（依赖 hal + core-tensor）
12. `sptorch-distributed`（依赖 core-tensor + serialize）
13. `sptorch-live-evolution`（依赖 core-tensor + core-ops + nn + optim + versioning）
14. `sptorch`（依赖以上框架能力）

> Note: `sptorch-mock-npu`, `sptorch-cli-train`, and `sptorch-cli-train-gpu` are not published to crates.io. Text2SQL and Studio are independent repositories, not framework workspace publish targets.

## C. 发布执行命令模板

```bash
cargo package -p <crate>
cargo publish -p <crate>
```

推荐每发布一个 crate 后等待索引同步，再发布下一个（避免“no matching package found”）。

## D. 当前阻塞（已识别）

- `cargo package -p sptorch` 失败原因：依赖 crate（如 `sptorch-core-autograd`）尚未在 registry 可解析。
- 结论：需要先完成 B 序列中前置 crate 的发布。

## E. 发布演练结果（2026-05-05）

按 B 序列执行 `cargo package -p <crate> --allow-dirty --no-verify`：

- ✅ 通过：`sptorch-core-tensor`、`sptorch-data`、`sptorch-versioning`
- ❌ 未通过（预期）：其余 crate 因前置依赖尚未在 registry 可解析（例如 `sptorch-core-tensor` / `sptorch-core-ops` / `sptorch-core-autograd` 未发布）。

这说明发布链路设计正确，后续需按顺序真实发布并等待索引同步后继续。

## G. 首批可发布 crate 最终检查（2026-05-05）

目标 crate：`sptorch-core-tensor`、`sptorch-data`、`sptorch-versioning`

- 已补齐 metadata：`description` / `license` / `repository` / `homepage` / `documentation` / `readme` / `keywords` / `categories`
- 已新增 crate 级 `README.md`
- `cargo package --list` 检查通过（包内容包含 README 与源码）
- `cargo publish --dry-run --registry crates-io` 检查通过

注意：

- 当前环境默认 registry 被替换为 `ustc`，真实发布或 dry-run 需显式带 `--registry crates-io`。
- Naming-conflict risk is closed: public package names use the `sptorch-*` namespace instead of short generic names.

## F. 演练脚本

- Bash: `scripts/publish-rehearsal.sh`
- PowerShell: `scripts/publish-rehearsal.ps1`

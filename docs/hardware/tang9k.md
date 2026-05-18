# SPTorch Tang9k Wiki

这页是 SPTorch 仓内自带的 Tang Nano 9K / Tang9k 硬件 wiki。它的目标不是替代 Sipeed 官方资料，而是把“这块板在 SPTorch 里承担什么角色、怎样烧录、怎样跑通 host 探测、怎样判断链路是否可信”沉淀成一页可执行的工程手册。

外部板卡资料仍以 Sipeed 官方 Tang Nano 9K Wiki 为准：<https://wiki.sipeed.com/hardware/zh/tang/Tang-Nano-9K/Nano-9K.html>。本页只记录 SPTorch 项目内已经验证或正在推进的约定。

- 硬件文档总入口：[SPTorch Hardware Wiki](README.md)
- 线协议标准：[Tang9k Serial Protocol v1](../tang9k-serial-protocol-v1.md)
- responder 工程页：[Tang9k UART Responder README](../../hardware/tang9k/uart_responder/README.md)
- 新板卡页面模板：[Hardware Board Wiki Template](board-template.md)

## 页面导航

- [当前定位](#当前定位)
- [板卡速览](#板卡速览)
- [当前验收状态](#当前验收状态)
- [连接与数据流](#连接与数据流)
- [仓库地图](#仓库地图)
- [命令速查](#命令速查)
- [工具准备](#工具准备)
- [构建 bitstream](#构建-bitstream)
- [SRAM 烧录](#sram-烧录)
- [Host 验收阶梯](#host-验收阶梯)
- [当前真实板卡记录](#当前真实板卡记录)
- [LED 观测](#led-观测)
- [故障排查](#故障排查)
- [协议演进规范](#协议演进规范)
- [后续路线](#后续路线)

## 当前定位

Tang9k 是 SPTorch HAL 的第一块真实 FPGA 控制面验证板。它现在不承担完整训练加速，不承诺矩阵结果已经来自真实 PE 阵列；当前阶段先把下面几件事练扎实：

- Host 能发现正确串口，而不是误连 Windows 内置 `COM1`。
- Host 和 FPGA 对 serial v1 帧头、payload、padding、checksum、sequence 的理解完全一致。
- FPGA 能返回 `DeviceInfo`，让 host 知道当前烧进去的是哪版 responder。
- FPGA 能完成 `Ping -> Pong`、`Matmul32x32 -> Ack/Ok`、scratch write/read、result-window 读回、越界读拒绝。
- 所有真实板卡结果都必须能回溯到 Gowin SRAM 烧录日志、COM 口、波特率和 raw bytes。

这条线后续会自然长成 DMA、片上 PE 阵列、多板互联和分布式 dry-run 的硬件入口。现在我们刻意保持 responder 很小，是为了先把“不说谎的硬件闭环”做稳。

## 板卡速览

| 项目 | SPTorch 当前记录 |
| --- | --- |
| 板卡 | Sipeed Tang Nano 9K |
| FPGA | Gowin `GW1NR-9C` / 官方资料中的 `GW1NR-LV9QN88PC6/I5` 系列 |
| 逻辑资源 | 约 `8640 LUT4`，适合做控制面、窄数据面、早期 PE 阵列实验 |
| 板载时钟 | `27 MHz`，当前约束写在 `hardware/tang9k/uart_responder/src/tang9k_uart_responder.sdc` |
| 调试器 | 板载 BL702，提供 USB-JTAG 下载和 USB-UART 串口 |
| 下载线 | Gowin Programmer 中选择 `USB Debugger A` |
| 当前 UART | 本机实测枚举为 `COM3` |
| 串口参数 | `115200 8N1` |
| 当前 FPGA 约束 | `uart_tx=17`、`uart_rx=18`，LED 低有效 |

官方 wiki 还列出了 HDMI、RGB/SPI 屏幕接口、SPI Flash、PSRAM、用户按键、LED、TF 卡座和扩展排针等板级资源。SPTorch 当前只依赖 USB-JTAG、USB-UART、27 MHz 时钟和 LED 观测；其他资源先不纳入核心假设，避免硬件 bring-up 阶段过早扩张。

## 当前验收状态

| 能力 | Host / 测试 | RTL | COM3 真板 | 说明 |
| --- | --- | --- | --- | --- |
| `Ping -> Pong` | 已接入 | 已接入 | 真板已通过 | 最小 UART 闭环 |
| `Matmul32x32 -> Ack/Ok` | 已接入 | 已接入 | 真板已通过 | 当前仍是控制面 smoke，不是真实矩阵计算 |
| `ScratchWrite32/ScratchRead32` | 已接入 | 已接入 | 真板已通过 | 第一条非 ACK 数据面回环 |
| `ResultRead32/ResultValue32` | 已接入 | 已接入 | 真板已通过 | 读取确定性摘要窗口 |
| `ResultWindowStatus` | 已接入 | 已接入 | 真板已通过 | 暴露 valid/base/stride/last-sequence |
| OOB 拒绝 | 已接入 | 已接入 | 真板已通过 | 越界读返回 `HardwareFault` |
| `DeviceInfo` | 已接入 | 已接入 | 待复测 | 最新 RTL 还需重新 build、烧录、COM3 验收 |

状态词遵循硬件 wiki 总入口的验收等级：`已接入` 不等于 `真板已通过`，`待复测` 表示最近实现已经变化，必须重新 build/烧录/COM3 跑通后才能升级。

## 连接与数据流

```text
Windows host
  |
  |  USB
  v
Tang Nano 9K 板载 BL702
  | \
  |  \__ USB-JTAG  -> Gowin Programmer / USB Debugger A
  |
  \_____ USB-UART  -> COM3 @ 115200 8N1
                  -> SPTorch tang9k_probe
                  -> serial v1 frames
                  -> FPGA UART responder RTL
```

从软件栈角度看，当前闭环是：

```text
tang9k_probe
  -> UartTang9kTransport
  -> SerialFrame / SerialSubmitQueue
  -> USB-UART / COM3
  -> tang9k_uart_responder.v
  -> DeviceInfo / Pong / Ack / ScratchValue32 / ResultValue32
```

这张图里最值得守住的边界，是 `SerialFrame`。只要 host 和 FPGA 对这一层字节语义保持一致，后面的 UART、DMA 甚至多板 transport 才能逐步替换，而不会把每层都重写一遍。

## 仓库地图

| 路径 | 作用 |
| --- | --- |
| `hardware/tang9k/uart_responder/` | 最小 Gowin 工程，生成 Tang9k UART responder bitstream |
| `hardware/tang9k/uart_responder/src/tang9k_uart_responder.v` | 当前 FPGA 端协议 responder RTL |
| `hardware/tang9k/uart_responder/src/tang9k_uart_responder.cst` | Tang Nano 9K 管脚约束 |
| `hardware/tang9k/uart_responder/src/tang9k_uart_responder.sdc` | 27 MHz 时钟与时序约束 |
| `docs/tang9k-serial-protocol-v1.md` | serial v1 字节级线协议标准 |
| `crates/hal/src/serial.rs` | host 侧帧结构、opcode、checksum、golden vector 语义 |
| `crates/hal-ffi/src/serial_backend.rs` | UART transport、dry-run backend、探测辅助逻辑 |
| `crates/hal-ffi/src/bin/tang9k_probe.rs` | 本机 bring-up CLI |
| `crates/hal/tests/tang9k_serial_golden.rs` | 协议字节级黄金样例 |

原则很简单：RTL、host parser、golden vector、协议文档必须一起前进。只改一边很容易制造“看起来能跑，实际已经漂移”的硬件幽灵。

## 命令速查

| 目标 | 命令 |
| --- | --- |
| 只列出串口 | `cargo run -p sptorch-hal-ffi --bin tang9k_probe -- --list` |
| 查询 responder 身份 | `cargo run -p sptorch-hal-ffi --bin tang9k_probe -- --port COM3 --device-info --baud 115200 --timeout-ms 1000 --dump-raw` |
| 最小 UART 闭环 | `cargo run -p sptorch-hal-ffi --bin tang9k_probe -- --port COM3 --baud 115200 --timeout-ms 1000` |
| 命令生命周期 smoke | `cargo run -p sptorch-hal-ffi --bin tang9k_probe -- --port COM3 --matmul-smoke --baud 115200 --timeout-ms 1000 --dump-raw` |
| scratch 数据面 | `cargo run -p sptorch-hal-ffi --bin tang9k_probe -- --port COM3 --scratch-smoke --baud 115200 --timeout-ms 1000 --dump-raw` |
| 结果窗口状态 | `cargo run -p sptorch-hal-ffi --bin tang9k_probe -- --port COM3 --result-window-status-smoke --baud 115200 --timeout-ms 1000 --dump-raw` |
| 结果窗口 4-word 读回 | `cargo run -p sptorch-hal-ffi --bin tang9k_probe -- --port COM3 --result-window-smoke --baud 115200 --timeout-ms 1000 --dump-raw` |
| 结果窗口越界拒绝 | `cargo run -p sptorch-hal-ffi --bin tang9k_probe -- --port COM3 --result-oob-smoke --baud 115200 --timeout-ms 1000 --dump-raw` |
| 完整 bring-up suite | `cargo run -p sptorch-hal-ffi --bin tang9k_probe -- --port COM3 --bringup-suite --baud 115200 --timeout-ms 1000 --dump-raw` |

给任意 probe 追加 `--record-json <path>`，可以生成机器可读的验收记录。这个文件适合直接附到 issue、wiki 更新或后续 Studio 硬件面板中：

```powershell
cargo run -p sptorch-hal-ffi --bin tang9k_probe -- --port COM3 --bringup-suite --baud 115200 --timeout-ms 1000 --dump-raw --record-json target\tang9k\bringup-suite.json
```

JSON 记录包含 schema、时间戳、命令、端口、波特率、timeout、每条 request/response raw bytes、opcode、sequence 和已知 payload 的解码字段。失败时也会写入同一 schema，并保留错误消息以及已收集到的 raw bytes。

## 工具准备

Windows 当前推荐路径：

```powershell
C:\Gowin\Gowin_V1.9.12.02_SP2_x64
```

需要确认：

- Gowin IDE 的 `gw_sh.exe` 可运行。
- Gowin Programmer 的 `programmer_cli.exe` 可运行。
- License 正常。最近一次命令行构建曾遇到 `License verification failed  Server not responding.`，这属于 Gowin 授权或网络问题，不是 RTL 编译错误。
- Rust workspace 可构建，至少能运行 `cargo run -p sptorch-hal-ffi --bin tang9k_probe -- --list`。
- Tang Nano 9K 已连接电脑，串口列表中能看到 `COM3`。

先列串口，不写板卡：

```powershell
cargo run -p sptorch-hal-ffi --bin tang9k_probe -- --list
```

如果只看到 `COM1 Unknown`，先不要把它当 Tang9k。很多 Windows 机器会有内置 ACPI 串口占着 `COM1`，真正的 USB-UART 往往是另一个端口；当前这台机器上是 `COM3`。

## 构建 bitstream

在仓库根目录执行：

```powershell
& 'C:\Gowin\Gowin_V1.9.12.02_SP2_x64\IDE\bin\gw_sh.exe' hardware\tang9k\uart_responder\build.tcl
```

成功后应生成：

```text
hardware/tang9k/uart_responder/impl/pnr/tang9k_uart_responder.fs
```

如果这里报 license，不要继续做串口验收，因为板上大概率还是旧 bitstream。先修 Gowin 授权，再重新 build。

## SRAM 烧录

bring-up 阶段优先烧 SRAM，不写 Flash。这样 pinout、UART 方向或协议有问题时，断电即可恢复，不会把实验固件长期留在板子上。

```powershell
$fs = (Resolve-Path hardware\tang9k\uart_responder\impl\pnr\tang9k_uart_responder.fs).Path
& 'C:\Gowin\Gowin_V1.9.12.02_SP2_x64\Programmer\bin\programmer_cli.exe' `
  --device GW1NR-9C `
  --operation_index 2 `
  --fsFile $fs `
  --cable "USB Debugger A" `
  --frequency 2.5MHz
```

`programmer_cli` 对相对路径比较挑，`--fsFile` 建议传 `Resolve-Path` 得到的绝对路径。

已记录过一次成功 SRAM Program：

```text
USB Debugger A -> GW1NR-9C, Status Code 0x0003F020
```

这只证明当时的 bitstream 已经写入 SRAM，不自动证明当前工作区最新 RTL 已经重新烧进板子。

## Host 验收阶梯

建议每次重新烧录后按这个顺序跑。顺序很重要：它让问题尽早停在最小边界上。

### 1. DeviceInfo

```powershell
cargo run -p sptorch-hal-ffi --bin tang9k_probe -- --port COM3 --device-info --baud 115200 --timeout-ms 1000 --dump-raw
```

期望字段：

```text
protocol=1
kind=1
responder_version=1
capabilities=0x0000000f
clk_hz=27000000
baud=115200
result_words=4
result_stride=4
build_id=0x20260518
```

`DeviceInfo` 已接入 host、协议 golden vector 和 RTL responder，但它还需要下一次 Gowin 重新 build、SRAM 烧录、`COM3` 实测后，才能被标记为“当前最新 RTL 的真实板卡已验证结果”。这类状态必须诚实记录，不能用旧板卡日志冒充新 RTL 验收。

### 2. Ping

```powershell
cargo run -p sptorch-hal-ffi --bin tang9k_probe -- --port COM3 --baud 115200 --timeout-ms 1000
```

期望：

```text
OK: response opcode=Pong, sequence=0, payload_len=0
```

Ping 只证明最小收发链路和 checksum 闭环，不证明命令状态机或数据面。

### 3. MatMul 命令生命周期

```powershell
cargo run -p sptorch-hal-ffi --bin tang9k_probe -- --port COM3 --matmul-smoke --baud 115200 --timeout-ms 1000 --dump-raw
```

期望：

```text
OK: response opcode=Ack, sequence=1, payload_len=8
```

这里的 `Matmul32x32` 仍然是控制帧 smoke test。当前 responder 收到命令后更新确定性 result window，并不表示已经完成真实矩阵乘法。

### 4. Scratch 数据面

```powershell
cargo run -p sptorch-hal-ffi --bin tang9k_probe -- --port COM3 --scratch-smoke --baud 115200 --timeout-ms 1000 --dump-raw
```

期望：

```text
OK: scratch write opcode=Ack, sequence=2, payload_len=8
OK: scratch read opcode=ScratchValue32, sequence=3, offset=0x00000044, value=0x11223344
```

Scratch 证明 FPGA 能保存一个 32-bit 值并读回，是后续 DMA/result buffer 的最小前置能力。

### 5. Result Window

```powershell
cargo run -p sptorch-hal-ffi --bin tang9k_probe -- --port COM3 --result-window-status-smoke --baud 115200 --timeout-ms 1000 --dump-raw
cargo run -p sptorch-hal-ffi --bin tang9k_probe -- --port COM3 --result-window-smoke --baud 115200 --timeout-ms 1000 --dump-raw
cargo run -p sptorch-hal-ffi --bin tang9k_probe -- --port COM3 --result-oob-smoke --baud 115200 --timeout-ms 1000 --dump-raw
```

期望核心结果：

```text
ResultWindowStatus valid=true, words=4, stride=4, base=0x00002000, last_sequence=4
[0x00003005, 0x9e3749bc, 0x3c6ec377, 0xdaa65d2e]
Error HardwareFault detail=0x00002010
```

这三步合起来证明：窗口被命令更新、host 能连续读 4 个 word、越界 offset 不会被伪装成合法结果。

### 6. Bring-up Suite

```powershell
cargo run -p sptorch-hal-ffi --bin tang9k_probe -- --port COM3 --bringup-suite --baud 115200 --timeout-ms 1000 --dump-raw
```

`--bringup-suite` 会顺序跑当前验收集：`DeviceInfoRead`、`Ping`、`Matmul32x32`、scratch write/read、result-window status、4-word result-window 读回和 OOB 拒绝。它适合作为每次烧录后的最终冒烟检查。

## 当前真实板卡记录

已实测通过：

```text
date: 2026-05-18
repo_commit: 未记录；这些结果来自 DeviceInfo 接入前的 responder 实测链路
host_os: Windows
toolchain: Gowin V1.9.12.02 SP2
device: Tang Nano 9K / GW1NR-9C
transport: USB Debugger A + COM3 @ 115200 8N1
bitstream_or_firmware: hardware/tang9k/uart_responder/impl/pnr/tang9k_uart_responder.fs
Gowin SRAM Program: USB Debugger A -> GW1NR-9C, Status Code 0x0003F020
Host probe: COM3 @ 115200 -> OK: response opcode=Pong, sequence=0, payload_len=0
Host command lifecycle: COM3 @ 115200 -> OK: response opcode=Ack, sequence=1, payload_len=8
Host scratch data-plane: COM3 @ 115200 -> OK: ScratchValue32 offset=0x00000044, value=0x11223344
Host result window: COM3 @ 115200 -> OK: ResultValue32 offset=0x00002000, value=0x00003005
Host result window 4-word smoke: COM3 @ 115200 -> [0x00003005, 0x9e3749bc, 0x3c6ec377, 0xdaa65d2e]
Host result window status: COM3 @ 115200 -> OK: ResultWindowStatus valid=true, words=4, stride=4, base=0x00002000, last_sequence=4
Host result window OOB: COM3 @ 115200 -> Error HardwareFault detail=0x00002010
Host bring-up suite: COM3 @ 115200 -> OK: bringup suite completed sequentially
```

待重新实测：

```text
DeviceInfo expected response: protocol=1, kind=1, responder_version=1, capabilities=0x0000000f, clk_hz=27000000, baud=115200, result_words=4, result_stride=4, build_id=0x20260518
```

原因：`DeviceInfo` 是最近接入的 host/protocol/RTL 能力，最新 RTL 还卡在 Gowin license 构建问题上，尚未完成新 bitstream 的 COM3 实板验收。

## LED 观测

`led` 为低有效输出：

| LED | 语义 |
| --- | --- |
| `led[0]` | 心跳 |
| `led[1]` | 收到 UART 字节后翻转 |
| `led[2]` | 发送 UART 字节后翻转 |
| `led[3]` | 结构性协议错误 |
| `led[4]` | 合法命令帧被接受并排队响应 |
| `led[5]` | checksum 错误 |

如果 host 超时但 `led[1]` 在动，说明 FPGA 可能收到了字节但没有形成合法帧；优先看 checksum、payload 长度、padding 和 opcode。若 `led[1]` 完全不动，优先检查串口号、烧录状态、TX/RX 方向和约束。

## 故障排查

| 现象 | 优先判断 | 处理方式 |
| --- | --- | --- |
| `License verification failed  Server not responding.` | Gowin 授权或网络不可用 | 先修 license，再重新执行 `gw_sh.exe build.tcl` |
| 只看到 `COM1 Unknown` | 可能是系统内置串口 | 不要直接 probe；先用 `--list` 找 USB-UART，当前机器使用 `COM3` |
| `DeviceInfo` 超时 | bitstream 未更新、端口错误或 UART 方向不对 | 重新 SRAM 烧录，确认 `COM3`，必要时检查 `uart_tx` / `uart_rx` 约束 |
| `Ping` 能过，`MatmulSmoke` checksum 错 | 可能是旧 RTL 的 checksum 准备逻辑 | 使用当前 RTL 重新 build + SRAM 烧录 |
| result window 读出旧值 | 可能没有先发送 MatMul 命令，或窗口 valid 状态过旧 | 先跑 `--result-window-status-smoke`，再跑 `--result-window-smoke` |
| OOB 没有返回 `HardwareFault` | 窗口边界判断漂移 | 停止向上推进，先修 RTL 和 golden vector |
| `programmer_cli` 找不到 `.fs` | 相对路径解析失败 | 用 `Resolve-Path` 传绝对 `--fsFile` |

## 协议演进规范

每次改 serial v1 或 responder 行为，必须同时满足：

- `docs/tang9k-serial-protocol-v1.md` 更新 wire contract。
- `crates/hal/src/serial.rs` 更新 opcode、payload 或 checksum 语义。
- `crates/hal/tests/tang9k_serial_golden.rs` 增加或更新 byte-level golden vector。
- `crates/hal-ffi/src/bin/tang9k_probe.rs` 暴露可重复执行的 host 验收命令。
- `hardware/tang9k/uart_responder/src/tang9k_uart_responder.v` 与 host 语义一致。
- README 或本页更新真实板卡状态，明确哪些已经 COM3 实测，哪些只是 host/RTL/golden vector 已接入。

最重要的一条：没有重新 Gowin build、SRAM 烧录和 COM3 实测，就不要把某个新能力写成“真实板卡已验证”。这不是保守，这是硬件项目里最便宜的防幻觉手段。

## 后续路线

- 修复 Gowin license 后，重新 build 并烧录带 `DeviceInfo` 的最新 responder。
- 把 `--bringup-suite --dump-raw` 的最新输出补回本页和 `hardware/tang9k/uart_responder/README.md`。
- 在 result window 之后推进真实矩阵结果缓冲区，而不是只返回确定性摘要。
- 设计最小 DMA/streaming data-plane，让 host 能上传 tile 数据、触发计算、读回结果。
- 把单板协议稳定后，再接入多 Tang9k board 的拓扑验证、ring allreduce dry-run 和 matmul shard plan。

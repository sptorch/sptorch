# Tang9k UART Responder Bring-Up

这个工程是 SPTorch HAL 串行协议的第一块真实板卡烟测固件。它不进入训练路径，先把控制面练稳：通过 Tang Nano 9K / Tang9k 的 USB-UART 收到 SPTorch serial v1 `DeviceInfoRead` 后回传 responder 协议版本、能力位和 build id；收到 `Ping` 后回传 checksum 正确、sequence 相同的 `Pong`；收到最小 `Matmul32x32` 控制帧后回传 `Ack/Ok`；收到 `ScratchWrite32` 后保存一个 32-bit 槽位，并在 `ScratchRead32` 时用 `ScratchValue32` 读回；收到 MatMul 后还会写入一个确定性的 4-word 结果窗口，供 `ResultRead32 -> ResultValue32` 烟测读取，并通过 `ResultWindowStatusRead -> ResultWindowStatus` 暴露窗口 valid/base/stride/last-sequence 元信息。真正 PE 阵列、DMA 和完整矩阵结果缓冲区会在下一层接入。

## 目标板与链路

- FPGA：Gowin `GW1NR-9C` / Tang Nano 9K 常见板型。
- 下载器：Gowin Programmer 选择 `USB Debugger A`。
- UART：当前机器上枚举为 `COM3`。
- 串口参数：`115200 8N1`。
- 时钟假设：板载 `27 MHz`，约束在 `src/tang9k_uart_responder.sdc`。

## 构建

```powershell
& 'C:\Gowin\Gowin_V1.9.12.02_SP2_x64\IDE\bin\gw_sh.exe' hardware\tang9k\uart_responder\build.tcl
```

成功后通常会生成：

```text
hardware/tang9k/uart_responder/impl/pnr/tang9k_uart_responder.fs
```

## SRAM 烧录

先用 SRAM Program，避免在 pinout 或 UART 方向确认前写入 Flash：

```powershell
$fs = (Resolve-Path hardware\tang9k\uart_responder\impl\pnr\tang9k_uart_responder.fs).Path
& 'C:\Gowin\Gowin_V1.9.12.02_SP2_x64\Programmer\bin\programmer_cli.exe' `
  --device GW1NR-9C `
  --operation_index 2 `
  --fsFile $fs `
  --cable "USB Debugger A" `
  --frequency 2.5MHz
```

`programmer_cli` 对相对 `--fsFile` 路径比较挑剔，建议像上面这样传绝对路径。

## Host 验收

```powershell
cargo run -p sptorch-hal-ffi --bin tang9k_probe -- --port COM3 --baud 115200 --timeout-ms 1000
cargo run -p sptorch-hal-ffi --bin tang9k_probe -- --port COM3 --device-info --baud 115200 --timeout-ms 1000 --dump-raw
cargo run -p sptorch-hal-ffi --bin tang9k_probe -- --port COM3 --bringup-suite --baud 115200 --timeout-ms 1000
cargo run -p sptorch-hal-ffi --bin tang9k_probe -- --port COM3 --matmul-smoke --baud 115200 --timeout-ms 1000
cargo run -p sptorch-hal-ffi --bin tang9k_probe -- --port COM3 --matmul-smoke --baud 115200 --timeout-ms 1000 --dump-raw
cargo run -p sptorch-hal-ffi --bin tang9k_probe -- --port COM3 --result-smoke --baud 115200 --timeout-ms 1000 --dump-raw
cargo run -p sptorch-hal-ffi --bin tang9k_probe -- --port COM3 --result-window-smoke --baud 115200 --timeout-ms 1000 --dump-raw
cargo run -p sptorch-hal-ffi --bin tang9k_probe -- --port COM3 --result-window-status-smoke --baud 115200 --timeout-ms 1000 --dump-raw
cargo run -p sptorch-hal-ffi --bin tang9k_probe -- --port COM3 --result-oob-smoke --baud 115200 --timeout-ms 1000 --dump-raw
cargo run -p sptorch-hal-ffi --bin tang9k_probe -- --port COM3 --scratch-smoke --baud 115200 --timeout-ms 1000 --dump-raw
```

`Ping` 期望输出类似：

```text
OK: response opcode=Pong, sequence=0, payload_len=0
```

`DeviceInfo` 期望输出类似：

```text
OK: device info opcode=DeviceInfo, sequence=11, protocol=1, kind=1, responder_version=1, capabilities=0x0000000f, clk_hz=27000000, baud=115200, result_words=4, result_stride=4, build_id=0x20260518
```

`BringupSuite` 期望输出会把 `DeviceInfo`、`Ping`、`MatmulSmoke`、`ScratchSmoke`、`ResultWindowStatusSmoke`、`ResultWindowSmoke` 和 `ResultOobSmoke` 顺序串起来，每一步都保留自己的 `OK:` 行和 raw bytes。这个命令适合每次重新烧录后第一时间跑一遍，确认板卡已经回到可用状态。

新版 `BringupSuite` 期望总结行：

```text
OK: bringup suite completed sequentially
OK: suite device info opcode=DeviceInfo, sequence=11, protocol=1, kind=1, responder_version=1, capabilities=0x0000000f, clk_hz=27000000, baud=115200, result_words=4, result_stride=4, build_id=0x20260518
OK: suite status opcode=ResultWindowStatus, sequence=10, valid=true, words=4, stride=4, base=0x00002000, last_sequence=4
OK: suite oob rejected read opcode=Error, sequence=9, status=HardwareFault, detail=0x00002010
```

说明：`DeviceInfo` 已经接入 host、协议 golden vector 和 RTL responder，但还需要重新 Gowin build、SRAM 烧录、COM3 实测后，才能把上面的 device-info 行标记为当前实测结果。

`MatmulSmoke` 期望输出类似：

```text
OK: response opcode=Ack, sequence=1, payload_len=8
```

`ScratchSmoke` 期望输出类似：

```text
OK: scratch write opcode=Ack, sequence=2, payload_len=8
OK: scratch read opcode=ScratchValue32, sequence=3, offset=0x00000044, value=0x11223344
```

`ResultSmoke` 期望输出类似：

```text
OK: result matmul opcode=Ack, sequence=4, payload_len=8
OK: result read opcode=ResultValue32, sequence=5, offset=0x00002000, value=0x00003005
```

`ResultWindowSmoke` 期望输出类似：

```text
OK: result window matmul opcode=Ack, sequence=4, payload_len=8
OK: result window read 0 opcode=ResultValue32, sequence=5, offset=0x00002000, value=0x00003005
OK: result window read 1 opcode=ResultValue32, sequence=6, offset=0x00002004, value=0x9e3749bc
OK: result window read 2 opcode=ResultValue32, sequence=7, offset=0x00002008, value=0x3c6ec377
OK: result window read 3 opcode=ResultValue32, sequence=8, offset=0x0000200c, value=0xdaa65d2e
```

`ResultWindowStatusSmoke` 期望输出类似：

```text
OK: result window status matmul opcode=Ack, sequence=4, payload_len=8
OK: result window status opcode=ResultWindowStatus, sequence=10, valid=true, words=4, stride=4, base=0x00002000, last_sequence=4
```

`ResultOobSmoke` 期望输出类似：

```text
OK: result oob rejected read opcode=Error, sequence=9, status=HardwareFault, detail=0x00002010
```

当前实测的 `MatmulSmoke` ACK raw response：

```text
53 50 01 7e 01 00 00 00 08 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 b8 59 20 24 00 00 00 00
```

`DeviceInfo` raw response 应该稳定为：

```text
53 50 01 04 0b 00 00 00 18 00 00 00 00 00 00 00 01 01 01 00 0f 00 00 00 c0 fc 9b 01 00 c2 01 00 04 04 00 00 18 05 26 20 f2 2b 98 83 00 00 00 00
```

当前实测的 `ResultSmoke` result raw response：

```text
53 50 01 31 05 00 00 00 08 00 00 00 00 00 00 00 00 20 00 00 05 30 00 00 d4 d0 6f 06 00 00 00 00
```

当前实测的 `ResultWindowStatusSmoke` status raw response：

```text
53 50 01 33 0a 00 00 00 10 00 00 00 00 00 00 00 01 04 04 00 00 20 00 00 04 00 00 00 00 00 00 00 75 00 79 ea 00 00 00 00
```

`ScratchSmoke` 是第一条非 ACK 数据面回环：它不证明 DDR/DMA 已经可用，只证明 target 能按协议保存一个 32-bit 值并作为数据帧读回。`ResultSmoke` 在这条边界上再推进一步：MatMul 控制帧会更新板端结果窗口，host 随后用 `ResultRead32` 读回摘要值。这个摘要还不是矩阵结果，只是证明“命令触发结果窗口更新”这条闭环已经存在。

`ResultWindowSmoke` 把单槽摘要扩展到 4 个连续 word。它依旧不是矩阵结果，但已经开始验证 result buffer 的核心行为：同一条 MatMul 命令写入多个可区分 word，host 按 offset 连续读回，RTL 必须做正确地址选择。`ResultWindowStatusSmoke` 再补一层窗口问诊：host 不读数据也能知道窗口是否有效、窗口有几个 word、步长是多少、base offset 是多少、最后一次写入来自哪个 MatMul sequence。`ResultOobSmoke` 则验证窗口边界：第一个越界 offset 必须返回 `Error/HardwareFault`，不能把默认值或旧值伪装成合法结果。

如果仍然超时，优先检查 `uart_tx` / `uart_rx` 约束是否需要对调。当前约束使用常见 Tang Nano 9K USB-UART 引脚：FPGA `uart_tx=17`、`uart_rx=18`。

## RTL 约束

响应帧 checksum 必须逐字节、逐拍计算。早期版本把 ACK 的 24 次 FNV-1a feed 串成单周期组合链，Pong 偶然通过，但 `Matmul32x32 -> Ack/Ok` 在真实 `GW1NR-9C` 上会出现 checksum drift。当前实现先用 `tx_prepare_active` 在 16/24 个时钟周期内生成 checksum，再进入 UART TX；这点准备延迟远小于串口发送时间，但能保证控制面稳定。

## LED 观测

`led` 为低有效输出：

- `led[0]`：心跳。
- `led[1]`：收到 UART 字节后翻转。
- `led[2]`：发送 UART 字节后翻转。
- `led[3]`：结构性协议错误。
- `led[4]`：合法命令帧被接受并排队响应。
- `led[5]`：checksum 错误。

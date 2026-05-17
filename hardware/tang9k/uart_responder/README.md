# Tang9k UART Responder Bring-Up

这个工程是 SPTorch HAL 串行协议的第一块真实板卡烟测固件。它不进入训练路径，先把控制面练稳：通过 Tang Nano 9K / Tang9k 的 USB-UART 收到 SPTorch serial v1 `Ping` 后回传 checksum 正确、sequence 相同的 `Pong`；收到最小 `Matmul32x32` 控制帧后回传 `Ack/Ok`；收到 `ScratchWrite32` 后保存一个 32-bit 槽位，并在 `ScratchRead32` 时用 `ScratchValue32` 读回。真正的 PE 阵列、DMA 和矩阵结果回读会在下一层接入。

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
cargo run -p sptorch-hal-ffi --bin tang9k_probe -- --port COM3 --matmul-smoke --baud 115200 --timeout-ms 1000
cargo run -p sptorch-hal-ffi --bin tang9k_probe -- --port COM3 --matmul-smoke --baud 115200 --timeout-ms 1000 --dump-raw
cargo run -p sptorch-hal-ffi --bin tang9k_probe -- --port COM3 --scratch-smoke --baud 115200 --timeout-ms 1000 --dump-raw
```

`Ping` 期望输出类似：

```text
OK: response opcode=Pong, sequence=0, payload_len=0
```

`MatmulSmoke` 期望输出类似：

```text
OK: response opcode=Ack, sequence=1, payload_len=8
```

`ScratchSmoke` 期望输出类似：

```text
OK: scratch write opcode=Ack, sequence=2, payload_len=8
OK: scratch read opcode=ScratchValue32, sequence=3, offset=0x00000044, value=0x11223344
```

当前实测的 `MatmulSmoke` ACK raw response：

```text
53 50 01 7e 01 00 00 00 08 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 b8 59 20 24 00 00 00 00
```

`ScratchSmoke` 是第一条非 ACK 数据面回环：它不证明 DDR/DMA 已经可用，只证明 target 能按协议保存一个 32-bit 值并作为数据帧读回。后续 MatMul 结果回读会沿着这条边界继续扩展。

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

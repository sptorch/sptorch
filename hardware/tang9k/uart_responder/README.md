# Tang9k UART Responder Bring-Up

这个工程是 SPTorch HAL 串行协议的第一块真实板卡烟测固件。它不实现 MatMul，也不进入训练路径，只做一件事：通过 Tang Nano 9K / Tang9k 的 USB-UART 收到 SPTorch serial v1 `Ping` 后，回传 checksum 正确、sequence 相同的 `Pong`。

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
```

期望输出类似：

```text
OK: response opcode=Pong, sequence=0, payload_len=0
```

如果仍然超时，优先检查 `uart_tx` / `uart_rx` 约束是否需要对调。当前约束使用常见 Tang Nano 9K USB-UART 引脚：FPGA `uart_tx=17`、`uart_rx=18`。

## LED 观测

`led` 为低有效输出：

- `led[0]`：心跳。
- `led[1]`：收到 UART 字节后翻转。
- `led[2]`：发送 UART 字节后翻转。
- `led[3]`：结构性协议错误。
- `led[4]`：合法 Ping 被接受。
- `led[5]`：checksum 错误。

# Hardware Board Wiki Template

复制这份模板时，把 `BOARD_NAME`、`BOARD_SHORT_NAME`、`VENDOR`、`HOST_PORT` 等占位符替换为真实信息。不要为了“看起来完整”提前填未验证的内容；硬件文档最重要的价值，是让后来的人知道哪些已经踩实，哪些还只是计划。

## BOARD_NAME Wiki

一句话说明这块板在 SPTorch 里的角色。例如：它是 FPGA 控制面验证板、外部 NPU FFI 后端、CUDA 运行时对照平台，还是多板互联拓扑节点。

- 硬件文档总入口：[SPTorch Hardware Wiki](README.md)
- 相关协议文档：`TODO`
- 最小工程路径：`TODO`

## 当前定位

- 这块板当前解决什么问题。
- 它暂时不解决什么问题。
- 它和 `hal`、`hal-ffi`、`distributed`、`runtime-*` 的关系是什么。

## 板卡速览

| 项目 | 记录 |
| --- | --- |
| 板卡 | `BOARD_NAME` |
| 芯片 | `TODO` |
| 时钟 | `TODO` |
| 调试/下载 | `TODO` |
| Host 连接 | `HOST_PORT` |
| 当前最小依赖 | `TODO` |
| 不纳入当前假设的资源 | `TODO` |

## 当前验收状态

| 能力 | Host / 测试 | 设备端实现 | 真板实测 | 证据 |
| --- | --- | --- | --- | --- |
| 最小连通 | 未开始 | 未开始 | 未开始 | `TODO` |

状态词只使用硬件 wiki 约定：

- `未开始`
- `已设计`
- `已接入`
- `dry-run 通过`
- `真板已通过`
- `待复测`
- `阻塞`

## 连接与数据流

```text
Host
  |
  v
BOARD_SHORT_NAME transport
  |
  v
Device firmware / RTL / runtime
```

这里要写清楚 host 到设备的真实路径：USB-UART、USB-JTAG、PCIe、Ethernet、SPI、DMA 或 vendor runtime。不要只写“连接板子”。

## 仓库地图

| 路径 | 作用 |
| --- | --- |
| `TODO` | `TODO` |

## 命令速查

| 目标 | 命令 |
| --- | --- |
| 只读枚举 | `TODO` |
| 最小闭环 | `TODO` |
| 完整验收 | `TODO` |

## 构建与烧录

记录完整、可复制的命令。若依赖 GUI，也要写出 GUI 中的关键选项和可导出的命令行形式。

```powershell
TODO
```

## Host 验收阶梯

从最小、低风险、只读命令开始，再进入写命令、数据面、压力测试。

1. 枚举设备。
2. 查询设备身份。
3. 最小 ping/echo。
4. 命令生命周期。
5. 数据面读写。
6. 错误路径。
7. 完整 suite。

## 当前真实板卡记录

每条记录至少包含：

- 日期。
- commit。
- 工具版本。
- 设备型号。
- host 端口。
- 命令。
- 结果摘要。
- raw bytes 或可复现日志路径。

```text
YYYY-MM-DD
commit: TODO
toolchain: TODO
device: TODO
host: TODO
command: TODO
result: TODO
raw: TODO
```

## LED / 指示灯 / 观测点

| 观测点 | 语义 |
| --- | --- |
| `TODO` | `TODO` |

## 故障排查

| 现象 | 优先判断 | 处理方式 |
| --- | --- | --- |
| `TODO` | `TODO` | `TODO` |

## 协议或接口演进规范

- 改 host parser 时，必须同步测试。
- 改设备端协议时，必须同步文档。
- 改 wire format 时，必须同步 golden vector。
- 新增真板能力时，必须补真实验收记录。

## 后续路线

- `TODO`

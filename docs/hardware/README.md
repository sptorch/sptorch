# SPTorch Hardware Wiki

这里是 SPTorch 框架仓的硬件知识入口。框架可以同时面向 CPU、CUDA、外部 NPU、FPGA 和未来的多板互联，但每一种硬件都必须先经过同一套纪律：边界清楚、协议可复现、实测状态诚实、文档和代码同步演进。

## 维护原则

- **先把控制面练稳，再扩数据面。** 在 checksum、sequence、错误码都不稳定时，过早堆 DMA 或算子只会放大排障成本。
- **区分“已接入”和“已实测”。** host、RTL、golden vector 都写好了，不等于最新 bitstream 已经在真板上跑过。
- **一页一块板。** 每块板都要有自己的资源概览、接线、烧录、验收、故障排查和演进边界。
- **协议改动要成套落地。** 线协议、host parser、测试向量、CLI、RTL 和文档必须一起改。
- **把 dry-run 当成真硬件前置能力。** 多板拓扑、ring allreduce、matmul shard 先在 `hal` / `distributed` 中稳定，真链路接入时才不会从零开始。

## 当前板卡目录

| 板卡 | 角色 | 当前状态 | 页面 |
| --- | --- | --- | --- |
| Tang Nano 9K / Tang9k | FPGA 控制面、serial v1、后续多板互联入口 | `Ping`、命令生命周期、scratch、result window `真板已通过`；`DeviceInfo` `已接入` 但待最新 bitstream 复测 | [Tang9k Wiki](tang9k.md) |

## 当前硬件主线

| 层级 | 现在在做什么 | 通过标准 |
| --- | --- | --- |
| Host 协议层 | 固定帧头、checksum、stream decoder、submit queue | golden vector 与 host tests 稳定 |
| 单板控制层 | `DeviceInfo`、`Ping`、命令 ACK、scratch、result window | 每次重新烧录后能跑完整 bring-up suite |
| 单板数据层 | 真实矩阵结果缓冲区、后续 DMA/streaming | 不再返回摘要，而是能读回真实计算结果 |
| 多板抽象层 | topology、ring allreduce dry-run、matmul shard plan | 断链、单向链路、成本估算都可复现 |
| 多板实链路 | 多 Tang9k 板串联/并行验证 | 真链路握手、吞吐、错误恢复都能被记录和回归 |

## 验收等级

硬件页统一使用下面这些状态词，避免“已经好了”“差不多通了”这类不好追溯的描述：

| 状态 | 含义 | 可以怎么写 |
| --- | --- | --- |
| `未开始` | 只有想法，没有设计或代码 | 不列入已完成能力 |
| `已设计` | 文档、接口或计划已经明确，但代码还没接上 | 可以写路线，不写验收结果 |
| `已接入` | host、RTL/firmware/runtime 或测试已写好，但未完成真链路验证 | 必须说明还缺哪一步 |
| `dry-run 通过` | mock、loopback、golden vector 或仿真通过 | 不能冒充真板 |
| `真板已通过` | 最新相关实现已经在真实设备上跑过，并有命令/日志证据 | 可以进入当前板卡记录 |
| `待复测` | 旧能力曾通过，但最近改过相关实现，或当前 bitstream 不是最新 | 必须重新 build/烧录/运行 |
| `阻塞` | 被 license、工具链、硬件连接、供电或缺件卡住 | 写清阻塞点和下一步 |

## 验收记录格式

真实板卡记录至少包含下面信息。缺少 commit、工具版本或 raw bytes 时，也要明确写“未记录”，不要用模糊语言补过去。

```text
date: YYYY-MM-DD
repo_commit: <git sha>
host_os: Windows / Linux / macOS
toolchain: <vendor tool + version>
device: <board + chip>
transport: <COMx / PCIe / Ethernet / USB / vendor runtime>
bitstream_or_firmware: <path or build id>
command: <exact command>
result: <one-line summary>
raw_or_log: <raw bytes, log path, or artifact path>
notes: <known caveats>
```

这个格式有一点“笨”，但硬件 bring-up 正需要这种笨拙的诚实。几周后回头排查时，救命的往往不是一句“当时能跑”，而是那一串看起来不起眼的 raw bytes。

## 文档地图

| 文档 | 适合什么时候看 |
| --- | --- |
| [Tang9k Wiki](tang9k.md) | 你要接板、烧录、跑 COM3、判断当前板卡状态 |
| [Tang9k Serial Protocol v1](../tang9k-serial-protocol-v1.md) | 你要改 opcode、payload、checksum 或做协议实现 |
| [Tang9k UART Responder README](../../hardware/tang9k/uart_responder/README.md) | 你只想看 responder 工程自己的构建与验收细节 |
| [Board Wiki Template](board-template.md) | 你要接入新板卡，或者给第二块 Tang9k 建独立记录 |
| [Roadmap](../roadmap.md) | 你想看 Tang9k 在框架长期路线里的位置 |

## 新板卡接入清单

一块新硬件要进入 SPTorch 主线，至少应具备：

1. 明确的板卡页，记录型号、资源、连接方式、时钟、供电和最小依赖。
2. 一个可重复构建的最小工程，能证明主机到设备的最短闭环。
3. 一套 host CLI 验收命令，不能只靠 GUI 或肉眼观察。
4. 一组 byte-level 或 equivalent golden tests，锁住协议边界。
5. 一个“已实测 / 未实测”状态表，避免仓库知识逐渐神话化。
6. 至少一个故障排查章节，把第一次踩过的坑变成后来者少踩一次的坑。

## 多板记录建议

当 Tang9k 从单板进入多板互联时，不要把所有记录都塞进单板页。建议按这个结构拆：

- `tang9k.md`：板卡公共知识、单板烧录、单板验收。
- `tang9k-board-N.md`：某一块实体板的序列、端口、异常和烧录历史。
- `tang9k-cluster-*.md`：某一次多板拓扑实验，记录连线、ring 顺序、allreduce dry-run、真实链路结果。

这样做有点像给硬件建“病历本”。板多起来以后，最容易出问题的不是算法，而是某块板、某根线、某次烧录状态被大家混在一起讲。

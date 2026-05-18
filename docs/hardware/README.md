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
| Tang Nano 9K / Tang9k | FPGA 控制面、serial v1、后续多板互联入口 | `Ping`、命令生命周期、scratch、result window 已真板通过；`DeviceInfo` 已接入但待最新 bitstream 复测 | [Tang9k Wiki](tang9k.md) |

## 当前硬件主线

| 层级 | 现在在做什么 | 通过标准 |
| --- | --- | --- |
| Host 协议层 | 固定帧头、checksum、stream decoder、submit queue | golden vector 与 host tests 稳定 |
| 单板控制层 | `DeviceInfo`、`Ping`、命令 ACK、scratch、result window | 每次重新烧录后能跑完整 bring-up suite |
| 单板数据层 | 真实矩阵结果缓冲区、后续 DMA/streaming | 不再返回摘要，而是能读回真实计算结果 |
| 多板抽象层 | topology、ring allreduce dry-run、matmul shard plan | 断链、单向链路、成本估算都可复现 |
| 多板实链路 | 多 Tang9k 板串联/并行验证 | 真链路握手、吞吐、错误恢复都能被记录和回归 |

## 文档地图

| 文档 | 适合什么时候看 |
| --- | --- |
| [Tang9k Wiki](tang9k.md) | 你要接板、烧录、跑 COM3、判断当前板卡状态 |
| [Tang9k Serial Protocol v1](../tang9k-serial-protocol-v1.md) | 你要改 opcode、payload、checksum 或做协议实现 |
| [Tang9k UART Responder README](../../hardware/tang9k/uart_responder/README.md) | 你只想看 responder 工程自己的构建与验收细节 |
| [Roadmap](../roadmap.md) | 你想看 Tang9k 在框架长期路线里的位置 |

## 新板卡接入清单

一块新硬件要进入 SPTorch 主线，至少应具备：

1. 明确的板卡页，记录型号、资源、连接方式、时钟、供电和最小依赖。
2. 一个可重复构建的最小工程，能证明主机到设备的最短闭环。
3. 一套 host CLI 验收命令，不能只靠 GUI 或肉眼观察。
4. 一组 byte-level 或 equivalent golden tests，锁住协议边界。
5. 一个“已实测 / 未实测”状态表，避免仓库知识逐渐神话化。
6. 至少一个故障排查章节，把第一次踩过的坑变成后来者少踩一次的坑。


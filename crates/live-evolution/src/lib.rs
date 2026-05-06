//! 在线进化运行时。
//!
//! 这个 crate 承担“模型不中断服务时继续学习”的底座能力：推理读取
//! active 参数，训练写入 shadow 参数；当监控确认新参数稳定后，通过 fence
//! 序列和版本提交完成切换。它不绑定某个产品形态，Text2SQL、Studio 或
//! 未来硬件控制面都应通过这里暴露的事件语义观察训练过程。
//!
//! - [`double_buffer`]：维护 active/shadow 两份参数，并用原子状态完成切换。
//! - [`incremental`]：把持续到来的样本积累成 micro-batch。
//! - [`ewc`]：用 Fisher 对角近似约束重要参数，降低灾难性遗忘。
//! - [`monitor`]：用滚动 loss 监控退化，并触发回滚。
//! - [`events`] / [`runtime`]：向外发布 metrics、version commit、fence 和硬件状态。

pub mod double_buffer;
pub mod events;
pub mod ewc;
pub mod incremental;
pub mod monitor;
pub mod runtime;

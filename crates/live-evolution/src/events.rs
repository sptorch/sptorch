use std::sync::OnceLock;

use sptorch_versioning::{EvolutionMetrics, FenceState, HardwareState, VersionNode};
use tokio::sync::broadcast;

/// 在线进化运行时向外发布的事件。
///
/// 这些事件是框架与 Studio/产品控制面之间的轻量协议：训练指标用于观察
/// 收敛，版本提交用于 request anchoring，fence 和硬件状态用于确认异构
/// 设备切换是否安全。
#[derive(Debug, Clone)]
pub enum LiveEvolutionEvent {
    Metrics(EvolutionMetrics),
    VersionCommit(VersionNode),
    Fence(FenceState),
    HardwareState(HardwareState),
}

static EVENT_BUS: OnceLock<broadcast::Sender<LiveEvolutionEvent>> = OnceLock::new();

/// 返回全局事件总线。
///
/// 使用 `broadcast` 是为了让多个观察者各自消费同一条运行时事件流。订阅者
/// 如果处理太慢可能出现 lag，这比阻塞训练循环更符合在线系统的优先级。
fn bus() -> &'static broadcast::Sender<LiveEvolutionEvent> {
    EVENT_BUS.get_or_init(|| {
        let (tx, _rx) = broadcast::channel(1024);
        tx
    })
}

/// 发布一条在线进化事件。
///
/// 没有订阅者时 `send` 会返回错误，这里主动忽略：训练运行时不应该因为
/// Studio 未打开或测试中没有监听器而失败。
pub fn publish(event: LiveEvolutionEvent) {
    let _ = bus().send(event);
}

/// 创建一个新的事件订阅者。
///
/// 每个订阅者拥有独立游标；UI、日志器和测试可以并行监听，不会互相抢走
/// 消息。调用者需要自行处理 `broadcast::error::RecvError::Lagged`。
pub fn subscribe() -> broadcast::Receiver<LiveEvolutionEvent> {
    bus().subscribe()
}

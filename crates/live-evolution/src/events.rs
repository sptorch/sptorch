use std::sync::OnceLock;

use sptorch_versioning::{EvolutionMetrics, FenceState, HardwareState, VersionNode};
use tokio::sync::broadcast;

#[derive(Debug, Clone)]
pub enum LiveEvolutionEvent {
    Metrics(EvolutionMetrics),
    VersionCommit(VersionNode),
    Fence(FenceState),
    HardwareState(HardwareState),
}

static EVENT_BUS: OnceLock<broadcast::Sender<LiveEvolutionEvent>> = OnceLock::new();

// 中文注释：关键逻辑说明。
fn bus() -> &'static broadcast::Sender<LiveEvolutionEvent> {
    EVENT_BUS.get_or_init(|| {
        let (tx, _rx) = broadcast::channel(1024);
        tx
    })
}
/// `publish`：中文注释，说明函数用途、输入约束与输出语义。
pub fn publish(event: LiveEvolutionEvent) {
    let _ = bus().send(event);
}
/// `subscribe`：中文注释，说明函数用途、输入约束与输出语义。
pub fn subscribe() -> broadcast::Receiver<LiveEvolutionEvent> {
    bus().subscribe()
}

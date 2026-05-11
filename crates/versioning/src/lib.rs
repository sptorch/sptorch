//! SPTorch 的版本化张量协议。
//!
//! 这个 crate 是框架、Studio 和 live-evolution 之间共享的“线格式契约”。它只定义
//! 可序列化的状态快照和事件载荷，不持有真实 Tensor，也不读取硬件指针。这样可以
//! 保证 IDE、训练进程和未来硬件遥测服务在不同进程中仍能用同一套版本语义对齐。

use serde::{Deserialize, Serialize};

/// 实时指标事件名。
pub const EVENT_METRICS: &str = "studio://metrics";
/// 版本提交事件名。
pub const EVENT_VERSION_COMMIT: &str = "studio://version-commit";
/// 硬件 fence 状态事件名。
pub const EVENT_FENCE: &str = "studio://fence";
/// 后端在线状态和队列深度事件名。
pub const EVENT_HARDWARE_STATE: &str = "studio://hardware-state";

/// 层参数更新策略。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum UpdatePolicy {
    /// 单缓冲：直接更新活跃参数，节省显存但不提供影子版本。
    Single,
    /// 双缓冲：更新 shadow 参数，提交时通过版本切换保证推理一致性。
    Double,
}

/// 单个模型层的更新策略配置。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LayerPolicy {
    pub layer_name: String,
    pub policy: UpdatePolicy,
}

/// 张量活跃缓冲和影子缓冲的逻辑指针。
///
/// v1 使用稳定字符串标识，不承诺是真实物理地址；真实硬件地址应留在 HAL/驱动侧，
/// 避免跨进程 UI 暴露不安全指针。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BufferPointers {
    pub active_ptr: String,
    pub shadow_ptr: Option<String>,
    pub active_version: u64,
    pub shadow_version: Option<u64>,
}

/// 前端可展示的张量布局快照。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TensorLayoutSnapshot {
    pub tensor_id: String,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub offset: usize,
    pub numel: usize,
    pub dtype: String,
    pub device: String,
    pub pointers: BufferPointers,
}

/// 版本链中的一个提交节点。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct VersionNode {
    pub version_id: u64,
    pub parent_version: Option<u64>,
    pub committed_at_ms: u64,
    pub reason: String,
}

/// 版本化存储的完整可观测快照。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct VersionedStorage {
    pub global_version: u64,
    pub active_version: u64,
    pub chain: Vec<VersionNode>,
    pub layer_policies: Vec<LayerPolicy>,
    pub tensors: Vec<TensorLayoutSnapshot>,
}

/// 原子切换/fence 状态机阶段。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum FencePhase {
    Idle,
    Prepare,
    WaitFence,
    Swap,
    Commit,
    Done,
    Error,
}

/// 一次硬件同步或影子缓冲切换的进度快照。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FenceState {
    pub phase: FencePhase,
    pub progress: f32,
    pub queue_depth: u32,
    pub message: String,
}

/// 实时训练/演化指标。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EvolutionMetrics {
    pub ts_ms: u64,
    pub loss: f32,
    pub grad_norm: f32,
    pub grad_scale_factor: f32,
    pub accum_current: u32,
    pub accum_target: u32,
    pub version_id: u64,
    pub fence: Option<FenceState>,
}

/// 硬件后端的最小在线状态。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HardwareState {
    pub backend: String,
    pub queue_depth: u32,
    pub online: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    // VersionedStorage 是 Studio 和引擎之间最重要的 JSON 契约，roundtrip 失败说明协议破坏。
    #[test]
    fn test_versioned_storage_json_roundtrip() {
        let storage = VersionedStorage {
            global_version: 3,
            active_version: 3,
            chain: vec![
                VersionNode {
                    version_id: 2,
                    parent_version: Some(1),
                    committed_at_ms: 1711000000,
                    reason: "shadow_swap".into(),
                },
                VersionNode {
                    version_id: 3,
                    parent_version: Some(2),
                    committed_at_ms: 1711000100,
                    reason: "online_commit".into(),
                },
            ],
            layer_policies: vec![LayerPolicy {
                layer_name: "transformer.block0.attn".into(),
                policy: UpdatePolicy::Double,
            }],
            tensors: vec![TensorLayoutSnapshot {
                tensor_id: "t0".into(),
                shape: vec![2, 4],
                strides: vec![4, 1],
                offset: 0,
                numel: 8,
                dtype: "F32".into(),
                device: "CPU".into(),
                pointers: BufferPointers {
                    active_ptr: "arc:0x1".into(),
                    shadow_ptr: Some("arc:0x2".into()),
                    active_version: 3,
                    shadow_version: Some(4),
                },
            }],
        };

        let json = serde_json::to_string(&storage).expect("serialize storage");
        let out: VersionedStorage = serde_json::from_str(&json).expect("deserialize storage");
        assert_eq!(out, storage);
    }

    // Metrics payload 覆盖梯度累积、scale 因子和 fence，避免 UI 订阅字段漂移。
    #[test]
    fn test_evolution_metrics_json_roundtrip() {
        let m = EvolutionMetrics {
            ts_ms: 1711000200,
            loss: 1.23,
            grad_norm: 0.45,
            grad_scale_factor: 0.5,
            accum_current: 2,
            accum_target: 4,
            version_id: 3,
            fence: Some(FenceState {
                phase: FencePhase::Swap,
                progress: 0.75,
                queue_depth: 6,
                message: "atomic swap".into(),
            }),
        };

        let json = serde_json::to_string(&m).expect("serialize metrics");
        let out: EvolutionMetrics = serde_json::from_str(&json).expect("deserialize metrics");
        assert_eq!(out, m);
    }
}

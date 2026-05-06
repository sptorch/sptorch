//! Versioned tensor protocol for SPTorch Studio.

use serde::{Deserialize, Serialize};

pub const EVENT_METRICS: &str = "studio://metrics";
pub const EVENT_VERSION_COMMIT: &str = "studio://version-commit";
pub const EVENT_FENCE: &str = "studio://fence";
pub const EVENT_HARDWARE_STATE: &str = "studio://hardware-state";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum UpdatePolicy {
    Single,
    Double,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LayerPolicy {
    pub layer_name: String,
    pub policy: UpdatePolicy,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BufferPointers {
    pub active_ptr: String,
    pub shadow_ptr: Option<String>,
    pub active_version: u64,
    pub shadow_version: Option<u64>,
}

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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct VersionNode {
    pub version_id: u64,
    pub parent_version: Option<u64>,
    pub committed_at_ms: u64,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct VersionedStorage {
    pub global_version: u64,
    pub active_version: u64,
    pub chain: Vec<VersionNode>,
    pub layer_policies: Vec<LayerPolicy>,
    pub tensors: Vec<TensorLayoutSnapshot>,
}

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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FenceState {
    pub phase: FencePhase,
    pub progress: f32,
    pub queue_depth: u32,
    pub message: String,
}

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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HardwareState {
    pub backend: String,
    pub queue_depth: u32,
    pub online: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    // 中文注释：关键逻辑说明。
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

    // 中文注释：关键逻辑说明。
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

//! SPTorch 的版本化张量协议。
//!
//! 这个 crate 是框架、Studio 和 live-evolution 之间共享的“线格式契约”。它只定义
//! 可序列化的状态快照和事件载荷，不持有真实 Tensor，也不读取硬件指针。这样可以
//! 保证 IDE、训练进程和未来硬件遥测服务在不同进程中仍能用同一套版本语义对齐。

use serde::{Deserialize, Serialize};

/// 实时训练指标事件名。
///
/// 事件载荷通常是 [`EvolutionMetrics`]。Studio 或外部监控面可以订阅这个事件
/// 来刷新 loss、梯度范数、梯度累积进度和当前版本号。
pub const EVENT_METRICS: &str = "studio://metrics";
/// 版本提交事件名。
///
/// 事件载荷通常是 [`VersionNode`] 或包含版本链的 [`VersionedStorage`]。它用于
/// 表示一次 shadow 参数切换、在线学习提交或回滚已经成为新的可观察版本。
pub const EVENT_VERSION_COMMIT: &str = "studio://version-commit";
/// 硬件 fence 状态事件名。
///
/// 事件载荷通常是 [`FenceState`]。它描述一次原子切换或硬件同步正在经历的阶段，
/// 让 UI 能把等待、交换、提交和失败状态展示出来。
pub const EVENT_FENCE: &str = "studio://fence";
/// 后端在线状态和队列深度事件名。
///
/// 事件载荷通常是 [`HardwareState`]，用于显示当前硬件后端是否在线以及排队压力。
pub const EVENT_HARDWARE_STATE: &str = "studio://hardware-state";
/// 双文件 checkpoint 清单的 schema 名称。
///
/// 所有通过 `serialize::save_state_dict_bundle` 写出的 manifest 都会使用这个值。
/// 加载端必须检查 schema，避免把其他格式误当成本协议解析。
pub const CHECKPOINT_MANIFEST_SCHEMA: &str = "sptorch.checkpoint_manifest.v1";
/// 双文件 checkpoint 清单的格式版本。
///
/// 当前版本号只描述 manifest 自身，不描述权重文件的 state_dict schema。
pub const CHECKPOINT_MANIFEST_FORMAT_VERSION: u32 = 1;
/// 训练闭环里常见的 state_dict / checkpoint 清单契约。
///
/// 这一层只记录“文件是什么、来自哪个模型、里面装了什么语义”，不直接
/// 依赖 Tensor。这样 versioning crate 可以继续作为纯协议层，被 Studio、训练器
/// 和未来产品仓复用，而不会反向绑住底层数值实现。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CheckpointManifest {
    /// manifest 自身的 schema 名称。
    pub schema: String,
    /// manifest 自身的格式版本。
    pub format_version: u32,
    /// 产出这份 checkpoint 的模型名称或逻辑别名。
    pub model_name: String,
    /// 保存类型，例如 `state_dict`。
    pub save_kind: String,
    /// 与 manifest 配套的权重文件名，不包含目录时表示与 manifest 同目录。
    pub weights_file: String,
    /// manifest 记录的参数数量，加载时会和模型命名参数数量对齐。
    pub parameter_count: usize,
    /// 参数稳定名称列表，顺序应与 state_dict 文件中的条目顺序一致。
    pub parameter_names: Vec<String>,
    /// 配套权重文件使用的 state_dict schema。
    pub state_dict_schema: String,
    /// 生成 manifest 的时间戳，单位毫秒。
    pub created_at_ms: u64,
    /// 给训练器或人工排查使用的自由文本说明。
    pub note: String,
}

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
    /// 策略作用的逻辑层名。
    pub layer_name: String,
    /// 该层参数更新时使用单缓冲还是双缓冲。
    pub policy: UpdatePolicy,
}

/// 张量活跃缓冲和影子缓冲的逻辑指针。
///
/// v1 使用稳定字符串标识，不承诺是真实物理地址；真实硬件地址应留在 HAL/驱动侧，
/// 避免跨进程 UI 暴露不安全指针。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BufferPointers {
    /// 当前推理读取的逻辑缓冲标识。
    pub active_ptr: String,
    /// 训练写入但尚未切换成 active 的逻辑缓冲标识。
    pub shadow_ptr: Option<String>,
    /// active 缓冲对应的版本号。
    pub active_version: u64,
    /// shadow 缓冲对应的候选版本号。
    pub shadow_version: Option<u64>,
}

/// 前端可展示的张量布局快照。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TensorLayoutSnapshot {
    /// 张量在观测系统中的稳定 ID。
    pub tensor_id: String,
    /// 逻辑 shape。
    pub shape: Vec<usize>,
    /// 逻辑 stride，用于展示 view 与物理连续布局的关系。
    pub strides: Vec<usize>,
    /// view 相对底层 storage 的元素偏移。
    pub offset: usize,
    /// 该张量 view 可访问的元素总数。
    pub numel: usize,
    /// dtype 名称，例如 `F32`、`F16` 或 `BF16`。
    pub dtype: String,
    /// 逻辑设备名称，例如 `CPU`、`CUDA(0)` 或未来的 Tank9k 标识。
    pub device: String,
    /// active/shadow 缓冲的逻辑指针信息。
    pub pointers: BufferPointers,
}

/// 版本链中的一个提交节点。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct VersionNode {
    /// 当前提交生成的版本号。
    pub version_id: u64,
    /// 父版本号；根版本没有父节点。
    pub parent_version: Option<u64>,
    /// 提交发生时间，单位毫秒。
    pub committed_at_ms: u64,
    /// 提交原因，例如 `shadow_swap`、`online_commit` 或 `rollback`。
    pub reason: String,
}

/// 版本化存储的完整可观测快照。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct VersionedStorage {
    /// 系统已知的最大版本号。
    pub global_version: u64,
    /// 当前对外服务正在使用的版本号。
    pub active_version: u64,
    /// 从旧到新的版本提交链。
    pub chain: Vec<VersionNode>,
    /// 各层采用的参数更新策略。
    pub layer_policies: Vec<LayerPolicy>,
    /// 当前可展示的张量布局集合。
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
    /// 当前 fence 所处阶段。
    pub phase: FencePhase,
    /// 进度，建议使用 `0.0..=1.0`。
    pub progress: f32,
    /// 后端队列深度，便于 UI 区分“卡住”和“仍在排队”。
    pub queue_depth: u32,
    /// 面向人类的状态说明或错误信息。
    pub message: String,
}

/// 实时训练/演化指标。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EvolutionMetrics {
    /// 指标采样时间，单位毫秒。
    pub ts_ms: u64,
    /// 当前训练 loss。
    pub loss: f32,
    /// 当前梯度全局范数。
    pub grad_norm: f32,
    /// 混合精度或梯度稳定策略使用的缩放因子。
    pub grad_scale_factor: f32,
    /// 当前梯度累积步数。
    pub accum_current: u32,
    /// 目标梯度累积步数。
    pub accum_target: u32,
    /// 这些指标对应的模型版本。
    pub version_id: u64,
    /// 如果当前正在切换或同步硬件，则携带 fence 状态。
    pub fence: Option<FenceState>,
}

/// 硬件后端的最小在线状态。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HardwareState {
    /// 后端名称，例如 `cpu`、`cuda`、`tank9k-serial`。
    pub backend: String,
    /// 后端当前队列深度。
    pub queue_depth: u32,
    /// 后端是否可被训练或观测路径使用。
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

    #[test]
    fn test_checkpoint_manifest_json_roundtrip() {
        let manifest = CheckpointManifest {
            schema: CHECKPOINT_MANIFEST_SCHEMA.into(),
            format_version: CHECKPOINT_MANIFEST_FORMAT_VERSION,
            model_name: "tiny-gpt".into(),
            save_kind: "state_dict".into(),
            weights_file: "tiny-gpt.weights.json".into(),
            parameter_count: 17,
            parameter_names: vec!["token_emb.weight".into(), "lm_head.weight".into()],
            state_dict_schema: "sptorch.state_dict.v1".into(),
            created_at_ms: 1711000300,
            note: "keep model weights and metadata aligned".into(),
        };

        let json = serde_json::to_string(&manifest).expect("serialize manifest");
        let out: CheckpointManifest = serde_json::from_str(&json).expect("deserialize manifest");
        assert_eq!(out, manifest);
    }
}

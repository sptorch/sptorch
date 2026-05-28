//! SPTorch 的稳定门面层。
//!
//! 这个 crate 不承担具体训练实现，而是把框架里最常用、最稳定的一组 API
//! 重新组织成外部产品更容易消费的入口。产品仓、IDE 仓和测试工具都应优先
//! 依赖 `sptorch::v1::*`，而不是直接深入内部 crate 路径。
//!
//! 设计目标：
//! - 把训练、序列化、版本化和硬件边界收敛到统一命名空间。
//! - 让对外暴露的 API 更稳定，内部 crate 可以继续按演进需要重构。
//! - 给产品侧提供“该用什么”的答案，而不是让调用方自己拼装内部模块。

pub mod v1 {
    /// 核心张量与梯度模式入口。
    ///
    /// 这里暴露的是框架最底层、最稳定的数值对象和与之配套的广播/梯度
    /// 控制工具。调用方如果只是做训练、推理或参数导出，通常不需要更深层
    /// 访问 `core-tensor` 的内部实现。
    pub mod core {
        pub use sptorch_core_tensor::{
            broadcast_shape, can_broadcast, is_grad_enabled, no_grad, set_grad_enabled, DType, Device, Tensor,
        };
    }

    /// 神经网络模块入口。
    ///
    /// 这一层主要给产品仓和实验脚本使用，包含小模型训练、生成和受约束
    /// 解码需要的模块。它尽量保持“能直接搭建模型”的粒度，而不是暴露所有
    /// 内部构件。
    pub mod nn {
        pub use sptorch_nn::{
            activate_inference_batch, activate_qwen_like_inference_batch, decode_constrained_next_token,
            decode_inference_batch_round, decode_next_token, decode_qwen_like_inference_batch_round,
            decode_qwen_like_next_token, generate_constrained, generate_qwen_like_with_config, generate_with_config,
            generate_with_sampling, sampling_candidates, DecodeStep, DecodeStopReason, GenerationConfig,
            InferenceAdmissionError, InferenceBatch, InferenceBatchState, InferenceBatchStep, InferenceRequest,
            InferenceScheduler, InferenceSchedulerConfig, InferenceSession, InferenceState, KvCache, KvCacheLayer,
            KvCacheSpec, QwenLikeGPT, TokenCandidate, TokenTrie, GPT,
        };
    }

    /// 数据与 tokenizer 入口。
    ///
    /// 这层把框架里最常用的字符级/BPE tokenizer 和 next-token 数据管道集中导出，
    /// 方便产品仓用更稳定的命名空间组装 SFT 样本。
    pub mod data {
        pub use sptorch_data::{BpeTokenizer, CharTokenizer, DataLoader, Dataset, TextDataset, Tokenizer};
    }

    /// 优化器与梯度工具入口。
    ///
    /// 这里统一导出参数更新、梯度裁剪和学习率调度相关工具。它们的职责边界
    /// 很明确：只管参数更新，不替代训练循环本身的调度逻辑。
    pub mod optim {
        pub use sptorch_optim::{clip_grad_norm, scale_gradients, zero_grad, AdamW, Optimizer, SGD};
    }

    /// 可微分算子入口。
    ///
    /// 这是一组训练语义上最常见的算子：矩阵乘、广播、归约、softmax、激活、
    /// loss 等。产品侧应该优先通过这里组装前向和反向路径，而不是直接拼接
    /// 内部实现文件。
    pub mod ops {
        pub use sptorch_core_ops::{
            add, batch_matmul, broadcast_add, concat, cross_entropy_loss, cross_entropy_loss_ignore_index,
            embedding_lookup, exp, gelu, log, log_softmax, masked_fill, masked_softmax, matmul, mean, mean_dim, mul,
            neg, relu, reshape, rms_norm, scale, softmax, sub, sum, sum_dim, swiglu, transpose,
        };
    }

    /// 权重与 checkpoint 入口。
    ///
    /// 这里同时保留单文件 checkpoint、JSON state_dict 和双文件 bundle 三种
    /// 路径。调试、轻量产品和训练恢复可以按需要选用不同粒度，但都应该经过这层
    /// 稳定门面。
    pub mod checkpoint {
        pub use sptorch_serialize::safetensors::SafeTensorsFile;
        pub use sptorch_serialize::{
            export_state_dict, load_checkpoint, load_named_state_dict, load_state_dict, load_state_dict_bundle,
            load_state_dict_file, save_checkpoint, save_named_state_dict, save_state_dict, save_state_dict_bundle,
            NamedStateDict, StateDictEntry,
        };
    }

    /// 硬件抽象入口。
    ///
    /// 这一层提供的是框架侧可见的硬件边界，而不是具体驱动实现。规划器、监控
    /// 工具和 Studio 可以依赖这里的类型来描述 fence、队列和后端状态。
    pub mod hal {
        pub use sptorch_hal::{
            CpuKvCacheBuffer, FenceState, KvCacheBuffer, KvCacheBufferSpec, KvCacheWindowState, QueueState,
        };
    }

    /// 版本化与遥测协议入口。
    ///
    /// 这里集中导出训练过程中的状态快照、清单协议和事件名。它的职责是让
    /// 训练进程、Studio 和未来的硬件观测面说同一种“版本语言”。
    pub mod versioning {
        pub use sptorch_versioning::{
            BufferPointers, CheckpointManifest, EvolutionMetrics, FencePhase, FenceState, HardwareState, LayerPolicy,
            TensorLayoutSnapshot, UpdatePolicy, VersionNode, VersionedStorage, EVENT_FENCE, EVENT_HARDWARE_STATE,
            EVENT_METRICS, EVENT_VERSION_COMMIT,
        };
    }

    /// 面向产品侧的常用预导入。
    ///
    /// 如果一个产品仓希望尽量少写 `use`，可以直接从这里导入训练闭环最常用
    /// 的 API。这个 prelude 不是“全部导出”，而是框架当前推荐的最小高频集合。
    pub mod prelude {
        pub use super::checkpoint::{
            export_state_dict, load_checkpoint, load_named_state_dict, load_state_dict, load_state_dict_bundle,
            load_state_dict_file, save_checkpoint, save_named_state_dict, save_state_dict, save_state_dict_bundle,
            NamedStateDict, SafeTensorsFile, StateDictEntry,
        };
        pub use super::core::{
            broadcast_shape, can_broadcast, is_grad_enabled, no_grad, set_grad_enabled, DType, Device, Tensor,
        };
        pub use super::data::{BpeTokenizer, CharTokenizer, DataLoader, Dataset, TextDataset, Tokenizer};
        pub use super::nn::{
            activate_inference_batch, activate_qwen_like_inference_batch, decode_constrained_next_token,
            decode_inference_batch_round, decode_next_token, decode_qwen_like_inference_batch_round,
            decode_qwen_like_next_token, generate_constrained, generate_qwen_like_with_config, generate_with_config,
            generate_with_sampling, sampling_candidates, DecodeStep, DecodeStopReason, GenerationConfig,
            InferenceAdmissionError, InferenceBatch, InferenceBatchState, InferenceBatchStep, InferenceRequest,
            InferenceScheduler, InferenceSchedulerConfig, InferenceSession, InferenceState, KvCache, KvCacheLayer,
            KvCacheSpec, QwenLikeGPT, TokenCandidate, TokenTrie, GPT,
        };
        pub use super::ops::{
            add, batch_matmul, broadcast_add, concat, cross_entropy_loss, cross_entropy_loss_ignore_index,
            embedding_lookup, exp, gelu, log, log_softmax, masked_fill, masked_softmax, matmul, mean, mean_dim, mul,
            neg, relu, reshape, rms_norm, scale, softmax, sub, sum, sum_dim, swiglu, transpose,
        };
        pub use super::optim::{clip_grad_norm, scale_gradients, zero_grad, AdamW, Optimizer, SGD};
        pub use super::versioning::{
            BufferPointers, CheckpointManifest, EvolutionMetrics, FencePhase, HardwareState, LayerPolicy,
            TensorLayoutSnapshot, UpdatePolicy, VersionNode, VersionedStorage, EVENT_FENCE, EVENT_HARDWARE_STATE,
            EVENT_METRICS, EVENT_VERSION_COMMIT,
        };
    }
}

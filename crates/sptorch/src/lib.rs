//! Public facade crate for SPTorch framework.
//!
//! External products should prefer `sptorch::v1::*` as the stable API surface.

pub mod v1 {
    /// 稳定的核心张量入口。
    pub mod core {
        pub use sptorch_core_tensor::{
            broadcast_shape, can_broadcast, is_grad_enabled, no_grad, set_grad_enabled, DType, Device, Tensor,
        };
    }

    /// Stable neural API used by product runtimes.
    pub mod nn {
        pub use sptorch_nn::{generate_constrained, TokenTrie, GPT};
    }

    /// Stable optimizer API used by product runtimes.
    pub mod optim {
        pub use sptorch_optim::{clip_grad_norm, scale_gradients, zero_grad, AdamW, Optimizer, SGD};
    }

    /// Stable ops API used by product runtimes.
    pub mod ops {
        pub use sptorch_core_ops::{
            add, batch_matmul, broadcast_add, concat, cross_entropy_loss, embedding_lookup, exp, gelu, log,
            log_softmax, masked_fill, masked_softmax, matmul, mean, mean_dim, mul, neg, relu, reshape, rms_norm, scale,
            softmax, sub, sum, sum_dim, swiglu, transpose,
        };
    }

    /// Stable checkpoint API used by product runtimes.
    pub mod checkpoint {
        pub use sptorch_serialize::safetensors::SafeTensorsFile;
        pub use sptorch_serialize::{
            export_state_dict, load_checkpoint, load_state_dict, save_checkpoint, StateDictEntry,
        };
    }

    /// Stable hardware boundary used by planners and observability tools.
    pub mod hal {
        pub use sptorch_hal::{FenceState, QueueState};
    }

    /// Convenience prelude for product-side imports.
    pub mod prelude {
        pub use super::checkpoint::{
            export_state_dict, load_checkpoint, load_state_dict, save_checkpoint, SafeTensorsFile, StateDictEntry,
        };
        pub use super::core::{
            broadcast_shape, can_broadcast, is_grad_enabled, no_grad, set_grad_enabled, DType, Device, Tensor,
        };
        pub use super::nn::{generate_constrained, TokenTrie, GPT};
        pub use super::ops::{
            add, batch_matmul, broadcast_add, concat, cross_entropy_loss, embedding_lookup, exp, gelu, log,
            log_softmax, masked_fill, masked_softmax, matmul, mean, mean_dim, mul, neg, relu, reshape, rms_norm, scale,
            softmax, sub, sum, sum_dim, swiglu, transpose,
        };
        pub use super::optim::{clip_grad_norm, scale_gradients, zero_grad, AdamW, Optimizer, SGD};
    }
}

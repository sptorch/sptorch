use std::sync::{
    atomic::{AtomicBool, Ordering},
    OnceLock,
};
use std::time::{SystemTime, UNIX_EPOCH};

use sptorch_core_tensor::Tensor;
use sptorch_optim::{scale_gradients, Optimizer, SGD};
use sptorch_versioning::{EvolutionMetrics, FencePhase, FenceState, HardwareState, VersionNode};
use tokio::time::{sleep, Duration};

use crate::double_buffer::DoubleBufferParams;
use crate::events::{publish, LiveEvolutionEvent};
use crate::incremental::IncrementalTrainer;
use crate::monitor::{MonitorAction, TrainingMonitor};

static STARTED: OnceLock<AtomicBool> = OnceLock::new();

/// 返回运行时启动标志。
///
/// 使用 `OnceLock<AtomicBool>` 是为了让库在没有显式 Runtime 对象的情况下也能
/// 提供一个全局演化流，同时避免多次调用启动多个后台训练循环。
fn started() -> &'static AtomicBool {
    STARTED.get_or_init(|| AtomicBool::new(false))
}

/// 启动在线进化后台循环，若已经启动则返回 `false`。
///
/// 该函数只负责启动一次事件源；调用者通常是 Studio bridge 或产品控制面。
/// 真正的事件通过 [`crate::events::subscribe`] 订阅，避免把异步任务句柄暴露
/// 给外部系统。
pub fn ensure_runtime_started() -> bool {
    if started()
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_err()
    {
        return false;
    }

    tokio::spawn(async move {
        run_loop().await;
    });
    true
}

/// 运行一个可观测的在线训练循环。
///
/// 当前循环仍是轻量模拟：它使用真实 `Tensor`、`SGD`、梯度缩放、双缓冲和
/// 监控器，但样本与梯度由确定性序列生成。这样 Studio 和测试可以订阅真实
/// 框架事件流，而不需要依赖某个产品仓或真实硬件。
async fn run_loop() {
    let params = vec![Tensor::with_grad(vec![0.5, 0.25, -0.1, 0.9], vec![2, 2], true)];
    let db = DoubleBufferParams::new(&params);
    let opt = SGD::new(params.clone(), 0.01, 0.9);
    let mut trainer = IncrementalTrainer::new(opt, params.clone(), 4);
    let mut monitor = TrainingMonitor::new(8, 0.35);

    let accum_target = 8u32;
    let mut accum_current = 0u32;
    let mut version_id = 1u64;
    let mut sample_id = 0u64;

    loop {
        sample_id = sample_id.wrapping_add(1);
        let should_train = trainer.push_sample(
            vec![(sample_id % 97) as usize, ((sample_id + 1) % 97) as usize],
            vec![((sample_id + 2) % 97) as usize, ((sample_id + 3) % 97) as usize],
        );

        accum_current = (accum_current % accum_target) + 1;
        let grad_scale_factor = 1.0 + ((sample_id % 7) as f32) * 0.05;
        let grad_norm = 0.2 + ((sample_id as f32) * 0.17).sin().abs();
        let base_loss = 1.8 / ((trainer.total_steps() + 1) as f32);
        let jitter = ((sample_id as f32) * 0.13).cos().abs() * 0.08;
        let loss = base_loss + jitter;

        emit_metrics(
            loss,
            grad_norm,
            grad_scale_factor,
            accum_current,
            accum_target,
            version_id,
            None,
        );

        if should_train {
            let _batch = trainer.drain_batch();
            trainer.optimizer_mut().zero_grad();

            for p in &params {
                let fake = Tensor::new(vec![0.05, -0.03, 0.02, -0.01], vec![2, 2]);
                p.accum_grad(&fake);
            }

            // 梯度累积完成后按目标步数缩放，Studio 会把这个 scale 作为稳定性指标展示。
            let scale = 1.0 / (accum_target as f32);
            scale_gradients(&params, scale);
            trainer.optimizer_mut().step();
            trainer.step_completed();

            match monitor.record_loss(loss) {
                MonitorAction::Continue => {
                    if trainer.total_steps() % 3 == 0 {
                        emit_fence_sequence(version_id).await;
                        db.swap();
                        version_id = version_id.wrapping_add(1);
                        emit_commit(version_id, "live_evolution_commit");
                    }
                }
                MonitorAction::Rollback { current_avg, best_avg } => {
                    emit_fence_error(format!(
                        "rollback triggered: current_avg={:.4}, best_avg={:.4}",
                        current_avg, best_avg
                    ));
                    db.sync_shadow_from_active();
                    monitor.reset_after_rollback();
                }
            }
        }

        sleep(Duration::from_millis(350)).await;
    }
}

/// 发布一次完整的 fence 状态机序列。
///
/// 真实硬件后端接入后，这里的 phase 应由 HAL FFI 的队列/fence 信号驱动。
/// 当前模拟序列仍保持 Prepare -> WaitFence -> Swap -> Commit -> Done 顺序，
/// 让 UI 和测试提前固定状态机契约。
async fn emit_fence_sequence(version_id: u64) {
    let phases = [
        (FencePhase::Prepare, 0.2, 8, "prepare"),
        (FencePhase::WaitFence, 0.45, 6, "wait fence"),
        (FencePhase::Swap, 0.7, 3, "swap pointers"),
        (FencePhase::Commit, 0.92, 1, "commit version"),
        (FencePhase::Done, 1.0, 0, "done"),
    ];

    for (phase, progress, queue_depth, msg) in phases {
        let fence = FenceState {
            phase,
            progress,
            queue_depth,
            message: msg.to_string(),
        };
        publish(LiveEvolutionEvent::Fence(fence.clone()));
        publish(LiveEvolutionEvent::Metrics(EvolutionMetrics {
            ts_ms: now_ms(),
            loss: 0.0,
            grad_norm: 0.0,
            grad_scale_factor: 1.0,
            accum_current: 0,
            accum_target: 0,
            version_id,
            fence: Some(fence.clone()),
        }));
        publish(LiveEvolutionEvent::HardwareState(HardwareState {
            backend: "live-evolution-sim".to_string(),
            queue_depth,
            online: !matches!(fence.phase, FencePhase::Error),
        }));
        sleep(Duration::from_millis(70)).await;
    }
}

/// 发布 fence 错误并把硬件状态标记为不可在线。
///
/// 这条路径用于监控器触发回滚时通知外部控制面：本次 shadow 更新没有提交，
/// 推理仍应继续锚定旧版本。
fn emit_fence_error(message: String) {
    let fence = FenceState {
        phase: FencePhase::Error,
        progress: 1.0,
        queue_depth: 0,
        message,
    };
    publish(LiveEvolutionEvent::Fence(fence.clone()));
    publish(LiveEvolutionEvent::HardwareState(HardwareState {
        backend: "live-evolution-sim".to_string(),
        queue_depth: fence.queue_depth,
        online: false,
    }));
}

/// 发布一个新的版本提交节点。
///
/// `parent_version` 用简单线性链表示当前实现的版本历史。未来若支持分支
/// 实验或多模型热切换，可以在 `VersionNode` 上扩展更丰富的 DAG 语义。
fn emit_commit(version_id: u64, reason: &str) {
    publish(LiveEvolutionEvent::VersionCommit(VersionNode {
        version_id,
        parent_version: version_id.checked_sub(1),
        committed_at_ms: now_ms(),
        reason: reason.to_string(),
    }));
}

/// 发布训练指标。
///
/// 指标携带当前 `version_id`，让请求级版本锚定可以把某次推理结果和当时的
/// 模型快照关联起来。`fence` 字段只在状态机事件中填充。
fn emit_metrics(
    loss: f32,
    grad_norm: f32,
    grad_scale_factor: f32,
    accum_current: u32,
    accum_target: u32,
    version_id: u64,
    fence: Option<FenceState>,
) {
    publish(LiveEvolutionEvent::Metrics(EvolutionMetrics {
        ts_ms: now_ms(),
        loss,
        grad_norm,
        grad_scale_factor,
        accum_current,
        accum_target,
        version_id,
        fence,
    }));
}

/// 返回 Unix epoch 毫秒时间戳。
///
/// 系统时间异常时退回 0，避免监控事件因为时间源问题 panic；外部消费者应
/// 把 0 视为不可用时间戳。
fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

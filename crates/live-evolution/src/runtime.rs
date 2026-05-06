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

// 中文注释：关键逻辑说明。
fn started() -> &'static AtomicBool {
    STARTED.get_or_init(|| AtomicBool::new(false))
}
/// `ensure_runtime_started`：中文注释，说明函数用途、输入约束与输出语义。
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

// 中文注释：关键逻辑说明。
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

// 中文注释：关键逻辑说明。
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

// 中文注释：关键逻辑说明。
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

// 中文注释：关键逻辑说明。
fn emit_commit(version_id: u64, reason: &str) {
    publish(LiveEvolutionEvent::VersionCommit(VersionNode {
        version_id,
        parent_version: version_id.checked_sub(1),
        committed_at_ms: now_ms(),
        reason: reason.to_string(),
    }));
}

// 中文注释：关键逻辑说明。
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

// 中文注释：关键逻辑说明。
fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

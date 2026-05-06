/// 在线训练质量监控器。
///
/// 监控器用滚动窗口观察最近 loss。如果当前窗口均值相对历史最佳均值恶化
/// 超过阈值，就建议回滚 shadow 参数。它不直接执行回滚，避免监控策略和
/// 参数存储实现耦合。
pub struct TrainingMonitor {
    /// 最近窗口内的 loss，越早的样本在窗口超长时被移除。
    loss_history: Vec<f32>,
    window_size: usize,
    /// 历史最佳窗口均值。初始为 `f32::MAX`，直到第一个完整窗口出现。
    best_avg_loss: f32,
    /// 退化阈值；0.2 表示当前均值比最佳均值高 20% 以上才回滚。
    degradation_threshold: f32,
    /// 已记录的 loss 样本总数，用于外部观测训练流量。
    total_samples: u64,
    /// 触发过的回滚次数。
    rollback_count: u32,
}

/// 记录一次 loss 后监控器给出的动作建议。
#[derive(Debug, Clone, PartialEq)]
pub enum MonitorAction {
    Continue,
    Rollback { current_avg: f32, best_avg: f32 },
}

impl TrainingMonitor {
    /// 创建滚动窗口监控器。
    ///
    /// `window_size` 决定敏感度：窗口越小越快发现退化，也越容易受噪声影响。
    /// `degradation_threshold` 应结合任务波动设置，太小会频繁误回滚。
    pub fn new(window_size: usize, degradation_threshold: f32) -> Self {
        TrainingMonitor {
            loss_history: Vec::new(),
            window_size,
            best_avg_loss: f32::MAX,
            degradation_threshold,
            total_samples: 0,
            rollback_count: 0,
        }
    }

    /// 记录一条 loss，并返回是否需要回滚。
    ///
    /// 在窗口填满前始终返回 [`MonitorAction::Continue`]，因为样本太少时均值
    /// 不稳定。窗口填满后，如果出现新的更低均值会刷新最佳基线；只有相对
    /// 基线恶化超过阈值才触发回滚。
    pub fn record_loss(&mut self, loss: f32) -> MonitorAction {
        self.loss_history.push(loss);
        self.total_samples += 1;

        if self.loss_history.len() > self.window_size {
            self.loss_history.remove(0);
        }

        if self.loss_history.len() < self.window_size {
            return MonitorAction::Continue;
        }

        let current_avg = self.rolling_avg();

        if current_avg < self.best_avg_loss {
            self.best_avg_loss = current_avg;
            return MonitorAction::Continue;
        }

        if current_avg > self.best_avg_loss * (1.0 + self.degradation_threshold) {
            self.rollback_count += 1;
            return MonitorAction::Rollback {
                current_avg,
                best_avg: self.best_avg_loss,
            };
        }

        MonitorAction::Continue
    }

    /// 回滚完成后清空当前窗口，但保留历史最佳基线。
    ///
    /// 保留 `best_avg_loss` 可以防止回滚后马上把较差状态重新当成新基线。
    pub fn reset_after_rollback(&mut self) {
        self.loss_history.clear();
    }

    /// 返回当前窗口均值。
    ///
    /// 窗口为空时返回 `f32::MAX`，让调用者能自然地把“尚无可用均值”视作
    /// 不可比较状态，而不是误判为 0 loss。
    pub fn rolling_avg(&self) -> f32 {
        if self.loss_history.is_empty() {
            return f32::MAX;
        }
        self.loss_history.iter().sum::<f32>() / self.loss_history.len() as f32
    }

    /// 返回历史最佳窗口均值。
    pub fn best_avg_loss(&self) -> f32 {
        self.best_avg_loss
    }

    /// 返回已记录的 loss 样本总数。
    pub fn total_samples(&self) -> u64 {
        self.total_samples
    }

    /// 返回累计触发的回滚次数。
    pub fn rollback_count(&self) -> u32 {
        self.rollback_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // 连续改善时应只刷新最佳基线，不触发回滚。
    #[test]
    fn test_monitor_improving_loss() {
        let mut mon = TrainingMonitor::new(3, 0.2);
        assert_eq!(mon.record_loss(3.0), MonitorAction::Continue);
        assert_eq!(mon.record_loss(2.5), MonitorAction::Continue);
        assert_eq!(mon.record_loss(2.0), MonitorAction::Continue); // window full, avg=2.5
        assert_eq!(mon.record_loss(1.5), MonitorAction::Continue); // avg=2.0, improving
        assert!(mon.best_avg_loss() < 2.5);
    }

    // 明显退化必须触发回滚建议，这是双缓冲安全提交前的最后一道闸门。
    #[test]
    fn test_monitor_triggers_rollback() {
        let mut mon = TrainingMonitor::new(3, 0.1); // 10% threshold
                                                    // 先建立一个稳定的好基线。
        mon.record_loss(1.0);
        mon.record_loss(1.0);
        mon.record_loss(1.0); // avg=1.0, best=1.0

        // 连续坏值填满窗口，避免偶发 spike 被误判。
        mon.record_loss(1.5);
        mon.record_loss(1.5);
        let action = mon.record_loss(1.5); // avg=1.5, > 1.0*1.1=1.1

        match action {
            MonitorAction::Rollback { current_avg, best_avg } => {
                assert!((current_avg - 1.5).abs() < 1e-6);
                assert!((best_avg - 1.0).abs() < 1e-6);
            }
            _ => panic!("expected rollback"),
        }
        assert!(mon.rollback_count() >= 1);
    }

    // 阈值内的小幅波动应被容忍，否则在线学习会在正常噪声下频繁回滚。
    #[test]
    fn test_monitor_no_rollback_within_threshold() {
        let mut mon = TrainingMonitor::new(3, 0.5); // 50% threshold (lenient)
        mon.record_loss(1.0);
        mon.record_loss(1.0);
        mon.record_loss(1.0); // best=1.0

        mon.record_loss(1.2);
        mon.record_loss(1.2);
        let action = mon.record_loss(1.2); // avg=1.2, < 1.0*1.5=1.5
        assert_eq!(action, MonitorAction::Continue);
    }

    // 回滚清空短期窗口但保留最佳值，保证后续比较仍有稳定锚点。
    #[test]
    fn test_monitor_reset_after_rollback() {
        let mut mon = TrainingMonitor::new(2, 0.1);
        mon.record_loss(1.0);
        mon.record_loss(1.0);
        mon.record_loss(5.0);
        mon.record_loss(5.0); // triggers rollback

        mon.reset_after_rollback();
        assert_eq!(mon.rolling_avg(), f32::MAX); // history cleared
        assert!((mon.best_avg_loss() - 1.0).abs() < 1e-6); // best preserved
    }

    // 样本计数用于外部监控吞吐，不能受窗口裁剪影响。
    #[test]
    fn test_monitor_sample_count() {
        let mut mon = TrainingMonitor::new(3, 0.2);
        mon.record_loss(1.0);
        mon.record_loss(2.0);
        mon.record_loss(3.0);
        assert_eq!(mon.total_samples(), 3);
    }
}
